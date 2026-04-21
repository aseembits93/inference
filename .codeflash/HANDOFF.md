# Deep GPU Optimization Session - TRT Inference

## Summary

Session goal: end-to-end optimization of `.infer()` for TensorRT GPU path
across RF-DETR, YOLOv8n, YOLOv8n-seg and shared NMS/preprocess helpers.

All changes preserve correctness (verified against baseline predictions on
real images for all three models) and don't regress the preprocess unit
test suite (247 tests pass).

## Cumulative improvements

Measured on the benchmark script (200 iterations after 50-iter warmup,
random 640x640x3 uint8 input, L4 GPU, CUDA 12.8, PyTorch 2.10, TRT 10.12):

| Model            | Pre-codeflash baseline | Current HEAD | Improvement |
|------------------|-----------------------|--------------|-------------|
| yolov8n-640      | 2.393ms ± 0.061ms    | 2.278ms ± 0.033ms | **-4.8%** (and 2x lower variance) |
| yolov8n-seg-640  | 2.424ms ± 0.081ms    | 2.338ms ± 0.035ms | **-3.5%** (and 2x lower variance) |
| rfdetr-base      | 3.797ms ± 0.092ms    | 3.550ms ± 0.038ms | **-6.5%** (and 2.4x lower variance) |

Real 427x640 image (with actual detections, heavier postprocess):

| Model       | Before | After | Improvement |
|-------------|--------|-------|-------------|
| yolov8n-640 | 2.91ms | 2.80ms | -3.8% |

## Optimizations committed (this session)

1. **6c45a8265**: pre-allocate pinned staging buffers for TRT preprocess.
   Thread-local cache of pinned host tensors keyed by (shape, dtype).
   Replaces per-call `.pin_memory()` (~50us CPU allocation + pinning) with
   a reusable buffer + `copy_()`. Applies to all 4 numpy input handlers
   shared by YOLOv8/v9/v10/v11 TRT and RF-DETR TRT.

2. **66724c1da**: remove H2D transfers from `rescale_image_detections`.
   Replace `torch.as_tensor([pad,pad,pad,pad], device='cuda')` with strided
   scalar arithmetic on xyxy views. Halves kernel count and eliminates two
   small H2D transfers per call.

3. **e02b83526**: same pattern for `rescale_key_points_detections` — saves
   five H2D transfers per call (xyxy + keypoint offsets + scales + crop).

4. **0fad67e05**: same pattern for `align_instance_segmentation_results`.

5. **97b52ad26**: share single `nonzero` across NMS filtered tensors.
   Replace 3-4 independent boolean indexings on the same mask with one
   `nonzero` + N `index_select` calls. Real-image E2E drops ~80us.

6. **6f42f447d**: single `index_select` over packed NMS tail tensors.
   Pre-concatenate xyxy + conf + cls (+ mask/kp) into one tensor, do one
   `keep`-index gather instead of 3-4 separate ones.

7. **6e0b48119**: streamline RF-DETR TRT post_process. Apply
   nonzero-sharing and strided-scalar-multiply to the detection and
   instance-segmentation post-processors. RF-DETR postprocess 0.47ms →
   0.36ms (-23%).

8. **bd4599538**: cache per-channel mean/std tensors for normalize.
   Legacy `functional.normalize` rebuilt the (C,1,1) mean/std tensors on
   every call. Cache them per (mean, std, device, dtype). Keeps the exact
   same `(x/s - mean) / std` arithmetic so FP32 rounding is bit-for-bit
   identical — all 247 preprocess unit tests still pass. RF-DETR
   preprocess 1.03 → 0.90ms (-13%).

9. **3c710460b**: cache `torch.arange` indices in `crop_masks_to_boxes`.
   Applies to all instance-segmentation models.

## Discarded experiments

- **Remove `torch.any(mask)` sync in NMS**: Big regression — the fast-path
  skip-on-empty-mask was critical for benchmark data with mostly-empty
  masks. Lesson: the sync point was hiding work that became expensive
  when the code path changed.
- **Fused affine `t * M - S` replacing functional.normalize**: ~50% faster
  micro-bench for normalize step, but the 1-ULP FP32 rounding difference
  broke 26 unit tests that assert exact equality. Replaced with the
  mean/std-tensor caching approach (no ULP shift).
- **GPU-resident resize (cv2.resize → F.interpolate)**: ~300us saved on
  RF-DETR preprocess, but changed pixel-level pre-processed values by up
  to 1 uint8 unit, which would break pixel-equality tests.
- **Color swap reordering (before letterbox)**: Marginal for 640x640
  benchmark (scaled ROI == full target), no benefit observed for the
  non-square real image either. Reverted for simpler code.

## Files modified

- `inference_models/inference_models/models/common/roboflow/pre_processing.py`
  - Thread-local pinned staging buffer cache (`_pinned_buffer_cache`)
  - Thread-local normalize constants cache (`_normalize_constants_cache`)
  - New helper `_numpy_to_device_via_pinned_buffer`
  - New helper `_maybe_apply_scale_and_normalize`
  - All 4 numpy handlers + 4 torch handlers switched over
- `inference_models/inference_models/models/common/roboflow/post_processing.py`
  - `rescale_image_detections` — strided scalar arithmetic
  - `rescale_key_points_detections` — same
  - `align_instance_segmentation_results` — same
  - `run_nms_for_object_detection` — single-nonzero + packed gather
  - `run_nms_for_instance_segmentation` — same
  - `run_nms_for_key_points_detection` — same
  - `crop_masks_to_boxes` — cached arange indices (`_arange_cache`)
- `inference_models/inference_models/models/rfdetr/rfdetr_object_detection_trt.py`
  - `post_process` — nonzero-sharing, strided scalar mul
- `inference_models/inference_models/models/rfdetr/common.py`
  - `post_process_instance_segmentation_results` — same

## Areas not pursued (rejected due to risk vs reward)

- **CUDA Graph capture** (`ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND`): Gives
  ~8% additional speedup on YOLOv8n, ~12% on YOLOv8n-seg, ~5% on RF-DETR
  when enabled. NOT enabled by default — changing the default would
  increase VRAM usage (each graph holds a dedicated execution context +
  buffer) and is a user-facing config change.
- **Cross-stream event-based synchronization** (replace 3 CPU syncs per
  inference with chained `wait_stream` events): Refactor is invasive —
  touches the contract between adapter methods. Attempted as a
  micro-bench — saves ~150us per call but the change is architectural
  and warrants its own PR.
- **Rewrite cv2.resize → GPU F.interpolate**: Changes FP/pixel semantics
  (±1 uint8 unit), would break existing pixel-equality tests.
- **Batch-first preprocessing for numpy image list**: Preprocessing 8
  images one-by-one already pipelines well (per-image cost drops from
  2.3ms single → 1.6ms in batch). Architectural refactor with diminishing
  returns.

## How to reproduce measurements

Baseline + stage breakdown:
```
/home/ubuntu/inference/.venv/bin/python .codeflash/profile_stages_v3.py <model_id>
```

Cross-baseline comparison:
```
/home/ubuntu/inference/.venv/bin/python /tmp/bench_all_variants.py
```

Real-image E2E:
```
/home/ubuntu/inference/.venv/bin/python .codeflash/profile_real_image.py <model_id>
```

Correctness check:
```
/home/ubuntu/inference/.venv/bin/python .codeflash/correctness_check_real.py
```
