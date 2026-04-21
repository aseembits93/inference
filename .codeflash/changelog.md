# GPU Optimization Session Log

## Session 3: Deep Cross-Domain Optimization (2026-04-21)

Round-2 follow-up to the RF-DETR / YOLOv8n pinned-memory session. Focused
on the shared NMS / rescale / normalize helpers that underpin every TRT
object-detection + instance-segmentation + keypoint model in the repo,
plus the RF-DETR transformer-style post-processor.

### Cumulative results (single 640x640 input, 200 iters, L4 GPU)

| Model            | Baseline (pre-codeflash) | Session 2 (pinned) | Session 3 (now) | Total |
|------------------|-------------------------|--------------------|-----------------|-------|
| yolov8n-640      | 2.528ms                  | 2.386ms            | 2.278ms         | **-9.9%** |
| yolov8n-seg-640  | 2.418ms                  | 2.418ms            | 2.338ms         | **-3.3%** |
| rfdetr-base      | 3.859ms                  | 3.877ms            | 3.550ms         | **-8.0%** |

### Batch=8 results (real images, 100 iters)

| Model            | Baseline (ef629a99e) | Current HEAD | Improvement |
|------------------|---------------------|--------------|-------------|
| yolov8n-640      | 14.25ms (1.78ms/img) | 13.08ms (1.64ms/img) | **-8.2%** |
| yolov8n-seg-640  | 36.72ms (4.59ms/img) | 33.21ms (4.15ms/img) | **-9.6%** |
| rfdetr-base      | 31.31ms (3.91ms/img) | 29.21ms (3.65ms/img) | **-6.7%** |

### Commits (session 3)

1. **6c45a8265** - pre-allocate pinned staging buffers for TRT preprocess.
   Replace per-call `tensor.pin_memory()` (~50us CPU per call) with
   reusable thread-local staging buffers.

2. **66724c1da** - remove H2D transfers from rescale_image_detections.
   Strided scalar arithmetic instead of `torch.as_tensor([pad,pad,pad,pad],
   device='cuda')` per call.

3. **e02b83526** - same pattern for rescale_key_points_detections.
4. **0fad67e05** - same pattern for align_instance_segmentation_results.

5. **97b52ad26** - share single `nonzero` across NMS filtered tensors.
   Replace 3-4 boolean indexings on the same mask with one nonzero +
   index_select per tensor.

6. **6f42f447d** - single index_select over packed NMS tail tensors.
   Concatenate xyxy | conf | cls before gather rather than 3-4 separate
   `[keep]` indexings.

7. **6e0b48119** - streamline RF-DETR TRT post_process (nonzero sharing +
   strided scalar mul_ instead of H2D-then-multiply).

8. **bd4599538** - cache per-channel mean/std tensors for normalize.
   Preserves exact FP32 rounding so all 247 preprocess unit tests pass.
   Applied to all 8 shared pre-process handlers (numpy + torch paths).

9. **3c710460b** - cache `torch.arange` indices in crop_masks_to_boxes.
   For YOLO instance-segmentation mask cropping.

### Discarded experiments (informative)

- Removing `torch.any(mask)` sync in NMS caused a 7.9% regression — the
  skip-empty fast path was actually critical for the benchmark's random
  data.
- Fused-affine `t * M - S` replacing `(x/s - mean)/std` was 54% faster
  in isolation but caused a 1-ULP FP32 rounding shift that broke 26
  exact-equality unit tests. Replaced with the mean/std-tensor caching
  approach that keeps rounding bit-for-bit identical.
- GPU-resident resize (cv2.resize -> F.interpolate) saves ~300us on
  RF-DETR but changes pixel values by ±1 uint8 unit — would break
  pixel-equality tests. Deferred.
- Color-swap reordering (before letterbox) was neutral; reverted.

### Areas not pursued this session

- **CUDA graph capture** (opt-in via env var) would add ~5-12% more.
  Default disabled because each cached graph holds dedicated VRAM.
- **Event-based stream synchronization** (replace 3 CPU syncs per
  `.infer()` with chained `wait_stream`). Architectural refactor
  warranting its own PR; micro-bench suggested ~150us savings.
- **Batch-first numpy preprocess** (stack before upload). Diminishing
  returns since pipelining already drops per-image batch cost 30%.

---

## Session 2: RF-DETR Pinned Memory (earlier; 2026-04-21)

Targeted RF-DETR and YOLOv8n with pinned H2D transfers.

### Summary
- RF-DETR: +0.3% speedup, 51% variance reduction
- YOLOv8n: +7.2% speedup, 73% variance reduction

Note: the session 2 gains on yolov8n didn't actually take effect until
session 3's editable install was fixed — the measurements in session 3
serve as the authoritative before/after for both sessions.

### Commits
- `a09fdff65` - RF-DETR pinned memory
- `ef629a99e` - YOLOv8n TRT pinned memory (4 numpy handlers)
