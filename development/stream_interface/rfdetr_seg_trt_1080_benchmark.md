# RF-DETR Seg TensorRT 1080p Variant Benchmark

This note records the June 4, 2026 check for the largest RF-DETR segmentation
variant that can run the `vehicles_1080p.mp4` stream workflow at 30 FPS on the
Jetson Orin NX 8GB target used for PR 2405.

## Context

The public non-nano RF-DETR segmentation TensorRT packages are built for L4/T4,
so they are not directly loadable on Jetson Orin. For this benchmark, local Orin
FP16 TensorRT packages were compiled from the public ONNX packages and wired into
the workflow as untracked local directories.

The Triton sparse RLE postprocess path previously rejected non-nano mask sizes
because it scanned the source mask with one Triton vector and capped source mask
area below the `small` model's 96x96 mask. The current patch adds a tiled source
mask bounds pass and raises the supported sparse path shape limit to RF-DETR Seg
2XLarge's 192x192 mask with 300 queries and COCO class logits.

## Benchmark Command

Use the stream workflow with the optimization flags enabled:

```bash
env \
  PYTHONPATH=/app/helloworld/inference/inference_models:/app/helloworld/inference \
  USE_INFERENCE_MODELS=True \
  ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES=True \
  ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES=True \
  INFERENCE_MODELS_RFDETR_TRITON_POSTPROC_ENABLED=true \
  INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED=true \
  RFDETR_PIPELINE_DEPTH=2 \
  ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND=true \
  python development/stream_interface/rfdetr_nano_seg_trt_workflow.py \
    --video_reference vehicles_1080p.mp4 \
    --model_id rfdetr-seg-large/1 \
    --backend trt
```

Change `--model_id` to the local package alias for each variant. A depth-3
sanity run was also performed for `xlarge`.

## Results

| Variant | Input size | Pipeline depth | FPS |
| --- | ---: | ---: | ---: |
| `rfdetr-seg-small/1` | 384 | 2 | 63.85 |
| `rfdetr-seg-large/1` | 504 | 2 | 35.49 |
| `rfdetr-seg-xlarge/1` | 624 | 2 | 20.94 |
| `rfdetr-seg-xlarge/1` | 624 | 3 | 20.91 |
| `rfdetr-seg-2xlarge/1` | 768 | 2 | 12.90 |

`large` is the largest tested non-nano RF-DETR Seg variant that clears 30 FPS on
this 1080p workload with all optimization flags enabled. `xlarge` remains below
30 FPS even when increasing pipeline depth from 2 to 3.

## Verification

The focused postprocess test suite passed after the 2XLarge shape-limit patch:

```bash
PYTHONPATH=/app/helloworld/inference/inference_models:/app/helloworld/inference \
  python -m pytest tests/unit_tests/models/rfdetr/test_triton_postprocess.py
```

Result:

```text
24 passed, 23 warnings
```
