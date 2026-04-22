# TRT optimization scratch

One-off profiling and micro-benchmark scripts from **earlier autonomous
optimization sessions** targeting YOLOv8n, YOLOv8n-seg, and RF-DETR on
the TensorRT path. These predate the SAM3 TRT work
(`development/sam3_tensorrt/`) and the YOLO26 archive
(`development/yolo26_trt/`).

> **Provenance.** Timestamps and script content show these files come
> from a prior session (dated 2026-04-21). They are archived here for
> reference, not as a maintained utility set. Many assume specific
> engine cache paths or intermediate pipeline state that no longer
> exist; they are useful as *patterns* more than as runnable tools.

## What's in here

### `scripts/` — YOLOv8 / RF-DETR profiling workbench

Small scripts (20-100 lines each) used to find hot spots during the
YOLOv8n / YOLOv8n-seg / RF-DETR TRT optimization sessions. Each one
isolates a specific stage or asks a specific diagnostic question.

- **Preprocess profiling** — `profile_pre.py`, `profile_pre_rfdetr.py`,
  `profile_rfdetr_pre_detail.py`, `profile_batch.py`
- **Postprocess profiling** — `profile_post.py`, `profile_inner_post.py`,
  `profile_real_yolov8n_post.py`, `profile_adapter_post.py`,
  `profile_adapter_detail.py`
- **Per-stage breakdown** — `profile_stages_v2.py`,
  `profile_stages_v3.py`, `profile_real_image.py`,
  `profile_torch_detail.py`
- **Micro-benchmarks** — `bench_fuse_d2h.py`, `bench_fuse_d2h2.py`
  (tests whether fusing 3 D2H transfers into 1 helps)
- **Correctness gates** — `correctness_check.py`,
  `correctness_check_real.py` (IoU >= 0.95, class match, score drift
  on real images)
- **Misc diagnostics** — `debug_pinned.py` (who's allocating pinned
  memory?), `check_network_dtypes.py` (inspect a parsed ONNX's tensor
  dtype distribution)

### `next_priorities.md`

The prior session's planned next steps (user-directed, 2026-04-21):
P1 FP16 engine rebuild for RF-DETR / YOLOv8n, P2 EfficientNMS_TRT
plugin integration, P3 sparsity / tactic flags. None of these were
completed before the session ended.

## Why this is archived rather than deleted

The scripts are short enough to be useful as starting templates for
future TRT profiling, even if the specific invocations won't replay
cleanly. The patterns for wiring `torch.profiler` around
`inference.get_model(...).infer()`, for pinned-memory inspection, and
for the IoU + class-match correctness check are non-obvious to
reconstruct from scratch.

## Not runnable without setup

Most scripts load models via `inference.get_model(model_id=...)` which
needs:
- `ROBOFLOW_API_KEY` (or `API_KEY=None` for public models; some scripts
  rely on this)
- A GPU with enough memory for the target model
- Hardcoded image paths in some scripts — these have been rewritten
  to relative `tests/inference/models_predictions_tests/assets/...`
  paths that resolve from the repo root.

## Relation to the other PRs

- `development/sam3_tensorrt/` (PR #17) — SAM3 TRT export pipeline, the
  current session's deliverable.
- `development/yolo26_trt/` (PR #18) — YOLO26 TRT optimization attempt
  (prior session, reached engine-build phase).
- **This directory** (PR #19) — smallest, oldest fragments from even
  earlier sessions. Read `next_priorities.md` first to get the
  original intent.
