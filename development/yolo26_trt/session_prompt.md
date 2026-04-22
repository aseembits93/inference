# YOLO26 TRT Optimization Session

AUTONOMOUS MODE: Work fully autonomously. Do NOT ask the user any questions. Time is no limit — iterate to plateau. Make all decisions yourself and document them in HANDOFF.md.

## Task

End-to-end optimization of the `.infer()` method of YOLO26 models on the TensorRT GPU path. Same methodology as the prior sessions in this repo.

## Prior state (branch: codeflash/optimize)

Three sessions have already landed ~10 commits optimizing shared TRT preprocess/postprocess infrastructure for YOLOv8n, YOLOv8n-seg, and RF-DETR. Since YOLO26 likely reuses that shared preprocessing (`inference_models/models/common/roboflow/pre_processing.py`) and postprocessing (same NMS/rescale path), it probably already benefits from:
- Pinned staging buffer cache in TRT preprocess
- Cached per-channel mean/std for normalize
- Strided scalar rescale
- Shared `nonzero` + packed `index_select` in NMS

A concurrent FP16/EfficientNMS agent is currently running on YOLOv8n/RF-DETR engine-level work. Do NOT rebuild YOLOv8n engines or touch post_processing.py in a way that would conflict with it — focus on YOLO26-specific wins and any preprocessing/postprocessing that YOLO26 uses but the prior sessions didn't profile.

A concurrent SAM3 session is also running — ignore anything SAM3-related.

Existing state in `.codeflash/` (HANDOFF.md, results.tsv, changelog.md, `bench_yolo26.py` — the prior agent left a benchmark script for YOLO26, check it out).

## Your workflow

1. **Locate YOLO26 variants in the repo.** There are multiple (object detection, segmentation, keypoints). Find the TRT backends.
2. **Pick the heaviest one as primary target.** Benchmark single-image and batch=8 on real images. Establish baseline with warmup + mean ± std.
3. **Profile** with torch.profiler (and nsys if useful) to find YOLO26-specific hotspots:
   - Preprocess path (might differ from YOLOv8 — check input shape, letterbox/stretch choice, any YOLO26-specific normalization)
   - TRT engine execution (kernels, H2D/D2H)
   - Postprocess (different detection head, possibly different anchor/decode math, NMS)
4. **Compare against YOLOv8n** — where is YOLO26 slower/faster? Is there a per-model hotspot that shared optimizations don't cover?
5. **Run experiments.** Typical candidates:
   - Pinned memory if any preprocess path bypasses the shared optimization
   - Cached tensors for any YOLO26-specific constants (anchor grids, stride tensors, class-index tensors)
   - Eliminate H2D/D2H in YOLO26-specific postprocess
   - torch.compile on hot paths (be careful with correctness and recompile overhead)
   - Buffer reuse in the adapter class
6. **Correctness gate.** For each kept change, verify detections match baseline within tolerance on real images (IoU >= 0.95 on kept boxes, class-match >= 99%, score drift <= 1%).
7. **Commit each accepted experiment separately** on `codeflash/optimize`. Label experiments `yolo26-exp001`, etc.
8. **Update** `.codeflash/HANDOFF.md`, `.codeflash/changelog.md`, `.codeflash/results.tsv` as you go.
9. **Stop** only when you've hit a genuine plateau (5 consecutive failed experiments, no remaining hotspot in profile).

## Deliverables

At end: baseline vs optimized for each YOLO26 variant benchmarked (single + batch=8), list of kept vs discarded experiments, and a short note on whether YOLO26-specific wins exist beyond the shared infra.

## Reporting

When sending progress messages back to the team lead, be concise. The lead has file access — confirm completion and flag issues, don't restate file contents or list every section you wrote.

## Environment

From setup.md:
- Python 3.12.3
- PyTorch 2.10.0+cu128
- TensorRT 10.12.0.36
- Test command: /home/ubuntu/inference/.venv/bin/pytest
- Virtual environment: /home/ubuntu/inference/.venv/

## Conventions

From conventions.md:
- Session Configuration (Autonomous Mode)
- Run tag: 2026-04-21
- Target: Heaviest model's .infer() method
- Focus: TensorRT GPU performance optimization
- Guard Command: Not specified (run tests after each optimization)

## Session History

From results.tsv and HANDOFF.md:
- Prior sessions optimized YOLOv8n, YOLOv8n-seg, RF-DETR
- 9 optimizations committed to shared infrastructure
- Cumulative improvements: 4.8-8.3% E2E speedup
- All correctness tests passing (247 preprocess tests)

Key prior optimizations that YOLO26 may already benefit from:
1. Pinned staging buffer cache (6c45a8265)
2. Strided scalar rescale (66724c1da, e02b83526, 0fad67e05)
3. Single nonzero + packed gather in NMS (97b52ad26, 6f42f447d)
4. Cached normalize constants (bd4599538)
5. Cached arange indices (3c710460b)

## Domain Knowledge: GPU/CUDA Optimization

Key principles from references/gpu/guide.md:
- Always use torch.profiler for GPU workloads (not cProfile)
- Look for: H2D/D2H transfers, kernel dispatch gaps, stream sync stalls, pipeline bubbles
- Per-stage measurement with torch.cuda.Event
- Warm-up required (10-20 iterations) before benchmarking
- Focus on: pinned memory, kernel fusion, cached constants, non_blocking transfers
- Monolithic kernel strategy when piecemeal optimizations don't register in E2E

## YOLO26 TRT Files Located

Three TRT backends:
- inference_models/inference_models/models/yolo26/yolo26_object_detection_trt.py
- inference_models/inference_models/models/yolo26/yolo26_instance_segmentation_trt.py
- inference_models/inference_models/models/yolo26/yolo26_key_points_detection_trt.py

Benchmark script already exists:
- .codeflash/bench_yolo26.py

## Related Repositories

None detected.

## Library Research

N/A (not using context7 in this session)
