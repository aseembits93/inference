# YOLO26 TensorRT optimization

Scripts and session notes from a prior optimization pass on the YOLO26 TRT
inference path (`inference_models/models/yolo26/*_trt.py`). This work
**predates the SAM3 TRT work in `development/sam3_tensorrt/`** and is
preserved here rather than left in the untracked `.codeflash/` scratch
directory.

> **Provenance note.** This cluster of scripts was produced in an earlier
> autonomous session (commit history / timestamps predate the SAM3 work).
> It is checked in as an archival record, not as a promise of
> reproducibility — the scripts reference `~/.cache/roboflow/...` engine
> paths and assume the prior session's L4 state. They will need rebuilt
> engines to run again.

## Files

- [SESSION_STATUS.md](SESSION_STATUS.md) — original progress log with
  objective, phases, environment, and open questions.
- [session_prompt.md](session_prompt.md) — the autonomous-mode prompt
  the session started from.

### `scripts/`

- `build_yolo26_engines.py` — builds TRT engines (FP16, max batch 8,
  8 GB workspace) from the three YOLO26 variants' ONNX files.
- `bench_yolo26_comprehensive.py` — first benchmark attempt (failed due
  to ONNX packaging mismatch).
- `bench_yolo26_all_variants.py` — second attempt using prebuilt TRT
  packages (failed because engines were built for a different GPU
  compute capability).
- `bench_yolo26_final.py` — the working benchmark against locally-built
  engines across all three variants (object detection, instance
  segmentation, keypoints).
- `profile_yolo26.py` — torch.profiler-based stage breakdown for
  identifying YOLO26-specific hotspots.
- `compare_yolo26_vs_yolov8.py` — differential profile vs YOLOv8n to
  find YOLO26-only slowdowns.

## Shared infrastructure context

YOLO26 sits on shared preprocessing / postprocessing helpers that were
optimized in prior sessions (YOLOv8n, YOLOv8n-seg, RF-DETR). See
`SESSION_STATUS.md` § "Shared Infrastructure Already Optimized" for the
specific commits. The scripts here were intended to determine whether
YOLO26 has additional YOLO26-specific wins beyond those shared gains.

## Status at hand-off

Phase 2 (engine build) was in progress when the session wrapped; Phases
3 (baseline), 4 (profile), and 5 (optimization loop) are planned but not
executed in this directory. `SESSION_STATUS.md` has the full plan.
