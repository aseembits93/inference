"""Direct-model correctness check for RFDETR_USE_TRITON_PREPROC against
the F.interpolate baseline.

Bypasses InferencePipeline / the workflow engine and drives the TRT IS
model directly. Dumps per-frame dets to two jsonl files:

  correctness_direct_tensor_baseline.jsonl  (F.interpolate path)
  correctness_direct_numpy_triton.jsonl     (Triton preproc path)

and pair-diffs them with greedy IoU matching.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = str(REPO / ".venv" / "bin" / "python3")
DUMP = str(REPO / "development" / "stream_interface" / "correctness_direct_dump.py")

# Reuse the diff implementation from the workflow check so we stay in sync.
sys.path.insert(0, str(REPO / "development" / "stream_interface"))
from correctness_check import diff_dumps  # noqa: E402


def _env_common() -> dict:
    # Disable optional model-types whose deps are not installed in this venv.
    return {
        "QWEN_3_5_ENABLED": "false",
        "QWEN_3_ENABLED": "false",
        "QWEN_2_5_ENABLED": "false",
        "PALIGEMMA_ENABLED": "false",
        "FLORENCE2_ENABLED": "false",
        "CORE_MODEL_SAM_ENABLED": "false",
        "CORE_MODEL_SAM2_ENABLED": "false",
        "CORE_MODEL_SAM3_ENABLED": "false",
        "CORE_MODEL_GAZE_ENABLED": "false",
        "CORE_MODEL_CLIP_ENABLED": "false",
        "CORE_MODEL_OWLV2_ENABLED": "false",
        "CORE_MODEL_PE_ENABLED": "false",
        "CORE_MODEL_DOCTR_ENABLED": "false",
        "CORE_MODEL_EASYOCR_ENABLED": "false",
        "CORE_MODEL_TROCR_ENABLED": "false",
        "CORE_MODEL_GROUNDINGDINO_ENABLED": "false",
        "CORE_MODEL_YOLO_WORLD_ENABLED": "false",
        "SMOLVLM2_ENABLED": "false",
        "DEPTH_ESTIMATION_ENABLED": "false",
        "MOONDREAM2_ENABLED": "false",
        "GLM_OCR_ENABLED": "false",
        "SAM3_3D_OBJECTS_ENABLED": "false",
    }


def run_dump(env_extra: dict, video: str, model_id: str, confidence: float,
             dump_path: str, input_mode: str, max_frames: int, timeout_s: int) -> int:
    env = os.environ.copy()
    env.update(_env_common())
    env.update(env_extra)
    cmd = [
        PY,
        DUMP,
        "--video_reference", video,
        "--model_id", model_id,
        "--confidence", str(confidence),
        "--dump_path", dump_path,
        "--input_mode", input_mode,
    ]
    if max_frames > 0:
        cmd += ["--max_frames", str(max_frames)]
    print(f"[run] mode={input_mode} env={env_extra} -> {dump_path}", flush=True)
    # cwd set to REPO/development so the editable `inference_models` finder wins
    # over the outer namespace dir at /home/ubuntu/inference/inference_models.
    proc = subprocess.run(
        cmd, env=env, cwd=str(REPO / "development"), timeout=timeout_s,
    )
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_reference", default=str(REPO / "vehicles_312px.mp4"))
    ap.add_argument("--model_id", default="rfdetr-seg-nano")
    ap.add_argument("--confidence", type=float, default=0.4)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--conf_tol", type=float, default=1e-4)
    ap.add_argument("--iou_tol", type=float, default=0.5)
    ap.add_argument("--dump_dir", default=str(REPO))
    ap.add_argument("--diff_only", action="store_true")
    args = ap.parse_args()

    dump_dir = Path(args.dump_dir)
    a_path = str(dump_dir / "correctness_direct_tensor_baseline.jsonl")
    b_path = str(dump_dir / "correctness_direct_numpy_triton.jsonl")

    if not args.diff_only:
        # Baseline: tensor input -> F.interpolate. Triton preproc ineligible.
        rc = run_dump(
            {"RFDETR_USE_TRITON_PREPROC": "false",
             "RFDETR_TRITON_FULLPOSTPROC": "false"},
            args.video_reference, args.model_id, args.confidence,
            a_path, "tensor", args.max_frames, args.timeout,
        )
        if rc != 0:
            print(f"FATAL: baseline run exited {rc}", file=sys.stderr)
            return 3

        # Triton: numpy input + fast path enabled.
        rc = run_dump(
            {"RFDETR_USE_TRITON_PREPROC": "true",
             "RFDETR_TRITON_FULLPOSTPROC": "false"},
            args.video_reference, args.model_id, args.confidence,
            b_path, "numpy", args.max_frames, args.timeout,
        )
        if rc != 0:
            print(f"FATAL: triton run exited {rc}", file=sys.stderr)
            return 3

    return diff_dumps(a_path, b_path, args.conf_tol, args.iou_tol)


if __name__ == "__main__":
    raise SystemExit(main())
