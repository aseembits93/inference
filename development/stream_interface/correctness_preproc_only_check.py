"""Runs correctness_preproc_only_dump.py for ref and triton preproc,
then diffs. Only the preproc function differs between runs; forward()
and post_process() are identical.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = str(REPO / ".venv" / "bin" / "python3")
DUMP = str(REPO / "development" / "stream_interface" / "correctness_preproc_only_dump.py")

sys.path.insert(0, str(REPO / "development" / "stream_interface"))
from correctness_check import diff_dumps  # noqa: E402


def _env_common() -> dict:
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


def run_dump(preproc: str, video: str, model_id: str, confidence: float,
             dump_path: str, max_frames: int, timeout_s: int) -> int:
    env = os.environ.copy()
    env.update(_env_common())
    cmd = [
        PY, DUMP,
        "--video_reference", video,
        "--model_id", model_id,
        "--confidence", str(confidence),
        "--dump_path", dump_path,
        "--preproc", preproc,
    ]
    if max_frames > 0:
        cmd += ["--max_frames", str(max_frames)]
    print(f"[run] preproc={preproc} -> {dump_path}", flush=True)
    proc = subprocess.run(cmd, env=env, cwd=str(REPO / "development"),
                          timeout=timeout_s)
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
    a_path = str(dump_dir / "correctness_preproc_only_ref.jsonl")
    b_path = str(dump_dir / "correctness_preproc_only_triton.jsonl")

    if not args.diff_only:
        rc = run_dump("ref", args.video_reference, args.model_id,
                      args.confidence, a_path, args.max_frames, args.timeout)
        if rc != 0:
            print(f"FATAL: ref run exited {rc}", file=sys.stderr)
            return 3
        rc = run_dump("triton", args.video_reference, args.model_id,
                      args.confidence, b_path, args.max_frames, args.timeout)
        if rc != 0:
            print(f"FATAL: triton run exited {rc}", file=sys.stderr)
            return 3

    return diff_dumps(a_path, b_path, args.conf_tol, args.iou_tol)


if __name__ == "__main__":
    raise SystemExit(main())
