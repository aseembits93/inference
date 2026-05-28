import argparse
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "development" / "stream_interface" / "rfdetr_nano_seg_trt_workflow.py"
VIDEO = ROOT / "vehicles_312px.mp4"
FPS_RE = re.compile(r"frames=\d+\s+elapsed=[0-9.]+s\s+fps=([0-9.]+)")


def run_once(env: dict, model_id: str, confidence: float) -> float:
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--video_reference",
        str(VIDEO),
        "--model_id",
        model_id,
        "--confidence",
        str(confidence),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    match = FPS_RE.search(proc.stdout)
    if match is None:
        raise RuntimeError(f"Could not parse fps from output:\n{proc.stdout}")
    return float(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--model_id", default="rfdetr-seg-nano")
    parser.add_argument("--confidence", type=float, default=0.4)
    args = parser.parse_args()

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", f"{ROOT}:{ROOT / 'inference_models'}")
    env.setdefault("RFDETR_TRITON_POSTPROC", "true")
    env.setdefault("INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED", "true")
    env.setdefault("RFDETR_PIPELINE_DEPTH", "2")
    env.setdefault("ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND", "true")
    env.setdefault("ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES", "true")

    fps_values = []
    for run_idx in range(args.runs):
        fps = run_once(
            env=env, model_id=args.model_id, confidence=args.confidence
        )
        fps_values.append(fps)
        print(f"run {run_idx + 1}: {fps:.2f} FPS", flush=True)

    mean_fps = statistics.mean(fps_values)
    median_fps = statistics.median(fps_values)
    min_fps = min(fps_values)
    max_fps = max(fps_values)
    print(
        f"summary: median={median_fps:.2f} mean={mean_fps:.2f} "
        f"min={min_fps:.2f} max={max_fps:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
