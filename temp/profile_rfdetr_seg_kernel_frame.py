import os
import time

import cv2
import torch

from inference_models import AutoModel


VIDEO_PATH = os.environ.get(
    "VIDEO_PATH", "/home/ubuntu/inference/vehicles_312px.mp4"
)
DEVICE = os.environ.get("DEVICE", "cuda:0")
WARMUP = int(os.environ.get("WARMUP", "10"))
CYCLES = int(os.environ.get("CYCLES", "40"))
MODEL_ID = os.environ.get("MODEL_ID", "rfdetr-seg-nano")


def load_frame(video_path: str):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read first frame from {video_path}.")
    return frame


def main() -> None:
    frame = load_frame(VIDEO_PATH)
    model = AutoModel.from_pretrained(
        model_id_or_path=MODEL_ID,
        device=torch.device(DEVICE),
        backend="trt",
    )

    for _ in range(WARMUP):
        _ = model(frame)

    start = time.perf_counter()
    for _ in range(CYCLES):
        _ = model(frame)
    elapsed = time.perf_counter() - start
    fps = CYCLES / elapsed if elapsed > 0 else 0.0
    print(f"cycles={CYCLES} elapsed={elapsed:.4f}s fps={fps:.2f}")


if __name__ == "__main__":
    main()
