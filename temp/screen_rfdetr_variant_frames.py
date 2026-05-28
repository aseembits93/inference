import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import torch

from inference_models import AutoModel


VIDEO_PATH = Path("/home/ubuntu/inference/vehicles_312px.mp4")


@dataclass
class FramePrediction:
    frame_id: int
    n: int
    xyxy: List[List[int]]
    cls: List[int]
    conf: List[float]


def load_frames(frame_ids: Iterable[int]) -> Dict[int, object]:
    wanted = set(frame_ids)
    frames: Dict[int, object] = {}
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        index += 1
        if index in wanted:
            frames[index] = frame.copy()
            if len(frames) == len(wanted):
                break
    cap.release()
    missing = wanted.difference(frames)
    if missing:
        raise RuntimeError(f"Missing frames from video: {sorted(missing)}")
    return frames


def predict_frames(model_id: str, frame_ids: List[int]) -> Dict[int, FramePrediction]:
    model = AutoModel.from_pretrained(
        model_id_or_path=model_id,
        device=torch.device("cuda:0"),
        backend="trt",
    )
    frames = load_frames(frame_ids)
    result: Dict[int, FramePrediction] = {}
    for frame_id in frame_ids:
        detections = model.infer(frames[frame_id])[0]
        xyxy = detections.xyxy.detach().cpu().tolist()
        cls = detections.class_id.detach().cpu().tolist()
        conf = [float(x) for x in detections.confidence.detach().cpu().tolist()]
        result[frame_id] = FramePrediction(
            frame_id=frame_id,
            n=len(xyxy),
            xyxy=xyxy,
            cls=cls,
            conf=conf,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-id", required=True)
    parser.add_argument("--candidate-model-id", required=True)
    parser.add_argument("--frames", nargs="+", type=int, required=True)
    args = parser.parse_args()

    frame_ids = list(args.frames)
    base = predict_frames(args.base_model_id, frame_ids)
    candidate = predict_frames(args.candidate_model_id, frame_ids)

    for frame_id in frame_ids:
        b = base[frame_id]
        c = candidate[frame_id]
        print(f"frame {frame_id}")
        print(f"  base n={b.n} cls={b.cls} conf={[round(x, 6) for x in b.conf]}")
        print(f"  cand n={c.n} cls={c.cls} conf={[round(x, 6) for x in c.conf]}")
        print(f"  base boxes={b.xyxy}")
        print(f"  cand boxes={c.xyxy}")


if __name__ == "__main__":
    main()
