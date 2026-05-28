import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, List, Optional

from inference import InferencePipeline
from inference.core.interfaces.stream.watchdog import BasePipelineWatchDog


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = ROOT / "vehicles_312px.mp4"


@dataclass
class FramePrediction:
    frame_id: int
    n: int
    cls: List[int]
    conf: List[float]
    xyxy: List[List[int]]


class _Recorder:
    def __init__(self, frames: Optional[Iterable[int]] = None) -> None:
        self._wanted = set(frames) if frames is not None else None
        self.records: Dict[int, FramePrediction] = {}
        self._lock = Lock()

    def sink(self, prediction: dict, video_frame) -> None:
        frame_id = int(video_frame.frame_id)
        if self._wanted is not None and frame_id not in self._wanted:
            return
        preds = prediction["predictions"]
        record = FramePrediction(
            frame_id=frame_id,
            n=len(preds),
            cls=[int(p["class_id"]) for p in preds],
            conf=[float(p["confidence"]) for p in preds],
            xyxy=[
                [
                    int(round(float(p["x"]) - (float(p["width"]) / 2.0))),
                    int(round(float(p["y"]) - (float(p["height"]) / 2.0))),
                    int(round(float(p["x"]) + (float(p["width"]) / 2.0))),
                    int(round(float(p["y"]) + (float(p["height"]) / 2.0))),
                ]
                for p in preds
            ],
        )
        with self._lock:
            self.records[frame_id] = record


def _max_box_drift(left: List[int], right: List[int]) -> int:
    return max(abs(int(l) - int(r)) for l, r in zip(left, right))


def _class_box_matches(
    base: FramePrediction,
    candidate: FramePrediction,
    box_drift_px: int,
) -> List[dict]:
    pairs = []
    for base_idx, base_cls in enumerate(base.cls):
        for candidate_idx, candidate_cls in enumerate(candidate.cls):
            if int(base_cls) != int(candidate_cls):
                continue
            drift = _max_box_drift(
                base.xyxy[base_idx],
                candidate.xyxy[candidate_idx],
            )
            if drift <= box_drift_px:
                pairs.append((drift, base_idx, candidate_idx))

    matches = []
    used_base = set()
    used_candidate = set()
    for drift, base_idx, candidate_idx in sorted(pairs):
        if base_idx in used_base or candidate_idx in used_candidate:
            continue
        used_base.add(base_idx)
        used_candidate.add(candidate_idx)
        matches.append(
            {
                "base_idx": base_idx,
                "candidate_idx": candidate_idx,
                "drift": drift,
            }
        )
    return matches


def collect_predictions(
    model_id: str,
    video_reference: str,
    frames: Optional[List[int]] = None,
    confidence: Optional[float] = 0.4,
) -> Dict[int, FramePrediction]:
    recorder = _Recorder(frames=frames)
    watchdog = BasePipelineWatchDog()
    pipeline = InferencePipeline.init(
        model_id=model_id,
        video_reference=video_reference,
        confidence=confidence,
        on_prediction=recorder.sink,
        watchdog=watchdog,
    )
    pipeline.start()
    pipeline.join()
    return recorder.records


def compare_predictions(
    base: Dict[int, FramePrediction],
    candidate: Dict[int, FramePrediction],
    box_drift_px: int,
) -> List[dict]:
    mismatches = []
    for frame_id in sorted(set(base).intersection(candidate)):
        b = base[frame_id]
        c = candidate[frame_id]
        matches = _class_box_matches(b, c, box_drift_px=box_drift_px)
        relaxed_frame_match = b.n == c.n == len(matches)
        if relaxed_frame_match:
            continue
        matched_drifts = [match["drift"] for match in matches]
        mismatches.append(
            {
                "frame_id": frame_id,
                "base_n": b.n,
                "candidate_n": c.n,
                "relaxed_matched": len(matches),
                "relaxed_mean_box_drift": (
                    round(sum(matched_drifts) / len(matched_drifts), 3)
                    if matched_drifts
                    else None
                ),
                "relaxed_max_box_drift": max(matched_drifts) if matched_drifts else None,
                "base_cls": b.cls,
                "candidate_cls": c.cls,
                "base_xyxy": b.xyxy,
                "candidate_xyxy": c.xyxy,
                "base_conf": [round(x, 6) for x in b.conf],
                "candidate_conf": [round(x, 6) for x in c.conf],
            }
        )
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-id", required=True)
    parser.add_argument("--candidate-model-id", required=True)
    parser.add_argument("--video_reference", default=str(DEFAULT_VIDEO))
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--frames", nargs="*", type=int, default=None)
    parser.add_argument(
        "--box-drift-px",
        type=int,
        default=int(os.environ.get("DIRECT_STREAM_PARITY_BOX_DRIFT_PX", "5")),
    )
    args = parser.parse_args()

    os.environ.setdefault(
        "PYTHONPATH", f"{ROOT}:{ROOT / 'inference_models'}"
    )
    os.environ.setdefault("RFDETR_TRITON_POSTPROC", "true")
    os.environ.setdefault(
        "INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED", "true"
    )
    os.environ.setdefault("RFDETR_PIPELINE_DEPTH", "2")
    os.environ.setdefault("ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND", "true")
    os.environ.setdefault(
        "ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES", "true"
    )

    base = collect_predictions(
        model_id=args.base_model_id,
        video_reference=args.video_reference,
        frames=args.frames,
        confidence=args.confidence,
    )
    candidate = collect_predictions(
        model_id=args.candidate_model_id,
        video_reference=args.video_reference,
        frames=args.frames,
        confidence=args.confidence,
    )
    mismatches = compare_predictions(
        base=base,
        candidate=candidate,
        box_drift_px=args.box_drift_px,
    )
    print(
        f"frames_compared={len(set(base).intersection(candidate))} "
        f"mismatches={len(mismatches)} "
        f"box_drift_px={args.box_drift_px}"
    )
    for mismatch in mismatches[:25]:
        print(mismatch)


if __name__ == "__main__":
    main()
