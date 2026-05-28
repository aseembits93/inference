import argparse
import os
from pathlib import Path
from typing import Dict, List

import cv2

from inference.core.models.inference_models_adapters import (
    InferenceModelsInstanceSegmentationAdapter,
)


VIDEO_PATH = Path("/home/ubuntu/inference/vehicles_312px.mp4")


def load_frames(frame_ids: List[int]) -> Dict[int, object]:
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
        raise RuntimeError(f"Missing frames: {sorted(missing)}")
    return frames


def summarize_det(det) -> dict:
    xyxy = det.xyxy.detach().cpu().round().int().tolist()
    conf = [round(float(x), 6) for x in det.confidence.detach().cpu().tolist()]
    cls = [int(x) for x in det.class_id.detach().cpu().tolist()]
    return {
        "n": len(xyxy),
        "cls": cls,
        "conf": conf,
        "xyxy": xyxy,
        "defer_count_to_adapter": getattr(det, "_defer_count_to_adapter", None),
        "combined_gpu_shape": (
            list(getattr(det, "_combined_gpu").shape)
            if getattr(det, "_combined_gpu", None) is not None
            else None
        ),
        "counter_gpu": (
            int(getattr(det, "_counter_gpu").detach().cpu().item())
            if getattr(det, "_counter_gpu", None) is not None
            else None
        ),
    }


def summarize_response(response) -> dict:
    preds = response.predictions
    return {
        "n": len(preds),
        "cls": [int(p.class_id) for p in preds],
        "conf": [round(float(p.confidence), 6) for p in preds],
        "xyxy": [
            [
                int(round(float(p.x) - float(p.width) / 2.0)),
                int(round(float(p.y) - float(p.height) / 2.0)),
                int(round(float(p.x) + float(p.width) / 2.0)),
                int(round(float(p.y) + float(p.height) / 2.0)),
            ]
            for p in preds
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--frame-a", type=int, required=True)
    parser.add_argument("--frame-b", type=int, required=True)
    parser.add_argument("--confidence", type=float, default=0.4)
    args = parser.parse_args()

    os.environ.setdefault("RFDETR_TRITON_POSTPROC", "true")
    os.environ.setdefault("INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED", "true")
    os.environ.setdefault("RFDETR_PIPELINE_DEPTH", "2")
    os.environ.setdefault("ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND", "true")
    os.environ.setdefault(
        "ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES", "true"
    )

    frames = load_frames([args.frame_a, args.frame_b])
    adapter = InferenceModelsInstanceSegmentationAdapter(model_id=args.model_id)
    common_kwargs = {"confidence": args.confidence}

    # Frame A primes the depth-2 pipeline and emits an empty placeholder.
    img_a, meta_a = adapter.preprocess(frames[args.frame_a], **common_kwargs)
    pred_a = adapter.predict(img_a, **common_kwargs)
    resp_a = adapter.postprocess(pred_a, meta_a, **common_kwargs)

    # Frame B's predict returns the future for frame A. That is the object
    # whose raw detections and final response we want to compare.
    img_b, meta_b = adapter.preprocess(frames[args.frame_b], **common_kwargs)
    pred_b = adapter.predict(img_b, **common_kwargs)
    fut = pred_b
    prev_meta = getattr(adapter, "_pending_flush_meta_prev", None)
    mapped_kwargs = getattr(fut, "_adapter_kwargs", {}).get("mapped_kwargs", {})
    fut._meta = prev_meta  # type: ignore[attr-defined]
    fut._kwargs = mapped_kwargs  # type: ignore[attr-defined]
    raw_det = fut.result()[0]
    final_resp = adapter.postprocess(pred_b, meta_b, **common_kwargs)

    print("frame_a_placeholder", summarize_response(resp_a[0]))
    print("frame_a_raw_det", summarize_det(raw_det))
    print("frame_a_response", summarize_response(final_resp[0]))

    # Drain frame B as well so callers can inspect the queued tail if needed.
    flushed = adapter.flush()
    if flushed:
        print("frame_b_flush_response", summarize_response(flushed[0]))


if __name__ == "__main__":
    main()
