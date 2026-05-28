import argparse
import os
import time
import traceback
from collections import defaultdict
from time import perf_counter

os.environ.setdefault(
    "ONNXRUNTIME_EXECUTION_PROVIDERS",
    "[TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider]",
)

from inference import InferencePipeline
import torch


def build_workflow(model_id: str, confidence: float) -> dict:
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": "roboflow_core/roboflow_instance_segmentation_model@v3",
                "name": "segmentation",
                "images": "$inputs.image",
                "model_id": model_id,
                "confidence_mode": "custom",
                "custom_confidence": confidence,
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "predictions",
                "selector": "$steps.segmentation.predictions",
            },
        ],
    }


SYNC_COUNTS = defaultdict(int)
SYNC_TIME_NS = defaultdict(int)
EVENT_COUNTS = defaultdict(int)
EVENT_TIME_NS = defaultdict(int)


def _callsite_key(depth: int = 6) -> str:
    stack = traceback.extract_stack(limit=depth + 8)
    trimmed = []
    for frame in reversed(stack[:-2]):
        path = frame.filename
        if "/temp/profile_stream_sync_sites.py" in path:
            continue
        if "torch/cuda" in path:
            continue
        trimmed.append(f"{os.path.basename(path)}:{frame.lineno}:{frame.name}")
        if len(trimmed) == 4:
            break
    return " <- ".join(trimmed) if trimmed else "<unknown>"


ORIG_STREAM_SYNC = torch.cuda.Stream.synchronize
ORIG_EVENT_SYNC = torch.cuda.Event.synchronize


def _wrapped_stream_sync(self):
    key = _callsite_key()
    start = time.perf_counter_ns()
    try:
        return ORIG_STREAM_SYNC(self)
    finally:
        SYNC_COUNTS[key] += 1
        SYNC_TIME_NS[key] += time.perf_counter_ns() - start


def _wrapped_event_sync(self):
    key = _callsite_key()
    start = time.perf_counter_ns()
    try:
        return ORIG_EVENT_SYNC(self)
    finally:
        EVENT_COUNTS[key] += 1
        EVENT_TIME_NS[key] += time.perf_counter_ns() - start


torch.cuda.Stream.synchronize = _wrapped_stream_sync
torch.cuda.Event.synchronize = _wrapped_event_sync


FRAME_COUNT = 0
START_TIME = None


def sink(predictions, _video_frames) -> None:
    global FRAME_COUNT, START_TIME
    del _video_frames
    if not isinstance(predictions, list):
        predictions = [predictions]
    FRAME_COUNT += sum(p is not None for p in predictions)
    if START_TIME is None:
        START_TIME = perf_counter()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_reference", default="/home/ubuntu/inference/vehicles_312px.mp4")
    parser.add_argument("--model_id", default="rfdetr-seg-nano")
    parser.add_argument("--confidence", type=float, default=0.4)
    args = parser.parse_args()

    pipeline = InferencePipeline.init_with_workflow(
        video_reference=args.video_reference,
        workflow_specification=build_workflow(args.model_id, args.confidence),
        on_prediction=sink,
    )
    pipeline.start()
    pipeline.join()

    elapsed = perf_counter() - START_TIME if START_TIME else 0.0
    fps = FRAME_COUNT / elapsed if elapsed > 0 else 0.0
    print(f"frames={FRAME_COUNT} elapsed={elapsed:.2f}s fps={fps:.2f}")

    print("\nStream.synchronize callsites:")
    for key, count in sorted(SYNC_COUNTS.items(), key=lambda item: SYNC_TIME_NS[item[0]], reverse=True):
        total_ms = SYNC_TIME_NS[key] / 1e6
        print(f"{count:4d} calls  {total_ms:8.3f} ms  {key}")

    print("\nEvent.synchronize callsites:")
    for key, count in sorted(EVENT_COUNTS.items(), key=lambda item: EVENT_TIME_NS[item[0]], reverse=True):
        total_ms = EVENT_TIME_NS[key] / 1e6
        print(f"{count:4d} calls  {total_ms:8.3f} ms  {key}")


if __name__ == "__main__":
    main()
