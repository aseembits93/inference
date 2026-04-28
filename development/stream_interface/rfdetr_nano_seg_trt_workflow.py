"""E2E benchmark: RF-DETR nano Segmentation (TensorRT CUDA engine) through a
single-block Workflow, run on a video stream via InferencePipeline.

Adapted from development/stream_interface/workflows_demo.py — swaps the object
detection block for `roboflow_core/roboflow_instance_segmentation_model@v1`
pointing at `rfdetr-seg-nano`, adds a mask visualization block so the sink
receives a pre-rendered preview, and pins the ONNX runtime provider list to
TensorRT (FP16, engine-cache on disk) via the `ONNXRUNTIME_EXECUTION_PROVIDERS`
env var. The first run compiles + caches the TRT engine under
TENSORRT_CACHE_PATH / MODEL_CACHE_DIR; subsequent runs reuse the cached plan.

Controls (interactive mode):
    i = watchdog report
    t = terminate
    p = pause stream
    m = mute stream
    r = resume stream
"""
import os

os.environ.setdefault(
    "ONNXRUNTIME_EXECUTION_PROVIDERS",
    "[TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider]",
)

import argparse
from threading import Thread
from time import perf_counter
from typing import List, Optional, Union

import cv2
import supervision as sv

from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.camera.video_source import (
    BufferConsumptionStrategy,
    BufferFillingStrategy,
)
from inference.core.interfaces.stream.watchdog import (
    BasePipelineWatchDog,
    PipelineWatchDog,
)
from inference.core.utils.drawing import create_tiles

MODEL_ID = "rfdetr-seg-nano"
STOP = False
HEADLESS = False
FRAME_COUNT = 0
START_TIME: Optional[float] = None
FPS_MONITOR = sv.FPSMonitor()


def build_workflow_specification(model_id: str, confidence: float) -> dict:
    return {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
        ],
        "steps": [
            {
                "type": "roboflow_core/roboflow_instance_segmentation_model@v1",
                "name": "segmentation",
                "images": "$inputs.image",
                "model_id": model_id,
                "confidence": confidence,
            },
            {
                "type": "roboflow_core/mask_visualization@v1",
                "name": "mask_visualiser",
                "predictions": "$steps.segmentation.predictions",
                "image": "$inputs.image",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "predictions",
                "selector": "$steps.segmentation.predictions",
            },
            {
                "type": "JsonField",
                "name": "preview",
                "selector": "$steps.mask_visualiser.image",
            },
        ],
    }


def workflows_sink(
    predictions: Union[dict, List[Optional[dict]]],
    video_frames: Union[VideoFrame, List[Optional[VideoFrame]]],
) -> None:
    global FRAME_COUNT, START_TIME
    FPS_MONITOR.tick()
    if not isinstance(predictions, list):
        predictions = [predictions]
        video_frames = [video_frames]

    images_to_show = []
    for prediction, frame in zip(predictions, video_frames):
        if prediction is None or frame is None:
            continue
        FRAME_COUNT += 1
        if START_TIME is None:
            START_TIME = perf_counter()

        detections: sv.Detections = prediction["predictions"]
        preview_field = prediction["preview"]
        preview = (
            preview_field.numpy_image
            if hasattr(preview_field, "numpy_image")
            else preview_field
        )

        if HEADLESS:
            if FRAME_COUNT % 10 == 0 or FRAME_COUNT <= 3:
                fps_value = (
                    FPS_MONITOR.fps if hasattr(FPS_MONITOR, "fps") else FPS_MONITOR()
                )
                print(
                    f"[frame {FRAME_COUNT}] detections={len(detections)} fps={fps_value:.2f}",
                    flush=True,
                )
            continue
        images_to_show.append(preview)

    if HEADLESS or not images_to_show:
        return

    fps_value = FPS_MONITOR.fps if hasattr(FPS_MONITOR, "fps") else FPS_MONITOR()
    tiles = create_tiles(images=images_to_show)
    cv2.putText(
        tiles,
        f"FPS: {fps_value:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )
    cv2.imshow("RF-DETR nano seg (TRT)", tiles)
    cv2.waitKey(1)
    print(f"FPS: {fps_value}")


def command_thread(pipeline: InferencePipeline, watchdog: PipelineWatchDog) -> None:
    global STOP
    while not STOP:
        key = input()
        if key == "i":
            print(watchdog.get_report())
        if key == "t":
            pipeline.terminate()
            STOP = True
        elif key == "p":
            pipeline.pause_stream()
        elif key == "m":
            pipeline.mute_stream()
        elif key == "r":
            pipeline.resume_stream()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_reference",
        required=True,
        help="Video file path, RTSP URL, or integer camera id.",
    )
    parser.add_argument("--max_fps", type=float, default=None)
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Skip cv2.imshow; log detection counts + FPS instead.",
    )
    parser.add_argument(
        "--stream_source",
        action="store_true",
        help="Use RTSP-style buffer strategies (DROP_OLDEST / EAGER). "
        "Leave off for video files — the default lets every frame through.",
    )
    return parser.parse_args()


def main() -> None:
    global STOP, HEADLESS
    args = parse_args()
    HEADLESS = args.headless

    try:
        video_reference: Union[str, int] = int(args.video_reference)
    except ValueError:
        video_reference = args.video_reference

    watchdog = BasePipelineWatchDog()
    workflow_specification = build_workflow_specification(
        model_id=args.model_id,
        confidence=args.confidence,
    )
    pipeline_kwargs = dict(
        video_reference=video_reference,
        workflow_specification=workflow_specification,
        watchdog=watchdog,
        on_prediction=workflows_sink,
        max_fps=args.max_fps,
    )
    if args.stream_source:
        pipeline_kwargs["source_buffer_filling_strategy"] = (
            BufferFillingStrategy.DROP_OLDEST
        )
        pipeline_kwargs["source_buffer_consumption_strategy"] = (
            BufferConsumptionStrategy.EAGER
        )
    pipeline = InferencePipeline.init_with_workflow(**pipeline_kwargs)

    control_thread: Optional[Thread] = None
    if not HEADLESS:
        control_thread = Thread(
            target=command_thread, args=(pipeline, watchdog), daemon=True
        )
        control_thread.start()

    pipeline.start()
    STOP = True
    pipeline.join()

    if not HEADLESS:
        cv2.destroyAllWindows()

    elapsed = perf_counter() - START_TIME if START_TIME is not None else 0.0
    avg_fps = (FRAME_COUNT - 1) / elapsed if elapsed > 0 and FRAME_COUNT > 1 else 0.0
    window_fps = FPS_MONITOR.fps if hasattr(FPS_MONITOR, "fps") else FPS_MONITOR()
    print(
        f"DONE: processed {FRAME_COUNT} frames in {elapsed:.2f}s, "
        f"avg FPS={avg_fps:.2f}, window FPS={window_fps:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
