import argparse
import os
from time import perf_counter


_ALL_BACKENDS = {
    "torch",
    "torch-script",
    "onnx",
    "trt",
    "hugging-face",
    "ultralytics",
    "mediapipe",
    "custom",
}


def _select_backend_from_argv() -> str:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--backend", choices=("trt", "onnx", "torch"), default="trt")
    args, _ = pre.parse_known_args()
    return args.backend


_BACKEND = _select_backend_from_argv()
os.environ.setdefault(
    "ONNXRUNTIME_EXECUTION_PROVIDERS",
    "[TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider]",
)
os.environ["DISABLED_INFERENCE_MODELS_BACKENDS"] = ",".join(
    sorted(_ALL_BACKENDS - {_BACKEND})
)
os.environ.setdefault("RFDETR_TRITON_POSTPROC", "true")
os.environ.setdefault("INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED", "true")
os.environ.setdefault("RFDETR_PIPELINE_DEPTH", "2")
os.environ.setdefault("ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND", "true")
os.environ.setdefault("ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES", "true")

import torch

from inference import InferencePipeline


FRAME_COUNT = 0
START_TIME = None
PROFILE_STARTED = False
PROFILE_STOPPED = False


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_reference", required=True)
    parser.add_argument("--model_id", default="rfdetr-seg-nano")
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--backend", choices=("trt", "onnx", "torch"), default="trt")
    parser.add_argument("--profile_start_frame", type=int, default=50)
    parser.add_argument("--profile_stop_frame", type=int, default=538)
    parser.add_argument("--progress_every", type=int, default=50)
    args = parser.parse_args()

    def sink(predictions, _video_frames) -> None:
        global FRAME_COUNT, START_TIME, PROFILE_STARTED, PROFILE_STOPPED
        del _video_frames
        if not isinstance(predictions, list):
            predictions = [predictions]
        FRAME_COUNT += sum(p is not None for p in predictions)
        if START_TIME is None:
            START_TIME = perf_counter()
        if (not PROFILE_STARTED) and FRAME_COUNT >= args.profile_start_frame:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            PROFILE_STARTED = True
        if FRAME_COUNT % args.progress_every == 0:
            fps = FRAME_COUNT / (perf_counter() - START_TIME)
            print(f"[progress] frames={FRAME_COUNT} fps={fps:.2f}", flush=True)
        if PROFILE_STARTED and (not PROFILE_STOPPED) and FRAME_COUNT >= args.profile_stop_frame:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            PROFILE_STOPPED = True

    pipeline = InferencePipeline.init_with_workflow(
        video_reference=args.video_reference,
        workflow_specification=build_workflow(args.model_id, args.confidence),
        on_prediction=sink,
    )
    pipeline.start()
    pipeline.join()

    elapsed = perf_counter() - START_TIME if START_TIME else 0.0
    fps = FRAME_COUNT / elapsed if elapsed > 0 else 0.0
    print(
        f"frames={FRAME_COUNT} elapsed={elapsed:.2f}s fps={fps:.2f} "
        f"profile_started={PROFILE_STARTED} profile_stopped={PROFILE_STOPPED}"
    )


if __name__ == "__main__":
    main()
