import argparse
import os
import threading
from time import perf_counter_ns


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
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backend", choices=("trt", "onnx", "torch"), default="trt")
    args, _ = parser.parse_known_args()
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

from inference import InferencePipeline
from inference.core.models import inference_models_adapters as adapter_mod
from inference.core.models.inference_models_adapters import (
    InferenceModelsInstanceSegmentationAdapter,
    LazyWorkflowSVDetections,
)
from inference.core.workflows.core_steps.models.roboflow.instance_segmentation.v3 import (
    RoboflowInstanceSegmentationModelBlockV3,
)
from inference_models.models.rfdetr import common as rfdetr_common
from inference_models.models.rfdetr import rfdetr_instance_segmentation_trt as rfdetr_trt
from inference_models.models.rfdetr.rfdetr_instance_segmentation_trt import (
    RFDetrForInstanceSegmentationTRT,
)
from inference_models.models.common.roboflow.post_processing import ConfidenceFilter


FRAME_COUNT = 0
START_TIME = None
PROGRESS_EVERY = 50
PROFILE = {}
PROFILE_LOCK = threading.Lock()


def _record_timing(name: str, elapsed_ns: int) -> None:
    with PROFILE_LOCK:
        count, total_ns = PROFILE.get(name, (0, 0))
        PROFILE[name] = (count + 1, total_ns + elapsed_ns)


def _wrap_method(cls, attr_name: str) -> None:
    original = getattr(cls, attr_name)

    def wrapped(*args, **kwargs):
        start = perf_counter_ns()
        try:
            return original(*args, **kwargs)
        finally:
            _record_timing(f"{cls.__name__}.{attr_name}", perf_counter_ns() - start)

    setattr(cls, attr_name, wrapped)


def _wrap_function(module, attr_name: str, label: str) -> None:
    original = getattr(module, attr_name)

    def wrapped(*args, **kwargs):
        start = perf_counter_ns()
        try:
            return original(*args, **kwargs)
        finally:
            _record_timing(label, perf_counter_ns() - start)

    setattr(module, attr_name, wrapped)


def install_wrappers() -> None:
    _wrap_method(
        RFDetrForInstanceSegmentationTRT,
        "_get_or_capture_combined_dense_graph_state",
    )
    _wrap_method(
        RFDetrForInstanceSegmentationTRT,
        "_capture_combined_dense_graph_state",
    )
    _wrap_method(
        RFDetrForInstanceSegmentationTRT,
        "_maybe_forward_async_combined_dense_graph",
    )
    _wrap_method(
        InferenceModelsInstanceSegmentationAdapter,
        "_build_responses_from_detections",
    )
    _wrap_method(LazyWorkflowSVDetections, "__init__")
    _wrap_method(
        RoboflowInstanceSegmentationModelBlockV3,
        "_post_process_result",
    )
    _wrap_method(ConfidenceFilter, "get_threshold")
    _wrap_function(
        rfdetr_trt,
        "build_deferred_dense_postproc_detection",
        "rfdetr_trt.build_deferred_dense_postproc_detection",
    )
    for attr_name in (
        "_prepare_threshold",
        "_get_class_mapping_int32",
        "get_rfdetr_triton_postproc_geometry",
    ):
        if hasattr(rfdetr_trt, attr_name):
            _wrap_function(
                rfdetr_trt,
                attr_name,
                f"rfdetr_trt.{attr_name}",
            )


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


def sink(predictions, _video_frames) -> None:
    global FRAME_COUNT, START_TIME
    del _video_frames
    if not isinstance(predictions, list):
        predictions = [predictions]
    FRAME_COUNT += sum(p is not None for p in predictions)
    if START_TIME is None:
        from time import perf_counter

        START_TIME = perf_counter()
    if FRAME_COUNT % PROGRESS_EVERY == 0:
        from time import perf_counter

        fps = FRAME_COUNT / (perf_counter() - START_TIME)
        print(f"[progress] frames={FRAME_COUNT} fps={fps:.2f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_reference", required=True)
    parser.add_argument("--model_id", default="rfdetr-seg-nano")
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument(
        "--backend",
        choices=("trt", "onnx", "torch"),
        default="trt",
    )
    args = parser.parse_args()

    install_wrappers()

    pipeline = InferencePipeline.init_with_workflow(
        video_reference=args.video_reference,
        workflow_specification=build_workflow(args.model_id, args.confidence),
        on_prediction=sink,
    )
    pipeline.start()
    pipeline.join()

    from time import perf_counter

    elapsed = perf_counter() - START_TIME if START_TIME else 0.0
    fps = FRAME_COUNT / elapsed if elapsed > 0 else 0.0
    print(f"frames={FRAME_COUNT} elapsed={elapsed:.2f}s fps={fps:.2f}")

    print("\n== hotspot summary ==")
    items = sorted(PROFILE.items(), key=lambda item: item[1][1], reverse=True)
    for name, (count, total_ns) in items:
        total_ms = total_ns / 1_000_000.0
        avg_us = total_ns / count / 1_000.0 if count else 0.0
        pct = (total_ns / (elapsed * 1_000_000_000.0) * 100.0) if elapsed > 0 else 0.0
        print(
            f"{name}: calls={count} total_ms={total_ms:.2f} "
            f"avg_us={avg_us:.2f} pct_of_elapsed={pct:.2f}"
        )


if __name__ == "__main__":
    main()
