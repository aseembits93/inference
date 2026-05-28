"""Minimal benchmark: RF-DETR instance segmentation through inference-models,
run via InferencePipeline on a single video source.

Workflow has exactly one block — the segmentation model. No annotators, no
buffer strategies, no rate limiting.

The `--backend` flag (trt | onnx | torch) is parsed before importing
`inference` and pins the auto-loader by setting
`DISABLED_INFERENCE_MODELS_BACKENDS` to every backend except the chosen one,
so the benchmark numbers correspond unambiguously to a single execution path.

Defaults: rfdetr-seg-nano @ confidence 0.4 on the native TRT backend.
"""
import argparse
import os
import sys
from functools import partial
from pathlib import Path
from typing import Optional

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

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_IMPORT_PATHS = (
    str(_REPO_ROOT / "inference_models"),
    str(_REPO_ROOT),
)
for _path in reversed(_LOCAL_IMPORT_PATHS):
    if _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)


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
os.environ.setdefault("ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES", "True")
os.environ["DISABLED_INFERENCE_MODELS_BACKENDS"] = ",".join(
    sorted(_ALL_BACKENDS - {_BACKEND})
)

from time import perf_counter

import inference
import inference_models
from inference import InferencePipeline
from inference.core.interfaces.stream.entities import ModelConfig
from inference.core.interfaces.stream.model_handlers.roboflow_models import (
    default_process_frame,
)
from inference.core.models.inference_models_adapters import (
    InferenceModelsInstanceSegmentationAdapter,
)

DEFAULT_LOCAL_TRT_MODEL_PACKAGE = _REPO_ROOT / "rfdetr-seg-nano-orin-trt-package"


def is_model_package_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    required_files = {"model_config.json", "inference_config.json", "class_names.txt"}
    package_files = {child.name for child in path.iterdir() if child.is_file()}
    return required_files.issubset(package_files)


def resolve_model_target(
    backend: str, model_id: str, model_package_path: Optional[str]
) -> str:
    if model_package_path:
        return str(Path(model_package_path).expanduser().resolve())
    candidate_path = Path(model_id).expanduser()
    if is_model_package_dir(candidate_path):
        return str(candidate_path.resolve())
    if backend == "trt" and DEFAULT_LOCAL_TRT_MODEL_PACKAGE.exists():
        return str(DEFAULT_LOCAL_TRT_MODEL_PACKAGE.resolve())
    return model_id


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

FRAME_COUNT = 0
START_TIME = None
PROGRESS_EVERY = 50


def sink(predictions, _video_frames) -> None:
    global FRAME_COUNT, START_TIME
    del _video_frames
    if not isinstance(predictions, list):
        predictions = [predictions]
    FRAME_COUNT += sum(p is not None for p in predictions)
    if START_TIME is None:
        START_TIME = perf_counter()
    if FRAME_COUNT % PROGRESS_EVERY == 0:
        fps = FRAME_COUNT / (perf_counter() - START_TIME)
        print(f"[progress] frames={FRAME_COUNT} fps={fps:.2f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_reference", required=True)
    parser.add_argument("--model_id", default="rfdetr-seg-nano")
    parser.add_argument(
        "--model_package_path",
        default=None,
        help=(
            "Absolute or relative path to a local model package directory. "
            "When omitted, TRT runs prefer the repo-local "
            "`rfdetr-seg-nano-orin-trt-package` if it exists."
        ),
    )
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument(
        "--backend",
        choices=("trt", "onnx", "torch"),
        default="trt",
        help="inference-models backend (consumed pre-import via env var).",
    )
    args = parser.parse_args()
    resolved_model_target = resolve_model_target(
        backend=args.backend,
        model_id=args.model_id,
        model_package_path=args.model_package_path,
    )
    local_model_package = Path(resolved_model_target).expanduser()
    uses_local_model_package = is_model_package_dir(local_model_package)
    workflow_model_id = args.model_id
    print(
        "[setup] "
        f"inference={Path(inference.__file__).resolve()} "
        f"inference_models={Path(inference_models.__file__).resolve()} "
        f"backend={args.backend} "
        f"model_target={workflow_model_id} "
        f"local_model_package={resolved_model_target}",
        flush=True,
    )
    if uses_local_model_package:
        inference_config = ModelConfig.init(confidence=args.confidence)
        model = InferenceModelsInstanceSegmentationAdapter(
            model_id=str(local_model_package.resolve())
        )
        pipeline = InferencePipeline.init_with_custom_logic(
            video_reference=args.video_reference,
            on_video_frame=partial(
                default_process_frame,
                model=model,
                inference_config=inference_config,
            ),
            on_prediction=sink,
        )
    else:
        pipeline = InferencePipeline.init_with_workflow(
            video_reference=args.video_reference,
            workflow_specification=build_workflow(workflow_model_id, args.confidence),
            on_prediction=sink,
        )
    pipeline.start()
    pipeline.join()

    elapsed = perf_counter() - START_TIME if START_TIME else 0.0
    fps = FRAME_COUNT / elapsed if elapsed > 0 else 0.0
    print(f"frames={FRAME_COUNT} elapsed={elapsed:.2f}s fps={fps:.2f}")


if __name__ == "__main__":
    main()
