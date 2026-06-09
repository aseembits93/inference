"""Minimal benchmark: RF-DETR instance segmentation through inference-models,
run via InferencePipeline on a single video source.

Workflow has exactly one block — the segmentation model. No annotators, no
buffer strategies, no rate limiting.

The `--backend` flag (trt | onnx | torch) is parsed before importing
`inference` and pins the auto-loader by setting
`DISABLED_INFERENCE_MODELS_BACKENDS` to every backend except the chosen one,
so the benchmark numbers correspond unambiguously to a single execution path.

Pass `--model_package_id` to download a specific registry package (cached under
`$INFERENCE_HOME/models-cache/`) and run the benchmark against that artefact
instead of auto-negotiation. A TRT package directory in the cwd is still used
when present and `--model_package_id` is not set.

Defaults: rfdetr-seg-nano @ confidence 0.4 on the native TRT backend.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

_ALL_BACKENDS = {
    "torch",
    "torch-script",
    "onnx",
    "trt",
    "hugging-face",
    "ultralytics",
    "custom",
}
_DEFAULT_MODEL_ID = "rfdetr-seg-nano"
_PREFERRED_LOCAL_TRT_PACKAGE = "rfdetr-seg-nano-orin-trt-package"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_INFERENCE_MODELS_ROOT = _REPO_ROOT / "inference_models"


def _is_local_trt_package(path: Path) -> bool:
    if not path.is_dir():
        return False
    required_files = ("engine.plan", "model_config.json", "inference_config.json")
    if not all((path / f).is_file() for f in required_files):
        return False
    try:
        model_config = json.loads((path / "model_config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return model_config.get("backend_type") == "trt"


def _find_local_trt_package() -> str | None:
    preferred = Path.cwd() / _PREFERRED_LOCAL_TRT_PACKAGE
    if _is_local_trt_package(preferred):
        return str(preferred.resolve())

    candidates = sorted(
        path.resolve() for path in Path.cwd().iterdir() if _is_local_trt_package(path)
    )
    if len(candidates) == 1:
        return str(candidates[0])
    return None


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
for path in (str(_INFERENCE_MODELS_ROOT), str(_REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
for module_name in list(sys.modules):
    if module_name == "inference" or module_name.startswith("inference."):
        del sys.modules[module_name]
    if module_name == "inference_models" or module_name.startswith("inference_models."):
        del sys.modules[module_name]

from time import perf_counter

_LOCAL_INFERENCE_SPEC = importlib.util.spec_from_file_location(
    "inference",
    _REPO_ROOT / "inference" / "__init__.py",
    submodule_search_locations=[str(_REPO_ROOT / "inference")],
)
if _LOCAL_INFERENCE_SPEC is None or _LOCAL_INFERENCE_SPEC.loader is None:
    raise RuntimeError("Could not load local inference package")
_LOCAL_INFERENCE_MODULE = importlib.util.module_from_spec(_LOCAL_INFERENCE_SPEC)
sys.modules["inference"] = _LOCAL_INFERENCE_MODULE
_LOCAL_INFERENCE_SPEC.loader.exec_module(_LOCAL_INFERENCE_MODULE)
InferencePipeline = _LOCAL_INFERENCE_MODULE.InferencePipeline


def _fetch_model_package(model_id: str, package_id: str, backend: str) -> str:
    from inference_models import AutoModel

    package_dirs: list[str] = []

    def capture_package_dir(path: str) -> None:
        package_dirs.append(path)

    AutoModel.from_pretrained(
        model_id_or_path=model_id,
        backend=backend,
        model_package_id=package_id,
        verbose=True,
        point_model_directory=capture_package_dir,
    )
    if not package_dirs:
        raise RuntimeError(
            f"Model package {package_id!r} for {model_id!r} did not report a cache path."
        )

    return package_dirs[0]


def _resolve_local_package(
    *,
    backend: str,
    model_id: str,
    model_package_id: str | None,
) -> str | None:
    if model_package_id is not None:
        package_dir = _fetch_model_package(
            model_id=model_id,
            package_id=model_package_id,
            backend=backend,
        )
        print(
            f"[model] fetched package_id={model_package_id} from {package_dir}",
            flush=True,
        )
        return package_dir

    if backend == "trt":
        return _find_local_trt_package()

    return None


def _resolve_model_id(model_id: str, local_package: str | None) -> str:
    if local_package is not None:
        return f"{model_id}/1"
    return model_id


def _prepare_local_workflow_model_bundle(
    workflow_model_id: str,
    local_package: str,
) -> None:
    model_dir = Path(workflow_model_id)
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    target_dir = Path(local_package)
    if not model_dir.exists():
        model_dir.symlink_to(target_dir, target_is_directory=True)

    model_cache_dir = (
        Path(os.environ.get("MODEL_CACHE_DIR", "/tmp/cache")) / workflow_model_id
    )
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    model_type_path = model_cache_dir / "model_type.json"
    model_metadata = {
        "project_task_type": "instance-segmentation",
        "model_type": "rfdetr-seg-nano",
    }
    model_type_path.write_text(json.dumps(model_metadata, indent=4))


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
                "enforce_dense_masks_in_inference_models": False,
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
    parser.add_argument("--model_id", default=_DEFAULT_MODEL_ID)
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument(
        "--backend",
        choices=("trt", "onnx", "torch"),
        default="trt",
        help="inference-models backend (consumed pre-import via env var).",
    )
    parser.add_argument(
        "--model_package_id",
        default=None,
        help=(
            "Registry package id to download and pin (via inference-models cache). "
            "Overrides auto-negotiation and any cwd TRT package discovery."
        ),
    )
    args = parser.parse_args()
    local_package = _resolve_local_package(
        backend=args.backend,
        model_id=args.model_id,
        model_package_id=args.model_package_id,
    )
    if local_package is not None:
        os.environ.setdefault(
            "ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES",
            "True",
        )

    workflow_model_id = _resolve_model_id(
        model_id=args.model_id,
        local_package=local_package,
    )
    if local_package is not None:
        _prepare_local_workflow_model_bundle(
            workflow_model_id=workflow_model_id,
            local_package=local_package,
        )
        print(
            f"[model] using package via workflow model id: {workflow_model_id}",
            flush=True,
        )

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
