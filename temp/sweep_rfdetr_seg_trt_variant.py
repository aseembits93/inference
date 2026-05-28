import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from inference_models.utils.download import download_files_to_directory
from inference_models.weights_providers.core import get_model_from_provider


REPO_ROOT = Path(__file__).resolve().parents[1]
DEV_ROOT = REPO_ROOT / "inference_models" / "development"
if str(DEV_ROOT) not in sys.path:
    sys.path.insert(0, str(DEV_ROOT))

from compilation.core import compile_model_to_trt  # noqa: E402


DEFAULT_VIDEO = str(REPO_ROOT / "vehicles_312px.mp4")
WORKFLOW_BENCH = REPO_ROOT / "development" / "stream_interface" / "rfdetr_nano_seg_trt_workflow.py"


def _select_package(model_id: str, backend: str, quantization: Optional[str]):
    metadata = get_model_from_provider(provider="roboflow", model_id=model_id)
    matches = []
    for package in metadata.model_packages:
        if package.backend.value != backend:
            continue
        if quantization is not None:
            if package.quantization is None or package.quantization.value != quantization:
                continue
        matches.append(package)
    if not matches:
        raise ValueError(
            f"Could not find package for model_id={model_id}, backend={backend}, "
            f"quantization={quantization}."
        )
    if len(matches) > 1 and backend == "trt":
        for package in matches:
            env = package.environment_requirements
            if env is not None and getattr(env, "cuda_device_name", "").lower() == "tesla-t4":
                return package
    return matches[0]


def _download_package(package, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    files_specs = [
        (artefact.file_handle, artefact.download_url, artefact.md5_hash)
        for artefact in package.package_artefacts
    ]
    download_files_to_directory(
        target_dir=str(target_dir),
        files_specs=files_specs,
        verbose=True,
    )


def _write_runtime_trt_config(target_dir: Path) -> None:
    trt_config = {
        "static_batch_size": 1,
        "dynamic_batch_size_min": None,
        "dynamic_batch_size_opt": None,
        "dynamic_batch_size_max": None,
    }
    (target_dir / "trt_config.json").write_text(json.dumps(trt_config))


def _materialize_variant(
    source_onnx_dir: Path,
    variant_dir: Path,
    *,
    workspace_size_gb: int,
    precision: str,
    trt_version_compatible: bool,
    same_compute_compatibility: bool,
    builder_optimization_level: Optional[int],
    max_aux_streams: Optional[int],
    tiling_optimization_level: Optional[str],
    profile_sharing_0806: bool,
    avg_timing_iterations: Optional[int],
    max_num_tactics: Optional[int],
    tactic_sources: Optional[List[str]],
    force_rebuild: bool,
) -> None:
    if variant_dir.exists() and force_rebuild:
        shutil.rmtree(variant_dir)
    variant_dir.mkdir(parents=True, exist_ok=True)

    for file_name in ["weights.onnx", "class_names.txt", "inference_config.json"]:
        shutil.copy2(source_onnx_dir / file_name, variant_dir / file_name)

    compile_model_to_trt(
        model_dir=str(variant_dir),
        precision=precision,
        workspace_size_gb=workspace_size_gb,
        trt_version_compatible=trt_version_compatible,
        same_compute_compatibility=same_compute_compatibility,
        builder_optimization_level=builder_optimization_level,
        max_aux_streams=max_aux_streams,
        tiling_optimization_level=tiling_optimization_level,
        profile_sharing_0806=profile_sharing_0806,
        avg_timing_iterations=avg_timing_iterations,
        max_num_tactics=max_num_tactics,
        tactic_sources=tactic_sources,
    )

    engine_suffix = ""
    if trt_version_compatible:
        engine_suffix += "-trt-version-compatible"
    if same_compute_compatibility:
        engine_suffix += "-same-cc"
    built_engine = variant_dir / f"engine-{precision}{engine_suffix}.plan"
    built_config = variant_dir / f"trt-config-{precision}{engine_suffix}.json"
    if not built_engine.exists():
        raise RuntimeError(f"Expected built engine at {built_engine}.")
    shutil.copy2(built_engine, variant_dir / "engine.plan")
    _write_runtime_trt_config(variant_dir)
    model_config = {
        "model_architecture": "rfdetr",
        "task_type": "instance-segmentation",
        "backend_type": "trt",
    }
    (variant_dir / "model_config.json").write_text(json.dumps(model_config))
    if built_config.exists():
        # Keep the detailed build metadata alongside the runtime config.
        detailed = variant_dir / "build_trt_config.json"
        shutil.copy2(built_config, detailed)


def _benchmark(model_id: str, video_reference: str) -> str:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{REPO_ROOT / 'inference_models'}"
    env["ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES"] = "true"
    env["RFDETR_TRITON_POSTPROC"] = "true"
    env["INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED"] = "true"
    env["RFDETR_PIPELINE_DEPTH"] = "2"
    env["ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND"] = "true"
    result = subprocess.run(
        [
            sys.executable,
            str(WORKFLOW_BENCH),
            "--video_reference",
            video_reference,
            "--model_id",
            model_id,
            "--backend",
            "trt",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="rfdetr-seg-nano")
    parser.add_argument("--precision", default="fp16", choices=("fp16", "fp32"))
    parser.add_argument("--variant_name", required=True)
    parser.add_argument("--workspace_size_gb", type=int, default=8)
    parser.add_argument("--builder_optimization_level", type=int, default=None)
    parser.add_argument("--max_aux_streams", type=int, default=None)
    parser.add_argument("--avg_timing_iterations", type=int, default=None)
    parser.add_argument("--max_num_tactics", type=int, default=None)
    parser.add_argument(
        "--tactic_source",
        action="append",
        dest="tactic_sources",
        default=None,
        choices=(
            "CUBLAS",
            "CUBLAS_LT",
            "CUDNN",
            "EDGE_MASK_CONVOLUTIONS",
            "JIT_CONVOLUTIONS",
        ),
    )
    parser.add_argument(
        "--tiling_optimization_level",
        choices=("none", "fast", "moderate", "full"),
        default=None,
    )
    parser.add_argument("--trt_version_compatible", action="store_true")
    parser.add_argument("--same_compute_compatibility", action="store_true")
    parser.add_argument("--profile_sharing_0806", action="store_true")
    parser.add_argument("--video_reference", default=DEFAULT_VIDEO)
    parser.add_argument(
        "--output_root",
        default="/tmp/rfdetr-seg-nano-trt-sweep",
    )
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--skip_benchmark", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    source_dir = output_root / "source-onnx"
    variant_dir = output_root / args.variant_name

    onnx_package = _select_package(
        model_id=args.model_id,
        backend="onnx",
        quantization="fp32",
    )
    if not (source_dir / "weights.onnx").exists():
        _download_package(onnx_package, source_dir)

    _materialize_variant(
        source_onnx_dir=source_dir,
        variant_dir=variant_dir,
        workspace_size_gb=args.workspace_size_gb,
        precision=args.precision,
        trt_version_compatible=args.trt_version_compatible,
        same_compute_compatibility=args.same_compute_compatibility,
        builder_optimization_level=args.builder_optimization_level,
        max_aux_streams=args.max_aux_streams,
        tiling_optimization_level=args.tiling_optimization_level,
        profile_sharing_0806=args.profile_sharing_0806,
        avg_timing_iterations=args.avg_timing_iterations,
        max_num_tactics=args.max_num_tactics,
        tactic_sources=args.tactic_sources,
        force_rebuild=args.force_rebuild,
    )

    print(f"variant_dir={variant_dir}")
    if args.skip_benchmark:
        return
    stdout = _benchmark(str(variant_dir), args.video_reference)
    print(stdout, end="")


if __name__ == "__main__":
    main()
