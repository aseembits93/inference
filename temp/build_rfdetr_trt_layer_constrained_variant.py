import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import tensorrt as trt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEV_ROOT = REPO_ROOT / "inference_models" / "development"
if str(DEV_ROOT) not in sys.path:
    sys.path.insert(0, str(DEV_ROOT))

from compilation.engine_builder import EngineBuilder  # noqa: E402
from compilation.core import translate_tactic_sources  # noqa: E402


def _copy_variant_tree(source_dir: Path, target_dir: Path) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("weights.onnx", "class_names.txt", "inference_config.json"):
        shutil.copy2(source_dir / name, target_dir / name)


def _matches(name: str, includes: Iterable[str], excludes: Iterable[str]) -> bool:
    if includes and not any(token in name for token in includes):
        return False
    if excludes and any(token in name for token in excludes):
        return False
    return True


def _apply_precision_constraints(
    builder: EngineBuilder,
    *,
    layer_name_includes: List[str],
    layer_name_excludes: List[str],
    layer_types: Optional[List[trt.LayerType]],
    precision: trt.DataType,
) -> int:
    builder.config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    count = 0
    for idx in range(builder.network.num_layers):
        layer = builder.network.get_layer(idx)
        if not _matches(layer.name, layer_name_includes, layer_name_excludes):
            continue
        if layer_types is not None and layer.type not in layer_types:
            continue
        layer.precision = precision
        for output_index in range(layer.num_outputs):
            layer.set_output_type(output_index, precision)
        print(f"set_precision {idx} {layer.type} {layer.name}")
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--builder-optimization-level", type=int, default=3)
    parser.add_argument(
        "--tactic-source",
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
        "--layer-name-include",
        action="append",
        dest="layer_name_includes",
        default=[],
    )
    parser.add_argument(
        "--layer-name-exclude",
        action="append",
        dest="layer_name_excludes",
        default=[],
    )
    parser.add_argument(
        "--layer-type",
        action="append",
        dest="layer_types",
        default=None,
        choices=(
            "convolution",
            "shuffle",
            "elementwise",
            "matrix_multiply",
            "softmax",
            "cast",
            "normalization",
        ),
    )
    parser.add_argument("--cache-in", default=None)
    parser.add_argument("--error-on-timing-cache-miss", action="store_true")
    parser.add_argument(
        "--precision",
        choices=("fp16", "fp32"),
        default="fp32",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    _copy_variant_tree(source_dir, target_dir)

    tactic_sources = translate_tactic_sources(args.tactic_sources)
    builder = EngineBuilder(
        workspace=8,
        builder_optimization_level=args.builder_optimization_level,
        tactic_sources=tactic_sources,
    )
    if args.error_on_timing_cache_miss:
        builder.config.set_flag(trt.BuilderFlag.ERROR_ON_TIMING_CACHE_MISS)
    if args.cache_in is not None:
        timing_cache = builder.config.create_timing_cache(
            Path(args.cache_in).read_bytes()
        )
        if not builder.config.set_timing_cache(timing_cache, False):
            raise RuntimeError("Could not attach timing cache to builder config.")
    builder.create_network(str(target_dir / "weights.onnx"))

    layer_type_map = {
        "convolution": trt.LayerType.CONVOLUTION,
        "shuffle": trt.LayerType.SHUFFLE,
        "elementwise": trt.LayerType.ELEMENTWISE,
        "matrix_multiply": trt.LayerType.MATRIX_MULTIPLY,
        "softmax": trt.LayerType.SOFTMAX,
        "cast": trt.LayerType.CAST,
        "normalization": trt.LayerType.NORMALIZATION,
    }
    selected_layer_types = (
        None if args.layer_types is None else [layer_type_map[name] for name in args.layer_types]
    )
    selected_precision = trt.float16 if args.precision == "fp16" else trt.float32
    changed = _apply_precision_constraints(
        builder,
        layer_name_includes=args.layer_name_includes,
        layer_name_excludes=args.layer_name_excludes,
        layer_types=selected_layer_types,
        precision=selected_precision,
    )
    print(f"constrained_layers={changed}")

    engine_path = target_dir / "engine-fp16.plan"
    builder.create_engine(
        engine_path=str(engine_path),
        precision="fp16",
        input_name="input",
        input_size=(312, 312),
        dynamic_batch_sizes=None,
        trt_version_compatible=False,
        same_compute_compatibility=False,
    )
    (target_dir / "engine.plan").write_bytes(engine_path.read_bytes())
    (target_dir / "trt_config.json").write_text(
        json.dumps(
            {
                "static_batch_size": 1,
                "dynamic_batch_size_min": None,
                "dynamic_batch_size_opt": None,
                "dynamic_batch_size_max": None,
            }
        )
    )
    (target_dir / "model_config.json").write_text(
        json.dumps(
            {
                "model_architecture": "rfdetr",
                "task_type": "instance-segmentation",
                "backend_type": "trt",
            }
        )
    )
    print(f"variant_dir={target_dir}")


if __name__ == "__main__":
    main()
