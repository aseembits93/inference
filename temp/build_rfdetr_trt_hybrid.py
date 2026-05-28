import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import onnxruntime
import tensorrt as trt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEV_ROOT = REPO_ROOT / "inference_models" / "development"
if str(DEV_ROOT) not in sys.path:
    sys.path.insert(0, str(DEV_ROOT))

from compilation.engine_builder import EngineBuilder  # noqa: E402


WEIGHTS_FILE_NAME = "weights.onnx"
DEFAULT_SOURCE = Path("/tmp/rfdetr-seg-nano-trt-sweep/source-onnx")
TACTIC_SOURCES = (
    "CUBLAS",
    "CUBLAS_LT",
    "CUDNN",
    "EDGE_MASK_CONVOLUTIONS",
    "JIT_CONVOLUTIONS",
)


def _normalize_name(name: str) -> str:
    name = name.rsplit("_myl", 1)[0]
    name = name.rsplit("_0x", 1)[0]
    return name


def _translate_tactic_sources(
    tactic_sources: Optional[Iterable[str]],
) -> Optional[List[trt.TacticSource]]:
    if tactic_sources is None:
        return None
    return [getattr(trt.TacticSource, source) for source in tactic_sources]


def _prepare_variant_dir(source_onnx_dir: Path, variant_dir: Path) -> Path:
    if variant_dir.exists():
        shutil.rmtree(variant_dir)
    variant_dir.mkdir(parents=True, exist_ok=True)
    for file_name in ["weights.onnx", "class_names.txt", "inference_config.json"]:
        shutil.copy2(source_onnx_dir / file_name, variant_dir / file_name)
    return variant_dir


def _write_runtime_files(variant_dir: Path) -> None:
    (variant_dir / "trt_config.json").write_text(
        json.dumps(
            {
                "static_batch_size": 1,
                "dynamic_batch_size_min": None,
                "dynamic_batch_size_opt": None,
                "dynamic_batch_size_max": None,
            }
        )
    )
    (variant_dir / "model_config.json").write_text(
        json.dumps(
            {
                "model_architecture": "rfdetr",
                "task_type": "instance-segmentation",
                "backend_type": "trt",
            }
        )
    )


def _algorithm_signature(algorithm: trt.IAlgorithm) -> Dict[str, int]:
    variant = algorithm.algorithm_variant
    return {
        "implementation": int(variant.implementation),
        "tactic": int(variant.tactic),
    }


class AlgorithmRecorder(trt.IAlgorithmSelector):
    def __init__(self) -> None:
        super().__init__()
        self.records: Dict[str, dict] = {}

    def select_algorithms(
        self, context: trt.IAlgorithmContext, choices: List[trt.IAlgorithm]
    ) -> List[int]:
        return list(range(len(choices)))

    def report_algorithms(
        self,
        contexts: List[trt.IAlgorithmContext],
        choices: List[trt.IAlgorithm],
    ) -> None:
        for context, choice in zip(contexts, choices):
            key = _normalize_name(context.name)
            self.records[key] = {
                "name": context.name,
                "normalized_name": key,
                "signature": _algorithm_signature(choice),
                "timing_msec": float(choice.timing_msec),
                "workspace_size": int(choice.workspace_size),
            }


class AlgorithmReplay(trt.IAlgorithmSelector):
    def __init__(self, chosen_algorithms: Dict[str, dict]) -> None:
        super().__init__()
        self.chosen_algorithms = chosen_algorithms
        self.matched = 0
        self.fallback = 0

    def select_algorithms(
        self, context: trt.IAlgorithmContext, choices: List[trt.IAlgorithm]
    ) -> List[int]:
        key = _normalize_name(context.name)
        target = self.chosen_algorithms.get(key)
        if target is None:
            self.fallback += 1
            return list(range(len(choices)))
        target_impl = int(target["signature"]["implementation"])
        target_tactic = int(target["signature"]["tactic"])
        for idx, choice in enumerate(choices):
            signature = _algorithm_signature(choice)
            if (
                signature["implementation"] == target_impl
                and signature["tactic"] == target_tactic
            ):
                self.matched += 1
                return [idx]
        self.fallback += 1
        return list(range(len(choices)))

    def report_algorithms(
        self,
        contexts: List[trt.IAlgorithmContext],
        choices: List[trt.IAlgorithm],
    ) -> None:
        return None


def _build_engine(
    *,
    source_onnx_dir: Path,
    variant_dir: Path,
    selector: trt.IAlgorithmSelector,
    builder_optimization_level: int,
    tactic_sources: Optional[List[str]],
    timing_cache_data: Optional[bytes] = None,
) -> None:
    variant_dir = _prepare_variant_dir(
        source_onnx_dir=source_onnx_dir, variant_dir=variant_dir
    )
    session = onnxruntime.InferenceSession(str(variant_dir / WEIGHTS_FILE_NAME))
    input_tensor = session.get_inputs()[0]
    input_shape = input_tensor.shape
    input_size = (int(input_shape[2]), int(input_shape[3]))

    engine_builder = EngineBuilder(
        workspace=8,
        builder_optimization_level=builder_optimization_level,
        tactic_sources=_translate_tactic_sources(tactic_sources),
    )
    if timing_cache_data is not None:
        timing_cache = engine_builder.config.create_timing_cache(timing_cache_data)
        if not engine_builder.config.set_timing_cache(timing_cache, False):
            raise RuntimeError("Could not attach timing cache to builder config.")
    engine_builder.config.algorithm_selector = selector
    engine_builder.config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    engine_builder.create_network(onnx_path=str(variant_dir / WEIGHTS_FILE_NAME))
    engine_builder.create_engine(
        engine_path=str(variant_dir / "engine.plan"),
        precision="fp16",
        input_name=input_tensor.name,
        input_size=input_size,
        dynamic_batch_sizes=None,
        trt_version_compatible=False,
        same_compute_compatibility=False,
    )
    _write_runtime_files(variant_dir=variant_dir)


def do_record(
    *,
    source_onnx_dir: Path,
    variant_dir: Path,
    record_out: Path,
    builder_optimization_level: int,
    tactic_sources: Optional[List[str]],
    cache_in: Optional[Path],
) -> None:
    selector = AlgorithmRecorder()
    _build_engine(
        source_onnx_dir=source_onnx_dir,
        variant_dir=variant_dir,
        selector=selector,
        builder_optimization_level=builder_optimization_level,
        tactic_sources=tactic_sources,
        timing_cache_data=cache_in.read_bytes() if cache_in is not None else None,
    )
    record_out.write_text(json.dumps(selector.records, indent=2, sort_keys=True))
    print(f"recorded_layers={len(selector.records)}")
    print(f"record_out={record_out}")


def do_hybrid(
    *,
    source_onnx_dir: Path,
    variant_dir: Path,
    base_record: Path,
    fast_record: Path,
    fast_prefixes: List[str],
    builder_optimization_level: int,
    tactic_sources: Optional[List[str]],
) -> None:
    base_map = json.loads(base_record.read_text())
    fast_map = json.loads(fast_record.read_text())
    chosen_map = dict(base_map)
    fast_applied = 0
    for key, value in fast_map.items():
        if any(key.startswith(prefix) for prefix in fast_prefixes):
            chosen_map[key] = value
            fast_applied += 1
    print(f"hybrid_fast_prefixes={fast_prefixes}")
    print(f"hybrid_fast_layers={fast_applied}")
    selector = AlgorithmReplay(chosen_algorithms=chosen_map)
    _build_engine(
        source_onnx_dir=source_onnx_dir,
        variant_dir=variant_dir,
        selector=selector,
        builder_optimization_level=builder_optimization_level,
        tactic_sources=tactic_sources,
    )
    print(f"selector_matched={selector.matched}")
    print(f"selector_fallback={selector.fallback}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    record = subparsers.add_parser("record")
    record.add_argument("--source_onnx_dir", default=str(DEFAULT_SOURCE))
    record.add_argument("--variant_dir", required=True)
    record.add_argument("--record_out", required=True)
    record.add_argument("--builder_optimization_level", type=int, default=3)
    record.add_argument("--cache_in", default=None)
    record.add_argument(
        "--tactic_source",
        action="append",
        dest="tactic_sources",
        default=None,
        choices=TACTIC_SOURCES,
    )

    hybrid = subparsers.add_parser("hybrid")
    hybrid.add_argument("--source_onnx_dir", default=str(DEFAULT_SOURCE))
    hybrid.add_argument("--variant_dir", required=True)
    hybrid.add_argument("--base_record", required=True)
    hybrid.add_argument("--fast_record", required=True)
    hybrid.add_argument(
        "--fast_prefix",
        action="append",
        dest="fast_prefixes",
        default=None,
    )
    hybrid.add_argument("--builder_optimization_level", type=int, default=3)
    hybrid.add_argument(
        "--tactic_source",
        action="append",
        dest="tactic_sources",
        default=None,
        choices=TACTIC_SOURCES,
    )

    args = parser.parse_args()
    if args.mode == "record":
        do_record(
            source_onnx_dir=Path(args.source_onnx_dir),
            variant_dir=Path(args.variant_dir),
            record_out=Path(args.record_out),
            builder_optimization_level=args.builder_optimization_level,
            tactic_sources=args.tactic_sources,
            cache_in=Path(args.cache_in) if args.cache_in is not None else None,
        )
        return
    do_hybrid(
        source_onnx_dir=Path(args.source_onnx_dir),
        variant_dir=Path(args.variant_dir),
        base_record=Path(args.base_record),
        fast_record=Path(args.fast_record),
        fast_prefixes=args.fast_prefixes or [],
        builder_optimization_level=args.builder_optimization_level,
        tactic_sources=args.tactic_sources,
    )


if __name__ == "__main__":
    main()
