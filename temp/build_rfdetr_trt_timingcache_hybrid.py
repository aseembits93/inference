import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Optional

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
INVALID_TACTIC_HASH = (1 << 64) - 1


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


def _is_valid_value(value: trt.TimingCacheValue) -> bool:
    return value.tacticHash != INVALID_TACTIC_HASH and value.timingMSec >= 0.0


def _build_engine(
    *,
    source_onnx_dir: Path,
    variant_dir: Path,
    builder_optimization_level: int,
    tactic_sources: Optional[List[str]],
    timing_cache_data: Optional[bytes],
    cache_out: Optional[Path],
    error_on_timing_cache_miss: bool,
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
    engine_builder.config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if error_on_timing_cache_miss:
        engine_builder.config.set_flag(trt.BuilderFlag.ERROR_ON_TIMING_CACHE_MISS)
    if timing_cache_data is not None:
        timing_cache = engine_builder.config.create_timing_cache(timing_cache_data)
        if not engine_builder.config.set_timing_cache(timing_cache, False):
            raise RuntimeError("Could not attach timing cache to builder config.")
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

    if cache_out is not None:
        timing_cache = engine_builder.config.get_timing_cache()
        if timing_cache is None:
            raise RuntimeError("Builder config did not expose a timing cache.")
        cache_out.write_bytes(bytes(timing_cache.serialize()))
        print(f"cache_out={cache_out}")


def do_record(
    *,
    source_onnx_dir: Path,
    variant_dir: Path,
    cache_out: Path,
    builder_optimization_level: int,
    tactic_sources: Optional[List[str]],
) -> None:
    _build_engine(
        source_onnx_dir=source_onnx_dir,
        variant_dir=variant_dir,
        builder_optimization_level=builder_optimization_level,
        tactic_sources=tactic_sources,
        timing_cache_data=b"",
        cache_out=cache_out,
        error_on_timing_cache_miss=False,
    )


def do_merge_min(
    *,
    base_cache: Path,
    fast_cache: Path,
    merged_cache_out: Path,
) -> None:
    builder = EngineBuilder(workspace=8, builder_optimization_level=3)
    base = builder.config.create_timing_cache(base_cache.read_bytes())
    fast = builder.config.create_timing_cache(fast_cache.read_bytes())
    if not base.combine(fast, False):
        raise RuntimeError("Could not combine timing caches.")

    fast_keys = fast.queryKeys()
    replaced = 0
    skipped = 0
    for key in fast_keys:
        fast_value = fast.query(key)
        if not _is_valid_value(fast_value):
            skipped += 1
            continue
        base_value = base.query(key)
        if not _is_valid_value(base_value) or fast_value.timingMSec < base_value.timingMSec:
            if base.update(key, fast_value):
                replaced += 1
            else:
                skipped += 1
    merged_cache_out.write_bytes(bytes(base.serialize()))
    print(f"merged_keys={len(base.queryKeys())}")
    print(f"replaced_entries={replaced}")
    print(f"skipped_entries={skipped}")
    print(f"merged_cache_out={merged_cache_out}")


def do_blend_topn(
    *,
    base_cache: Path,
    fast_cache: Path,
    blended_cache_out: Path,
    top_n: int,
    manifest_out: Optional[Path],
) -> None:
    builder = EngineBuilder(workspace=8, builder_optimization_level=3)
    base = builder.config.create_timing_cache(base_cache.read_bytes())
    fast = builder.config.create_timing_cache(fast_cache.read_bytes())

    candidates = []
    for key in base.queryKeys():
        base_value = base.query(key)
        fast_value = fast.query(key)
        if not _is_valid_value(base_value) or not _is_valid_value(fast_value):
            continue
        if (
            fast_value.tacticHash != base_value.tacticHash
            and fast_value.timingMSec < base_value.timingMSec
        ):
            candidates.append(
                {
                    "key": str(key),
                    "base_tactic_hash": int(base_value.tacticHash),
                    "fast_tactic_hash": int(fast_value.tacticHash),
                    "base_timing_msec": float(base_value.timingMSec),
                    "fast_timing_msec": float(fast_value.timingMSec),
                    "timing_gain_msec": float(
                        base_value.timingMSec - fast_value.timingMSec
                    ),
                }
            )
    candidates.sort(key=lambda item: item["timing_gain_msec"], reverse=True)
    selected = candidates[:top_n]

    updated = 0
    for item in selected:
        key = trt.TimingCacheKey.parse(item["key"])
        fast_value = fast.query(key)
        if base.update(key, fast_value):
            updated += 1

    blended_cache_out.write_bytes(bytes(base.serialize()))
    print(f"candidate_keys={len(candidates)}")
    print(f"selected_top_n={len(selected)}")
    print(f"updated_entries={updated}")
    print(f"blended_cache_out={blended_cache_out}")
    if manifest_out is not None:
        manifest_out.write_text(json.dumps(selected, indent=2, sort_keys=True))
        print(f"manifest_out={manifest_out}")


def _collect_improved_candidates(
    base: trt.ITimingCache, fast: trt.ITimingCache
) -> List[dict]:
    candidates = []
    for key in base.queryKeys():
        base_value = base.query(key)
        fast_value = fast.query(key)
        if not _is_valid_value(base_value) or not _is_valid_value(fast_value):
            continue
        if (
            fast_value.tacticHash != base_value.tacticHash
            and fast_value.timingMSec < base_value.timingMSec
        ):
            candidates.append(
                {
                    "key": str(key),
                    "base_tactic_hash": int(base_value.tacticHash),
                    "fast_tactic_hash": int(fast_value.tacticHash),
                    "base_timing_msec": float(base_value.timingMSec),
                    "fast_timing_msec": float(fast_value.timingMSec),
                    "timing_gain_msec": float(
                        base_value.timingMSec - fast_value.timingMSec
                    ),
                }
            )
    candidates.sort(key=lambda item: item["timing_gain_msec"], reverse=True)
    return candidates


def do_blend_indices(
    *,
    base_cache: Path,
    fast_cache: Path,
    blended_cache_out: Path,
    indices: List[int],
    manifest_out: Optional[Path],
) -> None:
    builder = EngineBuilder(workspace=8, builder_optimization_level=3)
    base = builder.config.create_timing_cache(base_cache.read_bytes())
    fast = builder.config.create_timing_cache(fast_cache.read_bytes())

    candidates = _collect_improved_candidates(base=base, fast=fast)
    selected = []
    for idx in indices:
        if idx < 1 or idx > len(candidates):
            raise ValueError(
                f"Requested candidate index {idx}, but valid range is 1..{len(candidates)}."
            )
        selected.append(candidates[idx - 1])

    updated = 0
    for item in selected:
        key = trt.TimingCacheKey.parse(item["key"])
        fast_value = fast.query(key)
        if base.update(key, fast_value):
            updated += 1

    blended_cache_out.write_bytes(bytes(base.serialize()))
    print(f"candidate_keys={len(candidates)}")
    print(f"selected_indices={indices}")
    print(f"updated_entries={updated}")
    print(f"blended_cache_out={blended_cache_out}")
    if manifest_out is not None:
        manifest_out.write_text(json.dumps(selected, indent=2, sort_keys=True))
        print(f"manifest_out={manifest_out}")


def do_build(
    *,
    source_onnx_dir: Path,
    variant_dir: Path,
    cache_in: Path,
    cache_out: Optional[Path],
    builder_optimization_level: int,
    tactic_sources: Optional[List[str]],
    error_on_timing_cache_miss: bool,
) -> None:
    _build_engine(
        source_onnx_dir=source_onnx_dir,
        variant_dir=variant_dir,
        builder_optimization_level=builder_optimization_level,
        tactic_sources=tactic_sources,
        timing_cache_data=cache_in.read_bytes(),
        cache_out=cache_out,
        error_on_timing_cache_miss=error_on_timing_cache_miss,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    record = subparsers.add_parser("record")
    record.add_argument("--source_onnx_dir", default=str(DEFAULT_SOURCE))
    record.add_argument("--variant_dir", required=True)
    record.add_argument("--cache_out", required=True)
    record.add_argument("--builder_optimization_level", type=int, default=3)
    record.add_argument(
        "--tactic_source",
        action="append",
        dest="tactic_sources",
        default=None,
        choices=TACTIC_SOURCES,
    )

    merge_min = subparsers.add_parser("merge-min")
    merge_min.add_argument("--base_cache", required=True)
    merge_min.add_argument("--fast_cache", required=True)
    merge_min.add_argument("--merged_cache_out", required=True)

    blend_topn = subparsers.add_parser("blend-topn")
    blend_topn.add_argument("--base_cache", required=True)
    blend_topn.add_argument("--fast_cache", required=True)
    blend_topn.add_argument("--blended_cache_out", required=True)
    blend_topn.add_argument("--top_n", type=int, required=True)
    blend_topn.add_argument("--manifest_out", default=None)

    blend_indices = subparsers.add_parser("blend-indices")
    blend_indices.add_argument("--base_cache", required=True)
    blend_indices.add_argument("--fast_cache", required=True)
    blend_indices.add_argument("--blended_cache_out", required=True)
    blend_indices.add_argument("--index", action="append", dest="indices", type=int, required=True)
    blend_indices.add_argument("--manifest_out", default=None)

    build = subparsers.add_parser("build")
    build.add_argument("--source_onnx_dir", default=str(DEFAULT_SOURCE))
    build.add_argument("--variant_dir", required=True)
    build.add_argument("--cache_in", required=True)
    build.add_argument("--cache_out", default=None)
    build.add_argument("--builder_optimization_level", type=int, default=3)
    build.add_argument("--error_on_timing_cache_miss", action="store_true")
    build.add_argument(
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
            cache_out=Path(args.cache_out),
            builder_optimization_level=args.builder_optimization_level,
            tactic_sources=args.tactic_sources,
        )
        return
    if args.mode == "merge-min":
        do_merge_min(
            base_cache=Path(args.base_cache),
            fast_cache=Path(args.fast_cache),
            merged_cache_out=Path(args.merged_cache_out),
        )
        return
    if args.mode == "blend-topn":
        do_blend_topn(
            base_cache=Path(args.base_cache),
            fast_cache=Path(args.fast_cache),
            blended_cache_out=Path(args.blended_cache_out),
            top_n=args.top_n,
            manifest_out=(
                Path(args.manifest_out) if args.manifest_out is not None else None
            ),
        )
        return
    if args.mode == "blend-indices":
        do_blend_indices(
            base_cache=Path(args.base_cache),
            fast_cache=Path(args.fast_cache),
            blended_cache_out=Path(args.blended_cache_out),
            indices=list(args.indices),
            manifest_out=(
                Path(args.manifest_out) if args.manifest_out is not None else None
            ),
        )
        return
    do_build(
        source_onnx_dir=Path(args.source_onnx_dir),
        variant_dir=Path(args.variant_dir),
        cache_in=Path(args.cache_in),
        cache_out=Path(args.cache_out) if args.cache_out is not None else None,
        builder_optimization_level=args.builder_optimization_level,
        tactic_sources=args.tactic_sources,
        error_on_timing_cache_miss=args.error_on_timing_cache_miss,
    )


if __name__ == "__main__":
    main()
