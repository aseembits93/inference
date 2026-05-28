import argparse
import ctypes
import json
import shutil
import subprocess
import sys
from pathlib import Path

import onnx
from onnx import helper
import tensorrt as trt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEV_ROOT = REPO_ROOT / "inference_models" / "development"
if str(DEV_ROOT) not in sys.path:
    sys.path.insert(0, str(DEV_ROOT))

from compilation.engine_builder import EngineBuilder  # noqa: E402


PLUGIN_NAME = "RfProbeEncoderAttentionCore"
PLUGIN_SO = Path("/tmp/rfprobe_native_plugin/libRfProbeEncoderAttentionCore.so")
PLUGIN_SRC = REPO_ROOT / "temp" / "native_encoder_attention_core_plugin.cpp"
TRT_INCLUDE = Path("/tmp/TensorRT-10.12/include")
TRT_LIB_DIR = (
    REPO_ROOT / ".venv" / "lib" / "python3.12" / "site-packages" / "tensorrt_libs"
)


def _copy_variant_tree(source_dir: Path, target_dir: Path) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("class_names.txt", "inference_config.json"):
        shutil.copy2(source_dir / name, target_dir / name)


def _compile_plugin() -> Path:
    if not TRT_INCLUDE.exists():
        raise FileNotFoundError(
            f"Expected TensorRT headers at {TRT_INCLUDE}. Clone release/10.12 first."
        )
    PLUGIN_SO.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "/usr/bin/nvcc",
        "-std=c++17",
        "-x",
        "cu",
        "-shared",
        "-Xcompiler=-fPIC",
        str(PLUGIN_SRC),
        f"-I{TRT_INCLUDE}",
        f"-L{TRT_LIB_DIR}",
        "-l:libnvinfer.so.10",
        "-l:libnvinfer_plugin.so.10",
        "-lcublas",
        "-lcudart",
        "-Xlinker",
        f"-rpath,{TRT_LIB_DIR}",
        "-Xlinker",
        "-rpath,/lib/x86_64-linux-gnu",
        "-o",
        str(PLUGIN_SO),
    ]
    subprocess.run(cmd, check=True)
    return PLUGIN_SO


def _load_plugin_library(plugin_so: Path) -> None:
    ctypes.CDLL(str(plugin_so), mode=ctypes.RTLD_GLOBAL)
    creator = trt.get_plugin_registry().get_creator(PLUGIN_NAME, "1", "")
    if creator is None:
        raise RuntimeError(f"Failed to register plugin creator for {PLUGIN_NAME}.")


def _replace_layers(model: onnx.ModelProto, *, layers: list[int]) -> int:
    nodes = list(model.graph.node)
    inserted = {}
    skipped = set()
    replaced = 0

    for layer in layers:
        prefix = (
            f"/backbone/backbone.0/encoder/encoder/encoder/layer.{layer}/attention/attention/"
        )
        query_matmul_idx = next(
            i for i, node in enumerate(nodes) if node.name == f"{prefix}query/MatMul"
        )
        query_add_idx = next(
            i for i, node in enumerate(nodes) if node.name == f"{prefix}query/Add"
        )
        key_matmul_idx = next(
            i for i, node in enumerate(nodes) if node.name == f"{prefix}key/MatMul"
        )
        key_add_idx = next(
            i for i, node in enumerate(nodes) if node.name == f"{prefix}key/Add"
        )
        value_matmul_idx = next(
            i for i, node in enumerate(nodes) if node.name == f"{prefix}value/MatMul"
        )
        value_add_idx = next(
            i for i, node in enumerate(nodes) if node.name == f"{prefix}value/Add"
        )
        insert_after = max(query_add_idx, key_add_idx, value_add_idx)
        keep_indices = {
            query_matmul_idx,
            query_add_idx,
            key_matmul_idx,
            key_add_idx,
            value_matmul_idx,
            value_add_idx,
        }
        keep_names = {
            f"{prefix}query/MatMul",
            f"{prefix}query/Add",
            f"{prefix}key/MatMul",
            f"{prefix}key/Add",
            f"{prefix}value/MatMul",
            f"{prefix}value/Add",
        }
        inserted[insert_after + 1] = helper.make_node(
            PLUGIN_NAME,
            inputs=[
                f"{prefix}query/Add_output_0",
                f"{prefix}key/Add_output_0",
                f"{prefix}value/Add_output_0",
            ],
            outputs=[f"{prefix}Reshape_3_output_0"],
            name=f"{prefix}AttentionCorePlugin",
            plugin_version="1",
            plugin_namespace="",
        )
        for idx, node in enumerate(nodes):
            if prefix not in node.name:
                continue
            if idx in keep_indices or node.name in keep_names:
                continue
            skipped.add(idx)
        replaced += 1

    new_nodes = []
    for idx, node in enumerate(nodes):
        if idx in inserted:
            new_nodes.append(inserted[idx])
        if idx in skipped:
            continue
        new_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return replaced


def _append_debug_outputs(model: onnx.ModelProto, *, layers: list[int]) -> None:
    existing = {value.name for value in model.graph.output}
    for layer in layers:
        prefix = (
            f"/backbone/backbone.0/encoder/encoder/encoder/layer.{layer}/attention/attention/"
        )
        output_names = [
            f"{prefix}query/Add_output_0",
            f"{prefix}key/Add_output_0",
            f"{prefix}value/Add_output_0",
            f"{prefix}Reshape_3_output_0",
        ]
        for output_name in output_names:
            if output_name in existing:
                continue
            model.graph.output.append(
                helper.make_tensor_value_info(
                    output_name,
                    onnx.TensorProto.FLOAT,
                    [1, 677, 384],
                )
            )
            existing.add(output_name)


def _write_runtime_files(target_dir: Path, plugin_so: Path) -> None:
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
    (target_dir / "native_plugin_path.txt").write_text(str(plugin_so) + "\n")


def _apply_plugin_precision_constraints(
    builder: EngineBuilder,
    *,
    plugin_precision: trt.DataType | None,
    precision_constraint_mode: str,
) -> int:
    if plugin_precision is None:
        return 0
    if precision_constraint_mode == "obey":
        builder.config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    else:
        builder.config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    constrained = 0
    for idx in range(builder.network.num_layers):
        layer = builder.network.get_layer(idx)
        if "AttentionCorePlugin" not in layer.name:
            continue
        layer.precision = plugin_precision
        for output_idx in range(layer.num_outputs):
            layer.set_output_type(output_idx, plugin_precision)
        constrained += 1
    return constrained


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 5])
    parser.add_argument("--builder-optimization-level", type=int, default=3)
    parser.add_argument(
        "--append-plugin-outputs",
        action="store_true",
        help="Mark replaced plugin outputs as graph outputs for direct TensorRT probing.",
    )
    parser.add_argument(
        "--plugin-precision",
        choices=("fp32",),
        default=None,
    )
    parser.add_argument(
        "--precision-constraint-mode",
        choices=("prefer", "obey"),
        default="prefer",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    _copy_variant_tree(source_dir=source_dir, target_dir=target_dir)

    plugin_so = _compile_plugin()
    _load_plugin_library(plugin_so)

    model = onnx.load(str(source_dir / "weights.onnx"))
    replaced = _replace_layers(model, layers=args.layers)
    if replaced != len(args.layers):
        raise ValueError(f"Expected to replace {len(args.layers)} layers, got {replaced}")
    if args.append_plugin_outputs:
        _append_debug_outputs(model, layers=args.layers)
    patched_onnx = target_dir / "weights.onnx"
    onnx.save(model, str(patched_onnx))

    builder = EngineBuilder(
        workspace=8,
        builder_optimization_level=args.builder_optimization_level,
    )
    builder.config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    builder.create_network(str(patched_onnx))
    constrained = _apply_plugin_precision_constraints(
        builder,
        plugin_precision=(trt.float32 if args.plugin_precision == "fp32" else None),
        precision_constraint_mode=args.precision_constraint_mode,
    )
    if args.plugin_precision is not None and constrained != len(args.layers):
        raise ValueError(
            f"Expected to constrain {len(args.layers)} plugin layers, got {constrained}"
        )
    builder.create_engine(
        engine_path=str(target_dir / "engine.plan"),
        precision="fp16",
        input_name="input",
        input_size=(312, 312),
        dynamic_batch_sizes=None,
        trt_version_compatible=False,
        same_compute_compatibility=False,
    )
    _write_runtime_files(target_dir=target_dir, plugin_so=plugin_so)
    print(f"variant_dir={target_dir}")
    print(f"plugin_so={plugin_so}")


if __name__ == "__main__":
    main()
