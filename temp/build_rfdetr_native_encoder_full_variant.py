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


ATTN_PLUGIN_NAME = "RfProbeEncoderAttentionCore"
ATTN_PLUGIN_SO = Path("/tmp/rfprobe_native_plugin/libRfProbeEncoderAttentionCore.so")
ATTN_PLUGIN_SRC = REPO_ROOT / "temp" / "native_encoder_attention_core_plugin.cpp"
PROJ_PLUGIN_NAME = "RfProbeProjectionMatmul"
PROJ_PLUGIN_SO = Path("/tmp/rfprobe_native_plugin/libRfProbeProjectionMatmul.so")
PROJ_PLUGIN_SRC = REPO_ROOT / "temp" / "native_projection_matmul_plugin.cpp"
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


def _compile_attention_plugin() -> Path:
    ATTN_PLUGIN_SO.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "/usr/bin/nvcc",
        "-std=c++17",
        "-x",
        "cu",
        "-shared",
        "-Xcompiler=-fPIC",
        str(ATTN_PLUGIN_SRC),
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
        str(ATTN_PLUGIN_SO),
    ]
    subprocess.run(cmd, check=True)
    return ATTN_PLUGIN_SO


def _compile_projection_plugin() -> Path:
    PROJ_PLUGIN_SO.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "/usr/bin/g++",
        "-std=c++17",
        "-shared",
        "-fPIC",
        str(PROJ_PLUGIN_SRC),
        f"-I{TRT_INCLUDE}",
        str(TRT_LIB_DIR / "libnvinfer.so.10"),
        str(TRT_LIB_DIR / "libnvinfer_plugin.so.10"),
        "-lcublasLt",
        "-lcublas",
        "-lcudart",
        f"-Wl,-rpath,{TRT_LIB_DIR}",
        "-Wl,-rpath,/lib/x86_64-linux-gnu",
        "-o",
        str(PROJ_PLUGIN_SO),
    ]
    subprocess.run(cmd, check=True)
    return PROJ_PLUGIN_SO


def _load_plugin_library(plugin_so: Path, plugin_name: str) -> None:
    ctypes.CDLL(str(plugin_so), mode=ctypes.RTLD_GLOBAL)
    creator = trt.get_plugin_registry().get_creator(plugin_name, "1", "")
    if creator is None:
        raise RuntimeError(f"Failed to register plugin creator for {plugin_name}.")


def _replace_attention_cores(model: onnx.ModelProto, *, layers: list[int]) -> int:
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
            ATTN_PLUGIN_NAME,
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


def _replace_projection_nodes(model: onnx.ModelProto, *, layers: list[int]) -> int:
    wanted = []
    for layer in layers:
        prefix = (
            f"/backbone/backbone.0/encoder/encoder/encoder/layer.{layer}/attention/attention/"
        )
        wanted.extend(
            [
                f"{prefix}query/MatMul",
                f"{prefix}key/MatMul",
                f"{prefix}value/MatMul",
            ]
        )
    wanted_set = set(wanted)
    replaced = 0
    for idx, node in enumerate(model.graph.node):
        if node.name not in wanted_set:
            continue
        model.graph.node.remove(node)
        model.graph.node.insert(
            idx,
            helper.make_node(
                PROJ_PLUGIN_NAME,
                inputs=list(node.input),
                outputs=list(node.output),
                name=node.name,
                plugin_version="1",
                plugin_namespace="",
            ),
        )
        replaced += 1
    return replaced


def _append_debug_outputs(model: onnx.ModelProto, *, layers: list[int]) -> None:
    existing = {value.name for value in model.graph.output}
    for layer in layers:
        layer_prefix = f"/backbone/backbone.0/encoder/encoder/encoder/layer.{layer}/"
        prefix = (
            f"{layer_prefix}attention/attention/"
        )
        output_names = [
            f"{layer_prefix}norm1/LayerNormalization_output_0",
            f"{prefix}query/MatMul_output_0",
            f"{prefix}key/MatMul_output_0",
            f"{prefix}value/MatMul_output_0",
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


def _write_runtime_files(target_dir: Path, plugin_paths: list[Path]) -> None:
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
    (target_dir / "native_plugin_path.txt").write_text(
        "".join(f"{path}\n" for path in plugin_paths)
    )


def _apply_attention_plugin_precision_constraints(
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
    parser.add_argument("--layers", type=int, nargs="+", default=[4])
    parser.add_argument("--builder-optimization-level", type=int, default=3)
    parser.add_argument("--append-plugin-outputs", action="store_true")
    parser.add_argument("--plugin-precision", choices=("fp32",), default=None)
    parser.add_argument(
        "--precision-constraint-mode",
        choices=("prefer", "obey"),
        default="prefer",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    _copy_variant_tree(source_dir=source_dir, target_dir=target_dir)

    attn_plugin_so = _compile_attention_plugin()
    proj_plugin_so = _compile_projection_plugin()
    _load_plugin_library(attn_plugin_so, ATTN_PLUGIN_NAME)
    _load_plugin_library(proj_plugin_so, PROJ_PLUGIN_NAME)

    model = onnx.load(str(source_dir / "weights.onnx"))
    replaced_proj = _replace_projection_nodes(model, layers=args.layers)
    expected_proj = len(args.layers) * 3
    if replaced_proj != expected_proj:
        raise ValueError(
            f"Expected to replace {expected_proj} projection nodes, got {replaced_proj}"
        )
    replaced_attn = _replace_attention_cores(model, layers=args.layers)
    if replaced_attn != len(args.layers):
        raise ValueError(
            f"Expected to replace {len(args.layers)} attention cores, got {replaced_attn}"
        )
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
    plugin_precision = trt.float32 if args.plugin_precision == "fp32" else None
    constrained = _apply_attention_plugin_precision_constraints(
        builder,
        plugin_precision=plugin_precision,
        precision_constraint_mode=args.precision_constraint_mode,
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
    _write_runtime_files(
        target_dir=target_dir,
        plugin_paths=[attn_plugin_so, proj_plugin_so],
    )
    print(f"variant_dir={target_dir}")
    print(f"attn_plugin_so={attn_plugin_so}")
    print(f"proj_plugin_so={proj_plugin_so}")
    print(f"attention_plugin_precision_constraints={constrained}")


if __name__ == "__main__":
    main()
