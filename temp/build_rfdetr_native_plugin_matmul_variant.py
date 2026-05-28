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


PLUGIN_NAME = "RfProbeProjectionMatmul"
PLUGIN_SO = Path("/tmp/rfprobe_native_plugin/libRfProbeProjectionMatmul.so")
PLUGIN_SRC = REPO_ROOT / "temp" / "native_projection_matmul_plugin.cpp"
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
        "/usr/bin/g++",
        "-std=c++17",
        "-shared",
        "-fPIC",
        str(PLUGIN_SRC),
        f"-I{TRT_INCLUDE}",
        str(TRT_LIB_DIR / "libnvinfer.so.10"),
        str(TRT_LIB_DIR / "libnvinfer_plugin.so.10"),
        "-lcublasLt",
        "-lcublas",
        "-lcudart",
        f"-Wl,-rpath,{TRT_LIB_DIR}",
        "-Wl,-rpath,/lib/x86_64-linux-gnu",
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


def _replace_nodes(model: onnx.ModelProto, *, node_names: list[str]) -> int:
    wanted = set(node_names)
    replaced = 0
    for idx, node in enumerate(model.graph.node):
        if node.name not in wanted:
            continue
        model.graph.node.remove(node)
        model.graph.node.insert(
            idx,
            helper.make_node(
                PLUGIN_NAME,
                inputs=list(node.input),
                outputs=list(node.output),
                name=node.name,
                plugin_version="1",
                plugin_namespace="",
            ),
        )
        replaced += 1
    return replaced


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument(
        "--node-name",
        action="append",
        default=None,
        help="Exact ONNX node name to replace. May be provided multiple times.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="If set, replace decoder self_attn MatMul_2 for these layer indices.",
    )
    parser.add_argument("--builder-optimization-level", type=int, default=3)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    _copy_variant_tree(source_dir=source_dir, target_dir=target_dir)

    plugin_so = _compile_plugin()
    _load_plugin_library(plugin_so)

    model = onnx.load(str(source_dir / "weights.onnx"))
    node_names = (
        [
            f"/transformer/decoder/layers.{layer}/self_attn/MatMul_2"
            for layer in args.layers
        ]
        if args.layers is not None
        else (
            args.node_name
            if args.node_name is not None
            else ["/transformer/decoder/layers.0/self_attn/MatMul_2"]
        )
    )
    replaced = _replace_nodes(model, node_names=node_names)
    if replaced != len(node_names):
        raise ValueError(
            f"Expected to replace {len(node_names)} nodes, got {replaced}"
        )
    patched_onnx = target_dir / "weights.onnx"
    onnx.save(model, str(patched_onnx))

    builder = EngineBuilder(
        workspace=8,
        builder_optimization_level=args.builder_optimization_level,
    )
    builder.config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    builder.create_network(str(patched_onnx))
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
