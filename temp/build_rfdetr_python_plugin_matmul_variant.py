import argparse
import json
import shutil
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
from temp.trt_exact_projection_matmul_plugin import (  # noqa: E402,F401
    AOT_TRITON_ADD_PLUGIN_ID,
    AOT_TRITON_PLUGIN_ID,
    PLUGIN_ID,
    TRITON_PLUGIN_ID,
)

_PLUGIN_IDS = {
    "torch": PLUGIN_ID,
    "triton": TRITON_PLUGIN_ID,
    "aot-triton": AOT_TRITON_PLUGIN_ID,
    "aot-add": AOT_TRITON_ADD_PLUGIN_ID,
}


def _copy_variant_tree(source_dir: Path, target_dir: Path) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("class_names.txt", "inference_config.json"):
        shutil.copy2(source_dir / name, target_dir / name)


def _replace_node(model: onnx.ModelProto, *, node_name: str, plugin_id: str) -> int:
    namespace, op_name = _PLUGIN_IDS[plugin_id].split("::")
    replaced = 0
    for idx, node in enumerate(model.graph.node):
        if node.name != node_name:
            continue
        model.graph.node.remove(node)
        model.graph.node.insert(
            idx,
            helper.make_node(
                op_name,
                inputs=list(node.input),
                outputs=list(node.output),
                name=node.name,
                plugin_version="1",
                plugin_namespace=namespace,
            ),
        )
        replaced += 1
    return replaced


def _write_runtime_files(target_dir: Path) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument(
        "--plugin-id",
        choices=("torch", "triton", "aot-triton", "aot-add"),
        default="torch",
    )
    parser.add_argument(
        "--node-name",
        default="/transformer/decoder/layers.0/self_attn/MatMul_2",
    )
    parser.add_argument("--builder-optimization-level", type=int, default=3)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    _copy_variant_tree(source_dir=source_dir, target_dir=target_dir)

    model = onnx.load(str(source_dir / "weights.onnx"))
    replaced = _replace_node(model, node_name=args.node_name, plugin_id=args.plugin_id)
    if replaced != 1:
        raise ValueError(
            f"Expected to replace exactly one node named {args.node_name}, got {replaced}"
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
    _write_runtime_files(target_dir=target_dir)
    print(f"variant_dir={target_dir}")


if __name__ == "__main__":
    main()
