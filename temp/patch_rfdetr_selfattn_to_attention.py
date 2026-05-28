import argparse
import math
from pathlib import Path
from typing import Iterable

import onnx
from onnx import helper, version_converter


_NUM_HEADS = 8
_HEAD_DIM = 32
_ATTN_SCALE = float(1.0 / math.sqrt(float(_HEAD_DIM)))


def _ensure_main_opset(model: onnx.ModelProto, version: int) -> None:
    for opset in model.opset_import:
        if opset.domain in ("", "main"):
            if opset.version < version:
                opset.version = version
            return
    model.opset_import.append(helper.make_operatorsetid("", version))


def patch_model(
    input_path: Path,
    output_path: Path,
    *,
    layers: Iterable[int],
    opset_version: int,
) -> None:
    model = onnx.load(str(input_path))
    main_opset = next(
        (opset.version for opset in model.opset_import if opset.domain in ("", "main")),
        0,
    )
    if main_opset and main_opset < opset_version:
        model = version_converter.convert_version(model, opset_version)
    _ensure_main_opset(model, opset_version)

    nodes = list(model.graph.node)
    by_name = {node.name: node for node in nodes}
    patched = 0

    for layer in layers:
        prefix = f"/transformer/decoder/layers.{layer}/self_attn/"
        q = by_name[f"{prefix}Reshape_3"]
        k = by_name[f"{prefix}Reshape_4"]
        v = by_name[f"{prefix}Reshape_5"]
        transpose_6 = by_name[f"{prefix}Transpose_6"]
        attention_out = f"{prefix}Attention_output_0"
        attention_node = helper.make_node(
            "Attention",
            inputs=[q.output[0], k.output[0], v.output[0]],
            outputs=[attention_out],
            name=f"{prefix}Attention",
            q_num_heads=_NUM_HEADS,
            kv_num_heads=_NUM_HEADS,
            scale=_ATTN_SCALE,
        )

        insert_at = nodes.index(v) + 1
        nodes.insert(insert_at, attention_node)
        transpose_6.input[0] = attention_out
        patched += 1

    del model.graph.node[:]
    model.graph.node.extend(nodes)
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    print(f"patched_self_attn_attention_layers={patched}")
    print(f"saved={output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[2, 3],
    )
    parser.add_argument("--opset", type=int, default=23)
    args = parser.parse_args()
    patch_model(
        input_path=Path(args.input),
        output_path=Path(args.output),
        layers=args.layers,
        opset_version=args.opset,
    )


if __name__ == "__main__":
    main()
