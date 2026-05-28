import argparse
from pathlib import Path
from typing import Dict, Iterable, Sequence

import onnx
from onnx import TensorProto, helper


_SELF_ATTN_RESHAPE_SHAPES = {
    "Reshape": [100, 8, 32],
    "Reshape_1": [100, 8, 32],
    "Reshape_2": [100, 8, 32],
    "Reshape_3": [1, 8, 100, 32],
    "Reshape_4": [1, 8, 100, 32],
    "Reshape_5": [1, 8, 100, 32],
    "Reshape_6": [100, 256],
    "Reshape_7": [100, 1, 256],
}


def _add_int64_initializer(
    graph: onnx.GraphProto,
    *,
    name: str,
    values: Sequence[int],
) -> str:
    tensor = helper.make_tensor(
        name=name,
        data_type=TensorProto.INT64,
        dims=[len(values)],
        vals=list(values),
    )
    graph.initializer.append(tensor)
    return name


def patch_model(
    input_path: Path,
    output_path: Path,
    *,
    layers: Iterable[int],
) -> None:
    model = onnx.load(str(input_path))
    target_layers = set(layers)
    patched = 0

    for node in model.graph.node:
        if node.op_type != "Reshape":
            continue
        for layer in target_layers:
            prefix = f"/transformer/decoder/layers.{layer}/self_attn/"
            if not node.name.startswith(prefix):
                continue
            suffix = node.name[len(prefix) :]
            shape = _SELF_ATTN_RESHAPE_SHAPES.get(suffix)
            if shape is None:
                break
            init_name = _add_int64_initializer(
                model.graph,
                name=f"{node.name.replace('/', '_').replace('.', '_')}_static_shape",
                values=shape,
            )
            if len(node.input) < 2:
                raise RuntimeError(f"Unexpected Reshape node without shape input: {node.name}")
            node.input[1] = init_name
            patched += 1
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))
    print(f"patched_self_attn_reshape_nodes={patched}")
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
        help="Decoder self-attention layer indices to patch.",
    )
    args = parser.parse_args()
    patch_model(
        input_path=Path(args.input),
        output_path=Path(args.output),
        layers=args.layers,
    )


if __name__ == "__main__":
    main()
