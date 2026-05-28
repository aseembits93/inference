import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import onnx
from onnx import TensorProto, helper


_RESHAPE_3D = [1, 8, 100, 32]
_RESHAPE_CTX = [100, 256]
_RESHAPE_OUT = [100, 1, 256]


def _add_int64_initializer(
    graph: onnx.GraphProto,
    *,
    name: str,
    values: List[int],
) -> str:
    tensor = helper.make_tensor(
        name=name,
        data_type=TensorProto.INT64,
        dims=[len(values)],
        vals=values,
    )
    graph.initializer.append(tensor)
    return name


def _find_index(nodes: List[onnx.NodeProto], name: str) -> int:
    for idx, node in enumerate(nodes):
        if node.name == name:
            return idx
    raise KeyError(f"Missing node: {name}")


def _make_static_reshape(
    graph: onnx.GraphProto,
    *,
    node_name: str,
    input_name: str,
    output_name: str,
    shape_values: List[int],
) -> onnx.NodeProto:
    shape_name = _add_int64_initializer(
        graph,
        name=f"{node_name.replace('/', '_').replace('.', '_')}_static_shape",
        values=shape_values,
    )
    return helper.make_node(
        "Reshape",
        inputs=[input_name, shape_name],
        outputs=[output_name],
        name=node_name,
    )


def patch_model(
    input_path: Path,
    output_path: Path,
    *,
    layers: Iterable[int],
) -> None:
    model = onnx.load(str(input_path))
    nodes = list(model.graph.node)

    insertions: Dict[int, List[onnx.NodeProto]] = {}
    skipped_indices: set[int] = set()
    patched = 0

    for layer in layers:
        prefix = f"/transformer/decoder/layers.{layer}/self_attn/"

        # Replace dynamic q/k/v reshaping with static equivalents at the original
        # Reshape node positions so topological order stays valid.
        reshape3_idx = _find_index(nodes, f"{prefix}Reshape_3")
        reshape4_idx = _find_index(nodes, f"{prefix}Reshape_4")
        reshape5_idx = _find_index(nodes, f"{prefix}Reshape_5")
        insertions[reshape3_idx] = [
            _make_static_reshape(
                model.graph,
                node_name=f"{prefix}Reshape_3",
                input_name=f"{prefix}Transpose_2_output_0",
                output_name=f"{prefix}Reshape_3_output_0",
                shape_values=_RESHAPE_3D,
            )
        ]
        insertions[reshape4_idx] = [
            _make_static_reshape(
                model.graph,
                node_name=f"{prefix}Reshape_4",
                input_name=f"{prefix}Transpose_3_output_0",
                output_name=f"{prefix}Reshape_4_output_0",
                shape_values=_RESHAPE_3D,
            )
        ]
        insertions[reshape5_idx] = [
            _make_static_reshape(
                model.graph,
                node_name=f"{prefix}Reshape_5",
                input_name=f"{prefix}Transpose_4_output_0",
                output_name=f"{prefix}Reshape_5_output_0",
                shape_values=_RESHAPE_3D,
            )
        ]
        skip_qkv_names = {
            f"{prefix}Shape_3",
            f"{prefix}Constant_5",
            f"{prefix}Gather_3",
            f"{prefix}Unsqueeze_3",
            f"{prefix}Unsqueeze_4",
            f"{prefix}Unsqueeze_5",
            f"{prefix}Concat_1",
            f"{prefix}Shape_4",
            f"{prefix}Constant_6",
            f"{prefix}Gather_4",
            f"{prefix}Unsqueeze_6",
            f"{prefix}Unsqueeze_7",
            f"{prefix}Unsqueeze_8",
            f"{prefix}Concat_2",
            f"{prefix}Shape_5",
            f"{prefix}Constant_7",
            f"{prefix}Gather_5",
            f"{prefix}Unsqueeze_9",
            f"{prefix}Constant_8",
            f"{prefix}Unsqueeze_10",
            f"{prefix}Unsqueeze_11",
            f"{prefix}Concat_3",
            f"{prefix}Unsqueeze_12",
            f"{prefix}Constant_9",
            f"{prefix}Unsqueeze_13",
            f"{prefix}Unsqueeze_14",
            f"{prefix}Concat_4",
            f"{prefix}Unsqueeze_15",
            f"{prefix}Constant_10",
            f"{prefix}Unsqueeze_16",
            f"{prefix}Unsqueeze_17",
            f"{prefix}Concat_5",
            f"{prefix}Reshape_3",
            f"{prefix}Reshape_4",
            f"{prefix}Reshape_5",
        }
        skipped_indices.update(
            idx for idx, node in enumerate(nodes) if node.name in skip_qkv_names
        )

        # Replace dynamic flattening of attention context before out-proj.
        reshape6_idx = _find_index(nodes, f"{prefix}Reshape_6")
        insertions[reshape6_idx] = [
            _make_static_reshape(
                model.graph,
                node_name=f"{prefix}Reshape_6",
                input_name=f"{prefix}Transpose_6_output_0",
                output_name=f"{prefix}Reshape_6_output_0",
                shape_values=_RESHAPE_CTX,
            )
        ]
        skip_ctx_names = {
            f"{prefix}Mul_3",
            f"{prefix}Unsqueeze_18",
            f"{prefix}Unsqueeze_19",
            f"{prefix}Concat_6",
            f"{prefix}Reshape_6",
        }
        skipped_indices.update(
            idx for idx, node in enumerate(nodes) if node.name in skip_ctx_names
        )

        # Replace dynamic restore of [S,B,H] after out-proj.
        reshape7_idx = _find_index(nodes, f"{prefix}Reshape_7")
        insertions[reshape7_idx] = [
            _make_static_reshape(
                model.graph,
                node_name=f"{prefix}Reshape_7",
                input_name=f"{prefix}Gemm_output_0",
                output_name=f"{prefix}Reshape_7_output_0",
                shape_values=_RESHAPE_OUT,
            )
        ]
        skip_out_names = {
            f"{prefix}Shape_7",
            f"{prefix}Constant_14",
            f"{prefix}Gather_6",
            f"{prefix}Unsqueeze_20",
            f"{prefix}Unsqueeze_21",
            f"{prefix}Unsqueeze_22",
            f"{prefix}Concat_7",
            f"{prefix}Reshape_7",
        }
        skipped_indices.update(
            idx for idx, node in enumerate(nodes) if node.name in skip_out_names
        )

        patched += 1

    new_nodes: List[onnx.NodeProto] = []
    for idx, node in enumerate(nodes):
        if idx in insertions:
            new_nodes.extend(insertions[idx])
        if idx in skipped_indices:
            continue
        new_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    print(f"patched_self_attn_canonical_static_layers={patched}")
    print(f"saved={output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", nargs="+", type=int, default=[3])
    args = parser.parse_args()
    patch_model(
        input_path=Path(args.input),
        output_path=Path(args.output),
        layers=args.layers,
    )


if __name__ == "__main__":
    main()
