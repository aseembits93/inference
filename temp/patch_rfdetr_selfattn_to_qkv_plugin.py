import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Tuple

import onnx
from onnx import TensorProto, helper


_HIDDEN_SIZE = 256
_NUM_HEADS = 8
_DQ_PROBS = 1.0
_HEAD_SIZE = _HIDDEN_SIZE // _NUM_HEADS


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


def _find_node_by_name(nodes: List[onnx.NodeProto], name: str) -> onnx.NodeProto:
    for node in nodes:
        if node.name == name:
            return node
    raise KeyError(f"Missing node: {name}")


def _make_plugin_rewrite(
    graph: onnx.GraphProto,
    *,
    prefix: str,
    q_input: str,
    k_input: str,
    v_input: str,
    layout: Literal["seq-first", "batch-first"],
    type_id: int,
) -> List[onnx.NodeProto]:
    base = prefix.replace("/", "_").replace(".", "_")
    qkv_head_shape = _add_int64_initializer(
        graph,
        name=f"{base}_qkv_head_shape",
        values=[0, 0, _NUM_HEADS, _HEAD_SIZE],
    )
    qkv_head_unsqueeze_axes = _add_int64_initializer(
        graph,
        name=f"{base}_qkv_head_unsqueeze_axes",
        values=[3],
    )
    qkv_packed_shape = _add_int64_initializer(
        graph,
        name=f"{base}_qkv_packed_shape",
        values=[0, 0, 3 * _HIDDEN_SIZE],
    )
    unsqueeze_axes = _add_int64_initializer(
        graph,
        name=f"{base}_packed_unsqueeze_axes",
        values=[3, 4],
    )
    squeeze_ctx_axes = _add_int64_initializer(
        graph,
        name=f"{base}_ctx_squeeze_axes",
        values=[3, 4],
    )
    squeeze_batch_axes = _add_int64_initializer(
        graph,
        name=f"{base}_batch_squeeze_axes",
        values=[1 if layout == "seq-first" else 0],
    )

    q_source = q_input
    k_source = k_input
    v_source = v_input
    qkv_prefix_nodes: List[onnx.NodeProto] = []
    if layout == "batch-first":
        q_source = f"{prefix}QBatchFirst_output_0"
        k_source = f"{prefix}KBatchFirst_output_0"
        v_source = f"{prefix}VBatchFirst_output_0"
        qkv_prefix_nodes.extend(
            [
                helper.make_node(
                    "Transpose",
                    inputs=[q_input],
                    outputs=[q_source],
                    name=f"{prefix}QBatchFirst",
                    perm=[1, 0, 2],
                ),
                helper.make_node(
                    "Transpose",
                    inputs=[k_input],
                    outputs=[k_source],
                    name=f"{prefix}KBatchFirst",
                    perm=[1, 0, 2],
                ),
                helper.make_node(
                    "Transpose",
                    inputs=[v_input],
                    outputs=[v_source],
                    name=f"{prefix}VBatchFirst",
                    perm=[1, 0, 2],
                ),
            ]
        )

    q_head = f"{prefix}QHead_output_0"
    k_head = f"{prefix}KHead_output_0"
    v_head = f"{prefix}VHead_output_0"
    q_head_unsqueezed = f"{prefix}QHeadUnsqueeze_output_0"
    k_head_unsqueezed = f"{prefix}KHeadUnsqueeze_output_0"
    v_head_unsqueezed = f"{prefix}VHeadUnsqueeze_output_0"
    qkv_concat = f"{prefix}PackedQKVConcat_output_0"
    qkv_packed_flat = f"{prefix}PackedQKVReshape_output_0"
    qkv_packed = f"{prefix}PackedQKVUnsqueeze_output_0"
    plugin_ctx = f"{prefix}CustomQKVToContextPluginDynamic_output_0"
    ctx_no_hw = f"{prefix}ContextSqueeze_output_0"
    ctx_flat = f"{prefix}ContextFlat_output_0"

    return qkv_prefix_nodes + [
        helper.make_node(
            "Reshape",
            inputs=[q_source, qkv_head_shape],
            outputs=[q_head],
            name=f"{prefix}QHead",
        ),
        helper.make_node(
            "Reshape",
            inputs=[k_source, qkv_head_shape],
            outputs=[k_head],
            name=f"{prefix}KHead",
        ),
        helper.make_node(
            "Reshape",
            inputs=[v_source, qkv_head_shape],
            outputs=[v_head],
            name=f"{prefix}VHead",
        ),
        helper.make_node(
            "Unsqueeze",
            inputs=[q_head, qkv_head_unsqueeze_axes],
            outputs=[q_head_unsqueezed],
            name=f"{prefix}QHeadUnsqueeze",
        ),
        helper.make_node(
            "Unsqueeze",
            inputs=[k_head, qkv_head_unsqueeze_axes],
            outputs=[k_head_unsqueezed],
            name=f"{prefix}KHeadUnsqueeze",
        ),
        helper.make_node(
            "Unsqueeze",
            inputs=[v_head, qkv_head_unsqueeze_axes],
            outputs=[v_head_unsqueezed],
            name=f"{prefix}VHeadUnsqueeze",
        ),
        helper.make_node(
            "Concat",
            inputs=[q_head_unsqueezed, k_head_unsqueezed, v_head_unsqueezed],
            outputs=[qkv_concat],
            name=f"{prefix}PackedQKVConcat",
            axis=3,
        ),
        helper.make_node(
            "Reshape",
            inputs=[qkv_concat, qkv_packed_shape],
            outputs=[qkv_packed_flat],
            name=f"{prefix}PackedQKVReshape",
        ),
        helper.make_node(
            "Unsqueeze",
            inputs=[qkv_packed_flat, unsqueeze_axes],
            outputs=[qkv_packed],
            name=f"{prefix}PackedQKVUnsqueeze",
        ),
        helper.make_node(
            "CustomQKVToContextPluginDynamic",
            inputs=[qkv_packed],
            outputs=[plugin_ctx],
            name=f"{prefix}CustomQKVToContextPluginDynamic",
            plugin_version="1",
            plugin_namespace="",
            type_id=type_id,
            hidden_size=_HIDDEN_SIZE,
            num_heads=_NUM_HEADS,
            has_mask=0,
            dq_probs=_DQ_PROBS,
        ),
        helper.make_node(
            "Squeeze",
            inputs=[plugin_ctx, squeeze_ctx_axes],
            outputs=[ctx_no_hw],
            name=f"{prefix}ContextSqueeze",
        ),
        helper.make_node(
            "Squeeze",
            inputs=[ctx_no_hw, squeeze_batch_axes],
            outputs=[ctx_flat],
            name=f"{prefix}ContextFlat",
        ),
    ]


def _slice_bounds(nodes: List[onnx.NodeProto], *, start_name: str, end_name: str) -> Tuple[int, int]:
    start_idx = None
    end_idx = None
    for idx, node in enumerate(nodes):
        if node.name == start_name:
            start_idx = idx
        if node.name == end_name:
            end_idx = idx
    if start_idx is None or end_idx is None or end_idx < start_idx:
        raise KeyError(f"Invalid slice bounds: {start_name} -> {end_name}")
    return start_idx, end_idx


def patch_model(
    input_path: Path,
    output_path: Path,
    *,
    layers: Iterable[int],
    layout: Literal["seq-first", "batch-first"],
    type_id: int,
) -> None:
    model = onnx.load(str(input_path))
    nodes = list(model.graph.node)

    insertions: Dict[int, List[onnx.NodeProto]] = {}
    skipped_indices: set[int] = set()
    patched = 0

    for layer in layers:
        prefix = f"/transformer/decoder/layers.{layer}/self_attn/"
        add_q = _find_node_by_name(nodes, f"{prefix}Add")
        add_k = _find_node_by_name(nodes, f"{prefix}Add_1")
        add_v = _find_node_by_name(nodes, f"{prefix}Add_2")
        gemm = _find_node_by_name(nodes, f"{prefix}Gemm")

        rewrite_nodes = _make_plugin_rewrite(
            model.graph,
            prefix=prefix,
            q_input=add_q.output[0],
            k_input=add_k.output[0],
            v_input=add_v.output[0],
            layout=layout,
            type_id=type_id,
        )
        gemm.input[0] = rewrite_nodes[-1].output[0]

        start_idx, end_idx = _slice_bounds(
            nodes,
            start_name=f"{prefix}Shape_3",
            end_name=f"{prefix}Reshape_6",
        )
        insertions[start_idx] = rewrite_nodes
        skipped_indices.update(range(start_idx, end_idx + 1))
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    print(f"patched_self_attn_plugin_layers={patched}")
    print(f"saved={output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", nargs="+", type=int, default=[2, 3])
    parser.add_argument(
        "--layout",
        choices=("seq-first", "batch-first"),
        default="seq-first",
    )
    parser.add_argument("--type-id", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    patch_model(
        input_path=Path(args.input),
        output_path=Path(args.output),
        layers=args.layers,
        layout=args.layout,
        type_id=args.type_id,
    )


if __name__ == "__main__":
    main()
