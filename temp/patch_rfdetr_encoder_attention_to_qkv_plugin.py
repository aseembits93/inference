import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import onnx
from onnx import TensorProto, helper


_HIDDEN_SIZE = 384
_NUM_HEADS = 6
_HEAD_SIZE = _HIDDEN_SIZE // _NUM_HEADS
_SEQ_LEN = 677


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


def _find_node(nodes: List[onnx.NodeProto], name: str) -> onnx.NodeProto:
    for node in nodes:
        if node.name == name:
            return node
    raise KeyError(f"Missing node: {name}")


def _make_plugin_nodes(graph: onnx.GraphProto, *, prefix: str) -> List[onnx.NodeProto]:
    base = prefix.replace("/", "_").replace(".", "_")
    to_seq_perm = [1, 0, 2]
    to_batch_perm = [1, 0, 2]

    head_shape = _add_int64_initializer(
        graph,
        name=f"{base}_head_shape",
        values=[_SEQ_LEN, 1, _NUM_HEADS, _HEAD_SIZE],
    )
    unsqueeze_head_axes = _add_int64_initializer(
        graph,
        name=f"{base}_head_unsqueeze_axes",
        values=[3],
    )
    packed_shape = _add_int64_initializer(
        graph,
        name=f"{base}_packed_shape",
        values=[_SEQ_LEN, 1, 3 * _HIDDEN_SIZE],
    )
    unsqueeze_hw_axes = _add_int64_initializer(
        graph,
        name=f"{base}_packed_unsqueeze_axes",
        values=[3, 4],
    )
    squeeze_hw_axes = _add_int64_initializer(
        graph,
        name=f"{base}_ctx_squeeze_axes",
        values=[3, 4],
    )

    q_seq = f"{prefix}QSeqFirst_output_0"
    k_seq = f"{prefix}KSeqFirst_output_0"
    v_seq = f"{prefix}VSeqFirst_output_0"
    q_head = f"{prefix}QHead_output_0"
    k_head = f"{prefix}KHead_output_0"
    v_head = f"{prefix}VHead_output_0"
    q_unsq = f"{prefix}QHeadUnsqueeze_output_0"
    k_unsq = f"{prefix}KHeadUnsqueeze_output_0"
    v_unsq = f"{prefix}VHeadUnsqueeze_output_0"
    qkv_concat = f"{prefix}PackedQKVConcat_output_0"
    qkv_flat = f"{prefix}PackedQKVReshape_output_0"
    qkv_packed = f"{prefix}PackedQKVUnsqueeze_output_0"
    plugin_ctx = f"{prefix}CustomQKVToContextPluginDynamic_output_0"
    ctx_no_hw = f"{prefix}ContextSqueeze_output_0"
    ctx_batch = f"{prefix}ContextBatchFirst_output_0"

    return [
        helper.make_node(
            "Transpose",
            inputs=[f"{prefix}query/Add_output_0"],
            outputs=[q_seq],
            name=f"{prefix}QSeqFirst",
            perm=to_seq_perm,
        ),
        helper.make_node(
            "Transpose",
            inputs=[f"{prefix}key/Add_output_0"],
            outputs=[k_seq],
            name=f"{prefix}KSeqFirst",
            perm=to_seq_perm,
        ),
        helper.make_node(
            "Transpose",
            inputs=[f"{prefix}value/Add_output_0"],
            outputs=[v_seq],
            name=f"{prefix}VSeqFirst",
            perm=to_seq_perm,
        ),
        helper.make_node(
            "Reshape",
            inputs=[q_seq, head_shape],
            outputs=[q_head],
            name=f"{prefix}QHead",
        ),
        helper.make_node(
            "Reshape",
            inputs=[k_seq, head_shape],
            outputs=[k_head],
            name=f"{prefix}KHead",
        ),
        helper.make_node(
            "Reshape",
            inputs=[v_seq, head_shape],
            outputs=[v_head],
            name=f"{prefix}VHead",
        ),
        helper.make_node(
            "Unsqueeze",
            inputs=[q_head, unsqueeze_head_axes],
            outputs=[q_unsq],
            name=f"{prefix}QHeadUnsqueeze",
        ),
        helper.make_node(
            "Unsqueeze",
            inputs=[k_head, unsqueeze_head_axes],
            outputs=[k_unsq],
            name=f"{prefix}KHeadUnsqueeze",
        ),
        helper.make_node(
            "Unsqueeze",
            inputs=[v_head, unsqueeze_head_axes],
            outputs=[v_unsq],
            name=f"{prefix}VHeadUnsqueeze",
        ),
        helper.make_node(
            "Concat",
            inputs=[q_unsq, k_unsq, v_unsq],
            outputs=[qkv_concat],
            name=f"{prefix}PackedQKVConcat",
            axis=3,
        ),
        helper.make_node(
            "Reshape",
            inputs=[qkv_concat, packed_shape],
            outputs=[qkv_flat],
            name=f"{prefix}PackedQKVReshape",
        ),
        helper.make_node(
            "Unsqueeze",
            inputs=[qkv_flat, unsqueeze_hw_axes],
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
            type_id=1,
            hidden_size=_HIDDEN_SIZE,
            num_heads=_NUM_HEADS,
            has_mask=0,
            dq_probs=1.0,
        ),
        helper.make_node(
            "Squeeze",
            inputs=[plugin_ctx, squeeze_hw_axes],
            outputs=[ctx_no_hw],
            name=f"{prefix}ContextSqueeze",
        ),
        helper.make_node(
            "Transpose",
            inputs=[ctx_no_hw],
            outputs=[ctx_batch],
            name=f"{prefix}ContextBatchFirst",
            perm=to_batch_perm,
        ),
    ]


def patch_model(input_path: Path, output_path: Path, *, layers: Iterable[int]) -> None:
    model = onnx.load(str(input_path))
    nodes = list(model.graph.node)

    insertions: Dict[int, List[onnx.NodeProto]] = {}
    skipped_indices: set[int] = set()
    patched = 0

    for layer in layers:
        prefix = (
            f"/backbone/backbone.0/encoder/encoder/encoder/layer.{layer}/attention/attention/"
        )
        dense = _find_node(
            nodes,
            f"/backbone/backbone.0/encoder/encoder/encoder/layer.{layer}/attention/output/dense/MatMul",
        )
        dense.input[0] = f"{prefix}ContextBatchFirst_output_0"
        start_idx = _find_index(nodes, f"{prefix}value/Add") + 1
        first_constant_idx = _find_index(nodes, f"{prefix}Constant")
        first_reshape_idx = _find_index(nodes, f"{prefix}Reshape")
        second_constant_idx = _find_index(nodes, f"{prefix}Constant_1")
        end_idx = _find_index(nodes, f"{prefix}Reshape_3")
        insertions[start_idx] = _make_plugin_nodes(model.graph, prefix=prefix)
        skipped_indices.update(range(first_constant_idx, first_reshape_idx + 1))
        skipped_indices.update(range(second_constant_idx, end_idx + 1))
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
    print(f"patched_encoder_attention_plugin_layers={patched}")
    print(f"saved={output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", nargs="+", type=int, default=[4, 5])
    args = parser.parse_args()
    patch_model(
        input_path=Path(args.input),
        output_path=Path(args.output),
        layers=args.layers,
    )


if __name__ == "__main__":
    main()
