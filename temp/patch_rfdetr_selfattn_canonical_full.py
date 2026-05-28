import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper


_RESHAPE_PRE = [100, 1, 8, 32]
_RESHAPE_CTX = [100, 256]
_RESHAPE_OUT = [100, 1, 256]
_SCALE_EACH = math.sqrt(1.0 / math.sqrt(32.0))


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


def _add_float_initializer(
    graph: onnx.GraphProto,
    *,
    name: str,
    values: List[float],
) -> str:
    arr = np.asarray(values, dtype=np.float32)
    tensor = helper.make_tensor(
        name=name,
        data_type=TensorProto.FLOAT,
        dims=[len(values)],
        vals=arr.tolist(),
    )
    graph.initializer.append(tensor)
    return name


def _find_index(nodes: List[onnx.NodeProto], name: str) -> int:
    for idx, node in enumerate(nodes):
        if node.name == name:
            return idx
    raise KeyError(f"Missing node: {name}")


def _make_pre_gemm_rewrite_nodes(graph: onnx.GraphProto, *, prefix: str) -> List[onnx.NodeProto]:
    pre_shape_name = _add_int64_initializer(
        graph,
        name=f"{prefix.replace('/', '_').replace('.', '_')}_pre_shape",
        values=_RESHAPE_PRE,
    )
    ctx_shape_name = _add_int64_initializer(
        graph,
        name=f"{prefix.replace('/', '_').replace('.', '_')}_ctx_shape",
        values=_RESHAPE_CTX,
    )
    scale_name = _add_float_initializer(
        graph,
        name=f"{prefix.replace('/', '_').replace('.', '_')}_scale_each",
        values=[_SCALE_EACH],
    )

    q_pre = f"{prefix}CanonicalPreQ_output_0"
    k_pre = f"{prefix}CanonicalPreK_output_0"
    v_pre = f"{prefix}CanonicalPreV_output_0"
    q_bhsd = f"{prefix}CanonicalQ_output_0"
    k_bhsd = f"{prefix}CanonicalK_output_0"
    v_bhsd = f"{prefix}CanonicalV_output_0"
    q_scaled = f"{prefix}CanonicalQScaled_output_0"
    k_t = f"{prefix}CanonicalKTranspose_output_0"
    k_scaled = f"{prefix}CanonicalKScaled_output_0"
    scores = f"{prefix}CanonicalScores_output_0"
    probs = f"{prefix}CanonicalSoftmax_output_0"
    ctx = f"{prefix}CanonicalContext_output_0"
    ctx_t = f"{prefix}CanonicalContextTranspose_output_0"

    return [
        helper.make_node(
            "Reshape",
            inputs=[f"{prefix}Add_output_0", pre_shape_name],
            outputs=[q_pre],
            name=f"{prefix}CanonicalPreQ",
        ),
        helper.make_node(
            "Reshape",
            inputs=[f"{prefix}Add_1_output_0", pre_shape_name],
            outputs=[k_pre],
            name=f"{prefix}CanonicalPreK",
        ),
        helper.make_node(
            "Reshape",
            inputs=[f"{prefix}Add_2_output_0", pre_shape_name],
            outputs=[v_pre],
            name=f"{prefix}CanonicalPreV",
        ),
        helper.make_node(
            "Transpose",
            inputs=[q_pre],
            outputs=[q_bhsd],
            name=f"{prefix}CanonicalTransposeQ",
            perm=[1, 2, 0, 3],
        ),
        helper.make_node(
            "Transpose",
            inputs=[k_pre],
            outputs=[k_bhsd],
            name=f"{prefix}CanonicalTransposeK",
            perm=[1, 2, 0, 3],
        ),
        helper.make_node(
            "Transpose",
            inputs=[v_pre],
            outputs=[v_bhsd],
            name=f"{prefix}CanonicalTransposeV",
            perm=[1, 2, 0, 3],
        ),
        helper.make_node(
            "Mul",
            inputs=[q_bhsd, scale_name],
            outputs=[q_scaled],
            name=f"{prefix}CanonicalMulQ",
        ),
        helper.make_node(
            "Transpose",
            inputs=[k_bhsd],
            outputs=[k_t],
            name=f"{prefix}CanonicalTransposeK2",
            perm=[0, 1, 3, 2],
        ),
        helper.make_node(
            "Mul",
            inputs=[k_t, scale_name],
            outputs=[k_scaled],
            name=f"{prefix}CanonicalMulK",
        ),
        helper.make_node(
            "MatMul",
            inputs=[q_scaled, k_scaled],
            outputs=[scores],
            name=f"{prefix}CanonicalMatMulScores",
        ),
        helper.make_node(
            "Softmax",
            inputs=[scores],
            outputs=[probs],
            name=f"{prefix}CanonicalSoftmax",
            axis=-1,
        ),
        helper.make_node(
            "MatMul",
            inputs=[probs, v_bhsd],
            outputs=[ctx],
            name=f"{prefix}CanonicalMatMulContext",
        ),
        helper.make_node(
            "Transpose",
            inputs=[ctx],
            outputs=[ctx_t],
            name=f"{prefix}CanonicalTransposeContext",
            perm=[2, 0, 1, 3],
        ),
        helper.make_node(
            "Reshape",
            inputs=[ctx_t, ctx_shape_name],
            outputs=[f"{prefix}Reshape_6_output_0"],
            name=f"{prefix}Reshape_6",
        ),
    ]


def _make_post_gemm_rewrite_node(graph: onnx.GraphProto, *, prefix: str) -> onnx.NodeProto:
    out_shape_name = _add_int64_initializer(
        graph,
        name=f"{prefix.replace('/', '_').replace('.', '_')}_out_shape",
        values=_RESHAPE_OUT,
    )
    return helper.make_node(
        "Reshape",
        inputs=[f"{prefix}Gemm_output_0", out_shape_name],
        outputs=[f"{prefix}Reshape_7_output_0"],
        name=f"{prefix}Reshape_7",
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
        start_idx = _find_index(nodes, f"{prefix}Add_2") + 1
        end_idx = _find_index(nodes, f"{prefix}Reshape_6")
        insertions[start_idx] = _make_pre_gemm_rewrite_nodes(model.graph, prefix=prefix)
        skipped_indices.update(range(start_idx, end_idx + 1))

        out_start = _find_index(nodes, f"{prefix}Shape_7")
        out_end = _find_index(nodes, f"{prefix}Reshape_7")
        insertions[out_start] = [_make_post_gemm_rewrite_node(model.graph, prefix=prefix)]
        skipped_indices.update(range(out_start, out_end + 1))
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
    print(f"patched_self_attn_canonical_full_layers={patched}")
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
