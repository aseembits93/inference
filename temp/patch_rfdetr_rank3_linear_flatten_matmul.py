import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import onnx
from onnx import TensorProto, helper, shape_inference


def _shape_map(model: onnx.ModelProto) -> Dict[str, List[int | str]]:
    result: Dict[str, List[int | str]] = {}

    def add_value_info(value_info: onnx.ValueInfoProto) -> None:
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            return
        dims: List[int | str] = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(dim.dim_value)
            elif dim.HasField("dim_param"):
                dims.append(dim.dim_param)
            else:
                dims.append("?")
        result[value_info.name] = dims

    for collection in (model.graph.input, model.graph.value_info, model.graph.output):
        for value_info in collection:
            add_value_info(value_info)
    return result


def _initializer_dims(model: onnx.ModelProto) -> Dict[str, List[int]]:
    return {initializer.name: list(initializer.dims) for initializer in model.graph.initializer}


def _add_small_initializer(
    graph: onnx.GraphProto,
    name: str,
    dtype: int,
    values: Sequence[int],
) -> str:
    tensor = helper.make_tensor(name=name, data_type=dtype, dims=[len(values)], vals=list(values))
    graph.initializer.append(tensor)
    return name


def _extract_linear_pattern(
    *,
    matmul_node: onnx.NodeProto,
    add_node: onnx.NodeProto,
    initializer_dims: Dict[str, List[int]],
    shape_map: Dict[str, List[int | str]],
) -> Optional[Tuple[str, str, int, int]]:
    if matmul_node.op_type != "MatMul" or add_node.op_type != "Add":
        return None
    matmul_output = matmul_node.output[0]
    if matmul_output not in add_node.input:
        return None

    weight_name = matmul_node.input[1]
    weight_dims = initializer_dims.get(weight_name)
    if weight_dims is None or len(weight_dims) != 2:
        return None

    bias_name = add_node.input[0] if add_node.input[1] == matmul_output else add_node.input[1]
    bias_dims = initializer_dims.get(bias_name)
    if bias_dims is None or len(bias_dims) != 1:
        return None

    input_shape = shape_map.get(matmul_node.input[0])
    if input_shape is None or len(input_shape) < 3:
        return None

    out_features = weight_dims[1]
    if bias_dims[0] != out_features:
        return None
    return weight_name, bias_name, out_features, len(input_shape)


def _rewrite_linear(
    *,
    graph: onnx.GraphProto,
    matmul_node: onnx.NodeProto,
    add_node: onnx.NodeProto,
    weight_name: str,
    bias_name: str,
    out_features: int,
    input_rank: int,
) -> List[onnx.NodeProto]:
    base = matmul_node.name.replace("/", "_").replace(".", "_")
    input_name = matmul_node.input[0]
    shape_out = f"{base}_shape"
    prefix_out = f"{base}_prefix_shape"
    flat_out = f"{base}_flat"
    matmul_out = f"{base}_matmul_out"
    add_out = f"{base}_add_out"
    outdim_name = f"{base}_outdim"
    reshape_shape_out = f"{base}_reshape_shape"

    starts = _add_small_initializer(
        graph=graph,
        name=f"{base}_slice_starts",
        dtype=TensorProto.INT64,
        values=[0],
    )
    ends = _add_small_initializer(
        graph=graph,
        name=f"{base}_slice_ends",
        dtype=TensorProto.INT64,
        values=[input_rank - 1],
    )
    axes = _add_small_initializer(
        graph=graph,
        name=f"{base}_slice_axes",
        dtype=TensorProto.INT64,
        values=[0],
    )
    steps = _add_small_initializer(
        graph=graph,
        name=f"{base}_slice_steps",
        dtype=TensorProto.INT64,
        values=[1],
    )
    _add_small_initializer(
        graph=graph,
        name=outdim_name,
        dtype=TensorProto.INT64,
        values=[out_features],
    )

    return [
        helper.make_node(
            "Shape",
            inputs=[input_name],
            outputs=[shape_out],
            name=f"{matmul_node.name}/Shape",
        ),
        helper.make_node(
            "Slice",
            inputs=[shape_out, starts, ends, axes, steps],
            outputs=[prefix_out],
            name=f"{matmul_node.name}/PrefixShape",
        ),
        helper.make_node(
            "Flatten",
            inputs=[input_name],
            outputs=[flat_out],
            name=f"{matmul_node.name}/Flatten",
            axis=input_rank - 1,
        ),
        helper.make_node(
            "MatMul",
            inputs=[flat_out, weight_name],
            outputs=[matmul_out],
            name=f"{matmul_node.name}/FlatMatMul",
        ),
        helper.make_node(
            "Add",
            inputs=[bias_name, matmul_out],
            outputs=[add_out],
            name=f"{add_node.name}/FlatAdd",
        ),
        helper.make_node(
            "Concat",
            inputs=[prefix_out, outdim_name],
            outputs=[reshape_shape_out],
            name=f"{matmul_node.name}/OutputShape",
            axis=0,
        ),
        helper.make_node(
            "Reshape",
            inputs=[add_out, reshape_shape_out],
            outputs=[add_node.output[0]],
            name=f"{matmul_node.name}/Reshape",
        ),
    ]


def patch_model(input_path: Path, output_path: Path) -> None:
    patch_model_filtered(
        input_path=input_path,
        output_path=output_path,
        include_substrings=None,
    )


def patch_model_filtered(
    input_path: Path,
    output_path: Path,
    include_substrings: Optional[Sequence[str]] = None,
) -> None:
    model = onnx.load(str(input_path))
    inferred = shape_inference.infer_shapes(model, strict_mode=False)
    shape_map = _shape_map(inferred)
    initializer_dims = _initializer_dims(inferred)

    patched = 0
    new_nodes: List[onnx.NodeProto] = []
    nodes = list(model.graph.node)
    idx = 0
    while idx < len(nodes):
        node = nodes[idx]
        if idx + 1 < len(nodes):
            next_node = nodes[idx + 1]
            if include_substrings and not any(
                substring in node.name for substring in include_substrings
            ):
                new_nodes.append(node)
                idx += 1
                continue
            pattern = _extract_linear_pattern(
                matmul_node=node,
                add_node=next_node,
                initializer_dims=initializer_dims,
                shape_map=shape_map,
            )
            if pattern is not None:
                weight_name, bias_name, out_features, input_rank = pattern
                new_nodes.extend(
                    _rewrite_linear(
                        graph=model.graph,
                        matmul_node=node,
                        add_node=next_node,
                        weight_name=weight_name,
                        bias_name=bias_name,
                        out_features=out_features,
                        input_rank=input_rank,
                    )
                )
                patched += 1
                idx += 2
                continue
        new_nodes.append(node)
        idx += 1

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    print(f"patched_rank3_linear_layers={patched}")
    print(f"saved={output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--include-substring",
        action="append",
        default=None,
        help="Only patch nodes whose MatMul name contains at least one of these substrings.",
    )
    args = parser.parse_args()
    patch_model_filtered(
        input_path=Path(args.input),
        output_path=Path(args.output),
        include_substrings=args.include_substring,
    )


if __name__ == "__main__":
    main()
