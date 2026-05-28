import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
    values: List[int],
) -> str:
    tensor = helper.make_tensor(name=name, data_type=dtype, dims=[len(values)], vals=values)
    graph.initializer.append(tensor)
    return name


def _extract_pattern(
    *,
    matmul_node: onnx.NodeProto,
    add_node: onnx.NodeProto,
    initializer_dims: Dict[str, List[int]],
    shape_map: Dict[str, List[int | str]],
) -> Optional[Tuple[str, str]]:
    if matmul_node.op_type != "MatMul" or add_node.op_type != "Add":
        return None
    matmul_output = matmul_node.output[0]
    if matmul_output not in add_node.input:
        return None
    input_shape = shape_map.get(matmul_node.input[0])
    if (
        input_shape is None
        or len(input_shape) != 3
        or input_shape[0] != 100
        or input_shape[2] != 256
    ):
        return None
    weight_name = matmul_node.input[1]
    weight_dims = initializer_dims.get(weight_name)
    if weight_dims != [256, 256]:
        return None
    bias_name = add_node.input[0] if add_node.input[1] == matmul_output else add_node.input[1]
    bias_dims = initializer_dims.get(bias_name)
    if bias_dims != [256]:
        return None
    return weight_name, bias_name


def _rewrite(
    *,
    graph: onnx.GraphProto,
    matmul_node: onnx.NodeProto,
    add_node: onnx.NodeProto,
    weight_name: str,
    bias_name: str,
) -> List[onnx.NodeProto]:
    base = matmul_node.name.replace("/", "_").replace(".", "_")
    squeeze_axes = _add_small_initializer(
        graph=graph,
        name=f"{base}_squeeze_axes",
        dtype=TensorProto.INT64,
        values=[1],
    )
    unsqueeze_axes = _add_small_initializer(
        graph=graph,
        name=f"{base}_unsqueeze_axes",
        dtype=TensorProto.INT64,
        values=[1],
    )
    squeezed = f"{base}_squeezed"
    gemm_out = f"{base}_gemm_out"
    return [
        helper.make_node(
            "Squeeze",
            inputs=[matmul_node.input[0], squeeze_axes],
            outputs=[squeezed],
            name=f"{matmul_node.name}/Squeeze",
        ),
        helper.make_node(
            "Gemm",
            inputs=[squeezed, weight_name, bias_name],
            outputs=[gemm_out],
            name=f"{matmul_node.name}/Gemm",
            alpha=1.0,
            beta=1.0,
            transB=0,
        ),
        helper.make_node(
            "Unsqueeze",
            inputs=[gemm_out, unsqueeze_axes],
            outputs=[add_node.output[0]],
            name=f"{matmul_node.name}/Unsqueeze",
        ),
    ]


def patch_model(
    input_path: Path,
    output_path: Path,
    *,
    layers: Iterable[int],
) -> None:
    model = onnx.load(str(input_path))
    inferred = shape_inference.infer_shapes(model, strict_mode=False)
    shape_map = _shape_map(inferred)
    initializer_dims = _initializer_dims(inferred)
    target_prefixes = {
        f"/transformer/decoder/layers.{layer}/self_attn/MatMul_2" for layer in layers
    }

    patched = 0
    new_nodes: List[onnx.NodeProto] = []
    nodes = list(model.graph.node)
    idx = 0
    while idx < len(nodes):
        node = nodes[idx]
        if idx + 1 < len(nodes):
            next_node = nodes[idx + 1]
            if node.name in target_prefixes:
                pattern = _extract_pattern(
                    matmul_node=node,
                    add_node=next_node,
                    initializer_dims=initializer_dims,
                    shape_map=shape_map,
                )
                if pattern is not None:
                    weight_name, bias_name = pattern
                    new_nodes.extend(
                        _rewrite(
                            graph=model.graph,
                            matmul_node=node,
                            add_node=next_node,
                            weight_name=weight_name,
                            bias_name=bias_name,
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
    print(f"patched_self_attn_vproj_layers={patched}")
    print(f"saved={output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", nargs="+", type=int, default=[2, 3])
    args = parser.parse_args()
    patch_model(
        input_path=Path(args.input),
        output_path=Path(args.output),
        layers=args.layers,
    )


if __name__ == "__main__":
    main()
