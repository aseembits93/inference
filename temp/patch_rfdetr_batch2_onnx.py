import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import AttributeProto, numpy_helper


def _constant_array(node: onnx.NodeProto):
    for attr in node.attribute:
        if attr.type == AttributeProto.TENSOR:
            return numpy_helper.to_array(attr.t)
    return None


def _set_constant_array(node: onnx.NodeProto, value: np.ndarray) -> None:
    for attr in node.attribute:
        if attr.type == AttributeProto.TENSOR:
            attr.t.CopyFrom(numpy_helper.from_array(value))
            return
    raise ValueError(f"Node {node.name} does not contain a tensor attribute.")


def patch_model(model: onnx.ModelProto, batch_size: int) -> dict[str, int]:
    out_to_node = {}
    for node in model.graph.node:
        for out in node.output:
            out_to_node[out] = node

    patched = {
        "decoder_tile_repeats": 0,
        "cross_attn_batch_scalars": 0,
        "cross_attn_batch_heads_scalars": 0,
    }

    # The already-dynamic graph still keeps learned query/value tiling at batch 1.
    for node in model.graph.node:
        if node.name in {"/transformer/Constant_62", "/transformer/Constant_63"}:
            _set_constant_array(
                node, np.array([batch_size, 1, 1], dtype=np.int64)
            )
            patched["decoder_tile_repeats"] += 1

    # Decoder cross-attention shape builders still bake batch=1 into several
    # reshape concat nodes. Promote those leading constants to the requested batch.
    for node in model.graph.node:
        if not node.name.startswith("/transformer/decoder/layers."):
            continue
        if "/cross_attn/" not in node.name or node.op_type != "Concat":
            continue
        if not node.input:
            continue
        first_constant = out_to_node.get(node.input[0])
        if first_constant is None or first_constant.op_type != "Constant":
            continue
        arr = _constant_array(first_constant)
        if arr is None or arr.shape != (1,):
            continue
        if int(arr[0]) == 1:
            _set_constant_array(first_constant, np.array([batch_size], dtype=np.int64))
            patched["cross_attn_batch_scalars"] += 1

    # In deformable cross-attention the value tensor reshape expects batch * heads.
    # The first scalar in each `Concat_2` still bakes in heads=16 only.
    for node in model.graph.node:
        if not node.name.startswith("/transformer/decoder/layers."):
            continue
        if "/cross_attn/Concat_2" not in node.name or node.op_type != "Concat":
            continue
        if not node.input:
            continue
        first_constant = out_to_node.get(node.input[0])
        if first_constant is None or first_constant.op_type != "Constant":
            continue
        arr = _constant_array(first_constant)
        if arr is None or arr.shape != (1,):
            continue
        if int(arr[0]) == 16:
            _set_constant_array(
                first_constant, np.array([16 * batch_size], dtype=np.int64)
            )
            patched["cross_attn_batch_heads_scalars"] += 1

    return patched


def validate_model(model_path: Path, batch_size: int) -> None:
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    sample = np.zeros((batch_size, 3, 312, 312), dtype=np.float32)
    outputs = session.run(None, {input_name: sample})
    print("validation: batch run ok")
    print("output_shapes:", [tuple(output.shape) for output in outputs])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/tmp/rfdetr-seg-nano-trt-sweep/source-onnx/weights-dynb-patched2.onnx",
    )
    parser.add_argument(
        "--output",
        default="/tmp/rfdetr-seg-nano-trt-sweep/source-onnx/weights-dynb-patched5.onnx",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--skip-validate", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    model = onnx.load(str(input_path))
    patched = patch_model(model=model, batch_size=args.batch_size)
    onnx.save(model, str(output_path))

    print(f"patched_model={output_path}")
    for name, count in patched.items():
        print(f"{name}={count}")

    if not args.skip_validate:
        validate_model(model_path=output_path, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
