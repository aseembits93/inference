import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import tensorrt as trt
import torch
from onnx import TensorProto, helper

from inference_models import AutoModel
from inference_models.models.common.trt import _trt_dtype_to_torch


DEFAULT_TARGETS = [
    "/transformer/decoder/layers.0/cross_attn/Reshape_3_output_0",
    "/transformer/decoder/layers.0/cross_attn/GridSample_output_0",
    "/transformer/decoder/layers.0/cross_attn/ReduceSum_output_0",
]


def load_two_frames(video_path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for _ in range(2):
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read 2 frames from {video_path}.")
        frames.append(frame)
    cap.release()
    return frames


def prepare_inputs(frames: list[np.ndarray], model_id: str, device: torch.device) -> torch.Tensor:
    model = AutoModel.from_pretrained(model_id_or_path=model_id, device=device, backend="trt")
    batches = []
    for frame in frames:
        tensor, _ = model.pre_process(frame)
        batches.append(tensor)
    return torch.cat(batches, dim=0).contiguous()


def build_ort_model_with_extra_outputs(onnx_path: Path, targets: list[str]) -> tuple[str, list[str]]:
    model = onnx.load(str(onnx_path))
    existing = {output.name for output in model.graph.output}
    for node in model.graph.node:
        for out in node.output:
            if out in targets and out not in existing:
                model.graph.output.append(
                    helper.make_tensor_value_info(out, TensorProto.FLOAT, None)
                )
                existing.add(out)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as handle:
        tmp_path = handle.name
    onnx.save(model, tmp_path)
    return tmp_path, [output.name for output in model.graph.output]


def run_ort(onnx_path: str, batch: np.ndarray) -> dict[str, np.ndarray]:
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        onnx_path,
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    outputs = session.run(None, {session.get_inputs()[0].name: batch})
    return dict(zip([output.name for output in session.get_outputs()], outputs))


def build_and_run_trt(
    onnx_path: Path,
    batch: torch.Tensor,
    targets: list[str],
    fp16: bool,
) -> dict[str, np.ndarray]:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as handle:
        if not parser.parse(handle.read()):
            for idx in range(parser.num_errors):
                print(parser.get_error(idx))
            raise RuntimeError("Failed to parse ONNX for TRT diagnostic build.")

    marked = []
    for layer_idx in range(network.num_layers):
        layer = network.get_layer(layer_idx)
        for output_idx in range(layer.num_outputs):
            tensor = layer.get_output(output_idx)
            if tensor is None:
                continue
            if tensor.name in targets and not tensor.is_network_output:
                network.mark_output(tensor)
                marked.append(tensor.name)
    print("trt_marked_outputs:", marked)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 * (2**30))
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    input_shape = tuple(batch.shape)
    profile.set_shape(input_name, input_shape, input_shape, input_shape)
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Failed to build TRT diagnostic engine.")

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    context = engine.create_execution_context()
    if not context.set_input_shape(input_name, input_shape):
        raise RuntimeError("Failed to set TRT input shape.")

    outputs = {}
    stream = torch.cuda.Stream(device=batch.device)
    with torch.cuda.stream(stream):
        batch_device = batch.contiguous()
        context.set_tensor_address(input_name, batch_device.data_ptr())
        for idx in range(engine.num_io_tensors):
            name = engine.get_tensor_name(idx)
            if engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
                continue
            shape = tuple(context.get_tensor_shape(name))
            dtype = _trt_dtype_to_torch(engine.get_tensor_dtype(name))
            buffer = torch.empty(shape, dtype=dtype, device=batch.device)
            context.set_tensor_address(name, buffer.data_ptr())
            outputs[name] = buffer
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TRT diagnostic execute_async_v3 failed.")
    stream.synchronize()
    return {
        name: tensor.detach().cpu().numpy() for name, tensor in outputs.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx",
        default="/tmp/rfdetr-seg-nano-trt-sweep/source-onnx/weights-dynb-patched5.onnx",
    )
    parser.add_argument(
        "--video",
        default="/home/ubuntu/inference/vehicles_312px.mp4",
    )
    parser.add_argument("--model-id", default="rfdetr-seg-nano")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--target", action="append", dest="targets")
    args = parser.parse_args()

    onnx_path = Path(args.onnx)
    targets = args.targets or list(DEFAULT_TARGETS)
    frames = load_two_frames(video_path=Path(args.video))
    batch = prepare_inputs(
        frames=frames,
        model_id=args.model_id,
        device=torch.device(args.device),
    )
    batch_np = batch.cpu().numpy()
    print("input_shape:", tuple(batch_np.shape))

    ort_onnx_path, _ = build_ort_model_with_extra_outputs(onnx_path, targets)
    ort_outputs = run_ort(ort_onnx_path, batch_np)
    trt_outputs = build_and_run_trt(
        onnx_path=onnx_path,
        batch=batch,
        targets=targets,
        fp16=args.fp16,
    )

    names = targets + ["dets", "labels", "4186"]
    for name in names:
        ort_value = ort_outputs[name].astype(np.float32)
        trt_value = trt_outputs[name].astype(np.float32)
        max_abs = float(np.max(np.abs(ort_value - trt_value)))
        mean_abs = float(np.mean(np.abs(ort_value - trt_value)))
        print(
            f"{name}: shape={ort_value.shape} max_abs={max_abs:.6g} "
            f"mean_abs={mean_abs:.6g}"
        )


if __name__ == "__main__":
    main()
