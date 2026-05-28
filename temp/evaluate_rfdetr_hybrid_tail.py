import argparse
import tempfile
import time
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


def prepare_batch(frames: list[np.ndarray], model_id: str, device: torch.device) -> torch.Tensor:
    model = AutoModel.from_pretrained(
        model_id_or_path=model_id,
        device=device,
        backend="trt",
    )
    batches = []
    for frame in frames:
        tensor, _ = model.pre_process(frame)
        batches.append(tensor)
    return torch.cat(batches, dim=0).contiguous()


def build_full_ort_outputs(
    onnx_path: Path,
    batch: np.ndarray,
    extra_outputs: list[str],
) -> dict[str, np.ndarray]:
    augmented = onnx.load(str(onnx_path))
    existing = {output.name for output in augmented.graph.output}
    for name in extra_outputs:
        if name not in existing:
            augmented.graph.output.append(
                helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
            )
            existing.add(name)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as handle:
        augmented_path = handle.name
    onnx.save(augmented, augmented_path)
    session = ort.InferenceSession(
        augmented_path,
        providers=["CPUExecutionProvider"],
    )
    outputs = session.run(None, {session.get_inputs()[0].name: batch})
    return dict(zip([output.name for output in session.get_outputs()], outputs))


def build_trt_cut_outputs(
    onnx_path: Path,
    batch: torch.Tensor,
    output_names: list[str],
) -> dict[str, np.ndarray]:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as handle:
        if not parser.parse(handle.read()):
            for idx in range(parser.num_errors):
                print(parser.get_error(idx))
            raise RuntimeError("Failed to parse ONNX for TRT cut-output build.")

    for layer_idx in range(network.num_layers):
        layer = network.get_layer(layer_idx)
        for output_idx in range(layer.num_outputs):
            tensor = layer.get_output(output_idx)
            if tensor is None:
                continue
            if tensor.name in output_names and not tensor.is_network_output:
                network.mark_output(tensor)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 * (2**30))
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    input_shape = tuple(batch.shape)
    profile.set_shape(input_name, input_shape, input_shape, input_shape)
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Failed to build TRT cut-output engine.")

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    context = engine.create_execution_context()
    if not context.set_input_shape(input_name, input_shape):
        raise RuntimeError("Failed to set TRT cut-output input shape.")

    stream = torch.cuda.Stream(device=batch.device)
    outputs = {}
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
            raise RuntimeError("TRT cut-output execute_async_v3 failed.")
    stream.synchronize()
    return {
        name: tensor.detach().cpu().numpy().astype(np.float32)
        for name, tensor in outputs.items()
    }


def build_tail_session(
    onnx_path: Path,
    cut_inputs: list[str],
    outputs: list[str],
    providers: list[str],
) -> ort.InferenceSession:
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as handle:
        tail_path = handle.name
    onnx.utils.extract_model(str(onnx_path), tail_path, cut_inputs, outputs)
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        tail_path,
        sess_options=session_options,
        providers=providers,
    )


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
    parser.add_argument("--cut", action="append", required=True)
    parser.add_argument("--output", action="append", default=["labels", "4186"])
    parser.add_argument("--cycles", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    if len(args.cut) < 1:
        raise ValueError("At least one --cut input is required.")

    onnx_path = Path(args.onnx)
    frames = load_two_frames(video_path=Path(args.video))
    batch = prepare_batch(
        frames=frames,
        model_id=args.model_id,
        device=torch.device(args.device),
    )
    batch_np = batch.cpu().numpy()

    extra_outputs = list(dict.fromkeys(args.cut + args.output))
    full_ort = build_full_ort_outputs(
        onnx_path=onnx_path,
        batch=batch_np,
        extra_outputs=extra_outputs,
    )
    trt_cut = build_trt_cut_outputs(
        onnx_path=onnx_path,
        batch=batch,
        output_names=extra_outputs,
    )
    providers = (
        ["CUDAExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    tail = build_tail_session(
        onnx_path=onnx_path,
        cut_inputs=args.cut,
        outputs=args.output,
        providers=providers,
    )

    feed = {name: trt_cut[name].astype(np.float32) for name in args.cut}
    hybrid_outputs = tail.run(None, feed)
    hybrid_map = dict(zip([output.name for output in tail.get_outputs()], hybrid_outputs))

    print("tail_providers:", providers)
    for name in args.cut:
        reference = full_ort[name].astype(np.float32)
        candidate = trt_cut[name].astype(np.float32)
        print(
            f"cut {name}: trt_vs_full_ort max_abs="
            f"{float(np.max(np.abs(reference - candidate))):.6g} mean_abs="
            f"{float(np.mean(np.abs(reference - candidate))):.6g}"
        )
    for name in args.output:
        reference = full_ort[name].astype(np.float32)
        hybrid = hybrid_map[name].astype(np.float32)
        trt_direct = trt_cut[name].astype(np.float32)
        print(
            f"hybrid {name}: max_abs={float(np.max(np.abs(reference - hybrid))):.6g} "
            f"mean_abs={float(np.mean(np.abs(reference - hybrid))):.6g}"
        )
        print(
            f"full_trt {name}: max_abs={float(np.max(np.abs(reference - trt_direct))):.6g} "
            f"mean_abs={float(np.mean(np.abs(reference - trt_direct))):.6g}"
        )

    for _ in range(args.warmup):
        tail.run(None, feed)
    start = time.perf_counter()
    for _ in range(args.cycles):
        tail.run(None, feed)
    elapsed = time.perf_counter() - start
    fps = 2 * args.cycles / elapsed if elapsed > 0 else 0.0
    ms_per_batch = elapsed * 1000.0 / args.cycles if args.cycles > 0 else 0.0
    print(f"tail_fps={fps:.2f}")
    print(f"tail_ms_per_batch={ms_per_batch:.4f}")


if __name__ == "__main__":
    main()
