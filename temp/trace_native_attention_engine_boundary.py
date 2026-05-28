import argparse
import ctypes
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import tensorrt as trt
import torch

from inference_models import AutoModel


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = REPO_ROOT / "vehicles_312px.mp4"
DEFAULT_ONNX = Path("/tmp/rfdetr-seg-nano-trt-sweep/source-onnx/weights.onnx")
NUM_HEADS = 6

TRT_TO_TORCH = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.int32: torch.int32,
    trt.int8: torch.int8,
    trt.bool: torch.bool,
}


def load_frame(video_path: Path, frame_id: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if frame_id > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_id} from {video_path}.")
    return frame


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    model = AutoModel.from_pretrained(
        model_id_or_path="rfdetr-seg-nano",
        device=torch.device("cuda:0"),
        backend="trt",
    )
    pre_processed, _ = model.pre_process(frame)
    return pre_processed.detach().cpu().numpy()


def make_prefix(layer: int) -> str:
    return (
        f"/backbone/backbone.0/encoder/encoder/encoder/layer.{layer}/attention/attention/"
    )


def summarize_diff(reference: np.ndarray, candidate: np.ndarray, label: str) -> None:
    diff = np.abs(reference - candidate)
    print(
        f"{label}: shape={reference.shape} max_abs={diff.max():.8f} "
        f"mean_abs={diff.mean():.8f} p99={np.percentile(diff, 99):.8f}"
    )


def load_ort_outputs(
    *,
    onnx_path: Path,
    input_tensor: np.ndarray,
    output_names: list[str],
) -> dict[str, np.ndarray]:
    model = onnx.load(str(onnx_path))
    existing = {value.name for value in model.graph.output}
    for name in output_names:
        if name in existing:
            continue
        model.graph.output.append(
            onnx.helper.make_tensor_value_info(
                name,
                onnx.TensorProto.FLOAT,
                [1, 677, 384],
            )
        )
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    values = session.run(output_names, {session.get_inputs()[0].name: input_tensor})
    return dict(zip(output_names, values))


def run_trt_outputs(
    *,
    engine_path: Path,
    plugin_sos: list[Path],
    input_tensor: np.ndarray,
    output_names: list[str],
) -> dict[str, np.ndarray]:
    for plugin_so in plugin_sos:
        ctypes.CDLL(str(plugin_so), mode=ctypes.RTLD_GLOBAL)
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    with engine_path.open("rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize {engine_path}.")

    context = engine.create_execution_context()
    stream = torch.cuda.Stream()
    input_name = engine.get_tensor_name(0)
    input_device = torch.from_numpy(np.ascontiguousarray(input_tensor)).to("cuda")
    outputs: dict[str, torch.Tensor] = {}

    if context.set_input_shape(input_name, tuple(input_device.shape)) is False:
        raise RuntimeError(f"Failed to set input shape for {input_name}.")

    old_stream = torch.cuda.current_stream()
    try:
        torch.cuda.set_stream(stream)
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                context.set_tensor_address(name, input_device.data_ptr())
                continue
            dtype = TRT_TO_TORCH[engine.get_tensor_dtype(name)]
            shape = tuple(context.get_tensor_shape(name))
            tensor = torch.empty(shape, dtype=dtype, device="cuda")
            outputs[name] = tensor
            context.set_tensor_address(name, tensor.data_ptr())
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("execute_async_v3 failed.")
    finally:
        torch.cuda.set_stream(old_stream)
    stream.synchronize()

    result = {}
    for name in output_names:
        if name not in outputs:
            raise KeyError(f"{name} not found in engine outputs: {list(outputs)}")
        result[name] = outputs[name].float().cpu().numpy()
    return result


def attention_core(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    device = "cuda"
    q_t = torch.from_numpy(np.ascontiguousarray(q)).to(device=device, dtype=torch.float32)
    k_t = torch.from_numpy(np.ascontiguousarray(k)).to(device=device, dtype=torch.float32)
    v_t = torch.from_numpy(np.ascontiguousarray(v)).to(device=device, dtype=torch.float32)

    batch, seq, hidden = q_t.shape
    head_dim = hidden // NUM_HEADS
    scale = float(np.sqrt(1.0 / np.sqrt(float(head_dim))))

    def split_heads(x: torch.Tensor) -> torch.Tensor:
        return x.view(batch, seq, NUM_HEADS, head_dim).permute(0, 2, 1, 3).contiguous()

    q_h = split_heads(q_t) * scale
    k_h = split_heads(k_t) * scale
    v_h = split_heads(v_t)
    scores = torch.matmul(q_h, k_h.transpose(-1, -2))
    weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(weights, v_h)
    out = out.permute(0, 2, 1, 3).contiguous().view(batch, seq, hidden)
    return out.cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--plugin-so", action="append", required=True)
    parser.add_argument("--frame-id", type=int, default=236)
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--video", default=str(DEFAULT_VIDEO))
    parser.add_argument("--onnx", default=str(DEFAULT_ONNX))
    args = parser.parse_args()

    prefix = make_prefix(args.layer)
    output_names = [
        f"{prefix}query/Add_output_0",
        f"{prefix}key/Add_output_0",
        f"{prefix}value/Add_output_0",
        f"{prefix}Reshape_3_output_0",
    ]

    frame = load_frame(Path(args.video), args.frame_id)
    input_tensor = preprocess_frame(frame)

    ort_outputs = load_ort_outputs(
        onnx_path=Path(args.onnx),
        input_tensor=input_tensor,
        output_names=output_names,
    )
    trt_outputs = run_trt_outputs(
        engine_path=Path(args.engine),
        plugin_sos=[Path(path) for path in args.plugin_so],
        input_tensor=input_tensor,
        output_names=output_names,
    )

    print(f"frame={args.frame_id} layer={args.layer}")
    for name in output_names:
        summarize_diff(ort_outputs[name], trt_outputs[name], f"trt_vs_ort {name}")

    ort_recomputed = attention_core(
        ort_outputs[output_names[0]],
        ort_outputs[output_names[1]],
        ort_outputs[output_names[2]],
    )
    summarize_diff(
        ort_outputs[output_names[3]],
        ort_recomputed,
        "ort_attention_formula_vs_ort_output",
    )

    trt_recomputed = attention_core(
        trt_outputs[output_names[0]],
        trt_outputs[output_names[1]],
        trt_outputs[output_names[2]],
    )
    summarize_diff(
        trt_outputs[output_names[3]],
        trt_recomputed,
        "torch_attention_on_trt_qkv_vs_trt_plugin_output",
    )
    summarize_diff(
        ort_outputs[output_names[3]],
        trt_recomputed,
        "torch_attention_on_trt_qkv_vs_ort_output",
    )


if __name__ == "__main__":
    main()
