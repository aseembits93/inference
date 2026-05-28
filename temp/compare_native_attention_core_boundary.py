import argparse
import ctypes
import os
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
DEFAULT_PLUGIN_SO = Path("/tmp/rfprobe_native_plugin/libRfProbeEncoderAttentionCore.so")
PLUGIN_NAME = "RfProbeEncoderAttentionCore"
NUM_HEADS = 6
HIDDEN = 384
SEQ = 677


def load_frame(video_path: Path, frame_id: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if frame_id > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_id} from {video_path}.")
    return frame


def make_prefix(layer: int) -> str:
    return (
        f"/backbone/backbone.0/encoder/encoder/encoder/layer.{layer}/attention/attention/"
    )


def extract_reference_tensors(
    *,
    onnx_path: Path,
    frame: np.ndarray,
    layer: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model = AutoModel.from_pretrained(
        model_id_or_path="rfdetr-seg-nano",
        device=torch.device("cuda:0"),
        backend="trt",
    )
    pre_processed, _ = model.pre_process(frame)
    inputs = pre_processed.detach().cpu().numpy()

    prefix = make_prefix(layer)
    outputs = [
        f"{prefix}query/Add_output_0",
        f"{prefix}key/Add_output_0",
        f"{prefix}value/Add_output_0",
        f"{prefix}Reshape_3_output_0",
    ]
    onnx_model = onnx.load(str(onnx_path))
    for name in outputs:
        onnx_model.graph.output.append(
            onnx.helper.make_tensor_value_info(
                name,
                onnx.TensorProto.FLOAT,
                [1, SEQ, HIDDEN],
            )
        )
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        onnx_model.SerializeToString(),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    ort_outputs = session.run(outputs, {session.get_inputs()[0].name: inputs})
    return tuple(ort_outputs)


def build_plugin_engine(dtype: trt.DataType, plugin_so: Path) -> trt.ICudaEngine:
    ctypes.CDLL(str(plugin_so), mode=ctypes.RTLD_GLOBAL)
    creator = trt.get_plugin_registry().get_creator(PLUGIN_NAME, "1", "")
    if creator is None:
        raise RuntimeError(f"Failed to register {PLUGIN_NAME}.")

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    q = network.add_input("q", dtype, (1, SEQ, HIDDEN))
    k = network.add_input("k", dtype, (1, SEQ, HIDDEN))
    v = network.add_input("v", dtype, (1, SEQ, HIDDEN))
    plugin = creator.create_plugin(
        "attncore",
        trt.PluginFieldCollection([]),
        trt.TensorRTPhase.BUILD,
    )
    layer = network.add_plugin_v3([q, k, v], [], plugin)
    out = layer.get_output(0)
    out.name = "out"
    network.mark_output(out)

    config = builder.create_builder_config()
    if dtype == trt.float16:
        config.set_flag(trt.BuilderFlag.FP16)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Failed to build plugin-only engine.")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("Failed to deserialize plugin-only engine.")
    return engine


def run_plugin_engine(
    engine: trt.ICudaEngine,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    dtype: trt.DataType,
) -> np.ndarray:
    context = engine.create_execution_context()
    stream = torch.cuda.Stream()
    np_dtype = np.float16 if dtype == trt.float16 else np.float32
    q_t = torch.from_numpy(np.ascontiguousarray(q.astype(np_dtype))).to("cuda")
    k_t = torch.from_numpy(np.ascontiguousarray(k.astype(np_dtype))).to("cuda")
    v_t = torch.from_numpy(np.ascontiguousarray(v.astype(np_dtype))).to("cuda")
    out_t = torch.empty_like(q_t)

    old_stream = torch.cuda.current_stream()
    try:
        torch.cuda.set_stream(stream)
        context.set_tensor_address("q", q_t.data_ptr())
        context.set_tensor_address("k", k_t.data_ptr())
        context.set_tensor_address("v", v_t.data_ptr())
        context.set_tensor_address("out", out_t.data_ptr())
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("execute_async_v3 failed.")
    finally:
        torch.cuda.set_stream(old_stream)
    stream.synchronize()
    return out_t.float().cpu().numpy()


def summarize_diff(reference: np.ndarray, candidate: np.ndarray, label: str) -> None:
    diff = np.abs(reference - candidate)
    print(
        f"{label}: max_abs={diff.max():.8f} "
        f"mean_abs={diff.mean():.8f} "
        f"p99={np.percentile(diff, 99):.8f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-id", type=int, default=236)
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--video", default=str(DEFAULT_VIDEO))
    parser.add_argument("--onnx", default=str(DEFAULT_ONNX))
    parser.add_argument("--plugin-so", default=str(DEFAULT_PLUGIN_SO))
    args = parser.parse_args()

    os.environ.setdefault("ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES", "true")
    os.environ.setdefault("PYTHONPATH", f"{REPO_ROOT}:{REPO_ROOT / 'inference_models'}")

    frame = load_frame(Path(args.video), args.frame_id)
    q, k, v, reference = extract_reference_tensors(
        onnx_path=Path(args.onnx),
        frame=frame,
        layer=args.layer,
    )
    print(
        f"reference frame={args.frame_id} layer={args.layer} "
        f"q={q.shape}/{q.dtype} k={k.shape}/{k.dtype} "
        f"v={v.shape}/{v.dtype} out={reference.shape}/{reference.dtype}"
    )

    engine_fp32 = build_plugin_engine(trt.float32, Path(args.plugin_so))
    output_fp32 = run_plugin_engine(engine_fp32, q, k, v, dtype=trt.float32)
    summarize_diff(reference, output_fp32, "plugin_fp32")

    engine_fp16 = build_plugin_engine(trt.float16, Path(args.plugin_so))
    output_fp16 = run_plugin_engine(engine_fp16, q, k, v, dtype=trt.float16)
    summarize_diff(reference, output_fp16, "plugin_fp16")

    os.environ["RFPROBE_ENCODER_ATTN_FULL_FP32_PATH"] = "1"
    engine_fp16_fp32path = build_plugin_engine(trt.float16, Path(args.plugin_so))
    output_fp16_fp32path = run_plugin_engine(
        engine_fp16_fp32path,
        q,
        k,
        v,
        dtype=trt.float16,
    )
    summarize_diff(reference, output_fp16_fp32path, "plugin_fp16_fullfp32path")


if __name__ == "__main__":
    main()
