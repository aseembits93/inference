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


def get_ort_reference(
    *,
    onnx_path: Path,
    input_tensor: np.ndarray,
    output_name: str,
) -> np.ndarray:
    model = onnx.load(str(onnx_path))
    model.graph.output.append(
        onnx.helper.make_tensor_value_info(
            output_name,
            onnx.TensorProto.FLOAT,
            None,
        )
    )
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    return session.run([output_name], {session.get_inputs()[0].name: input_tensor})[0]


def run_trt_engine(
    *,
    engine_path: Path,
    plugin_so: Path | None,
    input_tensor: np.ndarray,
    output_name: str,
) -> np.ndarray:
    if plugin_so is not None:
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
        raise RuntimeError(f"Failed to set shape for {input_name}.")

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

    if output_name not in outputs:
        raise KeyError(f"Output {output_name} not found in engine outputs: {list(outputs)}")
    return outputs[output_name].float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--frame-id", type=int, default=236)
    parser.add_argument("--video", default=str(DEFAULT_VIDEO))
    parser.add_argument("--onnx", default=str(DEFAULT_ONNX))
    parser.add_argument("--plugin-so", default=None)
    args = parser.parse_args()

    frame = load_frame(Path(args.video), args.frame_id)
    input_tensor = preprocess_frame(frame)
    reference = get_ort_reference(
        onnx_path=Path(args.onnx),
        input_tensor=input_tensor,
        output_name=args.output_name,
    )
    candidate = run_trt_engine(
        engine_path=Path(args.engine),
        plugin_so=(Path(args.plugin_so) if args.plugin_so else None),
        input_tensor=input_tensor,
        output_name=args.output_name,
    )
    diff = np.abs(reference - candidate)
    print(
        f"output={args.output_name} frame={args.frame_id} "
        f"shape={reference.shape} max_abs={diff.max():.8f} "
        f"mean_abs={diff.mean():.8f} p99={np.percentile(diff, 99):.8f}"
    )


if __name__ == "__main__":
    main()
