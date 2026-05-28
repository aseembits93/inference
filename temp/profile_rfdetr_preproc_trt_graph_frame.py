import os
import time

import cv2
import torch

from inference_models import AutoModel
from inference_models.models.common.cuda import use_cuda_context
from inference_models.models.common.trt import _trt_dtype_to_torch


VIDEO_PATH = os.environ.get(
    "VIDEO_PATH", "/home/ubuntu/inference/vehicles_312px.mp4"
)
DEVICE = os.environ.get("DEVICE", "cuda:0")
WARMUP = int(os.environ.get("WARMUP", "10"))
CYCLES = int(os.environ.get("CYCLES", "40"))
MODEL_ID = os.environ.get("MODEL_ID", "rfdetr-seg-nano")
CONSUME_OUTPUTS_INSIDE_GRAPH = (
    os.environ.get("CONSUME_OUTPUTS_INSIDE_GRAPH", "false").lower() == "true"
)


def load_frame(video_path: str):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read first frame from {video_path}.")
    return frame


class PreprocTRTGraphRunner:
    def __init__(self, model, frame):
        print("  build: init fast path", flush=True)
        model.pre_process(frame)
        state = model._fast_path_state
        if state is None:
            raise RuntimeError("Fast preprocess state was not initialized.")

        self.model = model
        self.frame = frame
        self.state = state
        self.preproc_out = state.out_buffers[0]
        self.device = model._device
        self.stream = torch.cuda.Stream(device=self.device)
        self.graph_context = model._engine.create_execution_context()
        status = self.graph_context.set_input_shape(
            model._input_name, tuple(self.preproc_out.shape)
        )
        if not status:
            raise RuntimeError("Failed to set TRT input shape.")
        status = self.graph_context.set_tensor_address(
            model._input_name, self.preproc_out.data_ptr()
        )
        if not status:
            raise RuntimeError("Failed to bind TRT input buffer.")

        self.output_buffers = []
        for output_name in model._output_names:
            output_shape = self.graph_context.get_tensor_shape(output_name)
            output_dtype = _trt_dtype_to_torch(
                model._engine.get_tensor_dtype(output_name)
            )
            output_buffer = torch.empty(
                tuple(output_shape), dtype=output_dtype, device=self.device
            )
            self.graph_context.set_tensor_address(output_name, output_buffer.data_ptr())
            self.output_buffers.append(output_buffer)
        self.consumer_buffers = (
            [torch.empty_like(buf) for buf in self.output_buffers]
            if CONSUME_OUTPUTS_INSIDE_GRAPH
            else None
        )

        model_module = __import__(
            "inference_models.models.rfdetr.rfdetr_instance_segmentation_trt",
            fromlist=["triton_preprocess_rfdetr_stretch"],
        )
        self.triton_preprocess = model_module.triton_preprocess_rfdetr_stretch
        self.target_h = model._inference_config.network_input.training_input_size.height
        self.target_w = model._inference_config.network_input.training_input_size.width
        self.means = tuple(
            float(v)
            for v in model._inference_config.network_input.normalization[0]
        )
        self.stds = tuple(
            float(v)
            for v in model._inference_config.network_input.normalization[1]
        )

        print("  build: warmup launch", flush=True)
        with use_cuda_context(context=model._cuda_context):
            with torch.cuda.stream(self.stream):
                self.state.src_gpu.copy_(self.state.pinned_host, non_blocking=True)
                self.triton_preprocess(
                    src=self.state.src_gpu,
                    tables=self.state.tables,
                    target_h=self.target_h,
                    target_w=self.target_w,
                    means=self.means,
                    stds=self.stds,
                    swap_rb=True,
                    out=self.preproc_out,
                )
                status = self.graph_context.execute_async_v3(
                    stream_handle=self.stream.cuda_stream
                )
                if not status:
                    raise RuntimeError("TRT warmup failed.")
                if self.consumer_buffers is not None:
                    for consumer, output in zip(self.consumer_buffers, self.output_buffers):
                        consumer.copy_(output, non_blocking=True)
            self.stream.synchronize()

        self.cuda_graph = torch.cuda.CUDAGraph()
        print("  build: capture graph", flush=True)
        with use_cuda_context(context=model._cuda_context):
            with torch.cuda.graph(self.cuda_graph, stream=self.stream):
                self.state.src_gpu.copy_(self.state.pinned_host, non_blocking=True)
                self.triton_preprocess(
                    src=self.state.src_gpu,
                    tables=self.state.tables,
                    target_h=self.target_h,
                    target_w=self.target_w,
                    means=self.means,
                    stds=self.stds,
                    swap_rb=True,
                    out=self.preproc_out,
                )
                status = self.graph_context.execute_async_v3(
                    stream_handle=self.stream.cuda_stream
                )
                if not status:
                    raise RuntimeError("TRT capture failed.")
                if self.consumer_buffers is not None:
                    for consumer, output in zip(self.consumer_buffers, self.output_buffers):
                        consumer.copy_(output, non_blocking=True)
        print("  build: done", flush=True)

    def replay_once(self):
        print("  replay: host copy", flush=True)
        self.state.pinned_host.numpy()[:] = self.frame
        with use_cuda_context(context=self.model._cuda_context):
            print("  replay: graph replay", flush=True)
            self.cuda_graph.replay()
            print("  replay: stream sync", flush=True)
            self.stream.synchronize()
        source = self.consumer_buffers if self.consumer_buffers is not None else self.output_buffers
        return [tensor.clone().cpu() for tensor in source]

    def benchmark(self, cycles: int) -> float:
        start = time.perf_counter()
        for _ in range(cycles):
            self.state.pinned_host.numpy()[:] = self.frame
            with use_cuda_context(context=self.model._cuda_context):
                self.cuda_graph.replay()
        with use_cuda_context(context=self.model._cuda_context):
            self.stream.synchronize()
        elapsed = time.perf_counter() - start
        return cycles / elapsed if elapsed > 0 else 0.0


def benchmark_current_preproc_trt(model, frame, cycles: int) -> float:
    last_evt = None
    start = time.perf_counter()
    for _ in range(cycles):
        pre_processed_images, _ = model.pre_process(frame)
        raw = model.forward(pre_processed_images)
        last_evt = getattr(raw[0], "_trt_produce_event", None)
    if last_evt is not None:
        last_evt.synchronize()
    elapsed = time.perf_counter() - start
    return cycles / elapsed if elapsed > 0 else 0.0


def main() -> None:
    print("stage: load frame", flush=True)
    frame = load_frame(VIDEO_PATH)
    print("stage: load model", flush=True)
    model = AutoModel.from_pretrained(
        model_id_or_path=MODEL_ID,
        device=torch.device(DEVICE),
        backend="trt",
    )

    print("stage: build graph runner", flush=True)
    runner = PreprocTRTGraphRunner(model, frame)
    print("stage: replay once", flush=True)
    graph_out = runner.replay_once()

    print("stage: baseline once", flush=True)
    pre_processed_images, _ = model.pre_process(frame)
    raw = model.forward(pre_processed_images)
    produce_evt = getattr(raw[0], "_trt_produce_event", None)
    if produce_evt is not None:
        produce_evt.synchronize()
    baseline_source = raw
    baseline_out = [tensor.clone().cpu() for tensor in baseline_source]

    for idx, (baseline, graph) in enumerate(zip(baseline_out, graph_out)):
        if not torch.equal(baseline, graph):
            max_abs = (baseline - graph).abs().max().item()
            raise RuntimeError(
                f"Output {idx} mismatch between baseline and graph replay; max_abs={max_abs}"
            )
    print("parity: raw TRT outputs matched exactly", flush=True)

    for _ in range(WARMUP):
        _ = runner.replay_once()
        pre_processed_images, _ = model.pre_process(frame)
        raw = model.forward(pre_processed_images)
        evt = getattr(raw[0], "_trt_produce_event", None)
        if evt is not None:
            evt.synchronize()

    current_fps = benchmark_current_preproc_trt(model, frame, CYCLES)
    graph_fps = runner.benchmark(CYCLES)
    print(f"current_preproc_trt_fps={current_fps:.2f}")
    print(f"graph_preproc_trt_fps={graph_fps:.2f}")


if __name__ == "__main__":
    main()
