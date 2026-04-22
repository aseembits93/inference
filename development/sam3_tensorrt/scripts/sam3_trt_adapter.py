#!/usr/bin/env python3
"""SAM3 TRT adapter: swaps PyTorch vision backbone with TRT engine.

The full SAM3 model stays in PyTorch except for the vision backbone
(forward_image), which accounts for the bulk of the ~230ms forward time.

Usage:
    from sam3_trt_adapter import patch_sam3_with_trt_backbone
    patch_sam3_with_trt_backbone(rf_model.model, engine_path)
    # rf_model.infer_from_request(...) now uses TRT for vision path
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import tensorrt as trt
import torch


def _trt_dtype_to_torch(trt_dtype):
    return {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT64: torch.int64,
        trt.DataType.BF16: torch.bfloat16,
    }[trt_dtype]


class Sam3VisionTRT:
    """Thin TRT runner for the SAM3 vision backbone ONNX export.

    - Pre-allocates output buffers (static shapes since export is static).
    - Keeps a dedicated execution context + CUDA stream.
    - Accepts fp16 CUDA tensor (1, 3, 1008, 1008).
    - Returns the dict matching backbone.forward_image() output shape.
    """

    def __init__(self, engine_path: str | Path, device: torch.device = torch.device("cuda:0")):
        self.device = device
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Discover tensor names/roles
        n = self.engine.num_io_tensors
        self.input_names = []
        self.output_names = []
        for i in range(n):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        assert len(self.input_names) == 1, f"Expected 1 input, got {self.input_names}"
        self.input_name = self.input_names[0]

        # Pre-allocate output buffers (static shapes)
        self.output_buffers: list[torch.Tensor] = []
        for out_name in self.output_names:
            shape = tuple(self.engine.get_tensor_shape(out_name))
            dtype = _trt_dtype_to_torch(self.engine.get_tensor_dtype(out_name))
            buf = torch.empty(shape, dtype=dtype, device=self.device)
            self.output_buffers.append(buf)
            self.context.set_tensor_address(out_name, buf.data_ptr())

        # Fix input shape (static)
        input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.input_shape = input_shape
        self.input_dtype = _trt_dtype_to_torch(
            self.engine.get_tensor_dtype(self.input_name)
        )
        self.context.set_input_shape(self.input_name, input_shape)

        # Pre-allocate a pinned-layout input buffer on GPU
        self.input_buffer = torch.empty(
            input_shape, dtype=self.input_dtype, device=self.device
        )
        self.context.set_tensor_address(self.input_name, self.input_buffer.data_ptr())

    def run(self, samples: torch.Tensor) -> dict:
        # Cast / copy input to the fp16 buffer
        if samples.dtype != self.input_dtype or not samples.is_contiguous():
            samples = samples.to(self.input_dtype).contiguous()
        self.input_buffer.copy_(samples, non_blocking=True)

        stream = torch.cuda.current_stream(self.device)
        ok = self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        if not ok:
            raise RuntimeError("TRT execute_async_v3 failed")

        # Output order follows onnx_export output_names
        vf = self.output_buffers[0]
        vision_pos_enc = [self.output_buffers[1], self.output_buffers[2], self.output_buffers[3]]
        backbone_fpn = [self.output_buffers[4], self.output_buffers[5], self.output_buffers[6]]
        return {
            "vision_features": vf,
            "vision_pos_enc": vision_pos_enc,
            "backbone_fpn": backbone_fpn,
            "sam2_backbone_out": None,
        }


def patch_sam3_with_trt_backbone(
    sam3_model: torch.nn.Module,
    engine_path: str | Path,
) -> Sam3VisionTRT:
    """Replace SAM3VLBackbone.forward_image with TRT runner.

    Returns the runner (caller should keep a reference so it isn't GC'd).
    """
    runner = Sam3VisionTRT(engine_path, device=torch.device("cuda:0"))
    backbone = sam3_model.backbone

    orig_forward_image = backbone.forward_image

    def forward_image_trt(samples: torch.Tensor):
        # Engine input dtype is runner.input_dtype — cast to that
        if samples.dtype != runner.input_dtype:
            samples = samples.to(runner.input_dtype)
        return runner.run(samples)

    backbone.forward_image = forward_image_trt
    backbone._orig_forward_image = orig_forward_image  # keep for fallback
    backbone._trt_runner = runner
    return runner


__all__ = ["Sam3VisionTRT", "patch_sam3_with_trt_backbone"]
