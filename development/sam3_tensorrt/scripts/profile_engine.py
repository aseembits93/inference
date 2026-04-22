#!/usr/bin/env python3
"""Profile a TRT engine using a profiler callback to find hot layers."""

from __future__ import annotations

import sys
from pathlib import Path

import tensorrt as trt
import torch


class Profiler(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.times: list[tuple[str, float]] = []

    def report_layer_time(self, layer_name: str, ms: float):
        self.times.append((layer_name, ms))


def main(engine_path: str) -> int:
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()

    # Set up I/O
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        shape = tuple(engine.get_tensor_shape(name))
        if mode == trt.TensorIOMode.INPUT:
            ctx.set_input_shape(name, shape)

    # Allocate all tensors
    bufs = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        dtype_map = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
                     trt.DataType.BF16: torch.bfloat16}
        dtype = dtype_map[engine.get_tensor_dtype(name)]
        buf = torch.empty(shape, dtype=dtype, device="cuda")
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            buf.normal_()
        bufs[name] = buf
        ctx.set_tensor_address(name, buf.data_ptr())

    # Warmup
    stream = torch.cuda.current_stream().cuda_stream
    for _ in range(3):
        ctx.execute_async_v3(stream_handle=stream)
    torch.cuda.synchronize()

    # Profile
    profiler = Profiler()
    ctx.profiler = profiler
    ctx.execute_async_v3(stream_handle=stream)
    torch.cuda.synchronize()

    total = sum(t for _, t in profiler.times)
    profiler.times.sort(key=lambda x: -x[1])
    print(f"Total GPU time: {total:.2f} ms")
    print(f"\nTop 20 layers by time:")
    for name, t in profiler.times[:20]:
        print(f"  {t:7.3f} ms  ({t/total*100:5.1f}%)  {name[:120]}")

    # Bucket by name pattern
    import re
    buckets = {
        "attn": 0.0, "mlp": 0.0, "convnext": 0.0, "patch_embed": 0.0,
        "neck_or_fpn": 0.0, "other": 0.0,
    }
    for name, t in profiler.times:
        n = name.lower()
        if "attn" in n:
            buckets["attn"] += t
        elif "mlp" in n:
            buckets["mlp"] += t
        elif "convnext" in n or "dconv" in n:
            buckets["convnext"] += t
        elif "patch_embed" in n:
            buckets["patch_embed"] += t
        elif "neck" in n or "fpn" in n or "convs" in n:
            buckets["neck_or_fpn"] += t
        else:
            buckets["other"] += t
    print(f"\nBuckets:")
    for k, v in sorted(buckets.items(), key=lambda x: -x[1]):
        print(f"  {k:15s}: {v:6.2f} ms  ({v/total*100:5.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
