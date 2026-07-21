#!/usr/bin/env python3
"""Benchmark ORT (CUDA and TRT providers) vs PyTorch and native TRT."""

from __future__ import annotations

import os, sys, time
from pathlib import Path
from statistics import mean, stdev

if "ROBOFLOW_API_KEY" not in os.environ:
    raise SystemExit("Set ROBOFLOW_API_KEY env var before running this script.")


import numpy as np
import torch

ONNX_PATH = "./sam3_onnx_exports/sam3_vision_backbone_fp16_native.onnx"
WARMUP = 3
ITERS = 20


def main() -> int:
    import onnxruntime as ort

    torch.manual_seed(42)
    x_np = torch.randn(1, 3, 1008, 1008, dtype=torch.float32).numpy()

    # PyTorch baseline
    print("Loading SAM3 for PT baseline...")
    from inference.models.sam3.segment_anything3 import SegmentAnything3
    rf = SegmentAnything3(model_id="sam3/sam3_final", api_key=os.environ["ROBOFLOW_API_KEY"])
    b = rf.model.backbone
    x = torch.from_numpy(x_np).cuda()
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for _ in range(WARMUP):
                _ = b.forward_image(x)
            torch.cuda.synchronize()
            lats = []
            for _ in range(ITERS):
                torch.cuda.synchronize(); t = time.perf_counter()
                _ = b.forward_image(x)
                torch.cuda.synchronize(); lats.append((time.perf_counter()-t)*1000)
    print(f"PyTorch bf16: {mean(lats):.2f}ms ± {stdev(lats):.2f}")

    # ORT CUDA
    print("\nLoading ORT (CUDA)...")
    sess = ort.InferenceSession(ONNX_PATH, providers=["CUDAExecutionProvider"])
    for _ in range(WARMUP):
        sess.run(None, {"samples": x_np})
    lats = []
    for _ in range(ITERS):
        t = time.perf_counter()
        sess.run(None, {"samples": x_np})
        lats.append((time.perf_counter()-t)*1000)
    print(f"ORT CUDA FP16: {mean(lats):.2f}ms ± {stdev(lats):.2f}")
    out = sess.run(None, {"samples": x_np})
    print(f"  vision_features range: {out[0].min():.3f}..{out[0].max():.3f}")

    # ORT TRT (with FP16)
    print("\nLoading ORT (TensorRT EP)...")
    trt_opts = {
        "trt_fp16_enable": True,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": "/tmp/ort_trt_cache",
    }
    os.makedirs("/tmp/ort_trt_cache", exist_ok=True)
    try:
        sess = ort.InferenceSession(
            ONNX_PATH,
            providers=[("TensorrtExecutionProvider", trt_opts), "CUDAExecutionProvider"],
        )
    except Exception as e:
        print(f"  Failed: {e}")
        return 0
    for _ in range(WARMUP):
        sess.run(None, {"samples": x_np})
    lats = []
    for _ in range(ITERS):
        t = time.perf_counter()
        sess.run(None, {"samples": x_np})
        lats.append((time.perf_counter()-t)*1000)
    print(f"ORT TRT FP16: {mean(lats):.2f}ms ± {stdev(lats):.2f}")
    out = sess.run(None, {"samples": x_np})
    print(f"  vision_features range: {out[0].min():.3f}..{out[0].max():.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
