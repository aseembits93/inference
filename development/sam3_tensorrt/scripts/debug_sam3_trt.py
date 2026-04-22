#!/usr/bin/env python3
"""Debug TRT backbone output vs PyTorch backbone output."""

from __future__ import annotations

import os
import sys
from pathlib import Path

if "ROBOFLOW_API_KEY" not in os.environ:
    raise SystemExit("Set ROBOFLOW_API_KEY env var before running this script.")


import torch
import numpy as np

import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parent))
from sam3_trt_adapter import Sam3VisionTRT

ENGINE_PATH = os.environ.get(
    "SAM3_ENGINE_PATH",
    "./sam3_onnx_exports/sam3_vision_backbone_fp16.engine",
)


def main() -> int:
    # Load models
    from inference.models.sam3.segment_anything3 import SegmentAnything3
    rf = SegmentAnything3(
        model_id="sam3/sam3_final",
        api_key=os.environ["ROBOFLOW_API_KEY"],
    )
    backbone_pt = rf.model.backbone

    # Load TRT engine
    runner = Sam3VisionTRT(ENGINE_PATH)
    print("=" * 60)
    print("TRT runner info")
    print("=" * 60)
    print(f"  input_names : {runner.input_names}")
    print(f"  output_names: {runner.output_names}")
    print(f"  input_shape : {runner.input_shape}")
    print(f"  input_dtype : {runner.input_dtype}")
    for i, n in enumerate(runner.output_names):
        buf = runner.output_buffers[i]
        print(f"  output[{i}] = {n}: shape={tuple(buf.shape)} dtype={buf.dtype}")

    # Random input
    torch.manual_seed(42)
    x_fp32 = torch.randn(1, 3, 1008, 1008, device="cuda", dtype=torch.float32)
    x_fp16 = x_fp32.to(torch.float16)

    # PyTorch baseline (exactly as used in SAM3 inference: bf16 autocast)
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pt_out = backbone_pt.forward_image(x_fp32)

    # Also run PyTorch in pure FP32 (no autocast) for reference
    with torch.inference_mode():
        pt_fp32_out = backbone_pt.forward_image(x_fp32)

    print("\n" + "=" * 60)
    print("PyTorch-FP32 (no autocast) output magnitudes")
    print("=" * 60)
    for k, v in pt_fp32_out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: min={v.float().min().item():.4f} max={v.float().max().item():.4f}")
        elif isinstance(v, list):
            for i, t in enumerate(v):
                if isinstance(t, torch.Tensor):
                    print(f"  {k}[{i}]: min={t.float().min().item():.4f} max={t.float().max().item():.4f}")

    # Also run the patched PyTorch backbone in fp16 to isolate precision effects
    # (backbone's parameters are still fp32 originally).
    # (Skip for now; see TRT vs PyTorch-bf16 first.)

    print("\n" + "=" * 60)
    print("PyTorch backbone output keys/shapes")
    print("=" * 60)
    for k, v in pt_out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype} min={v.float().min().item():.4f} max={v.float().max().item():.4f}")
        elif isinstance(v, list):
            for i, t in enumerate(v):
                if isinstance(t, torch.Tensor):
                    print(f"  {k}[{i}]: shape={tuple(t.shape)} dtype={t.dtype} min={t.float().min().item():.4f} max={t.float().max().item():.4f}")

    # TRT run
    trt_out = runner.run(x_fp16)

    print("\n" + "=" * 60)
    print("TRT output keys/shapes")
    print("=" * 60)
    for k, v in trt_out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype} min={v.float().min().item():.4f} max={v.float().max().item():.4f}")
        elif isinstance(v, list):
            for i, t in enumerate(v):
                if isinstance(t, torch.Tensor):
                    print(f"  {k}[{i}]: shape={tuple(t.shape)} dtype={t.dtype} min={t.float().min().item():.4f} max={t.float().max().item():.4f}")

    # Per-key cosine similarity
    print("\n" + "=" * 60)
    print("Cosine similarity (TRT vs PyTorch)")
    print("=" * 60)

    def cos(a, b):
        a = a.float().flatten()
        b = b.float().flatten()
        return (a @ b / (a.norm() * b.norm() + 1e-12)).item()

    def rel_err(a, b):
        a = a.float()
        b = b.float()
        return ((a - b).abs().mean() / (a.abs().mean() + 1e-12)).item()

    print("--- TRT vs PyTorch-bf16-autocast ---")
    print(f"  vision_features: cos={cos(trt_out['vision_features'], pt_out['vision_features']):.6f}  rel_err={rel_err(trt_out['vision_features'], pt_out['vision_features']):.6f}")
    for i in range(3):
        c = cos(trt_out["backbone_fpn"][i], pt_out["backbone_fpn"][i])
        e = rel_err(trt_out["backbone_fpn"][i], pt_out["backbone_fpn"][i])
        print(f"  backbone_fpn[{i}]: cos={c:.6f}  rel_err={e:.6f}")

    print("--- TRT vs PyTorch-FP32 (no autocast) ---")
    print(f"  vision_features: cos={cos(trt_out['vision_features'], pt_fp32_out['vision_features']):.6f}  rel_err={rel_err(trt_out['vision_features'], pt_fp32_out['vision_features']):.6f}")
    for i in range(3):
        c = cos(trt_out["backbone_fpn"][i], pt_fp32_out["backbone_fpn"][i])
        e = rel_err(trt_out["backbone_fpn"][i], pt_fp32_out["backbone_fpn"][i])
        print(f"  backbone_fpn[{i}]: cos={c:.6f}  rel_err={e:.6f}")

    print("--- PyTorch-bf16 vs PyTorch-FP32 ---")
    print(f"  vision_features: cos={cos(pt_out['vision_features'], pt_fp32_out['vision_features']):.6f}  rel_err={rel_err(pt_out['vision_features'], pt_fp32_out['vision_features']):.6f}")
    for i in range(3):
        c = cos(pt_out["backbone_fpn"][i], pt_fp32_out["backbone_fpn"][i])
        e = rel_err(pt_out["backbone_fpn"][i], pt_fp32_out["backbone_fpn"][i])
        print(f"  backbone_fpn[{i}]: cos={c:.6f}  rel_err={e:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
