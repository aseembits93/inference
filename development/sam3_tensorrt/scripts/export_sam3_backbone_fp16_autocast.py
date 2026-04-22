#!/usr/bin/env python3
"""Export SAM3 vision backbone mimicking PyTorch's autocast(fp16) semantics.

PyTorch autocast keeps LayerNorm, SoftMax, and a few other ops in FP32 even
when activations are FP16. This script exports an ONNX that follows the
same convention:
  - FP16 weights for Linear / Conv / MatMul
  - FP32 weights for LayerNorm (gamma + beta stay FP32)
  - Casts around LN so activation dtype remains FP16 outside LN, FP32 inside

Hypothesis: the strongly-typed FP16 ONNX (weights including LN in FP16) is
what causes TRT FP16 execution to amplify outputs by ~2.5×. Matching PT's
autocast convention should recover correctness in TRT.
"""

from __future__ import annotations

import os, sys
from pathlib import Path

if "ROBOFLOW_API_KEY" not in os.environ:
    raise SystemExit("Set ROBOFLOW_API_KEY env var before running this script.")


import torch
import torch.nn as nn

# same-directory imports
from export_sam3_backbone_v2 import patch_vitdet_rope_v2

EXPORT_DIR = Path("./sam3_onnx_exports")
IMAGE_SIZE = 1008
OPSET = 17


def _cast_except_layernorm(module: nn.Module, dtype: torch.dtype) -> None:
    """Cast all parameters/buffers to `dtype`, but keep LayerNorm in FP32."""
    for m in module.modules():
        is_ln = isinstance(m, nn.LayerNorm) or "LayerNorm" in m.__class__.__name__
        if is_ln:
            for p in m.parameters(recurse=False):
                p.data = p.data.to(torch.float32)
            for name, buf in m.named_buffers(recurse=False):
                m.register_buffer(name, buf.to(torch.float32))
            continue
        for p in m.parameters(recurse=False):
            p.data = p.data.to(dtype)
        for name, buf in m.named_buffers(recurse=False):
            m.register_buffer(name, buf.to(dtype))


class Sam3VisionAutocastWrapper(nn.Module):
    """Input FP32 → cast to FP16 at boundary. LN weights stay FP32 (PT
    autocast semantics). Output FP32."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        _cast_except_layernorm(backbone, torch.float16)
        self.backbone = backbone

    def forward(self, samples: torch.Tensor):
        # Use autocast to replicate the exact training-time convention.
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            x = samples.to(torch.float16)
            out = self.backbone.forward_image(x)
        return (
            out["vision_features"].float(),
            out["vision_pos_enc"][0].float(),
            out["vision_pos_enc"][1].float(),
            out["vision_pos_enc"][2].float(),
            out["backbone_fpn"][0].float(),
            out["backbone_fpn"][1].float(),
            out["backbone_fpn"][2].float(),
        )


def main() -> int:
    from inference.models.sam3.segment_anything3 import SegmentAnything3
    print("Loading ...", flush=True)
    rf = SegmentAnything3(
        model_id="sam3/sam3_final", api_key=os.environ["ROBOFLOW_API_KEY"],
    )
    backbone = rf.model.backbone.eval()
    n = patch_vitdet_rope_v2(backbone)
    print(f"Patched {n} RoPE modules")

    # Also patch LN? Actually autocast handles that automatically - but for
    # ONNX export we need to manually keep LN in FP32.
    wrap = Sam3VisionAutocastWrapper(backbone).eval()
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device="cuda", dtype=torch.float32)
    with torch.inference_mode():
        out = wrap(dummy)
    for i, t in enumerate(out):
        print(f"  out[{i}]: {tuple(t.shape)} {t.dtype}  range {t.min():.3f}..{t.max():.3f}")

    onnx_path = EXPORT_DIR / "sam3_vision_backbone_fp16_autocast.onnx"
    print(f"\nExporting to {onnx_path} ...")
    output_names = [
        "vision_features",
        "vision_pos_enc_0", "vision_pos_enc_1", "vision_pos_enc_2",
        "backbone_fpn_0", "backbone_fpn_1", "backbone_fpn_2",
    ]
    with torch.inference_mode():
        torch.onnx.export(
            wrap, (dummy,), str(onnx_path),
            input_names=["samples"], output_names=output_names,
            opset_version=OPSET, do_constant_folding=True, dynamo=False,
        )
    print(f"Exported: {onnx_path}  ({onnx_path.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
