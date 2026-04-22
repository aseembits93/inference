#!/usr/bin/env python3
"""Export SAM3 vision backbone as a strongly-typed FP16 ONNX graph.

Wrapper casts FP32 input to FP16 at the boundary. All weights and
activations are FP16. Returns FP32 outputs so downstream PyTorch gets
precision back. This forces TRT to actually run FP16 kernels (not
silently fall back to FP32 "for accuracy")."""

from __future__ import annotations

import os
import sys
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


class Sam3VisionFP16Wrapper(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone.to(torch.float16)

    def forward(self, samples: torch.Tensor):
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
    print(f"Patched {n} RoPE modules (v2)")

    wrap = Sam3VisionFP16Wrapper(backbone).eval()
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device="cuda", dtype=torch.float32)
    with torch.inference_mode():
        out = wrap(dummy)
    for i, t in enumerate(out):
        print(f"  out[{i}]: {tuple(t.shape)} {t.dtype}  range {t.min():.3f}..{t.max():.3f}")

    onnx_path = EXPORT_DIR / "sam3_vision_backbone_fp16_native.onnx"
    print(f"\nExporting to {onnx_path} (opset {OPSET}) ...")
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
