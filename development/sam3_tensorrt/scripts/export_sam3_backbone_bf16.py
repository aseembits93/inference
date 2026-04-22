#!/usr/bin/env python3
"""Export SAM3 vision backbone (forward_image) to ONNX as a BF16 graph.

The wrapper casts the FP32 input to BF16 and returns FP32 outputs. Internal
weights/activations are BF16 so TRT must use BF16 kernels throughout (no
silent fallback to FP32 as happens when you just set BuilderFlag.BF16 on an
FP32 network).

Uses the same real-arithmetic RoPE patch as export_sam3_backbone_onnx.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if "ROBOFLOW_API_KEY" not in os.environ:
    raise SystemExit("Set ROBOFLOW_API_KEY env var before running this script.")


import torch
import torch.nn as nn

# same-directory imports
from export_sam3_backbone_onnx import patch_vitdet_rope

EXPORT_DIR = Path("./sam3_onnx_exports")
IMAGE_SIZE = 1008
OPSET = 17


class Sam3VisionBF16Wrapper(nn.Module):
    """Wrap backbone.forward_image with BF16 weights & activations.

    Input: FP32 samples (casts to BF16 at the boundary).
    Output: FP32 tensors (casts back for downstream PyTorch precision).
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone.to(torch.bfloat16)

    def forward(self, samples: torch.Tensor):
        # Cast input to BF16 inside the graph (so TRT captures the cast)
        x = samples.to(torch.bfloat16)
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

    print("Loading SAM3 ...", flush=True)
    rf = SegmentAnything3(
        model_id="sam3/sam3_final", api_key=os.environ["ROBOFLOW_API_KEY"],
    )
    backbone = rf.model.backbone.eval()
    n = patch_vitdet_rope(backbone)
    print(f"Patched {n} RoPE buffer(s)")

    wrap = Sam3VisionBF16Wrapper(backbone).eval()

    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device="cuda", dtype=torch.float32)
    with torch.inference_mode():
        out = wrap(dummy)
    for i, t in enumerate(out):
        print(f"  out[{i}]: {tuple(t.shape)} {t.dtype}")

    onnx_path = EXPORT_DIR / "sam3_vision_backbone_bf16.onnx"
    print(f"\nExporting to {onnx_path} (opset {OPSET}) ...")

    output_names = [
        "vision_features",
        "vision_pos_enc_0", "vision_pos_enc_1", "vision_pos_enc_2",
        "backbone_fpn_0", "backbone_fpn_1", "backbone_fpn_2",
    ]

    with torch.inference_mode():
        torch.onnx.export(
            wrap,
            (dummy,),
            str(onnx_path),
            input_names=["samples"],
            output_names=output_names,
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
            verbose=False,
        )
    size_mb = onnx_path.stat().st_size / 1e6
    print(f"Exported: {onnx_path}  ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
