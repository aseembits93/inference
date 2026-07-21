#!/usr/bin/env python3
"""Re-export HF Sam3Model with torch.onnx.export(dynamo=True).

Hypothesis: the TorchScript-based exporter (dynamo=False) captured the
pred_logits computation incorrectly — possibly because of a conditional
branch, default arg, or unused codepath that only fires at inference.
Dynamo's export is more faithful to PyTorch runtime semantics, so a
dynamo export should either fix the issue or prove that the divergence
is intrinsic to the model structure (not exporter-dependent).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# requires HF_TOKEN env var

import torch
from PIL import Image

EXPORT_DIR = Path("./sam3_hf_onnx_dynamo")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


class Sam3FullOutputWrapper(torch.nn.Module):
    def __init__(self, sam3):
        super().__init__()
        self.sam3 = sam3

    def forward(self, pixel_values, input_ids, attention_mask):
        out = self.sam3(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return (
            out.pred_logits,
            out.pred_boxes,
            out.pred_masks,
            out.presence_logits,
            out.semantic_seg,
        )


def main() -> int:
    from transformers import Sam3Model, Sam3Processor

    device = "cpu"
    print(f"Loading HF Sam3Model on {device} ...", flush=True)
    proc = Sam3Processor.from_pretrained("facebook/sam3", token=os.environ["HF_TOKEN"])
    model = (
        Sam3Model.from_pretrained("facebook/sam3", token=os.environ["HF_TOKEN"])
        .to(device)
        .eval()
    )

    img = Path(
        os.environ.get("SAM3_ASSETS", "tests/workflows/integration_tests/execution/assets") + "/dogs.jpg"
    )
    image = Image.open(img).convert("RGB")
    inputs = proc(images=image, text="dog", return_tensors="pt").to(device)

    pixel_values = inputs["pixel_values"]
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    wrapper = Sam3FullOutputWrapper(model).eval()

    with torch.inference_mode():
        pl, pb, pm, prl, ss = wrapper(pixel_values, input_ids, attention_mask)
    print(f"Sanity: pl std={pl.std():.4f}, pm std={pm.std():.4f}")

    onnx_path = EXPORT_DIR / "sam3_full_dynamo.onnx"
    print(f"\nExporting to {onnx_path} via torch.onnx.export(dynamo=True) ...")
    torch.onnx.export(
        wrapper,
        (pixel_values, input_ids, attention_mask),
        str(onnx_path),
        input_names=["pixel_values", "input_ids", "attention_mask"],
        output_names=[
            "pred_logits",
            "pred_boxes",
            "pred_masks",
            "presence_logits",
            "semantic_seg",
        ],
        dynamo=True,
        opset_version=17,
    )
    print(f"Exported: {onnx_path}")
    # Show files in export dir
    for f in sorted(EXPORT_DIR.iterdir()):
        print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
