#!/usr/bin/env python3
"""Export HF Sam3Model to ONNX with the FULL set of outputs needed by
Sam3Processor.post_process_instance_segmentation:

  pred_logits, pred_boxes, pred_masks, presence_logits, semantic_seg

The existing export_hf_sam3_onnx.py only returned pred_masks + semantic_seg,
which caused post_process_instance_segmentation to raise
AttributeError("'SimpleNamespace' object has no attribute 'pred_logits'").
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# requires HF_TOKEN env var

import torch
from PIL import Image

EXPORT_DIR = Path("./sam3_hf_onnx_full")
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
    print(f"pixel_values: {tuple(pixel_values.shape)} {pixel_values.dtype}")
    print(f"input_ids:    {tuple(input_ids.shape)} {input_ids.dtype}")

    wrapper = Sam3FullOutputWrapper(model).eval()

    print("Sanity check wrapper ...")
    with torch.inference_mode():
        pl, pb, pm, prl, ss = wrapper(pixel_values, input_ids, attention_mask)
    print(f"  pred_logits:    {tuple(pl.shape)} {pl.dtype}")
    print(f"  pred_boxes:     {tuple(pb.shape)} {pb.dtype}")
    print(f"  pred_masks:     {tuple(pm.shape)} {pm.dtype}")
    print(f"  presence_logits:{tuple(prl.shape)} {prl.dtype}")
    print(f"  semantic_seg:   {tuple(ss.shape)} {ss.dtype}")

    onnx_path = EXPORT_DIR / "sam3_full.onnx"
    print(f"\nExporting to {onnx_path} ...")
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
        dynamo=False,
        opset_version=17,
    )
    print(f"Exported: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
