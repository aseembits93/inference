#!/usr/bin/env python3
"""Compare raw outputs of hf_trt_fixed vs hf_pt_dummy_box on one image.

The dummy box baked into the ONNX export should reproduce HF-PT-with-
dummy-box exactly (modulo FP16 precision). If cos is low, TRT is
introducing additional error; if high, TRT is faithful and the residual
correctness gap is purely TRT's FP16 precision.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

# requires HF_TOKEN env var

import numpy as np
import torch
from PIL import Image


def _cos(a, b):
    a = a.double().flatten(); b = b.double().flatten()
    return float(a @ b / (a.norm() * b.norm() + 1e-20))


def main() -> int:
    # Initialize TRT plugins so ROIAlign works
    import tensorrt as trt
    trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.WARNING), "")

    IMG = Path(os.environ.get("SAM3_ASSETS", "tests/workflows/integration_tests/execution/assets") + "/dogs.jpg")
    image = Image.open(IMG).convert("RGB")
    from transformers import Sam3Processor, Sam3Model
    proc = Sam3Processor.from_pretrained("facebook/sam3", token=os.environ["HF_TOKEN"])
    inputs = proc(images=image, text="dog", return_tensors="pt").to("cuda")

    # PT with dummy box
    print("Running HF PT with dummy box ...")
    hf = Sam3Model.from_pretrained("facebook/sam3", token=os.environ["HF_TOKEN"]).to("cuda").eval()
    dummy_box = torch.zeros(1, 1, 4, device="cuda")
    dummy_lab = torch.tensor([[-10]], dtype=torch.long, device="cuda")
    with torch.inference_mode():
        out_pt = hf(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            input_boxes=dummy_box,
            input_boxes_labels=dummy_lab,
        )
    pt_pl = out_pt.pred_logits.float().cpu()
    pt_pb = out_pt.pred_boxes.float().cpu()
    pt_pm = out_pt.pred_masks.float().cpu()
    pt_prl = out_pt.presence_logits.float().cpu()
    print(f"PT pred_logits std: {pt_pl.std():.4f}")
    del hf
    import gc; gc.collect(); torch.cuda.empty_cache()

    # TRT fixed
    print("Running HF TRT (fixed) ...")
    sys.path.insert(0, ".")
    from bench_three_way import HFTrtRunner
    runner = HFTrtRunner(Path("./sam3_hf_onnx_fixed/sam3_hf_fp16.engine"))
    outs = runner(inputs["pixel_values"], inputs["input_ids"], inputs["attention_mask"])
    names = runner.output_names
    m = {n: outs[i].float().cpu() for i, n in enumerate(names)}

    print(f"TRT pred_logits std: {m['pred_logits'].std():.4f}")
    print(f"TRT pred_logits shape: {tuple(m['pred_logits'].shape)}")
    print(f"PT pred_logits shape: {tuple(pt_pl.shape)}")

    # align shapes
    def squeeze_trailing(t):
        if t.ndim > 1 and t.shape[-1] == 1:
            return t.squeeze(-1)
        return t
    pt_pl_s = squeeze_trailing(pt_pl)
    trt_pl = squeeze_trailing(m["pred_logits"])

    print()
    print("=== Raw-output comparison: HF-TRT (fixed) vs HF-PT (with dummy box) ===")
    for k, pt_t in [
        ("pred_logits", pt_pl_s),
        ("pred_boxes", pt_pb),
        ("pred_masks", pt_pm),
        ("presence_logits", pt_prl),
    ]:
        trt_t = squeeze_trailing(m[k])
        if pt_t.shape != trt_t.shape:
            print(f"  {k}: SHAPE MISMATCH pt={tuple(pt_t.shape)} trt={tuple(trt_t.shape)}")
            continue
        cos = _cos(pt_t, trt_t)
        d = (pt_t - trt_t).abs()
        print(f"  {k:18s} cos={cos:.6f}  mean|Δ|={d.mean().item():.4e}  max|Δ|={d.max().item():.4f}  "
              f"pt_std={pt_t.std().item():.4f}  trt_std={trt_t.std().item():.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
