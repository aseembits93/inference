#!/usr/bin/env python3
"""Run HF and RF together; at every logical stage capture outputs, align
shapes/layouts, and report cos / mean |Δ| / max |Δ|. Find the earliest
stage where cos drops materially below 1.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# requires HF_TOKEN env var
# requires ROBOFLOW_API_KEY env var

import numpy as np
import torch
from PIL import Image


def _compare(label: str, a: torch.Tensor, b: torch.Tensor):
    if a is None or b is None:
        print(f"  {label:40s}  (missing)")
        return
    # Align shapes if they differ only by batch/seq permutation.
    # Typical mismatches: (seq, 1, C) vs (1, seq, C).
    aa, bb = a.detach().float().cpu(), b.detach().float().cpu()
    if aa.shape != bb.shape:
        # Try permuting bb
        permuted = None
        if aa.ndim == bb.ndim == 3:
            # Try (seq, B, C) -> (B, seq, C)
            if bb.shape == (aa.shape[1], aa.shape[0], aa.shape[2]):
                permuted = bb.permute(1, 0, 2)
            elif aa.shape == (bb.shape[1], bb.shape[0], bb.shape[2]):
                aa = aa.permute(1, 0, 2)
                permuted = bb
        if aa.ndim == bb.ndim == 4 and sorted(aa.shape) == sorted(bb.shape):
            # Try last dim swap
            if aa.shape[-1] == bb.shape[1] and aa.shape[1] == bb.shape[-1]:
                # HHWC -> BCHW
                bb_p = bb.permute(0, 2, 3, 1)
                if bb_p.shape == aa.shape:
                    permuted = bb_p
        if permuted is not None:
            bb = permuted
    if aa.shape != bb.shape:
        print(f"  {label:40s}  SHAPE {tuple(a.shape)} vs {tuple(b.shape)}")
        return
    af = aa.double().flatten(); bf = bb.double().flatten()
    cos = float(af @ bf / (af.norm() * bf.norm() + 1e-20))
    d = (aa - bb).abs()
    print(f"  {label:40s}  shape={str(tuple(aa.shape)):24s}  "
          f"cos={cos:.6f}  mean|Δ|={d.mean().item():.4e}  max|Δ|={d.max().item():.4f}")


def _grab(t):
    if isinstance(t, (list, tuple)):
        return _grab(t[0])
    if isinstance(t, dict):
        for k in t:
            return _grab(t[k])
    if torch.is_tensor(t):
        return t.detach().float().cpu()
    return None


def main() -> int:
    IMG = Path(os.environ.get("SAM3_ASSETS", "tests/workflows/integration_tests/execution/assets") + "/dogs.jpg")
    image = Image.open(IMG).convert("RGB")
    from transformers import Sam3Processor, Sam3Model
    proc = Sam3Processor.from_pretrained("facebook/sam3", token=os.environ["HF_TOKEN"])
    inputs = proc(images=image, text="dog", return_tensors="pt").to("cuda")

    # HF
    print("Running HF ...")
    hf = Sam3Model.from_pretrained("facebook/sam3", token=os.environ["HF_TOKEN"]).to("cuda").eval()
    hf_cap = {}
    def _hf_hook(name):
        def h(mod, inp, out):
            hf_cap[name] = _grab(out)
        return h

    for i, m in enumerate(hf.vision_encoder.neck.fpn_layers):
        m.register_forward_hook(_hf_hook(f"fpn_{i}"))
    hf.vision_encoder.neck.position_encoding.register_forward_hook(_hf_hook("vision_pos_enc"))
    hf.vision_encoder.register_forward_hook(_hf_hook("vision_encoder_out"))
    hf.text_encoder.register_forward_hook(_hf_hook("text_encoder_out"))
    hf.text_projection.register_forward_hook(_hf_hook("text_proj_out"))
    hf.detr_encoder.register_forward_hook(_hf_hook("detr_encoder_out"))
    for i, m in enumerate(hf.detr_decoder.layers):
        m.register_forward_hook(_hf_hook(f"detr_decoder_layer_{i}"))
    hf.detr_decoder.register_forward_hook(_hf_hook("detr_decoder_out"))
    hf.mask_decoder.register_forward_hook(_hf_hook("mask_decoder_out"))
    hf.dot_product_scoring.register_forward_hook(_hf_hook("dot_prod_scoring_out"))

    with torch.inference_mode():
        hf_out = hf(pixel_values=inputs["pixel_values"], input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"])
    # also capture the final pred_* tensors
    hf_cap["pred_logits"] = hf_out.pred_logits.detach().float().cpu()
    hf_cap["pred_masks"] = hf_out.pred_masks.detach().float().cpu()
    hf_cap["pred_boxes"] = hf_out.pred_boxes.detach().float().cpu()
    hf_cap["presence_logits"] = hf_out.presence_logits.detach().float().cpu()
    del hf
    import gc; gc.collect(); torch.cuda.empty_cache()

    # RF
    print("Running RF ...")
    from inference.models.sam3.segment_anything3 import SegmentAnything3
    from sam3.train.data.collator import collate_fn_api
    from sam3.train.data.sam3_image_dataset import Datapoint as Sam3Datapoint
    from sam3.train.data.sam3_image_dataset import Image as Sam3ImageDP
    from sam3.train.transforms.basic_for_api import ComposeAPI, NormalizeAPI, ToTensorAPI, RandomResizeAPI
    from inference.models.sam3.segment_anything3 import _build_text_query
    from inference.core.env import SAM3_IMAGE_SIZE
    from sam3.model.utils.misc import copy_data_to_device

    rf = SegmentAnything3(model_id="sam3/sam3_final", api_key=os.environ["ROBOFLOW_API_KEY"])
    rf_cap = {}
    def _rf_hook(name):
        def h(mod, inp, out):
            rf_cap[name] = _grab(out)
        return h

    vb = rf.model.backbone.vision_backbone
    for i, m in enumerate(vb.convs):
        m.register_forward_hook(_rf_hook(f"fpn_{i}"))
    vb.position_encoding.register_forward_hook(_rf_hook("vision_pos_enc"))
    lb = rf.model.backbone.language_backbone
    lb.register_forward_hook(_rf_hook("language_backbone_out"))
    # Transformer
    rf.model.transformer.register_forward_hook(_rf_hook("transformer_out"))
    if hasattr(rf.model.transformer, "encoder"):
        rf.model.transformer.encoder.register_forward_hook(_rf_hook("transformer_encoder"))
    if hasattr(rf.model.transformer, "decoder"):
        rf.model.transformer.decoder.register_forward_hook(_rf_hook("transformer_decoder"))
        if hasattr(rf.model.transformer.decoder, "layers"):
            for i, m in enumerate(rf.model.transformer.decoder.layers):
                m.register_forward_hook(_rf_hook(f"transformer_decoder_layer_{i}"))
    rf.model.segmentation_head.register_forward_hook(_rf_hook("segmentation_head_out"))
    rf.model.dot_prod_scoring.register_forward_hook(_rf_hook("dot_prod_scoring_out"))

    transform = ComposeAPI(transforms=[
        RandomResizeAPI(sizes=SAM3_IMAGE_SIZE, max_size=SAM3_IMAGE_SIZE, square=True, consistent_transform=False),
        ToTensorAPI(), NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    h, w = image.size[1], image.size[0]
    dummy = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
    dp = Sam3Datapoint(find_queries=[], images=[Sam3ImageDP(data=dummy, objects=[], size=(h, w))])
    dp.find_queries.append(_build_text_query(coco_id=0, h=h, w=w, text="dog"))
    dp = transform(dp)
    dp.images[0].data = inputs["pixel_values"][0].cpu().clone()

    from sam3.model.tokenizer_ve import SimpleTokenizer
    orig = SimpleTokenizer.__call__
    def patched(self, texts, context_length=77, **kwargs):
        device = next(rf.model.parameters()).device
        ids = inputs["input_ids"][0]; mask = inputs["attention_mask"][0]
        real = ids[mask.bool()]
        out = torch.zeros((1, context_length), dtype=torch.long, device=device)
        n = min(real.numel(), context_length)
        out[0, :n] = real[:n].to(device)
        return out
    SimpleTokenizer.__call__ = patched

    batch = collate_fn_api(batch=[dp], dict_key="x")["x"]
    batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)
    with torch.inference_mode():
        rf_out = rf.model(batch)
    SimpleTokenizer.__call__ = orig

    # RF returns a list
    out0 = rf_out[0]
    rf_cap["pred_logits"] = out0["pred_logits"].detach().float().cpu()
    rf_cap["pred_masks"] = out0["pred_masks"].detach().float().cpu()
    if "pred_boxes" in out0:
        rf_cap["pred_boxes"] = out0["pred_boxes"].detach().float().cpu()
    if "presence_logit_dec" in out0:
        rf_cap["presence_logits"] = out0["presence_logit_dec"].detach().float().cpu()

    # ===== Compare =====
    print("\n=== Vision neck / FPN ===")
    for i in range(4):
        _compare(f"fpn_{i}", hf_cap.get(f"fpn_{i}"), rf_cap.get(f"fpn_{i}"))
    _compare("vision_pos_enc", hf_cap.get("vision_pos_enc"), rf_cap.get("vision_pos_enc"))

    print("\n=== DETR encoder ===")
    _compare("detr_encoder / transformer_encoder",
             hf_cap.get("detr_encoder_out"), rf_cap.get("transformer_encoder"))

    print("\n=== DETR decoder per-layer ===")
    for i in range(6):
        _compare(f"decoder_layer_{i}",
                 hf_cap.get(f"detr_decoder_layer_{i}"),
                 rf_cap.get(f"transformer_decoder_layer_{i}"))

    print("\n=== Final outputs ===")
    _compare("pred_logits", hf_cap.get("pred_logits"), rf_cap.get("pred_logits"))
    _compare("pred_boxes", hf_cap.get("pred_boxes"), rf_cap.get("pred_boxes"))
    _compare("pred_masks", hf_cap.get("pred_masks"), rf_cap.get("pred_masks"))
    _compare("presence_logits", hf_cap.get("presence_logits"), rf_cap.get("presence_logits"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
