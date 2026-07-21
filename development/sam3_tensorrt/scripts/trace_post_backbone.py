#!/usr/bin/env python3
"""Trace HF and RF outputs at every post-vision-backbone stage.

Backbone outputs are bit-identical across 32 blocks (confirmed earlier).
Now hook:
 - vision neck / FPN output (multi-scale features)
 - position encoding (added to vision features)
 - text encoder output (pre-text projection)
 - text projection output
 - geometry encoder output
 - DETR encoder output (fused vision + text memory)
 - DETR decoder per-layer outputs
 - mask decoder output (pred_masks)
 - dot product scoring output (pred_logits)

Find the earliest stage where cos drops below 0.99.
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


def _cos(a, b):
    if a.shape != b.shape:
        return float("nan"), float("nan"), float("nan"), f"SHAPE {tuple(a.shape)} vs {tuple(b.shape)}"
    af = a.double().flatten(); bf = b.double().flatten()
    cos = float(af @ bf / (af.norm() * bf.norm() + 1e-20))
    d = (a - b).abs().float()
    return cos, float(d.mean()), float(d.max()), ""


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

    # ===== HF hooks =====
    print("Running HF ...")
    hf = Sam3Model.from_pretrained("facebook/sam3", token=os.environ["HF_TOKEN"]).to("cuda").eval()

    hf_cap = {}
    def _hook(name, store):
        def h(mod, inp, out):
            store[name] = _grab(out)
        return h

    # Vision path
    hf.vision_encoder.register_forward_hook(_hook("vision_encoder_out", hf_cap))
    for i, m in enumerate(hf.vision_encoder.neck.fpn_layers):
        m.register_forward_hook(_hook(f"fpn_{i}", hf_cap))
    hf.vision_encoder.neck.position_encoding.register_forward_hook(_hook("vision_pos_enc", hf_cap))
    # Text path
    hf.text_encoder.register_forward_hook(_hook("text_encoder_out", hf_cap))
    hf.text_projection.register_forward_hook(_hook("text_proj_out", hf_cap))
    # Encoder / decoder
    hf.detr_encoder.register_forward_hook(_hook("detr_encoder_out", hf_cap))
    for i, m in enumerate(hf.detr_decoder.layers):
        m.register_forward_hook(_hook(f"detr_decoder_layer_{i}", hf_cap))
    hf.detr_decoder.register_forward_hook(_hook("detr_decoder_out", hf_cap))
    hf.mask_decoder.register_forward_hook(_hook("mask_decoder_out", hf_cap))
    hf.dot_product_scoring.register_forward_hook(_hook("dot_prod_scoring_out", hf_cap))

    with torch.inference_mode():
        _ = hf(pixel_values=inputs["pixel_values"], input_ids=inputs["input_ids"],
               attention_mask=inputs["attention_mask"])
    del hf
    import gc; gc.collect(); torch.cuda.empty_cache()

    # ===== RF hooks =====
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

    # Vision path: backbone has forward_image method
    rf.model.backbone.register_forward_hook(_hook("backbone_out", rf_cap))
    # vision neck / FPN-like modules live in backbone.vision_backbone.convs
    vb = rf.model.backbone.vision_backbone
    for i, m in enumerate(vb.convs):
        m.register_forward_hook(_hook(f"vb_convs_{i}", rf_cap))
    vb.position_encoding.register_forward_hook(_hook("vision_pos_enc", rf_cap))
    # Language
    lb = rf.model.backbone.language_backbone
    lb.register_forward_hook(_hook("language_backbone_out", rf_cap))
    # Transformer (equivalent to detr encoder + decoder)
    rf.model.transformer.register_forward_hook(_hook("transformer_out", rf_cap))
    if hasattr(rf.model.transformer, "encoder"):
        rf.model.transformer.encoder.register_forward_hook(_hook("transformer_encoder", rf_cap))
    if hasattr(rf.model.transformer, "decoder"):
        rf.model.transformer.decoder.register_forward_hook(_hook("transformer_decoder", rf_cap))
        if hasattr(rf.model.transformer.decoder, "layers"):
            for i, m in enumerate(rf.model.transformer.decoder.layers):
                m.register_forward_hook(_hook(f"transformer_decoder_layer_{i}", rf_cap))
    rf.model.segmentation_head.register_forward_hook(_hook("segmentation_head_out", rf_cap))
    rf.model.dot_prod_scoring.register_forward_hook(_hook("dot_prod_scoring_out", rf_cap))

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
        _ = rf.model(batch)   # NO autocast - pure FP32
    SimpleTokenizer.__call__ = orig

    # ===== Print captures =====
    print("\n=== HF captures ===")
    for k, v in hf_cap.items():
        if v is None:
            print(f"  {k}: None")
        elif torch.is_tensor(v):
            print(f"  {k}: {tuple(v.shape)}  mean={v.mean():.4f} std={v.std():.4f}")
        else:
            print(f"  {k}: {type(v).__name__}")
    print("\n=== RF captures ===")
    for k, v in rf_cap.items():
        if v is None:
            print(f"  {k}: None")
        elif torch.is_tensor(v):
            print(f"  {k}: {tuple(v.shape)}  mean={v.mean():.4f} std={v.std():.4f}")
        else:
            print(f"  {k}: {type(v).__name__}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
