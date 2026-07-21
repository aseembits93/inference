#!/usr/bin/env python3
"""Per-layer DETR encoder comparison between HF and RF.

Vision features entering DETR encoder are bit-identical (verified).
Text features (tokens identical after monkey-patch) should be equivalent.
So if we hook every encoder layer output, we'll see exactly which layer
introduces the drift.
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


def _grab(t):
    if isinstance(t, (list, tuple)):
        return _grab(t[0])
    if isinstance(t, dict):
        for k in t:
            return _grab(t[k])
    if torch.is_tensor(t):
        return t.detach().float().cpu()
    return None


def _compare(label, a, b):
    if a is None or b is None:
        print(f"  {label}: None"); return
    aa, bb = a, b
    # Reshape / permute to compare (1, 5184, 256) vs (5184, 1, 256)
    if aa.ndim == bb.ndim == 3:
        if aa.shape == bb.shape:
            pass
        elif aa.shape == (bb.shape[1], bb.shape[0], bb.shape[2]):
            aa = aa.permute(1, 0, 2)
        elif bb.shape == (aa.shape[1], aa.shape[0], aa.shape[2]):
            bb = bb.permute(1, 0, 2)
    if aa.shape != bb.shape:
        print(f"  {label}: SHAPE {tuple(a.shape)} vs {tuple(b.shape)}")
        return
    af = aa.double().flatten(); bf = bb.double().flatten()
    cos = float(af @ bf / (af.norm() * bf.norm() + 1e-20))
    d = (aa - bb).abs()
    print(f"  {label:40s} cos={cos:.6f}  mean|Δ|={d.mean():.4e}  max|Δ|={d.max():.4f}")


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
    def hf_h(name):
        def h(mod, inp, out): hf_cap[name] = _grab(out)
        return h
    hf.text_encoder.register_forward_hook(hf_h("text_encoder_out"))
    hf.text_projection.register_forward_hook(hf_h("text_proj_out"))
    for i, m in enumerate(hf.detr_encoder.layers):
        m.register_forward_hook(hf_h(f"detr_enc_L{i}"))
    with torch.inference_mode():
        hf_out = hf(pixel_values=inputs["pixel_values"], input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"])
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
    def rf_h(name):
        def h(mod, inp, out): rf_cap[name] = _grab(out)
        return h
    lb = rf.model.backbone.language_backbone
    lb.register_forward_hook(rf_h("language_backbone_out"))
    # Language backbone internals
    if hasattr(lb, "encoder"):
        lb.encoder.register_forward_hook(rf_h("language_encoder_out"))
    if hasattr(lb, "resizer"):
        lb.resizer.register_forward_hook(rf_h("language_resizer_out"))

    enc = rf.model.transformer.encoder
    if hasattr(enc, "layers"):
        for i, m in enumerate(enc.layers):
            m.register_forward_hook(rf_h(f"transformer_enc_L{i}"))

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
        _ = rf.model(batch)
    SimpleTokenizer.__call__ = orig

    # ===== Compare text encoder =====
    print("\n=== Text encoder ===")
    print("HF captures:")
    for k, v in hf_cap.items():
        if "text" in k and v is not None:
            print(f"  {k}: {tuple(v.shape)}  mean={v.mean():.4f} std={v.std():.4f}")
    print("RF captures:")
    for k, v in rf_cap.items():
        if "language" in k and v is not None:
            print(f"  {k}: {tuple(v.shape)}  mean={v.mean():.4f} std={v.std():.4f}")

    # HF text_encoder returns a big scalar — possibly wrong capture
    # text_projection gives the projected text (1, 32, 256). That's the natural comparison.
    hf_txt = hf_cap.get("text_proj_out")
    rf_txt_resize = rf_cap.get("language_resizer_out")  # (77, 1, 256)?
    if hf_txt is not None and rf_txt_resize is not None:
        print(f"\nHF text_proj_out: {tuple(hf_txt.shape)}")
        print(f"RF language_resizer_out: {tuple(rf_txt_resize.shape)}")

    # ===== Compare DETR encoder per layer =====
    print("\n=== DETR encoder per layer ===")
    for i in range(6):
        _compare(f"enc_layer_{i}",
                 hf_cap.get(f"detr_enc_L{i}"),
                 rf_cap.get(f"transformer_enc_L{i}"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
