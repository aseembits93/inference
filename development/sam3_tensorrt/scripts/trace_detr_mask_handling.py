#!/usr/bin/env python3
"""Investigate how HF vs RF pass attention masks to the DETR encoder.

Theory: the text encoder outputs are bit-identical on the 3 real tokens.
But the DETR encoder cos is 0.995 — so either pad tokens influence
something, or there's a non-masking detail that differs.
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


def main() -> int:
    IMG = Path(os.environ.get("SAM3_ASSETS", "tests/workflows/integration_tests/execution/assets") + "/dogs.jpg")
    image = Image.open(IMG).convert("RGB")
    from transformers import Sam3Processor, Sam3Model
    proc = Sam3Processor.from_pretrained("facebook/sam3", token=os.environ["HF_TOKEN"])
    inputs = proc(images=image, text="dog", return_tensors="pt").to("cuda")

    # ===== HF: capture what feeds the first DETR encoder layer =====
    print("Running HF ...")
    hf = Sam3Model.from_pretrained("facebook/sam3", token=os.environ["HF_TOKEN"]).to("cuda").eval()

    hf_cap = {}
    # Hook the first encoder layer forward to grab its inputs AND outputs
    def hf_pre_hook(mod, args, kwargs):
        hf_cap["enc_L0_args"] = args
        hf_cap["enc_L0_kwargs"] = {k: (v.detach().float().cpu() if torch.is_tensor(v) else v)
                                    for k, v in kwargs.items()}
    def hf_fwd_hook(mod, inp, out):
        hf_cap["enc_L0_hidden"] = inp[0].detach().float().cpu() if len(inp) else None
        hf_cap["enc_L0_out"] = out.detach().float().cpu()

    hf.detr_encoder.layers[0].register_forward_pre_hook(hf_pre_hook, with_kwargs=True)
    hf.detr_encoder.layers[0].register_forward_hook(hf_fwd_hook)

    with torch.inference_mode():
        _ = hf(pixel_values=inputs["pixel_values"], input_ids=inputs["input_ids"],
               attention_mask=inputs["attention_mask"])
    del hf
    import gc; gc.collect(); torch.cuda.empty_cache()

    print("HF enc_L0 input hidden shape:", hf_cap["enc_L0_hidden"].shape if hf_cap.get("enc_L0_hidden") is not None else None)
    print("HF enc_L0 kwargs:", {k: (v.shape if torch.is_tensor(v) else v) for k, v in hf_cap["enc_L0_kwargs"].items()})
    # prompt_feats, prompt_cross_attn_mask, vision_pos_encoding, etc.
    for k, v in hf_cap["enc_L0_kwargs"].items():
        if torch.is_tensor(v):
            print(f"  {k}: {tuple(v.shape)} dtype={v.dtype}  mean={v.float().mean():.4f} std={v.float().std():.4f}")

    # ===== RF =====
    print("\nRunning RF ...")
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
    def rf_pre_hook(mod, args, kwargs):
        rf_cap["enc_L0_args"] = tuple(
            a.detach().float().cpu() if torch.is_tensor(a) else a for a in args
        )
        rf_cap["enc_L0_kwargs"] = {
            k: (v.detach().float().cpu() if torch.is_tensor(v) else v)
            for k, v in kwargs.items()
        }

    rf.model.transformer.encoder.layers[0].register_forward_pre_hook(rf_pre_hook, with_kwargs=True)

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
    orig_call = SimpleTokenizer.__call__
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
    SimpleTokenizer.__call__ = orig_call

    print(f"RF enc_L0 args: {len(rf_cap['enc_L0_args'])} positional")
    for i, a in enumerate(rf_cap["enc_L0_args"]):
        if torch.is_tensor(a):
            print(f"  arg[{i}]: {tuple(a.shape)} dtype={a.dtype}  mean={a.float().mean():.4f} std={a.float().std():.4f}")
        else:
            print(f"  arg[{i}]: {type(a).__name__}")
    for k, v in rf_cap["enc_L0_kwargs"].items():
        if torch.is_tensor(v):
            print(f"  kw {k}: {tuple(v.shape)} dtype={v.dtype}  mean={v.float().mean():.4f} std={v.float().std():.4f}")
        else:
            print(f"  kw {k}: {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
