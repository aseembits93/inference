#!/usr/bin/env python3
"""Hook both HF and RF models to capture the output of early layers
and compare. Track where they diverge in the vision backbone.
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

    # ===== HF =====
    print("Loading HF ...")
    hf = Sam3Model.from_pretrained("facebook/sam3", token=os.environ["HF_TOKEN"]).to("cuda").eval()
    hf_captures = {}

    def hf_embed_hook(mod, inp, out):
        hf_captures["embeddings"] = out.detach().float().cpu()
    def hf_block0_hook(mod, inp, out):
        hf_captures["block_0"] = (out[0] if isinstance(out, tuple) else out).detach().float().cpu()
    def hf_block1_hook(mod, inp, out):
        hf_captures["block_1"] = (out[0] if isinstance(out, tuple) else out).detach().float().cpu()
    def hf_layer_norm_hook(mod, inp, out):
        hf_captures["backbone_out"] = out.detach().float().cpu()

    hf.vision_encoder.backbone.embeddings.register_forward_hook(hf_embed_hook)
    hf.vision_encoder.backbone.layers[0].register_forward_hook(hf_block0_hook)
    hf.vision_encoder.backbone.layers[1].register_forward_hook(hf_block1_hook)
    hf.vision_encoder.backbone.layer_norm.register_forward_hook(hf_layer_norm_hook)

    with torch.inference_mode():
        _ = hf(pixel_values=inputs["pixel_values"], input_ids=inputs["input_ids"],
               attention_mask=inputs["attention_mask"])
    del hf
    import gc; gc.collect(); torch.cuda.empty_cache()

    # ===== RF =====
    print("Loading RF ...")
    from inference.models.sam3.segment_anything3 import SegmentAnything3
    from sam3.train.data.collator import collate_fn_api
    from sam3.train.data.sam3_image_dataset import Datapoint as Sam3Datapoint
    from sam3.train.data.sam3_image_dataset import Image as Sam3ImageDP
    from sam3.train.transforms.basic_for_api import ComposeAPI, NormalizeAPI, ToTensorAPI, RandomResizeAPI
    from inference.models.sam3.segment_anything3 import _build_text_query
    from inference.core.env import SAM3_IMAGE_SIZE
    from sam3.model.utils.misc import copy_data_to_device

    rf = SegmentAnything3(model_id="sam3/sam3_final", api_key=os.environ["ROBOFLOW_API_KEY"])
    trunk = rf.model.backbone.vision_backbone.trunk

    rf_captures = {}
    def rf_after_patch_and_pos_hook(mod, inp, out):
        """Captures the output of ln_pre, which is right after patch_embed + pos_embed."""
        rf_captures["ln_pre_out"] = out.detach().float().cpu()
    def rf_block0_hook(mod, inp, out):
        rf_captures["block_0"] = (out[0] if isinstance(out, tuple) else out).detach().float().cpu()
    def rf_block1_hook(mod, inp, out):
        rf_captures["block_1"] = (out[0] if isinstance(out, tuple) else out).detach().float().cpu()
    def rf_ln_post_hook(mod, inp, out):
        rf_captures["ln_post_out"] = out.detach().float().cpu()

    trunk.ln_pre.register_forward_hook(rf_after_patch_and_pos_hook)
    trunk.blocks[0].register_forward_hook(rf_block0_hook)
    trunk.blocks[1].register_forward_hook(rf_block1_hook)
    trunk.ln_post.register_forward_hook(rf_ln_post_hook)

    # Build RF batch with same pixel_values
    transform = ComposeAPI(transforms=[
        RandomResizeAPI(sizes=SAM3_IMAGE_SIZE, max_size=SAM3_IMAGE_SIZE,
                        square=True, consistent_transform=False),
        ToTensorAPI(),
        NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    h, w = image.size[1], image.size[0]
    dummy = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
    dp = Sam3Datapoint(find_queries=[], images=[Sam3ImageDP(data=dummy, objects=[], size=(h, w))])
    dp.find_queries.append(_build_text_query(coco_id=0, h=h, w=w, text="dog"))
    dp = transform(dp)
    dp.images[0].data = inputs["pixel_values"][0].cpu().clone()

    # Monkey-patch tokenizer
    from sam3.model.tokenizer_ve import SimpleTokenizer
    orig = SimpleTokenizer.__call__
    def patched(self, texts, context_length=77, **kwargs):
        device = next(rf.model.parameters()).device
        ids = inputs["input_ids"][0]
        mask = inputs["attention_mask"][0]
        real = ids[mask.bool()]
        out = torch.zeros((1, context_length), dtype=torch.long, device=device)
        n = min(real.numel(), context_length)
        out[0, :n] = real[:n].to(device)
        return out
    SimpleTokenizer.__call__ = patched

    batch = collate_fn_api(batch=[dp], dict_key="x")["x"]
    batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _ = rf.model(batch)
    SimpleTokenizer.__call__ = orig

    # ===== Compare =====
    print("\n=== Captured tensor shapes ===")
    print("HF captures:")
    for k, v in hf_captures.items():
        print(f"  {k}: {tuple(v.shape)}")
    print("RF captures:")
    for k, v in rf_captures.items():
        print(f"  {k}: {tuple(v.shape)}")

    # HF embeddings is (1, 5184, 1024) presumably (72*72)
    # RF ln_pre_out may be (1, 72, 72, 1024) or (1, 5184, 1024)
    print("\n=== Comparison: post-embedding+pos (first norm) ===")
    hf_emb = hf_captures["embeddings"]
    rf_pre = rf_captures["ln_pre_out"]
    # Flatten both to (1, L, C)
    def _flatten_tokens(t):
        if t.ndim == 4:  # (B, H, W, C)
            return t.reshape(t.shape[0], -1, t.shape[-1])
        return t
    hf_f = _flatten_tokens(hf_emb)
    rf_f = _flatten_tokens(rf_pre)
    print(f"  HF: {tuple(hf_f.shape)}, RF: {tuple(rf_f.shape)}")
    if hf_f.shape == rf_f.shape:
        d = (hf_f - rf_f).abs()
        cos = float((hf_f.flatten() @ rf_f.flatten())
                    / (hf_f.flatten().norm() * rf_f.flatten().norm() + 1e-12))
        print(f"  max|Δ| = {d.max():.4g}, mean|Δ| = {d.mean():.4g}, cos = {cos:.6f}")
    else:
        print(f"  shapes differ — HF includes ln_pre? RF structure differs?")

    print("\n=== Comparison: block 0 output ===")
    hf_b0 = _flatten_tokens(hf_captures["block_0"])
    rf_b0 = _flatten_tokens(rf_captures["block_0"])
    print(f"  HF: {tuple(hf_b0.shape)}, RF: {tuple(rf_b0.shape)}")
    if hf_b0.shape == rf_b0.shape:
        d = (hf_b0 - rf_b0).abs()
        cos = float((hf_b0.flatten() @ rf_b0.flatten())
                    / (hf_b0.flatten().norm() * rf_b0.flatten().norm() + 1e-12))
        print(f"  max|Δ| = {d.max():.4g}, mean|Δ| = {d.mean():.4g}, cos = {cos:.6f}")

    print("\n=== Comparison: block 1 output ===")
    hf_b1 = _flatten_tokens(hf_captures["block_1"])
    rf_b1 = _flatten_tokens(rf_captures["block_1"])
    print(f"  HF: {tuple(hf_b1.shape)}, RF: {tuple(rf_b1.shape)}")
    if hf_b1.shape == rf_b1.shape:
        d = (hf_b1 - rf_b1).abs()
        cos = float((hf_b1.flatten() @ rf_b1.flatten())
                    / (hf_b1.flatten().norm() * rf_b1.flatten().norm() + 1e-12))
        print(f"  max|Δ| = {d.max():.4g}, mean|Δ| = {d.mean():.4g}, cos = {cos:.6f}")

    print("\n=== Comparison: final backbone output ===")
    hf_bo = _flatten_tokens(hf_captures["backbone_out"])
    rf_bo = _flatten_tokens(rf_captures["ln_post_out"])
    print(f"  HF: {tuple(hf_bo.shape)}, RF: {tuple(rf_bo.shape)}")
    if hf_bo.shape == rf_bo.shape:
        d = (hf_bo - rf_bo).abs()
        cos = float((hf_bo.flatten() @ rf_bo.flatten())
                    / (hf_bo.flatten().norm() * rf_bo.flatten().norm() + 1e-12))
        print(f"  max|Δ| = {d.max():.4g}, mean|Δ| = {d.mean():.4g}, cos = {cos:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
