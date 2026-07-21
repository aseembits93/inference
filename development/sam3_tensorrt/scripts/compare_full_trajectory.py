#!/usr/bin/env python3
"""Compare HF and RF backbone outputs at EVERY block.

After the previous test showed that blocks 0 and 1 produce cos≈1.0 but
something else diverges, let's trace the full depth. Where does the
divergence emerge?
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


def _cos_mean_max(a, b):
    """Flatten and compare two tensors of the same shape."""
    if a.shape != b.shape:
        return float("nan"), float("nan"), float("nan")
    af = a.flatten().double()
    bf = b.flatten().double()
    cos = float(af @ bf / (af.norm() * bf.norm() + 1e-20))
    d = (a - b).abs()
    return cos, float(d.mean()), float(d.max())


def main() -> int:
    IMG = Path(os.environ.get("SAM3_ASSETS", "tests/workflows/integration_tests/execution/assets") + "/dogs.jpg")
    image = Image.open(IMG).convert("RGB")
    from transformers import Sam3Processor, Sam3Model
    proc = Sam3Processor.from_pretrained("facebook/sam3", token=os.environ["HF_TOKEN"])
    inputs = proc(images=image, text="dog", return_tensors="pt").to("cuda")

    N_BLOCKS = 32

    # ===== HF =====
    print("Running HF ...")
    hf = Sam3Model.from_pretrained("facebook/sam3", token=os.environ["HF_TOKEN"]).to("cuda").eval()
    hf_blk = {}
    def make_hook(name, store):
        def hook(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            store[name] = t.detach().float().cpu()
        return hook

    for i in range(N_BLOCKS):
        hf.vision_encoder.backbone.layers[i].register_forward_hook(make_hook(f"block_{i}", hf_blk))

    with torch.inference_mode():
        _ = hf(pixel_values=inputs["pixel_values"],
               input_ids=inputs["input_ids"],
               attention_mask=inputs["attention_mask"])
    del hf
    import gc; gc.collect(); torch.cuda.empty_cache()

    # ===== RF =====
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
    trunk = rf.model.backbone.vision_backbone.trunk
    rf_blk = {}
    for i in range(N_BLOCKS):
        trunk.blocks[i].register_forward_hook(make_hook(f"block_{i}", rf_blk))

    transform = ComposeAPI(transforms=[
        RandomResizeAPI(sizes=SAM3_IMAGE_SIZE, max_size=SAM3_IMAGE_SIZE,
                        square=True, consistent_transform=False),
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
        # NO autocast — pure FP32 for both
        _ = rf.model(batch)
    SimpleTokenizer.__call__ = orig_call

    # ===== Compare each block =====
    print("\n=== Per-block comparison ===")
    print(f"{'block':<6s} {'shape':<24s} {'cos':>10s} {'mean|Δ|':>12s} {'max|Δ|':>10s}")
    for i in range(N_BLOCKS):
        k = f"block_{i}"
        if k not in hf_blk or k not in rf_blk:
            print(f"  block {i}: MISSING"); continue
        a, b = hf_blk[k], rf_blk[k]
        cos, md, mx = _cos_mean_max(a, b)
        print(f"block {i:<2d} {str(tuple(a.shape)):<24s} {cos:>10.6f} {md:>12.6e} {mx:>10.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
