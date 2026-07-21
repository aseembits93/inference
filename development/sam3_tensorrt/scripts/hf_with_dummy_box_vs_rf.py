#!/usr/bin/env python3
"""Test hypothesis: HF with a dummy box (to trigger geometry_encoder +
cls_embed prepending) should match RF's output more closely than HF
without box.
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
    a = a.double().flatten(); b = b.double().flatten()
    return float(a @ b / (a.norm() * b.norm() + 1e-20))


def main() -> int:
    IMG = Path(os.environ.get("SAM3_ASSETS", "tests/workflows/integration_tests/execution/assets") + "/dogs.jpg")
    image = Image.open(IMG).convert("RGB")
    from transformers import Sam3Processor, Sam3Model
    proc = Sam3Processor.from_pretrained("facebook/sam3", token=os.environ["HF_TOKEN"])
    inputs = proc(images=image, text="dog", return_tensors="pt").to("cuda")

    # HF with NO box
    print("=== HF (no box, text-only) ===")
    hf = Sam3Model.from_pretrained("facebook/sam3", token=os.environ["HF_TOKEN"]).to("cuda").eval()
    with torch.inference_mode():
        out_no_box = hf(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

    # HF with DUMMY box (label=-10 → padding)
    print("=== HF (with dummy box, label=-10 padding) ===")
    dummy_box = torch.zeros(1, 1, 4, device="cuda")
    dummy_lab = torch.tensor([[-10]], dtype=torch.long, device="cuda")
    with torch.inference_mode():
        out_dummy_box = hf(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            input_boxes=dummy_box,
            input_boxes_labels=dummy_lab,
        )

    del hf
    import gc; gc.collect(); torch.cuda.empty_cache()

    # RF (native — always invokes geometry_encoder with cls_embed)
    print("=== RF (native path, text-only) ===")
    from inference.models.sam3.segment_anything3 import SegmentAnything3
    from sam3.train.data.collator import collate_fn_api
    from sam3.train.data.sam3_image_dataset import Datapoint as Sam3Datapoint
    from sam3.train.data.sam3_image_dataset import Image as Sam3ImageDP
    from sam3.train.transforms.basic_for_api import ComposeAPI, NormalizeAPI, ToTensorAPI, RandomResizeAPI
    from inference.models.sam3.segment_anything3 import _build_text_query
    from inference.core.env import SAM3_IMAGE_SIZE
    from sam3.model.utils.misc import copy_data_to_device

    rf = SegmentAnything3(model_id="sam3/sam3_final", api_key=os.environ["ROBOFLOW_API_KEY"])
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
        rf_raw = rf.model(batch)
    SimpleTokenizer.__call__ = orig

    rf_logits = rf_raw[0]["pred_logits"].float().cpu()
    rf_boxes = rf_raw[0]["pred_boxes"].float().cpu() if "pred_boxes" in rf_raw[0] else None
    rf_masks = rf_raw[0]["pred_masks"].float().cpu()

    # Normalize shapes
    hf_nb_logits = out_no_box.pred_logits.float().cpu()  # (1, 200)
    hf_nb_boxes = out_no_box.pred_boxes.float().cpu()     # (1, 200, 4)
    hf_nb_masks = out_no_box.pred_masks.float().cpu()     # (1, 200, 288, 288)
    hf_db_logits = out_dummy_box.pred_logits.float().cpu()
    hf_db_boxes = out_dummy_box.pred_boxes.float().cpu()
    hf_db_masks = out_dummy_box.pred_masks.float().cpu()
    # RF logits may be (1, 200, 1)
    if rf_logits.ndim == 3 and rf_logits.shape[-1] == 1:
        rf_logits = rf_logits.squeeze(-1)

    print("\n=== Three-way comparison ===")
    print(f"Shapes: HF_nb={tuple(hf_nb_logits.shape)} HF_db={tuple(hf_db_logits.shape)} RF={tuple(rf_logits.shape)}")
    print(f"\npred_logits:")
    print(f"  HF_no_box vs RF:            cos = {_cos(hf_nb_logits, rf_logits):.6f}")
    print(f"  HF_dummy_box vs RF:         cos = {_cos(hf_db_logits, rf_logits):.6f}")
    print(f"  HF_no_box vs HF_dummy_box:  cos = {_cos(hf_nb_logits, hf_db_logits):.6f}")

    print(f"\npred_boxes:")
    print(f"  HF_no_box vs RF:            cos = {_cos(hf_nb_boxes, rf_boxes):.6f}")
    print(f"  HF_dummy_box vs RF:         cos = {_cos(hf_db_boxes, rf_boxes):.6f}")
    print(f"  HF_no_box vs HF_dummy_box:  cos = {_cos(hf_nb_boxes, hf_db_boxes):.6f}")

    print(f"\npred_masks:")
    print(f"  HF_no_box vs RF:            cos = {_cos(hf_nb_masks, rf_masks):.6f}")
    print(f"  HF_dummy_box vs RF:         cos = {_cos(hf_db_masks, rf_masks):.6f}")
    print(f"  HF_no_box vs HF_dummy_box:  cos = {_cos(hf_nb_masks, hf_db_masks):.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
