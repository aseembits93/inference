#!/usr/bin/env python3
"""Direct one-image sanity check: HF canonical inference snippet vs
Roboflow SegmentAnything3, on the exact image + prompt from the HF
model card (COCO val2017/000000077595.jpg, prompt="ear").

Outputs: per-config detection count, scores, mask IoU against HF-PT,
and raw bounding-box comparison. No autocast tricks, no TRT — just
the two PyTorch code paths as published.
"""

from __future__ import annotations

import io
import os
import sys
import urllib.request
from pathlib import Path

# requires HF_TOKEN env var
# requires ROBOFLOW_API_KEY env var

import numpy as np
import torch
from PIL import Image

# Canonical HF example inputs
IMG_URL = "http://images.cocodataset.org/val2017/000000077595.jpg"
PROMPT = "ear"


def fetch_image(url=IMG_URL):
    cache = Path("/tmp/coco_077595.jpg")
    if not cache.exists():
        print(f"Downloading {url} ...")
        cache.write_bytes(urllib.request.urlopen(url, timeout=30).read())
    return Image.open(cache).convert("RGB")


def run_hf(image):
    """Exact snippet from facebook/sam3 model card."""
    from transformers import Sam3Processor, Sam3Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Sam3Model.from_pretrained(
        "facebook/sam3", token=os.environ["HF_TOKEN"],
    ).to(device)
    processor = Sam3Processor.from_pretrained(
        "facebook/sam3", token=os.environ["HF_TOKEN"],
    )

    inputs = processor(images=image, text=PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]
    return results, outputs


def run_rf(image):
    """Roboflow SegmentAnything3 equivalent path."""
    import base64
    from io import BytesIO
    from inference.models.sam3.segment_anything3 import SegmentAnything3
    from inference.core.entities.requests.sam3 import (
        Sam3SegmentationRequest, Sam3Prompt,
    )

    m = SegmentAnything3(
        model_id="sam3/sam3_final", api_key=os.environ["ROBOFLOW_API_KEY"],
    )
    # Encode PIL image to base64 so we can use the same ingestion path as production
    buf = BytesIO()
    image.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    req = Sam3SegmentationRequest(
        image={"type": "base64", "value": img_b64},
        prompts=[Sam3Prompt(text=PROMPT)],
        output_prob_thresh=0.5,
        format="rle",
    )
    resp = m.infer_from_request(req)
    preds = resp.prompt_results[0].predictions
    return preds


def _rle_to_mask(rle):
    from pycocotools import mask as mu
    if isinstance(rle.get("counts"), str):
        rle = {"size": rle["size"], "counts": rle["counts"].encode()}
    return mu.decode(rle)


def _mask_iou(a, b):
    if a.shape != b.shape:
        return 0.0
    a = (a > 0).astype(np.uint8)
    b = (b > 0).astype(np.uint8)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def _mask_from_pred(p):
    """Convert Roboflow Sam3SegmentationPrediction -> np.ndarray."""
    if p.format == "rle":
        return _rle_to_mask(p.masks)
    raise ValueError(f"unsupported format {p.format}")


def main() -> int:
    print(f"Image: {IMG_URL}")
    print(f"Prompt: {PROMPT!r}\n")
    image = fetch_image()
    print(f"PIL size: {image.size}")

    print("\n=== Running HF canonical snippet ===")
    hf_results, hf_raw = run_hf(image)
    hf_scores = hf_results["scores"].cpu().numpy() if torch.is_tensor(hf_results["scores"]) else np.asarray(hf_results["scores"])
    hf_boxes = hf_results["boxes"].cpu().numpy() if torch.is_tensor(hf_results["boxes"]) else np.asarray(hf_results["boxes"])
    hf_masks_t = hf_results["masks"]
    if torch.is_tensor(hf_masks_t):
        hf_masks = hf_masks_t.cpu().numpy().astype(np.uint8)
        if hf_masks.ndim == 4 and hf_masks.shape[1] == 1:
            hf_masks = hf_masks[:, 0]
    else:
        hf_masks = np.asarray(hf_masks_t, dtype=np.uint8)
    print(f"  HF found {len(hf_scores)} objects")
    print(f"  HF scores: {[round(float(s), 4) for s in hf_scores]}")
    print(f"  HF boxes (xyxy abs): {hf_boxes.tolist()}")
    print(f"  HF raw pred_logits: shape={tuple(hf_raw.pred_logits.shape)} "
          f"std={hf_raw.pred_logits.float().std().item():.4f}")

    # Free HF GPU memory
    import gc
    del hf_raw
    gc.collect(); torch.cuda.empty_cache()

    print("\n=== Running Roboflow SegmentAnything3 ===")
    rf_preds = run_rf(image)
    print(f"  RF found {len(rf_preds)} objects")
    print(f"  RF scores: {[round(float(p.confidence), 4) for p in rf_preds]}")
    rf_masks = [_mask_from_pred(p) for p in rf_preds]

    # Align masks by greedy IoU matching
    print("\n=== Mask alignment (greedy IoU) ===")
    if len(hf_masks) == 0 and len(rf_masks) == 0:
        print("  Both found 0 objects (trivially matching)")
    elif len(hf_masks) == 0 or len(rf_masks) == 0:
        print(f"  MISMATCH: HF={len(hf_masks)}, RF={len(rf_masks)}")
    else:
        pairs = []
        for i in range(len(hf_masks)):
            for j in range(len(rf_masks)):
                iou = _mask_iou(hf_masks[i], rf_masks[j])
                pairs.append((i, j, iou))
        pairs.sort(key=lambda p: -p[2])
        used_i, used_j = set(), set()
        print(f"  {'HF #':>4} {'RF #':>4} {'IoU':>8} {'HF score':>10} {'RF score':>10}")
        for i, j, iou in pairs:
            if i in used_i or j in used_j:
                continue
            used_i.add(i); used_j.add(j)
            print(f"  {i:>4d} {j:>4d} {iou:>8.4f} "
                  f"{float(hf_scores[i]):>10.4f} {float(rf_preds[j].confidence):>10.4f}")
        print(f"\n  Unmatched HF: {sorted(set(range(len(hf_masks))) - used_i)}")
        print(f"  Unmatched RF: {sorted(set(range(len(rf_masks))) - used_j)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
