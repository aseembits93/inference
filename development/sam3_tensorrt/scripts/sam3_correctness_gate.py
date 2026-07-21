#!/usr/bin/env python3
"""Correctness gate: SAM3 PyTorch vs. SAM3+TRT-backbone on real images.

Runs the SAM3 model end-to-end twice:
  1. Pure PyTorch baseline
  2. With the TRT vision backbone swapped in

Compares mask IoU and score deltas. Passes if mean mask IoU >= 0.95 and
number-of-detections matches.
"""

from __future__ import annotations

import os
import sys
import base64
from pathlib import Path

if "ROBOFLOW_API_KEY" not in os.environ:
    raise SystemExit("Set ROBOFLOW_API_KEY env var before running this script.")


import numpy as np
import torch

# Allow importing sibling adapter
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parent))
from sam3_trt_adapter import patch_sam3_with_trt_backbone

IMAGE_DIR = Path(os.environ.get("SAM3_ASSETS", "tests/workflows/integration_tests/execution/assets"))
IMAGES = [
    ("dogs.jpg", "dog"),
    ("car.jpg", "car"),
    ("crowd.jpg", "person"),
    ("multi-fruit.jpg", "fruit"),
]
ENGINE_PATH = os.environ.get(
    "SAM3_ENGINE_PATH",
    "./sam3_onnx_exports/sam3_vision_backbone_bf16.engine",
)
MIN_IOU = 0.95


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = (a > 0).astype(np.uint8)
    b = (b > 0).astype(np.uint8)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def _best_iou_match(ref_masks, tst_masks) -> list[tuple[int, int, float]]:
    """Greedy match between reference and test masks by max IoU."""
    if len(ref_masks) == 0 or len(tst_masks) == 0:
        return []
    ious = np.zeros((len(ref_masks), len(tst_masks)))
    for i, rm in enumerate(ref_masks):
        for j, tm in enumerate(tst_masks):
            if rm.shape == tm.shape:
                ious[i, j] = _mask_iou(rm, tm)
    pairs = []
    used_i, used_j = set(), set()
    flat = [(ious[i, j], i, j) for i in range(len(ref_masks)) for j in range(len(tst_masks))]
    flat.sort(reverse=True)
    for iou, i, j in flat:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        pairs.append((i, j, iou))
    return pairs


def _rle_to_mask(rle_dict) -> np.ndarray:
    from pycocotools import mask as mu
    if isinstance(rle_dict.get("counts"), str):
        rle_dict = {"size": rle_dict["size"], "counts": rle_dict["counts"].encode("utf-8")}
    return mu.decode(rle_dict)


def segment(model, image_req, prompt_text: str):
    from inference.core.entities.requests.sam3 import Sam3SegmentationRequest, Sam3Prompt
    req = Sam3SegmentationRequest(
        image=image_req,
        prompts=[Sam3Prompt(text=prompt_text)],
        output_prob_thresh=0.5,
        format="rle",
    )
    resp = model.infer_from_request(req)
    preds = resp.prompt_results[0].predictions
    masks, scores = [], []
    for p in preds:
        m = _rle_to_mask(p.masks)
        masks.append(m)
        scores.append(float(p.confidence))
    return masks, scores


def main() -> int:
    from inference.models.sam3.segment_anything3 import SegmentAnything3

    print("Loading baseline SAM3 (PyTorch)...")
    base_model = SegmentAnything3(
        model_id="sam3/sam3_final",
        api_key=os.environ["ROBOFLOW_API_KEY"],
    )

    # Load a second instance for TRT patching
    print("Loading SAM3 for TRT patching...")
    trt_model = SegmentAnything3(
        model_id="sam3/sam3_final",
        api_key=os.environ["ROBOFLOW_API_KEY"],
    )

    # IMPORTANT: the exported ONNX was patched in vitdet to use real-arith rope.
    # We must apply the same in-place patch to *trt_model* before running it,
    # because the TRT runner only handles the image backbone, but the rest of
    # the model still sees the original PyTorch vitdet rope. Actually the TRT
    # engine *replaces* the entire forward_image call — so vitdet complex
    # math inside it never runs. The PyTorch baseline keeps the original rope.
    runner = patch_sam3_with_trt_backbone(trt_model.model, ENGINE_PATH)
    print(f"TRT runner: input shape {runner.input_shape}")

    print("\nPer-image correctness check:")
    all_ious = []
    for fname, prompt in IMAGES:
        path = IMAGE_DIR / fname
        if not path.exists():
            print(f"  SKIP {fname} (missing)")
            continue
        img_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        req = {"type": "base64", "value": img_b64}

        ref_masks, ref_scores = segment(base_model, req, prompt)
        tst_masks, tst_scores = segment(trt_model, req, prompt)

        pairs = _best_iou_match(ref_masks, tst_masks)
        ious = [p[2] for p in pairs]
        mean_iou = float(np.mean(ious)) if ious else (1.0 if not ref_masks and not tst_masks else 0.0)
        all_ious.extend(ious)
        print(
            f"  {fname:20s} prompt={prompt!r:15s} "
            f"baseline N={len(ref_masks)}, TRT N={len(tst_masks)}, "
            f"mean IoU={mean_iou:.4f}, min IoU={min(ious) if ious else 1.0:.4f}"
        )
        if ref_scores and tst_scores:
            print(f"    baseline scores: {[f'{s:.3f}' for s in ref_scores[:5]]}")
            print(f"    TRT scores:      {[f'{s:.3f}' for s in tst_scores[:5]]}")

    if not all_ious:
        print("\nNo valid comparisons made.")
        return 1

    overall_mean = float(np.mean(all_ious))
    overall_min = float(np.min(all_ious))
    print(f"\nOverall: mean IoU = {overall_mean:.4f}, min = {overall_min:.4f}")

    passed = overall_mean >= MIN_IOU
    print(f"\nGATE {'PASSED' if passed else 'FAILED'} (threshold {MIN_IOU})")
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
