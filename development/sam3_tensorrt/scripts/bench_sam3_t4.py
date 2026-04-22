#!/usr/bin/env python3
"""T4-friendly benchmark: runs baseline and TRT in SEPARATE subprocess-like
passes so only one SAM3 model is resident in GPU memory at a time (T4 has
15GB; two SAM3 models + the engine don't fit)."""

from __future__ import annotations

import base64, os, sys, time, gc
from pathlib import Path
from statistics import mean, stdev

if "ROBOFLOW_API_KEY" not in os.environ:
    raise SystemExit("Set ROBOFLOW_API_KEY env var before running this script.")


import numpy as np
import torch

import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parent))
from sam3_trt_adapter import patch_sam3_with_trt_backbone

ENGINE_PATH = Path(os.environ.get(
    "SAM3_ENGINE_PATH",
    "./sam3_onnx_exports/sam3_vision_backbone_fp16_rope_fp32_d10.engine",
))
ASSET_DIR = Path(os.environ.get("SAM3_ASSETS", "tests/workflows/integration_tests/execution/assets"))
IMAGES = [
    ("dogs.jpg", "dog"),
    ("car.jpg", "car"),
    ("crowd.jpg", "person"),
    ("multi-fruit.jpg", "fruit"),
]
BENCH_IMAGE, BENCH_PROMPT = IMAGES[0]
WARMUP = 3
ITERS = 15


def _build_req(path, prompt):
    from inference.core.entities.requests.sam3 import Sam3SegmentationRequest, Sam3Prompt
    return Sam3SegmentationRequest(
        image={"type": "base64", "value": base64.b64encode(path.read_bytes()).decode()},
        prompts=[Sam3Prompt(text=prompt)],
        output_prob_thresh=0.5,
        format="rle",
    )


def _rle_to_mask(rle):
    from pycocotools import mask as mu
    if isinstance(rle.get("counts"), str):
        rle = {"size": rle["size"], "counts": rle["counts"].encode()}
    return mu.decode(rle)


def _mask_iou(a, b):
    a = (a > 0).astype(np.uint8); b = (b > 0).astype(np.uint8)
    inter = np.logical_and(a, b).sum(); union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def _bench(model, req):
    for _ in range(WARMUP):
        model.infer_from_request(req)
    torch.cuda.synchronize()
    lats = []
    for _ in range(ITERS):
        torch.cuda.synchronize(); t = time.perf_counter()
        model.infer_from_request(req)
        torch.cuda.synchronize()
        lats.append((time.perf_counter() - t) * 1000)
    return lats


def _collect_masks(model, req):
    resp = model.infer_from_request(req)
    preds = resp.prompt_results[0].predictions
    masks = [_rle_to_mask(p.masks) for p in preds]
    scores = [float(p.confidence) for p in preds]
    return masks, scores


def run_pt_pass():
    """Load PT, run correctness preds + bench, save results, free."""
    from inference.models.sam3.segment_anything3 import SegmentAnything3
    m = SegmentAnything3(model_id="sam3/sam3_final", api_key=os.environ["ROBOFLOW_API_KEY"])
    results = {}
    for fname, prompt in IMAGES:
        path = ASSET_DIR / fname
        if not path.exists():
            continue
        req = _build_req(path, prompt)
        masks, scores = _collect_masks(m, req)
        results[fname] = {"masks": masks, "scores": scores, "prompt": prompt}
    bench_req = _build_req(ASSET_DIR / BENCH_IMAGE, BENCH_PROMPT)
    lats = _bench(m, bench_req)
    del m
    gc.collect(); torch.cuda.empty_cache()
    return results, lats


def run_trt_pass():
    from inference.models.sam3.segment_anything3 import SegmentAnything3
    m = SegmentAnything3(model_id="sam3/sam3_final", api_key=os.environ["ROBOFLOW_API_KEY"])
    patch_sam3_with_trt_backbone(m.model, ENGINE_PATH)
    results = {}
    for fname, prompt in IMAGES:
        path = ASSET_DIR / fname
        if not path.exists():
            continue
        req = _build_req(path, prompt)
        masks, scores = _collect_masks(m, req)
        results[fname] = {"masks": masks, "scores": scores, "prompt": prompt}
    bench_req = _build_req(ASSET_DIR / BENCH_IMAGE, BENCH_PROMPT)
    lats = _bench(m, bench_req)
    del m
    gc.collect(); torch.cuda.empty_cache()
    return results, lats


def main() -> int:
    print(f"Engine: {ENGINE_PATH.name} ({ENGINE_PATH.stat().st_size / 1e6:.0f} MB)")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Free GPU mem at start: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")

    print("\n=== Pass 1: PyTorch baseline ===")
    pt_results, pt_lats = run_pt_pass()
    pt_m = mean(pt_lats); pt_s = stdev(pt_lats)
    print(f"PT E2E: {pt_m:.1f}ms ± {pt_s:.1f} (min={min(pt_lats):.1f} max={max(pt_lats):.1f})")

    print("\n=== Pass 2: TRT-patched ===")
    trt_results, trt_lats = run_trt_pass()
    trt_m = mean(trt_lats); trt_s = stdev(trt_lats)
    print(f"TRT E2E: {trt_m:.1f}ms ± {trt_s:.1f} (min={min(trt_lats):.1f} max={max(trt_lats):.1f})")

    print(f"\nSpeedup: {pt_m/trt_m:.2f}x  ({(trt_m-pt_m)/pt_m*100:+.1f}%)")

    print("\n=== Correctness (mask IoU) ===")
    all_ious = []
    for fname, _ in IMAGES:
        if fname not in pt_results or fname not in trt_results:
            continue
        ref_masks = pt_results[fname]["masks"]
        tst_masks = trt_results[fname]["masks"]
        ious = []
        if ref_masks and tst_masks:
            pairs = [(i, j, _mask_iou(ref_masks[i], tst_masks[j]))
                     for i in range(len(ref_masks)) for j in range(len(tst_masks))
                     if ref_masks[i].shape == tst_masks[j].shape]
            pairs.sort(key=lambda x: -x[2])
            used_i, used_j = set(), set()
            for i, j, iou in pairs:
                if i not in used_i and j not in used_j:
                    used_i.add(i); used_j.add(j); ious.append(iou)
        miou = float(np.mean(ious)) if ious else float("nan")
        mn = float(min(ious)) if ious else float("nan")
        all_ious.extend(ious)
        prompt = pt_results[fname]["prompt"]
        print(f"  {fname:20s} prompt={prompt!r:12s} "
              f"PT N={len(ref_masks)}  TRT N={len(tst_masks)}  "
              f"mean IoU={miou:.4f}  min IoU={mn:.4f}")

    overall = float(np.mean(all_ious)) if all_ious else float("nan")
    overall_min = float(np.min(all_ious)) if all_ious else float("nan")
    print(f"\nOverall mean IoU: {overall:.4f}   min IoU: {overall_min:.4f}")
    print(f"Gate (>=0.95): {'PASS' if overall >= 0.95 else 'FAIL'}")
    return 0 if overall >= 0.95 else 1


if __name__ == "__main__":
    sys.exit(main())
