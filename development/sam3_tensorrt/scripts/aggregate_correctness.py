#!/usr/bin/env python3
"""Aggregate correctness metrics for the four-config 100-image sweep.

Reference: PT-bf16 (the repo default that would ship without any of
our changes). For each tested config {PT-fp16, TRT-swap, HF-TRT},
compute:

  - Detection count match: per-image |n_test - n_ref|
  - Recall: fraction of reference detections matched at IoU >= 0.5
  - Precision: fraction of test detections matched at IoU >= 0.5
  - Per-match IoU statistics (greedy matching on IoU, IoU >= 0.1 to pair)
  - Per-match score delta
  - Per-class breakdown

Also report how often the test config produces zero detections when
the reference produces >=1 (silent failures).
"""

from __future__ import annotations

import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import List, Tuple

import numpy as np

OUT_DIR = Path(os.environ.get("SAM3_BENCH_DIR", "/tmp"))
REF = OUT_DIR / "coco_sweep_pt_bf16.pkl"
CONFIGS = ["pt_fp16", "trt_swap", "hf_trt"]

MATCH_IOU = 0.5       # counts as a matched detection at this IoU
MIN_IOU_FOR_PAIR = 0.1  # only consider pairs with at least this IoU for greedy matching


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    a = (a > 0).astype(np.uint8)
    b = (b > 0).astype(np.uint8)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def _greedy_match(ref_masks, tst_masks, min_iou=MIN_IOU_FOR_PAIR) -> List[Tuple[int, int, float]]:
    """Return list of (ref_idx, tst_idx, iou), 1-to-1 greedy by IoU desc."""
    if not ref_masks or not tst_masks:
        return []
    pairs = []
    for i in range(len(ref_masks)):
        for j in range(len(tst_masks)):
            iou = _mask_iou(ref_masks[i], tst_masks[j])
            if iou >= min_iou:
                pairs.append((i, j, iou))
    pairs.sort(key=lambda p: -p[2])
    used_i, used_j = set(), set()
    out = []
    for i, j, iou in pairs:
        if i in used_i or j in used_j:
            continue
        used_i.add(i); used_j.add(j)
        out.append((i, j, iou))
    return out


def _load(name):
    path = OUT_DIR / f"coco_sweep_{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def _per_image_metrics(ref_rec, tst_rec):
    """Compute one image's metrics: counts, recall, precision, iou stats, score drift."""
    if "error" in tst_rec or "error" in ref_rec:
        return None
    ref_m, ref_s = ref_rec["masks"], ref_rec["scores"]
    tst_m, tst_s = tst_rec["masks"], tst_rec["scores"]
    n_ref, n_tst = len(ref_m), len(tst_m)
    matches = _greedy_match(ref_m, tst_m, MIN_IOU_FOR_PAIR)
    n_good = sum(1 for _, _, iou in matches if iou >= MATCH_IOU)

    return {
        "n_ref": n_ref,
        "n_tst": n_tst,
        "n_good": n_good,   # matches at IoU >= MATCH_IOU
        "n_any_match": len(matches),  # at MIN_IOU_FOR_PAIR
        "ious": [iou for _, _, iou in matches],
        "score_deltas": [tst_s[j] - ref_s[i] for i, j, iou in matches if iou >= MATCH_IOU],
        "matched_scores_ref": [ref_s[i] for i, j, iou in matches if iou >= MATCH_IOU],
        "matched_scores_tst": [tst_s[j] for i, j, iou in matches if iou >= MATCH_IOU],
        "silent_zero": n_ref > 0 and n_tst == 0,
        "extra_when_ref_empty": n_ref == 0 and n_tst > 0,
    }


def _aggregate(ref, tst, label):
    per_image = []
    per_class = defaultdict(list)
    for iid, r_rec in ref.items():
        t_rec = tst.get(iid)
        if t_rec is None:
            continue
        m = _per_image_metrics(r_rec, t_rec)
        if m is None:
            continue
        m["image_id"] = iid
        m["prompt"] = r_rec.get("prompt")
        per_image.append(m)
        per_class[r_rec.get("prompt")].append(m)

    n = len(per_image)
    if n == 0:
        print(f"{label}: no valid pairs"); return

    total_ref = sum(m["n_ref"] for m in per_image)
    total_tst = sum(m["n_tst"] for m in per_image)
    total_good = sum(m["n_good"] for m in per_image)

    # Recall = fraction of reference detections matched (at IoU>=0.5)
    recall = total_good / total_ref if total_ref > 0 else 0.0
    precision = total_good / total_tst if total_tst > 0 else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0

    # Silent failures: images where ref finds something, tst finds nothing
    silent = sum(1 for m in per_image if m["silent_zero"])
    # Ghost: images where tst finds something but ref doesn't
    ghost = sum(1 for m in per_image if m["extra_when_ref_empty"])

    # Absolute count match: images where n_tst == n_ref
    exact_count = sum(1 for m in per_image if m["n_tst"] == m["n_ref"])

    # IoU stats over matched detections at IoU>=MATCH_IOU
    match_ious = []
    score_deltas = []
    for m in per_image:
        for iou in m["ious"]:
            if iou >= MATCH_IOU:
                match_ious.append(iou)
        score_deltas.extend(m["score_deltas"])

    def _stat(arr, digits=4):
        if not arr:
            return {"n": 0}
        s = sorted(arr)
        return {
            "n": len(s),
            "mean": round(mean(s), digits),
            "median": round(median(s), digits),
            "min": round(min(s), digits),
            "p05": round(s[int(0.05 * (len(s) - 1))], digits),
            "p95": round(s[int(0.95 * (len(s) - 1))], digits),
            "max": round(max(s), digits),
        }

    print(f"\n=== {label} vs PT-bf16 (N={n} images) ===")
    print(f"  Totals: ref={total_ref} dets, test={total_tst} dets, matched>=0.5 IoU: {total_good}")
    print(f"  Recall      (matched/ref):  {recall*100:.1f}%")
    print(f"  Precision   (matched/test): {precision*100:.1f}%")
    print(f"  F1:                         {f1*100:.1f}%")
    print(f"  Exact-count images (n_tst == n_ref): {exact_count}/{n}")
    print(f"  Silent failures (ref>0 & test==0):   {silent}/{n}")
    print(f"  Ghost detections (ref==0 & test>0):  {ghost}/{n}")

    iou_s = _stat(match_ious)
    print(f"  Match IoU (matched dets only):")
    print(f"    n={iou_s['n']}  mean={iou_s.get('mean','-')}  median={iou_s.get('median','-')}  "
          f"min={iou_s.get('min','-')}  p05={iou_s.get('p05','-')}  p95={iou_s.get('p95','-')}")

    sd_s = _stat(score_deltas)
    print(f"  Score delta (tst - ref):")
    print(f"    n={sd_s['n']}  mean={sd_s.get('mean','-')}  median={sd_s.get('median','-')}  "
          f"p05={sd_s.get('p05','-')}  p95={sd_s.get('p95','-')}")

    # Per-class summary
    print(f"\n  Top 5 worst-recall classes (min 3 ref detections):")
    class_recalls = []
    for cls, ms in per_class.items():
        r_cls = sum(m["n_ref"] for m in ms)
        g_cls = sum(m["n_good"] for m in ms)
        if r_cls >= 3:
            class_recalls.append((cls, g_cls / r_cls, r_cls))
    class_recalls.sort(key=lambda x: x[1])
    for cls, rc, n_ref_cls in class_recalls[:5]:
        print(f"    {cls:25s} recall={rc*100:5.1f}%  (n_ref={n_ref_cls})")

    # Latency per image
    latencies = [t_rec["latency_ms"] for iid, t_rec in tst.items() if "latency_ms" in t_rec]
    if latencies:
        lat_s = _stat(latencies, digits=1)
        print(f"\n  E2E latency (ms): mean={lat_s['mean']} median={lat_s['median']} "
              f"p05={lat_s['p05']} p95={lat_s['p95']}")


def main() -> int:
    if not REF.exists():
        print(f"Missing reference {REF}. Run pt_bf16 sweep first."); return 1

    ref = _load("pt_bf16")
    print(f"Reference (PT-bf16): {len(ref)} images, "
          f"{sum(len(r.get('masks', [])) for r in ref.values())} detections")

    # Reference's own latency distribution for context
    from statistics import median as _median
    ref_lats = [r["latency_ms"] for r in ref.values() if "latency_ms" in r]
    if ref_lats:
        print(f"  PT-bf16 latency: median={_median(ref_lats):.1f} ms")

    for cfg in CONFIGS:
        try:
            tst = _load(cfg)
        except FileNotFoundError:
            print(f"skip {cfg}: sweep file missing"); continue
        _aggregate(ref, tst, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
