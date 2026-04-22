#!/usr/bin/env python3
"""Head-to-head: HF Sam3Model PyTorch vs Roboflow SegmentAnything3
PyTorch, on the 100-image COCO subset. Neither uses TRT.

Prints per-image detection counts + scores for the cases where HF and
RF disagree, plus aggregate correctness + score/IoU distributions.
"""

from __future__ import annotations

import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np

OUT_DIR = Path(os.environ.get("SAM3_BENCH_DIR", "/tmp"))


def _mask_iou(a, b):
    if a.shape != b.shape:
        return 0.0
    a = (a > 0).astype(np.uint8)
    b = (b > 0).astype(np.uint8)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def _greedy(a_masks, b_masks, min_iou=0.1):
    if not a_masks or not b_masks:
        return []
    pairs = [
        (i, j, _mask_iou(a_masks[i], b_masks[j]))
        for i in range(len(a_masks)) for j in range(len(b_masks))
    ]
    pairs = [p for p in pairs if p[2] >= min_iou]
    pairs.sort(key=lambda p: -p[2])
    used_i, used_j, out = set(), set(), []
    for i, j, iou in pairs:
        if i in used_i or j in used_j:
            continue
        used_i.add(i); used_j.add(j); out.append((i, j, iou))
    return out


def main() -> int:
    hf = pickle.load(open(OUT_DIR / "coco_sweep_hf_pt.pkl", "rb"))
    rf = pickle.load(open(OUT_DIR / "coco_sweep_pt_bf16.pkl", "rb"))
    print(f"HF records: {len(hf)}")
    print(f"RF records: {len(rf)}")

    # Aggregate counters
    agree_exact_count = 0
    per_class_hf_minus_rf = defaultdict(list)  # n_hf - n_rf
    all_match_ious = []
    all_score_deltas = []  # hf_score - rf_matched_score
    disagreements = []  # collect cases where n_hf != n_rf

    for iid, hf_rec in hf.items():
        rf_rec = rf.get(iid)
        if rf_rec is None or "error" in hf_rec or "error" in rf_rec:
            continue

        hf_masks = hf_rec["masks"]; hf_scores = hf_rec["scores"]
        rf_masks = rf_rec["masks"]; rf_scores = rf_rec["scores"]
        n_hf, n_rf = len(hf_masks), len(rf_masks)
        prompt = hf_rec.get("prompt")

        per_class_hf_minus_rf[prompt].append(n_hf - n_rf)
        if n_hf == n_rf:
            agree_exact_count += 1
        else:
            disagreements.append({
                "image_id": iid,
                "prompt": prompt,
                "file": hf_rec.get("file_name"),
                "n_hf": n_hf, "n_rf": n_rf,
                "hf_scores": [round(float(s), 3) for s in hf_scores],
                "rf_scores": [round(float(s), 3) for s in rf_scores],
            })

        matches = _greedy(hf_masks, rf_masks)
        for i, j, iou in matches:
            if iou >= 0.5:
                all_match_ious.append(iou)
                all_score_deltas.append(float(hf_scores[i]) - float(rf_scores[j]))

    print(f"\n=== Detection count agreement ===")
    n = sum(1 for iid in hf if iid in rf and "error" not in hf[iid] and "error" not in rf[iid])
    print(f"Images where n_hf == n_rf: {agree_exact_count}/{n}")

    # Total detections from each
    total_hf = sum(len(hf[iid].get("masks", [])) for iid in hf if "error" not in hf[iid])
    total_rf = sum(len(rf[iid].get("masks", [])) for iid in rf if "error" not in rf[iid])
    print(f"Total HF detections: {total_hf}")
    print(f"Total RF detections: {total_rf}")
    print(f"Matched at IoU>=0.5: {len(all_match_ious)}")

    print(f"\n=== Matched-pair IoU ===")
    if all_match_ious:
        arr = sorted(all_match_ious)
        print(f"  n={len(arr)}  mean={mean(arr):.4f}  median={median(arr):.4f}")
        print(f"  min={arr[0]:.4f}  p05={arr[len(arr)//20]:.4f}  p95={arr[int(0.95*len(arr))]:.4f}  max={arr[-1]:.4f}")

    print(f"\n=== Score delta (HF - RF) on matched pairs ===")
    if all_score_deltas:
        arr = sorted(all_score_deltas)
        print(f"  n={len(arr)}  mean={mean(arr):+.4f}  median={median(arr):+.4f}")
        print(f"  p05={arr[len(arr)//20]:+.4f}  p95={arr[int(0.95*len(arr))]:+.4f}  "
              f"min={arr[0]:+.4f}  max={arr[-1]:+.4f}")
        n_hf_higher = sum(1 for d in arr if d > 0.005)
        n_rf_higher = sum(1 for d in arr if d < -0.005)
        n_tie = len(arr) - n_hf_higher - n_rf_higher
        print(f"  HF > RF: {n_hf_higher}/{len(arr)}, RF > HF: {n_rf_higher}/{len(arr)}, ~tie: {n_tie}/{len(arr)}")

    print(f"\n=== Per-class count delta (n_hf - n_rf) ===")
    print(f"{'class':<20s} {'n_imgs':>7s} {'mean_delta':>11s} {'median_delta':>13s} {'n_hf_more':>10s} {'n_rf_more':>10s}")
    per_class_sorted = sorted(
        per_class_hf_minus_rf.items(),
        key=lambda kv: -abs(mean(kv[1])),
    )
    for cls, deltas in per_class_sorted[:15]:
        md = mean(deltas)
        mdm = median(deltas)
        n_hf_more = sum(1 for d in deltas if d > 0)
        n_rf_more = sum(1 for d in deltas if d < 0)
        print(f"{cls:<20s} {len(deltas):>7d} {md:>+11.2f} {mdm:>+13.1f} {n_hf_more:>10d} {n_rf_more:>10d}")

    print(f"\n=== Worst disagreements ({len(disagreements)} total; showing top 15 by |Δn|) ===")
    disagreements.sort(key=lambda d: -abs(d["n_hf"] - d["n_rf"]))
    for d in disagreements[:15]:
        print(f"  {d['file']:<25s} prompt={d['prompt']:<12s} "
              f"n_hf={d['n_hf']:>3d}  n_rf={d['n_rf']:>3d}  "
              f"Δ={d['n_hf']-d['n_rf']:+3d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
