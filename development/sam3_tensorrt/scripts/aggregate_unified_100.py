#!/usr/bin/env python3
"""Aggregate the unified pre/post 100-image sweep.

Compares HF-vs-RF with:
  - Same preprocessing (HF Sam3Processor)
  - Same tokenizer (monkey-patched so RF uses HF's tokens)
  - Same post-processing (HF Sam3Processor.post_process_instance_segmentation)

Residual differences are pure model-graph differences.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from statistics import mean, median, stdev
from collections import defaultdict

import numpy as np

OUT_DIR = Path(os.environ.get("SAM3_BENCH_DIR", "/tmp"))


def _mask_iou(a, b):
    if a.shape != b.shape: return 0.0
    a = (a > 0).astype(np.uint8); b = (b > 0).astype(np.uint8)
    i = np.logical_and(a, b).sum(); u = np.logical_or(a, b).sum()
    return float(i) / u if u > 0 else 0.0


def _greedy_match(ref_masks, tst_masks, min_iou=0.1):
    if len(ref_masks) == 0 or len(tst_masks) == 0:
        return []
    pairs = [(i, j, _mask_iou(ref_masks[i], tst_masks[j]))
             for i in range(len(ref_masks)) for j in range(len(tst_masks))
             if ref_masks[i].shape == tst_masks[j].shape]
    pairs = [p for p in pairs if p[2] >= min_iou]
    pairs.sort(key=lambda p: -p[2])
    used_i, used_j, out = set(), set(), []
    for i, j, iou in pairs:
        if i in used_i or j in used_j: continue
        used_i.add(i); used_j.add(j); out.append((i, j, iou))
    return out


def main():
    hf = pickle.load(open(OUT_DIR / "coco_sweep_hf_pt_unified.pkl", "rb"))
    rf = pickle.load(open(OUT_DIR / "coco_sweep_rf_pt_unified.pkl", "rb"))

    n_hf_ok = sum(1 for v in hf.values() if "error" not in v)
    n_rf_ok = sum(1 for v in rf.values() if "error" not in v)
    print(f"HF ok: {n_hf_ok}/{len(hf)}, RF ok: {n_rf_ok}/{len(rf)}")

    # Raw-logit-level comparison
    print("\n=== Raw-output comparison (HF vs RF, same inputs) ===")
    cos_pred_logits, cos_pred_boxes = [], []
    pmn_ratios = []  # pred_masks norm ratio RF/HF
    pms_ratios = []  # pred_masks std ratio RF/HF

    for iid, hr in hf.items():
        rr = rf.get(iid)
        if rr is None or "error" in hr or "error" in rr:
            continue
        # pred_logits shape: HF (1, 200), RF (1, 200, 1)
        hl = hr["pred_logits"].flatten()
        rl = rr["pred_logits"].flatten()
        if hl.shape == rl.shape:
            cos = float(hl @ rl / (np.linalg.norm(hl) * np.linalg.norm(rl) + 1e-12))
            cos_pred_logits.append(cos)
        # pred_boxes
        hb = hr["pred_boxes"].flatten(); rb = rr["pred_boxes"].flatten() if rr["pred_boxes"] is not None else None
        if rb is not None and hb.shape == rb.shape:
            cos = float(hb @ rb / (np.linalg.norm(hb) * np.linalg.norm(rb) + 1e-12))
            cos_pred_boxes.append(cos)
        # pred_masks summary
        if hr["pred_masks_norm"] > 0:
            pmn_ratios.append(rr["pred_masks_norm"] / hr["pred_masks_norm"])
        if hr["pred_masks_std"] > 0:
            pms_ratios.append(rr["pred_masks_std"] / hr["pred_masks_std"])

    def s(arr, digits=4):
        if not arr: return "(empty)"
        a = sorted(arr)
        return (f"n={len(a)} mean={mean(a):.{digits}f} median={median(a):.{digits}f} "
                f"min={min(a):.{digits}f} p05={a[len(a)//20]:.{digits}f} p95={a[int(0.95*len(a))]:.{digits}f}")

    print(f"  pred_logits cos: {s(cos_pred_logits)}")
    print(f"  pred_boxes  cos: {s(cos_pred_boxes)}")
    print(f"  pred_masks norm ratio (RF/HF): {s(pmn_ratios)}")
    print(f"  pred_masks std  ratio (RF/HF): {s(pms_ratios)}")

    # Post-process agreement
    print("\n=== Post-processed detection agreement ===")
    n_pairs = 0
    n_exact_count = 0
    n_both_empty = 0
    total_hf_det = 0
    total_rf_det = 0
    n_silent_fail_rf = 0  # HF found, RF empty
    n_silent_fail_hf = 0
    all_match_ious = []
    all_score_deltas = []  # hf_score - rf_score
    per_class_delta = defaultdict(list)

    for iid, hr in hf.items():
        rr = rf.get(iid)
        if rr is None or "error" in hr or "error" in rr:
            continue
        n_pairs += 1
        n_h, n_r = hr["n_det"], rr["n_det"]
        total_hf_det += n_h; total_rf_det += n_r
        if n_h == n_r:
            n_exact_count += 1
        if n_h == 0 and n_r == 0:
            n_both_empty += 1
        if n_h > 0 and n_r == 0:
            n_silent_fail_rf += 1
        if n_h == 0 and n_r > 0:
            n_silent_fail_hf += 1
        per_class_delta[hr["prompt"]].append(n_h - n_r)
        # Match
        matches = _greedy_match(hr["masks"], rr["masks"])
        for i, j, iou in matches:
            if iou >= 0.5:
                all_match_ious.append(iou)
                all_score_deltas.append(float(hr["scores"][i]) - float(rr["scores"][j]))

    print(f"  N pairs: {n_pairs}")
    print(f"  Images where n_hf == n_rf: {n_exact_count}/{n_pairs}")
    print(f"  Both empty: {n_both_empty}")
    print(f"  Silent fail (HF found, RF empty): {n_silent_fail_rf}")
    print(f"  Silent fail (RF found, HF empty): {n_silent_fail_hf}")
    print(f"  Total HF detections: {total_hf_det}, total RF detections: {total_rf_det}")
    print(f"  Matched @ IoU >= 0.5: {len(all_match_ious)}")
    print(f"\n  Match IoU stats: {s(all_match_ious)}")
    print(f"  Score delta (HF - RF) stats: {s(all_score_deltas)}")

    # Per-class count delta
    print(f"\n=== Per-class count delta (|mean| > 0.1, n_imgs >= 2) ===")
    rows = []
    for cls, deltas in per_class_delta.items():
        if len(deltas) < 2 or abs(mean(deltas)) < 0.1:
            continue
        rows.append((cls, len(deltas), mean(deltas)))
    rows.sort(key=lambda r: -abs(r[2]))
    for cls, n, md in rows[:15]:
        print(f"  {cls:<20s} n={n:>2d}  mean Δ(hf - rf) = {md:+.2f}")


if __name__ == "__main__":
    main()
