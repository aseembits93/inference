#!/usr/bin/env python3
"""Compute HF-TRT correctness vs HF-PT (not vs Roboflow PT-bf16).

Same greedy-IoU matching as aggregate_correctness.py but uses
hf_pt as reference so we can see TRT's own divergence separate
from the HF-vs-Roboflow implementation differences.
"""

import os, pickle, sys
from pathlib import Path
from statistics import mean, median
from collections import defaultdict

import numpy as np

sys.path.insert(0, ".")
from aggregate_correctness import _greedy_match, _per_image_metrics, MATCH_IOU

OUT_DIR = Path(os.environ.get("SAM3_BENCH_DIR", "/tmp"))


def _load(name):
    return pickle.load(open(OUT_DIR / f"coco_sweep_{name}.pkl", "rb"))


def _aggregate(ref, tst, label):
    per_image = []
    per_class = defaultdict(list)
    for iid, r_rec in ref.items():
        t_rec = tst.get(iid)
        if t_rec is None: continue
        m = _per_image_metrics(r_rec, t_rec)
        if m is None: continue
        m["image_id"] = iid
        m["prompt"] = r_rec.get("prompt")
        per_image.append(m)
        per_class[r_rec.get("prompt")].append(m)

    n = len(per_image)
    total_ref = sum(m["n_ref"] for m in per_image)
    total_tst = sum(m["n_tst"] for m in per_image)
    total_good = sum(m["n_good"] for m in per_image)

    recall = total_good / total_ref if total_ref > 0 else 0.0
    precision = total_good / total_tst if total_tst > 0 else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
    silent = sum(1 for m in per_image if m["silent_zero"])
    ghost = sum(1 for m in per_image if m["extra_when_ref_empty"])
    exact = sum(1 for m in per_image if m["n_tst"] == m["n_ref"])

    match_ious, score_deltas = [], []
    for m in per_image:
        for iou in m["ious"]:
            if iou >= MATCH_IOU:
                match_ious.append(iou)
        score_deltas.extend(m["score_deltas"])

    print(f"\n=== {label} (N={n}) ===")
    print(f"  ref={total_ref}, test={total_tst}, matched>=0.5: {total_good}")
    print(f"  Recall    {recall*100:5.1f}%   Precision {precision*100:5.1f}%   F1 {f1*100:5.1f}%")
    print(f"  Exact-count {exact}/{n}   silent_fail {silent}/{n}   ghost {ghost}/{n}")
    if match_ious:
        print(f"  Match IoU: mean={mean(match_ious):.4f} median={median(match_ious):.4f} min={min(match_ious):.4f}")
    if score_deltas:
        print(f"  Score delta: mean={mean(score_deltas):+.4f} median={median(score_deltas):+.4f}")


def main() -> int:
    print("HF-TRT variants compared against HF-PT (not Roboflow PT-bf16)")
    hf_pt = _load("hf_pt")

    for cfg in ["hf_trt", "hf_trt_decoder_fp32", "hf_trt_backbone_only",
                "hf_trt_attn_fp32", "hf_trt_fp32", "hf_trt_inferred",
                "hf_trt_nofuse_decoder", "hf_trt_nofuse_all"]:
        try:
            tst = _load(cfg)
        except FileNotFoundError:
            continue
        _aggregate(hf_pt, tst, f"{cfg} vs hf_pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
