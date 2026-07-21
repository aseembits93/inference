#!/usr/bin/env python3
"""Compare mask outputs across runs to compute IoU."""

import os
import numpy as np
from pathlib import Path


def load_masks(path):
    d = np.load(path)
    if "empty" in d.files:
        return []
    return [d[k] for k in d.files]


def mask_iou(a, b):
    a = (a > 0).astype(np.uint8)
    b = (b > 0).astype(np.uint8)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def iou_pairs(ref, tst):
    if not ref or not tst:
        return []
    pairs = [(i, j, mask_iou(ref[i], tst[j]))
             for i in range(len(ref)) for j in range(len(tst))
             if ref[i].shape == tst[j].shape]
    pairs.sort(key=lambda x: -x[2])
    used_i, used_j, ious = set(), set(), []
    for i, j, iou in pairs:
        if i not in used_i and j not in used_j:
            used_i.add(i); used_j.add(j); ious.append(iou)
    return ious


ref = load_masks(f"{os.environ.get('SAM3_BENCH_DIR', '/tmp')}/sam3_bench_bf16_masks.npz")
print(f"Reference (PT-bf16): N={len(ref)}")

for tag in ["fp16", "fp32", "trt"]:
    p = Path(f"{os.environ.get('SAM3_BENCH_DIR', '/tmp')}/sam3_bench_{tag}_masks.npz")
    if not p.exists():
        print(f"{tag}: missing")
        continue
    tst = load_masks(p)
    ious = iou_pairs(ref, tst)
    miou = float(np.mean(ious)) if ious else float("nan")
    print(f"  vs PT-{tag}: N={len(tst)} mean IoU={miou:.4f} "
          f"(ious={[f'{x:.4f}' for x in ious]})")
