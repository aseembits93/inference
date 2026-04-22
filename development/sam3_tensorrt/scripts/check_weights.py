#!/usr/bin/env python3
"""Compare weights between HF Sam3Model and Roboflow SegmentAnything3.

Same published model (Meta's SAM3), but two different PyTorch code
paths. Are the weights bit-identical? If yes, the HF-PT divergence
is purely implementation (different ops, different order of ops).
If no, the checkpoints themselves differ.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# requires HF_TOKEN env var
# requires ROBOFLOW_API_KEY env var

import torch


def main() -> int:
    print("Loading HF Sam3Model ...")
    from transformers import Sam3Model
    hf = Sam3Model.from_pretrained("facebook/sam3", token=os.environ["HF_TOKEN"]).eval()
    hf_sd = hf.state_dict()
    print(f"  HF state_dict: {len(hf_sd)} tensors, "
          f"{sum(t.numel() for t in hf_sd.values()) / 1e6:.1f}M params")

    print("\nLoading Roboflow SegmentAnything3 ...")
    from inference.models.sam3.segment_anything3 import SegmentAnything3
    rf = SegmentAnything3(
        model_id="sam3/sam3_final", api_key=os.environ["ROBOFLOW_API_KEY"],
    )
    # Roboflow wraps a PyTorch module in .model
    rf_sd = rf.model.state_dict()
    print(f"  RF state_dict: {len(rf_sd)} tensors, "
          f"{sum(t.numel() for t in rf_sd.values()) / 1e6:.1f}M params")

    # Collect summary stats per tensor (dtype, shape, flat-fingerprint)
    def fingerprint(t):
        t = t.detach().float().flatten().cpu()
        n = t.numel()
        return {
            "numel": n,
            "shape": tuple(t.shape),  # flat
            "dtype": str(t.dtype),
            "mean": float(t.mean()) if n else 0.0,
            "std": float(t.std()) if n > 1 else 0.0,
            "min": float(t.min()) if n else 0.0,
            "max": float(t.max()) if n else 0.0,
            "sum": float(t.sum()) if n else 0.0,
        }

    # Key-level comparison: match keys by basic sorted order + identity
    print("\nSample HF keys (first 10):")
    for k in list(hf_sd.keys())[:10]:
        print(f"  {k}: {tuple(hf_sd[k].shape)} {hf_sd[k].dtype}")
    print("\nSample RF keys (first 10):")
    for k in list(rf_sd.keys())[:10]:
        print(f"  {k}: {tuple(rf_sd[k].shape)} {rf_sd[k].dtype}")

    # Count overlap by shape signature
    from collections import Counter
    hf_shapes = Counter(tuple(t.shape) for t in hf_sd.values())
    rf_shapes = Counter(tuple(t.shape) for t in rf_sd.values())

    print(f"\nHF distinct shapes: {len(hf_shapes)}")
    print(f"RF distinct shapes: {len(rf_shapes)}")

    common_shapes = set(hf_shapes) & set(rf_shapes)
    print(f"Common shapes: {len(common_shapes)}")

    # Aggregate statistics: global mean + std of all weights concatenated
    # (a sanity fingerprint for whether the two are the same checkpoint)
    hf_all = torch.cat([t.detach().float().flatten().cpu() for t in hf_sd.values()])
    rf_all = torch.cat([t.detach().float().flatten().cpu() for t in rf_sd.values()])

    print()
    print("Global weight statistics:")
    print(f"  HF:  n={hf_all.numel():>11d}  mean={hf_all.mean():+.6f}  "
          f"std={hf_all.std():.6f}  sum={hf_all.sum():+.4f}  "
          f"sum_abs={hf_all.abs().sum():.4f}")
    print(f"  RF:  n={rf_all.numel():>11d}  mean={rf_all.mean():+.6f}  "
          f"std={rf_all.std():.6f}  sum={rf_all.sum():+.4f}  "
          f"sum_abs={rf_all.abs().sum():.4f}")

    if hf_all.numel() == rf_all.numel():
        diff = (hf_all - rf_all).abs()
        print(f"\nIf sorted-param-order matches:")
        print(f"  abs-diff max  = {diff.max():.6e}")
        print(f"  abs-diff mean = {diff.mean():.6e}")

    # Sort both per-tensor fingerprint lists by (numel, shape, mean) to make
    # a rough global fingerprint that doesn't depend on naming conventions
    hf_fp = sorted(
        (fingerprint(t) for t in hf_sd.values()),
        key=lambda d: (d["numel"], d["shape"], round(d["mean"], 6), round(d["std"], 6)),
    )
    rf_fp = sorted(
        (fingerprint(t) for t in rf_sd.values()),
        key=lambda d: (d["numel"], d["shape"], round(d["mean"], 6), round(d["std"], 6)),
    )

    n_match = 0
    n_shape_mismatch = 0
    for a, b in zip(hf_fp, rf_fp):
        if a["shape"] != b["shape"]:
            n_shape_mismatch += 1
            continue
        # Compare mean/std/sum fingerprint tight (float32 precision)
        if (abs(a["mean"] - b["mean"]) < 1e-6 and
            abs(a["std"] - b["std"]) < 1e-6 and
            abs(a["sum"] - b["sum"]) < 1e-4):
            n_match += 1
    print(f"\nPer-tensor fingerprint match (sorted by shape+stats):")
    print(f"  matched: {n_match} / {min(len(hf_fp), len(rf_fp))}")
    print(f"  shape-mismatch at same rank: {n_shape_mismatch}")
    print(f"  HF total tensors: {len(hf_fp)}")
    print(f"  RF total tensors: {len(rf_fp)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
