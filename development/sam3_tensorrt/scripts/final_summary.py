#!/usr/bin/env python3
"""Combine latency benchmarks with logit-level correctness into a single
per-configuration summary, using PT-bf16 as reference."""

import os
from pathlib import Path

import numpy as np
BENCH_DIR = os.environ.get("SAM3_BENCH_DIR", "/tmp")
REF = f"{BENCH_DIR}/sam3_logits_bf16.npz"
CONFIGS = [
    ("PT-bf16",            f"{BENCH_DIR}/sam3_logits_bf16.npz",       2786.5),
    ("PT-fp32",            f"{BENCH_DIR}/sam3_logits_fp32.npz",       1743.5),
    ("PT-fp16",            f"{BENCH_DIR}/sam3_logits_fp16.npz",        488.0),
    ("TRT rope_fp32_d10",  f"{BENCH_DIR}/sam3_logits_trt_ropefp32.npz", 974.0),
    ("TRT rope_windowed_d8",f"{BENCH_DIR}/sam3_logits_trt.npz",        870.4),
    ("TRT fp16 (broken)",  f"{BENCH_DIR}/sam3_logits_trt_bad.npz",     748.0),
]

# Tensors we'll highlight: the ones that actually feed decisions.
KEY = [
    "out[0]/pred_masks",
    "out[0]/pred_logits",
    "out[0]/semantic_seg",
    "out[0]/queries",
    "out[0]/prev_encoder_out/backbone_out/vision_features",
]


def stats(ref_arr, tst_arr):
    a = ref_arr.flatten().astype(np.float64)
    b = tst_arr.flatten().astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos = float(a @ b / (na * nb + 1e-12)) if na > 0 and nb > 0 else float("nan")
    ratio = float((b.std() + 1e-12) / (a.std() + 1e-12))
    max_abs = float(np.abs(a - b).max())
    return cos, ratio, max_abs


def main():
    ref = dict(np.load(REF))

    print(f"{'config':26s} {'E2E (ms)':>10s} {'speedup':>8s} "
          f"{'min_cos':>8s} {'worst_ratio':>12s} {'pred_masks_cos':>16s} {'pred_masks_std':>16s}")
    pt_bf16_ms = None
    for name, path, ms in CONFIGS:
        if name == "PT-bf16":
            pt_bf16_ms = ms
            break

    for name, path, ms in CONFIGS:
        if not Path(path).exists():
            print(f"{name:26s} MISSING")
            continue
        tst = dict(np.load(path))

        # Overall worst cos + worst ratio deviation over all float tensors
        min_cos = 1.0
        worst_ratio_dev = 0.0
        worst_ratio = 1.0
        for k in ref:
            if k not in tst: continue
            if ref[k].dtype.kind != "f": continue
            if ref[k].shape != tst[k].shape: continue
            if ref[k].size == 0: continue
            cos, ratio, _ = stats(ref[k], tst[k])
            if np.isnan(cos): continue
            if cos < min_cos:
                min_cos = cos
            dev = abs(np.log(ratio + 1e-12))
            if dev > worst_ratio_dev:
                worst_ratio_dev = dev
                worst_ratio = ratio

        # Key tensor: pred_masks
        pm_cos, pm_ratio, pm_maxd = stats(ref["out[0]/pred_masks"], tst["out[0]/pred_masks"])

        speedup = pt_bf16_ms / ms
        print(f"{name:26s} {ms:10.1f} {speedup:7.2f}x "
              f"{min_cos:8.4f} {worst_ratio:12.4f} "
              f"{pm_cos:16.4f} {pm_ratio:16.4f}")

    print()
    print("Per-key-tensor cosine similarity (higher = better; -1 = inverted):")
    print(f"{'tensor':55s} " + " ".join(f"{n[:20]:>20s}" for n, _, _ in CONFIGS))
    for k in KEY:
        row = [f"{k:55s}"]
        for name, path, _ in CONFIGS:
            if not Path(path).exists():
                row.append(f"{'—':>20s}"); continue
            tst = dict(np.load(path))
            if k not in tst:
                row.append(f"{'—':>20s}"); continue
            cos, _, _ = stats(ref[k], tst[k])
            row.append(f"{cos:20.4f}")
        print(" ".join(row))

    print()
    print("Per-key-tensor std ratio (1.0 = same spread as bf16):")
    print(f"{'tensor':55s} " + " ".join(f"{n[:20]:>20s}" for n, _, _ in CONFIGS))
    for k in KEY:
        row = [f"{k:55s}"]
        for name, path, _ in CONFIGS:
            if not Path(path).exists():
                row.append(f"{'—':>20s}"); continue
            tst = dict(np.load(path))
            if k not in tst:
                row.append(f"{'—':>20s}"); continue
            _, ratio, _ = stats(ref[k], tst[k])
            row.append(f"{ratio:20.4f}")
        print(" ".join(row))


if __name__ == "__main__":
    main()
