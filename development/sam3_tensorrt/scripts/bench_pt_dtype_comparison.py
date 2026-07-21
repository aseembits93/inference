#!/usr/bin/env python3
"""Compare PyTorch baselines with different autocast dtypes on T4.

Runs SAM3 infer_from_request with three PT configurations:
 1. bfloat16 autocast (the repo default, emulated on T4)
 2. float16 autocast (what T4 Tensor Cores natively support)
 3. no autocast (pure FP32)

Plus the best TRT engine (fp16_rope_windowed_d8) for reference.

Each configuration is run in its own subprocess pass so memory doesn't
overflow on T4 (15 GB).
"""

from __future__ import annotations

import base64
import gc
import os
import sys
import time
from pathlib import Path
from statistics import mean, stdev

if "ROBOFLOW_API_KEY" not in os.environ:
    raise SystemExit("Set ROBOFLOW_API_KEY env var before running this script.")


import numpy as np
import torch

ASSET_DIR = Path(os.environ.get("SAM3_ASSETS", "tests/workflows/integration_tests/execution/assets"))
IMAGE_PATH = ASSET_DIR / "dogs.jpg"
PROMPT = "dog"
WARMUP = 3
ITERS = 15

ENGINE_PATH = Path(os.environ.get(
    "SAM3_ENGINE_PATH",
    "./sam3_onnx_exports/sam3_vision_backbone_fp16_rope_windowed_d8.engine",
))


def _build_req():
    from inference.core.entities.requests.sam3 import Sam3SegmentationRequest, Sam3Prompt
    img_b64 = base64.b64encode(IMAGE_PATH.read_bytes()).decode()
    return Sam3SegmentationRequest(
        image={"type": "base64", "value": img_b64},
        prompts=[Sam3Prompt(text=PROMPT)],
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
    return [_rle_to_mask(p.masks) for p in preds]


def _iou_vs_ref(ref_masks, tst_masks):
    if not ref_masks or not tst_masks:
        return float("nan"), len(ref_masks), len(tst_masks)
    pairs = [(i, j, _mask_iou(ref_masks[i], tst_masks[j]))
             for i in range(len(ref_masks)) for j in range(len(tst_masks))
             if ref_masks[i].shape == tst_masks[j].shape]
    pairs.sort(key=lambda x: -x[2])
    used_i, used_j, ious = set(), set(), []
    for i, j, iou in pairs:
        if i not in used_i and j not in used_j:
            used_i.add(i); used_j.add(j); ious.append(iou)
    return (float(np.mean(ious)) if ious else float("nan"),
            len(ref_masks), len(tst_masks))


def run_pt_pass(dtype_name: str):
    """Load SAM3 with autocast dtype patched to `dtype_name` ('bf16'/'fp16'/'fp32').

    We patch `torch.amp.autocast_mode.autocast.__init__` BEFORE loading so the
    model's autocast context uses our chosen dtype. For 'fp32' we disable
    autocast entirely.
    """
    if dtype_name == "bf16":
        pass  # repo default
    elif dtype_name == "fp16":
        orig_init = torch.amp.autocast_mode.autocast.__init__
        def new_init(self, device_type, dtype=None, enabled=True, cache_enabled=None):
            orig_init(self, device_type=device_type, dtype=torch.float16,
                      enabled=enabled, cache_enabled=cache_enabled)
        torch.amp.autocast_mode.autocast.__init__ = new_init
    elif dtype_name == "fp32":
        orig_init = torch.amp.autocast_mode.autocast.__init__
        def new_init(self, device_type, dtype=None, enabled=True, cache_enabled=None):
            orig_init(self, device_type=device_type, dtype=torch.float32,
                      enabled=False, cache_enabled=cache_enabled)
        torch.amp.autocast_mode.autocast.__init__ = new_init
    else:
        raise ValueError(dtype_name)

    from inference.models.sam3.segment_anything3 import SegmentAnything3
    m = SegmentAnything3(model_id="sam3/sam3_final", api_key=os.environ["ROBOFLOW_API_KEY"])
    req = _build_req()
    masks = _collect_masks(m, req)
    lats = _bench(m, req)
    del m
    gc.collect(); torch.cuda.empty_cache()
    return masks, lats


def run_trt_pass():
    from inference.models.sam3.segment_anything3 import SegmentAnything3
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sam3_trt_adapter import patch_sam3_with_trt_backbone
    m = SegmentAnything3(model_id="sam3/sam3_final", api_key=os.environ["ROBOFLOW_API_KEY"])
    patch_sam3_with_trt_backbone(m.model, ENGINE_PATH)
    req = _build_req()
    masks = _collect_masks(m, req)
    lats = _bench(m, req)
    del m
    gc.collect(); torch.cuda.empty_cache()
    return masks, lats


def main() -> int:
    which = sys.argv[1]
    if which in ("bf16", "fp16", "fp32"):
        masks, lats = run_pt_pass(which)
        label = f"PT-{which}"
    elif which == "trt":
        masks, lats = run_trt_pass()
        label = "TRT"
    else:
        print("usage: bench_pt_dtype_comparison.py {bf16|fp16|fp32|trt}")
        return 1

    print(f"{label}: {mean(lats):.1f}ms ± {stdev(lats):.1f}  "
          f"(min={min(lats):.1f} max={max(lats):.1f} N={ITERS})")
    print(f"{label} N_detections: {len(masks)}")
    # Save masks to compare later
    bench_dir = os.environ.get("SAM3_BENCH_DIR", "/tmp")
    out = Path(f"{bench_dir}/sam3_bench_{which}_masks.npz")
    if masks:
        np.savez(out, *masks)
        print(f"Saved {len(masks)} masks to {out}")
    else:
        np.savez(out, empty=np.zeros(1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
