#!/usr/bin/env python3
"""Final SAM3 TRT benchmark: PyTorch (bf16 autocast) vs TRT bf16_io engine.

Reports:
  - Backbone-only latency (the swapped piece)
  - E2E .infer_from_request() latency
  - Correctness: per-image mask IoU across 4 real images
"""

from __future__ import annotations

import base64
import os
import sys
import time
from pathlib import Path
from statistics import mean, stdev

if "ROBOFLOW_API_KEY" not in os.environ:
    raise SystemExit("Set ROBOFLOW_API_KEY env var before running this script.")


import numpy as np
import torch

import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parent))
from sam3_trt_adapter import Sam3VisionTRT, patch_sam3_with_trt_backbone

ENGINE_PATH = Path(os.environ.get(
    "SAM3_ENGINE_PATH",
    "./sam3_onnx_exports/sam3_vision_backbone_bf16_in.engine",
))
ASSET_DIR = Path(
    os.environ.get("SAM3_ASSETS", "tests/workflows/integration_tests/execution/assets")
)
IMAGES = [
    ("dogs.jpg", "dog"),
    ("car.jpg", "car"),
    ("crowd.jpg", "person"),
    ("multi-fruit.jpg", "fruit"),
]
BENCH_IMAGE, BENCH_PROMPT = IMAGES[0]

WARMUP = 5
ITERS = 30


def _rle_to_mask(rle_dict) -> np.ndarray:
    from pycocotools import mask as mu
    if isinstance(rle_dict.get("counts"), str):
        rle_dict = {"size": rle_dict["size"], "counts": rle_dict["counts"].encode("utf-8")}
    return mu.decode(rle_dict)


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = (a > 0).astype(np.uint8); b = (b > 0).astype(np.uint8)
    inter = np.logical_and(a, b).sum(); union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def _build_req(path: Path, prompt: str):
    from inference.core.entities.requests.sam3 import Sam3SegmentationRequest, Sam3Prompt
    return Sam3SegmentationRequest(
        image={"type": "base64", "value": base64.b64encode(path.read_bytes()).decode("ascii")},
        prompts=[Sam3Prompt(text=prompt)],
        output_prob_thresh=0.5,
        format="rle",
    )


def bench(model, req, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        model.infer_from_request(req)
    torch.cuda.synchronize()
    lats = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t = time.perf_counter()
        model.infer_from_request(req)
        torch.cuda.synchronize()
        lats.append((time.perf_counter() - t) * 1000)
    return lats


def bench_backbone(rf, iters=50):
    backbone = rf.model.backbone
    x = torch.randn(1, 3, 1008, 1008, device="cuda", dtype=torch.float32)
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for _ in range(5):
                _ = backbone.forward_image(x)
            torch.cuda.synchronize()
            lats = []
            for _ in range(iters):
                torch.cuda.synchronize()
                t = time.perf_counter()
                _ = backbone.forward_image(x)
                torch.cuda.synchronize()
                lats.append((time.perf_counter() - t) * 1000)
    return lats


def bench_trt_backbone(engine_path, iters=50):
    runner = Sam3VisionTRT(engine_path)
    x = torch.randn(1, 3, 1008, 1008, device="cuda", dtype=torch.float32)
    for _ in range(5):
        _ = runner.run(x)
    torch.cuda.synchronize()
    lats = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t = time.perf_counter()
        _ = runner.run(x)
        torch.cuda.synchronize()
        lats.append((time.perf_counter() - t) * 1000)
    return lats, runner


def main() -> int:
    from inference.models.sam3.segment_anything3 import SegmentAnything3

    print(f"Engine: {ENGINE_PATH.name} ({ENGINE_PATH.stat().st_size/1e6:.0f} MB)")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()

    # Load both models
    print("Loading baseline PyTorch SAM3...")
    base_model = SegmentAnything3(
        model_id="sam3/sam3_final", api_key=os.environ["ROBOFLOW_API_KEY"],
    )

    print("Loading TRT-patched SAM3...")
    trt_model = SegmentAnything3(
        model_id="sam3/sam3_final", api_key=os.environ["ROBOFLOW_API_KEY"],
    )
    patch_sam3_with_trt_backbone(trt_model.model, ENGINE_PATH)

    # ---- Correctness (run first so state is clean) ----
    print("\n=== Correctness (mask IoU across 4 images) ===")
    all_ious = []
    for fname, prompt in IMAGES:
        path = ASSET_DIR / fname
        if not path.exists():
            print(f"  SKIP {fname}")
            continue
        req = _build_req(path, prompt)

        ref_resp = base_model.infer_from_request(req)
        tst_resp = trt_model.infer_from_request(req)
        ref_preds = ref_resp.prompt_results[0].predictions
        tst_preds = tst_resp.prompt_results[0].predictions
        ref_masks = [_rle_to_mask(p.masks) for p in ref_preds]
        tst_masks = [_rle_to_mask(p.masks) for p in tst_preds]

        # Greedy IoU match
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
        print(f"  {fname:20s} prompt={prompt!r:12s} "
              f"baseline N={len(ref_masks)}  TRT N={len(tst_masks)}  "
              f"mean IoU={miou:.4f}  min IoU={mn:.4f}")

    overall = float(np.mean(all_ious)) if all_ious else float("nan")
    overall_min = float(np.min(all_ious)) if all_ious else float("nan")
    print(f"\nOverall:  mean IoU = {overall:.4f}   min IoU = {overall_min:.4f}")
    gate = overall >= 0.95
    print(f"Gate (>=0.95):  {'PASS' if gate else 'FAIL'}")

    # ---- Backbone-only latency ----
    print("\n=== Backbone-only latency (GPU sync'd) ===")
    pt_lats = bench_backbone(base_model)
    pt_m = mean(pt_lats)
    print(f"  PyTorch bf16 autocast:  {pt_m:>7.2f} ms ± {stdev(pt_lats):.2f}")
    trt_lats, _ = bench_trt_backbone(ENGINE_PATH)
    trt_m = mean(trt_lats)
    print(f"  TRT engine:             {trt_m:>7.2f} ms ± {stdev(trt_lats):.2f}")
    print(f"  Speedup:                {pt_m/trt_m:.2f}x  "
          f"({(trt_m-pt_m)/pt_m*100:+.1f}%)")

    # ---- E2E latency ----
    print("\n=== E2E latency (.infer_from_request) ===")
    bench_req = _build_req(ASSET_DIR / BENCH_IMAGE, BENCH_PROMPT)
    pt_lats = bench(base_model, bench_req)
    pt_m = mean(pt_lats)
    print(f"  PyTorch:  {pt_m:>7.2f} ms ± {stdev(pt_lats):.2f}  "
          f"(min={min(pt_lats):.2f} max={max(pt_lats):.2f})")
    trt_lats = bench(trt_model, bench_req)
    trt_m = mean(trt_lats)
    print(f"  TRT:      {trt_m:>7.2f} ms ± {stdev(trt_lats):.2f}  "
          f"(min={min(trt_lats):.2f} max={max(trt_lats):.2f})")
    print(f"  Speedup:  {pt_m/trt_m:.2f}x  ({(trt_m-pt_m)/pt_m*100:+.1f}%)")
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
