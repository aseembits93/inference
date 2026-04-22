#!/usr/bin/env python3
"""Run each config across all 100 COCO images and save per-image predictions.

One pass per config in its own subprocess so T4's 15 GB VRAM doesn't OOM.

Outputs:
  /tmp/coco_sweep_{config}.pkl
with structure: {image_id: {'masks': List[np.ndarray bool], 'scores': List[float], 'n_det': int}}
"""

from __future__ import annotations

import base64
import gc
import json
import os
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# requires ROBOFLOW_API_KEY env var
# requires HF_TOKEN env var

import numpy as np
import torch
from PIL import Image

SUBSET_DIR = Path(os.environ.get("COCO_SUBSET", "/tmp/coco_val2017_subset"))
MANIFEST = SUBSET_DIR / "manifest.json"
OUT_DIR = Path(os.environ.get("SAM3_BENCH_DIR", "/tmp"))

REPO_TRT_ENGINE = Path(
    "./sam3_onnx_exports/"
    "sam3_vision_backbone_fp16_rope_windowed_d8.engine"
)
HF_TRT_ENGINE = Path(
    "./sam3_hf_onnx_full/sam3_hf_fp16.engine"
)

# Threshold for accepting a detection (matches post_process default)
OUTPUT_PROB_THRESH = 0.5


# ---------------------------------------------------------------------
# Patch autocast (used by every config)
# ---------------------------------------------------------------------
def _patch_autocast_to(want_dtype: torch.dtype):
    orig = torch.amp.autocast_mode.autocast.__init__
    def new_init(self, device_type, dtype=None, enabled=True, cache_enabled=None):
        orig(self, device_type=device_type, dtype=want_dtype,
             enabled=enabled, cache_enabled=cache_enabled)
    torch.amp.autocast_mode.autocast.__init__ = new_init


def _rle_to_mask(rle):
    from pycocotools import mask as mu
    if isinstance(rle.get("counts"), str):
        rle = {"size": rle["size"], "counts": rle["counts"].encode()}
    return mu.decode(rle)


# ---------------------------------------------------------------------
# SAM3 repo configs (pt_bf16, pt_fp16, trt_swap)
# ---------------------------------------------------------------------
def _collect_repo(mode: str):
    """mode in {'pt_bf16', 'pt_fp16', 'trt_swap'}"""
    if mode == "pt_fp16":
        _patch_autocast_to(torch.float16)
    elif mode == "trt_swap":
        _patch_autocast_to(torch.float16)
    # pt_bf16 uses repo default

    sys.path.insert(0, ".")
    from inference.models.sam3.segment_anything3 import SegmentAnything3
    from inference.core.entities.requests.sam3 import (
        Sam3SegmentationRequest, Sam3Prompt,
    )

    m = SegmentAnything3(
        model_id="sam3/sam3_final", api_key=os.environ["ROBOFLOW_API_KEY"],
    )
    if mode == "trt_swap":
        from sam3_trt_adapter import patch_sam3_with_trt_backbone
        patch_sam3_with_trt_backbone(m.model, REPO_TRT_ENGINE)

    manifest = json.loads(MANIFEST.read_text())
    results = {}
    t_total_start = time.perf_counter()
    for i, entry in enumerate(manifest):
        path = Path(entry["local_path"])
        if not path.exists():
            continue
        img_b64 = base64.b64encode(path.read_bytes()).decode()
        req = Sam3SegmentationRequest(
            image={"type": "base64", "value": img_b64},
            prompts=[Sam3Prompt(text=entry["prompt"])],
            output_prob_thresh=OUTPUT_PROB_THRESH,
            format="rle",
        )
        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            resp = m.infer_from_request(req)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000
            preds = resp.prompt_results[0].predictions
            masks = [_rle_to_mask(p.masks) for p in preds]
            scores = [float(p.confidence) for p in preds]
            results[entry["image_id"]] = {
                "masks": masks,
                "scores": scores,
                "n_det": len(preds),
                "latency_ms": dt,
                "prompt": entry["prompt"],
                "file_name": entry["file_name"],
            }
        except Exception as e:
            print(f"  [{i}] {entry['file_name']} FAILED: {e}")
            results[entry["image_id"]] = {
                "error": repr(e),
                "prompt": entry["prompt"],
                "file_name": entry["file_name"],
            }
        if (i + 1) % 10 == 0:
            elapsed = time.perf_counter() - t_total_start
            print(f"  [{i+1}/{len(manifest)}] elapsed {elapsed:.1f}s", flush=True)

    del m
    gc.collect(); torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------
# HF-TRT config
# ---------------------------------------------------------------------
def _collect_hf_trt():
    from transformers import Sam3Processor
    sys.path.insert(0, ".")
    from bench_three_way import HFTrtRunner

    proc = Sam3Processor.from_pretrained(
        "facebook/sam3", token=os.environ["HF_TOKEN"],
    )
    runner = HFTrtRunner(HF_TRT_ENGINE)

    manifest = json.loads(MANIFEST.read_text())
    results = {}
    t_total_start = time.perf_counter()
    for i, entry in enumerate(manifest):
        path = Path(entry["local_path"])
        if not path.exists():
            continue
        image = Image.open(path).convert("RGB")
        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            inputs = proc(
                images=image, text=entry["prompt"], return_tensors="pt",
            ).to("cuda")
            outs = runner(
                inputs["pixel_values"], inputs["input_ids"], inputs["attention_mask"],
            )
            names = runner.output_names
            out_map = {n: outs[i] for i, n in enumerate(names)}
            pseudo = SimpleNamespace(**out_map)
            target_sizes = (
                inputs.get("original_sizes").tolist()
                if "original_sizes" in inputs else [image.size[::-1]]
            )
            proc_results = proc.post_process_instance_segmentation(
                pseudo, threshold=OUTPUT_PROB_THRESH,
                mask_threshold=0.5, target_sizes=target_sizes,
            )
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000
            r0 = proc_results[0] if proc_results else {}
            # HF returns masks as (N, H, W) tensor or list
            masks_list = []
            if "masks" in r0 and r0["masks"] is not None and len(r0["masks"]) > 0:
                masks_tensor = r0["masks"]
                if torch.is_tensor(masks_tensor):
                    masks_np = masks_tensor.detach().cpu().numpy().astype(np.uint8)
                else:
                    masks_np = np.asarray(masks_tensor, dtype=np.uint8)
                if masks_np.ndim == 4 and masks_np.shape[1] == 1:
                    masks_np = masks_np[:, 0]
                elif masks_np.ndim == 2:
                    masks_np = masks_np[None, ...]
                masks_list = [masks_np[j] for j in range(masks_np.shape[0])]
            scores = (
                r0["scores"].detach().cpu().tolist()
                if "scores" in r0 and torch.is_tensor(r0["scores"])
                else list(r0.get("scores", []))
            )
            results[entry["image_id"]] = {
                "masks": masks_list,
                "scores": scores,
                "n_det": len(masks_list),
                "latency_ms": dt,
                "prompt": entry["prompt"],
                "file_name": entry["file_name"],
            }
        except Exception as e:
            print(f"  [{i}] {entry['file_name']} FAILED: {e}")
            results[entry["image_id"]] = {
                "error": repr(e),
                "prompt": entry["prompt"],
                "file_name": entry["file_name"],
            }
        if (i + 1) % 10 == 0:
            elapsed = time.perf_counter() - t_total_start
            print(f"  [{i+1}/{len(manifest)}] elapsed {elapsed:.1f}s", flush=True)

    return results


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else None
    if which in ("pt_bf16", "pt_fp16", "trt_swap"):
        results = _collect_repo(which)
    elif which == "hf_trt":
        results = _collect_hf_trt()
    else:
        print("usage: sweep_100_images.py {pt_bf16|pt_fp16|trt_swap|hf_trt}")
        return 1

    out = OUT_DIR / f"coco_sweep_{which}.pkl"
    with open(out, "wb") as f:
        pickle.dump(results, f)
    n_ok = sum(1 for v in results.values() if "error" not in v)
    print(f"[{which}] saved {len(results)} records ({n_ok} ok) to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
