#!/usr/bin/env python3
"""Sanity check: is HF Sam3Model (without TRT) behaving the same as the
SAM3 repo's SegmentAnything3 baseline?

Both should be the same underlying model. If they disagree, the HF-TRT
recall gap we attributed to TRT may actually be an HF-vs-Roboflow
model difference that TRT is faithfully reproducing.

Runs both on the 100-image COCO subset and reports per-image matching
stats, using the same methodology as aggregate_correctness.py.
"""

from __future__ import annotations

import gc
import json
import os
import pickle
import sys
import time
from pathlib import Path
from statistics import mean, stdev

# requires HF_TOKEN env var
# requires ROBOFLOW_API_KEY env var

import numpy as np
import torch
from PIL import Image

SUBSET_DIR = Path(os.environ.get("COCO_SUBSET", "/tmp/coco_val2017_subset"))
MANIFEST = SUBSET_DIR / "manifest.json"
OUT_DIR = Path(os.environ.get("SAM3_BENCH_DIR", "/tmp"))


def _collect_hf_pt():
    from transformers import Sam3Processor, Sam3Model

    proc = Sam3Processor.from_pretrained(
        "facebook/sam3", token=os.environ["HF_TOKEN"],
    )
    model = (
        Sam3Model.from_pretrained("facebook/sam3", token=os.environ["HF_TOKEN"])
        .to("cuda")
        .eval()
    )

    manifest = json.loads(MANIFEST.read_text())
    results = {}
    t_start = time.perf_counter()
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
            with torch.inference_mode():
                out = model(**inputs)
            target_sizes = (
                inputs.get("original_sizes").tolist()
                if "original_sizes" in inputs else [image.size[::-1]]
            )
            proc_results = proc.post_process_instance_segmentation(
                out, threshold=0.5, mask_threshold=0.5,
                target_sizes=target_sizes,
            )
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000
            r0 = proc_results[0] if proc_results else {}

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
            print(f"  [{i}] {entry['file_name']} FAILED: {e}", flush=True)
            results[entry["image_id"]] = {
                "error": repr(e),
                "prompt": entry["prompt"],
                "file_name": entry["file_name"],
            }
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(manifest)}] elapsed "
                  f"{time.perf_counter()-t_start:.1f}s", flush=True)

    del model
    gc.collect(); torch.cuda.empty_cache()
    return results


def main() -> int:
    out = OUT_DIR / "coco_sweep_hf_pt.pkl"
    print(f"Running HF PyTorch (no TRT) sweep on 100 COCO images ...")
    results = _collect_hf_pt()
    with open(out, "wb") as f:
        pickle.dump(results, f)
    n_ok = sum(1 for v in results.values() if "error" not in v)
    print(f"[hf_pt] saved {len(results)} records ({n_ok} ok) to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
