#!/usr/bin/env python3
"""Re-run HF on 100 COCO images, passing a dummy padding box to force
the geometry_encoder path. If the cls_embed-inclusion hypothesis is
correct, this should bring HF-PT into near-exact agreement with RF-PT.
"""

from __future__ import annotations

import gc
import json
import os
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# requires HF_TOKEN env var

import numpy as np
import torch
from PIL import Image

SUBSET_DIR = Path(os.environ.get("COCO_SUBSET", "/tmp/coco_val2017_subset"))
MANIFEST = SUBSET_DIR / "manifest.json"
OUT_DIR = Path(os.environ.get("SAM3_BENCH_DIR", "/tmp"))


def main() -> int:
    from transformers import Sam3Processor, Sam3Model
    proc = Sam3Processor.from_pretrained("facebook/sam3", token=os.environ["HF_TOKEN"])
    model = Sam3Model.from_pretrained(
        "facebook/sam3", token=os.environ["HF_TOKEN"],
    ).to("cuda").eval()

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
            # Force geometry_encoder by passing a dummy padding box
            dummy_box = torch.zeros(1, 1, 4, device="cuda")
            dummy_lab = torch.tensor([[-10]], dtype=torch.long, device="cuda")
            with torch.inference_mode():
                out = model(
                    pixel_values=inputs["pixel_values"],
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    input_boxes=dummy_box,
                    input_boxes_labels=dummy_lab,
                )
            target_sizes = (
                inputs.get("original_sizes").tolist()
                if "original_sizes" in inputs else [image.size[::-1]]
            )
            proc_results = proc.post_process_instance_segmentation(
                out, threshold=0.5, mask_threshold=0.5, target_sizes=target_sizes,
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
                  f"{time.perf_counter() - t_start:.1f}s", flush=True)

    out = OUT_DIR / "coco_sweep_hf_pt_dummy_box.pkl"
    with open(out, "wb") as f:
        pickle.dump(results, f)
    n_ok = sum(1 for v in results.values() if "error" not in v)
    print(f"[hf_pt_dummy_box] saved {len(results)} records ({n_ok} ok) to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
