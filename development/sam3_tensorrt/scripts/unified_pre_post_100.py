#!/usr/bin/env python3
"""Run unified pre/post cross-test across all 100 COCO images.

- Preprocess with HF Sam3Processor (once per image)
- Monkey-patch Roboflow's SimpleTokenizer to return HF's tokens
- Run both models with the same pixel_values + tokens
- Apply HF's post_process_instance_segmentation to both
- Aggregate: raw-output cosines per tensor, post-process agreement

Output: /tmp/coco_sweep_hf_pt_unified.pkl and
        /tmp/coco_sweep_rf_pt_unified.pkl
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

# requires HF_TOKEN env var
# requires ROBOFLOW_API_KEY env var

import numpy as np
import torch
from PIL import Image

SUBSET_DIR = Path(os.environ.get("COCO_SUBSET", "/tmp/coco_val2017_subset"))
MANIFEST = SUBSET_DIR / "manifest.json"
OUT_DIR = Path(os.environ.get("SAM3_BENCH_DIR", "/tmp"))


# ---------- Monkey-patch helper for Roboflow tokenizer ----------
def _patch_rf_tokenizer(rf_model, input_ids_per_image: dict,
                         attention_mask_per_image: dict):
    """Replace SimpleTokenizer.__call__ so it looks up tokens by a process
    -global variable `_CURRENT_IMG_ID`. Each call to the model sets that
    variable before invocation.
    """
    from sam3.model.tokenizer_ve import SimpleTokenizer
    orig_call = SimpleTokenizer.__call__

    def patched_call(self, texts, context_length=77, **kwargs):
        img_id = globals().get("_CURRENT_IMG_ID")
        if img_id is None:
            return orig_call(self, texts, context_length=context_length, **kwargs)

        device = next(rf_model.parameters()).device
        ids = input_ids_per_image[img_id][0]         # (seq_len,)
        mask = attention_mask_per_image[img_id][0]   # (seq_len,)
        real = ids[mask.bool()]                       # only the real tokens

        out = torch.zeros((1, context_length), dtype=torch.long, device=device)
        n = min(real.numel(), context_length)
        out[0, :n] = real[:n].to(device)
        return out

    SimpleTokenizer.__call__ = patched_call
    return orig_call


# ---------- Build RF batch from HF pixel_values ----------
def _build_rf_batch(pixel_values, prompt_text, h, w):
    from sam3.train.data.collator import collate_fn_api
    from sam3.train.data.sam3_image_dataset import Datapoint as Sam3Datapoint
    from sam3.train.data.sam3_image_dataset import Image as Sam3ImageDP
    from sam3.train.transforms.basic_for_api import (
        ComposeAPI, NormalizeAPI, ToTensorAPI, RandomResizeAPI,
    )
    from inference.models.sam3.segment_anything3 import _build_text_query
    from inference.core.env import SAM3_IMAGE_SIZE
    transform = ComposeAPI(transforms=[
        RandomResizeAPI(sizes=SAM3_IMAGE_SIZE, max_size=SAM3_IMAGE_SIZE,
                        square=True, consistent_transform=False),
        ToTensorAPI(),
        NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dummy = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
    dp = Sam3Datapoint(find_queries=[], images=[
        Sam3ImageDP(data=dummy, objects=[], size=(h, w))
    ])
    dp.find_queries.append(_build_text_query(coco_id=0, h=h, w=w, text=prompt_text))
    dp = transform(dp)
    # Replace image tensor with HF's pixel_values
    dp.images[0].data = pixel_values[0].clone()
    batch = collate_fn_api(batch=[dp], dict_key="x")["x"]
    return batch


# ---------- Unified post-process ----------
def _postprocess(raw, target_size, hf_proc):
    pl = raw["pred_logits"]
    if pl.ndim == 3 and pl.shape[-1] == 1:
        pl = pl.squeeze(-1)
    prl = raw["presence_logits"]
    if prl.ndim == 1:
        prl = prl.unsqueeze(0)
    pm = raw["pred_masks"]
    if pm.ndim == 3:
        pm = pm.unsqueeze(0)

    obj = SimpleNamespace(
        pred_logits=pl.to("cuda"),
        pred_boxes=raw["pred_boxes"].to("cuda") if raw["pred_boxes"] is not None else None,
        pred_masks=pm.to("cuda"),
        presence_logits=prl.to("cuda"),
    )
    return hf_proc.post_process_instance_segmentation(
        obj, threshold=0.5, mask_threshold=0.5, target_sizes=[target_size],
    )[0]


# ---------- Per-image pipelines ----------
def run_hf_pass():
    from transformers import Sam3Model, Sam3Processor
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
        h, w = image.size[1], image.size[0]
        try:
            inputs = proc(images=image, text=entry["prompt"], return_tensors="pt")
            with torch.inference_mode():
                out = model(
                    pixel_values=inputs["pixel_values"].to("cuda"),
                    input_ids=inputs["input_ids"].to("cuda"),
                    attention_mask=inputs["attention_mask"].to("cuda"),
                )
            post = _postprocess(
                {"pred_logits": out.pred_logits.float(),
                 "pred_boxes": out.pred_boxes.float(),
                 "pred_masks": out.pred_masks.float(),
                 "presence_logits": out.presence_logits.float()},
                (h, w), proc,
            )
            # Save raw tensor summaries (small) for later logit-level compare
            rec = {
                "prompt": entry["prompt"],
                "file_name": entry["file_name"],
                "hw": (h, w),
                "pred_logits": out.pred_logits.float().cpu().numpy(),  # (1, 200)
                "pred_boxes": out.pred_boxes.float().cpu().numpy(),    # (1, 200, 4)
                "presence_logits": out.presence_logits.float().cpu().numpy(),
                # Keep a summary of pred_masks to avoid bloat
                "pred_masks_std": float(out.pred_masks.float().std().item()),
                "pred_masks_norm": float(out.pred_masks.float().norm().item()),
                "pred_masks_mean": float(out.pred_masks.float().mean().item()),
                # Post-processed
                "scores": post["scores"].cpu().numpy(),
                "boxes_abs": post["boxes"].cpu().numpy(),
                "masks": post["masks"].cpu().numpy().astype(np.uint8),
                "n_det": len(post["scores"]),
                # For monkey-patch on Roboflow pass, save tokens
                "input_ids": inputs["input_ids"].cpu().numpy(),
                "attention_mask": inputs["attention_mask"].cpu().numpy(),
                "pixel_values": inputs["pixel_values"].cpu().numpy(),
            }
            results[entry["image_id"]] = rec
        except Exception as e:
            print(f"  [{i}] {entry['file_name']} FAILED: {e}", flush=True)
            results[entry["image_id"]] = {"error": repr(e),
                                          "prompt": entry["prompt"],
                                          "file_name": entry["file_name"]}
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(manifest)}] elapsed "
                  f"{time.perf_counter() - t_start:.1f}s", flush=True)
    del model
    gc.collect(); torch.cuda.empty_cache()
    return results


def run_rf_pass(hf_results):
    from inference.models.sam3.segment_anything3 import SegmentAnything3
    from transformers import Sam3Processor
    proc = Sam3Processor.from_pretrained("facebook/sam3", token=os.environ["HF_TOKEN"])

    m = SegmentAnything3(
        model_id="sam3/sam3_final", api_key=os.environ["ROBOFLOW_API_KEY"],
    )

    # Build dicts for the monkey-patch
    input_ids_per = {iid: torch.from_numpy(r["input_ids"])
                     for iid, r in hf_results.items() if "error" not in r}
    attn_per = {iid: torch.from_numpy(r["attention_mask"])
                for iid, r in hf_results.items() if "error" not in r}

    from sam3.model.tokenizer_ve import SimpleTokenizer
    orig_call = SimpleTokenizer.__call__

    def patched_call(self, texts, context_length=77, **kwargs):
        img_id = globals().get("_CURRENT_IMG_ID")
        if img_id is None or img_id not in input_ids_per:
            return orig_call(self, texts, context_length=context_length, **kwargs)
        device = next(m.model.parameters()).device
        ids = input_ids_per[img_id][0]
        mask = attn_per[img_id][0]
        real = ids[mask.bool()]
        out = torch.zeros((1, context_length), dtype=torch.long, device=device)
        n = min(real.numel(), context_length)
        out[0, :n] = real[:n].to(device)
        return out

    SimpleTokenizer.__call__ = patched_call

    try:
        from sam3.model.utils.misc import copy_data_to_device

        results = {}
        t_start = time.perf_counter()
        image_ids = sorted(hf_results.keys())
        for i, iid in enumerate(image_ids):
            hf_rec = hf_results[iid]
            if "error" in hf_rec:
                continue
            h, w = hf_rec["hw"]
            prompt = hf_rec["prompt"]
            pixel_values = torch.from_numpy(hf_rec["pixel_values"])

            globals()["_CURRENT_IMG_ID"] = iid
            try:
                batch = _build_rf_batch(pixel_values, prompt, h, w)
                batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)
                with torch.inference_mode():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        raw = m.model(batch)
                out = raw[0]
                post = _postprocess(
                    {"pred_logits": out["pred_logits"].float(),
                     "pred_boxes": out["pred_boxes"].float() if "pred_boxes" in out else None,
                     "pred_masks": out["pred_masks"].float(),
                     "presence_logits": out.get("presence_logit_dec",
                                                torch.tensor([0.0])).float()},
                    (h, w), proc,
                )
                rec = {
                    "prompt": prompt, "file_name": hf_rec["file_name"],
                    "hw": (h, w),
                    "pred_logits": out["pred_logits"].float().cpu().numpy(),
                    "pred_boxes": out["pred_boxes"].float().cpu().numpy() if "pred_boxes" in out else None,
                    "presence_logits": out.get("presence_logit_dec",
                                               torch.tensor([0.0])).float().cpu().numpy(),
                    "pred_masks_std": float(out["pred_masks"].float().std().item()),
                    "pred_masks_norm": float(out["pred_masks"].float().norm().item()),
                    "pred_masks_mean": float(out["pred_masks"].float().mean().item()),
                    "scores": post["scores"].cpu().numpy(),
                    "boxes_abs": post["boxes"].cpu().numpy(),
                    "masks": post["masks"].cpu().numpy().astype(np.uint8),
                    "n_det": len(post["scores"]),
                }
                results[iid] = rec
            except Exception as e:
                print(f"  [{i}] img_id={iid} FAILED: {e}", flush=True)
                results[iid] = {"error": repr(e),
                                "prompt": prompt,
                                "file_name": hf_rec["file_name"]}
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(image_ids)}] elapsed "
                      f"{time.perf_counter()-t_start:.1f}s", flush=True)
    finally:
        SimpleTokenizer.__call__ = orig_call
        globals()["_CURRENT_IMG_ID"] = None

    return results


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else None
    if which == "hf":
        print(f"Running HF unified pass on 100 images ...")
        results = run_hf_pass()
        out = OUT_DIR / "coco_sweep_hf_pt_unified.pkl"
        with open(out, "wb") as f:
            pickle.dump(results, f)
        print(f"saved {out}")
    elif which == "rf":
        hf_path = OUT_DIR / "coco_sweep_hf_pt_unified.pkl"
        if not hf_path.exists():
            print(f"ERROR: {hf_path} missing. Run 'hf' pass first.")
            return 1
        hf_results = pickle.load(open(hf_path, "rb"))
        print(f"Running RF unified pass on 100 images ...")
        results = run_rf_pass(hf_results)
        out = OUT_DIR / "coco_sweep_rf_pt_unified.pkl"
        with open(out, "wb") as f:
            pickle.dump(results, f)
        print(f"saved {out}")
    else:
        print("usage: unified_pre_post_100.py {hf|rf}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
