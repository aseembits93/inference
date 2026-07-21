#!/usr/bin/env python3
"""Cross-test: run HF and Roboflow models with IDENTICAL preprocessing
and IDENTICAL post-processing.

Procedure:
  1. Preprocess the image ONCE via Sam3Processor (HF). Produce pixel_values,
     input_ids (length 32), attention_mask.
  2. Feed the same pixel_values into the Roboflow model via a custom
     calling path that bypasses its native pre-tokenization (we pass the
     HF's already-tokenized ids as the language input).
  3. Collect raw outputs from both models: pred_logits, pred_boxes,
     pred_masks, presence_logits, semantic_seg.
  4. Post-process both with the SAME function (HF's Sam3Processor
     post_process_instance_segmentation).
  5. Report per-image detection counts + matched IoU + score delta.

If after unified pre/post the models still disagree, the gap is pure
model-graph difference (identical weights, different code paths). If
they now agree, the gap was in pre/post.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# requires HF_TOKEN env var
# requires ROBOFLOW_API_KEY env var

import numpy as np
import torch
from PIL import Image


def build_rf_batch_from_hf_tensors(pixel_values: torch.Tensor, input_ids: torch.Tensor,
                                     attention_mask: torch.Tensor, prompt_text: str,
                                     h: int, w: int):
    """Build a Roboflow BatchedDatapoint using the pre-tokenized HF inputs.

    The Roboflow forward path wants a BatchedDatapoint with:
      - img_batch (raw image tensor that matches what its transform would produce)
      - find_text_batch (a list of raw strings -- but those get tokenized
        internally by VETextEncoder)
      - find_metadatas, find_inputs, etc.

    We can't easily bypass the internal tokenization to feed HF's tokens
    directly. So we'll just pass the prompt string and rely on matching
    pixel_values. If Roboflow's internal CLIPTokenizer differs from HF's
    (padding to 77 vs 32, etc.), that's the residual pre-processing
    confound we flagged.

    For a TRUE apples-to-apples test we'd need to monkey-patch the
    language_backbone to skip its tokenizer. Let's do that.
    """
    from sam3.train.data.collator import collate_fn_api
    from sam3.train.data.sam3_image_dataset import Datapoint as Sam3Datapoint
    from sam3.train.data.sam3_image_dataset import Image as Sam3ImageDP
    from inference.models.sam3.segment_anything3 import _build_text_query

    # Build a Datapoint. We'll REPLACE its image tensor with pixel_values below.
    # PIL image is just a placeholder for the constructor.
    dummy_pil = Image.fromarray(
        (np.zeros((h, w, 3), dtype=np.uint8))
    )
    dp = Sam3Datapoint(
        find_queries=[],
        images=[Sam3ImageDP(data=dummy_pil, objects=[], size=(h, w))],
    )
    dp.find_queries.append(_build_text_query(coco_id=0, h=h, w=w, text=prompt_text))
    # Apply only the ToTensor + Normalize transforms so that non-image fields
    # (find_queries) are set up, then overwrite the image tensor.
    from sam3.train.transforms.basic_for_api import ComposeAPI, NormalizeAPI, ToTensorAPI, RandomResizeAPI
    from inference.core.env import SAM3_IMAGE_SIZE
    transform = ComposeAPI(transforms=[
        RandomResizeAPI(sizes=SAM3_IMAGE_SIZE, max_size=SAM3_IMAGE_SIZE,
                        square=True, consistent_transform=False),
        ToTensorAPI(),
        NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dp = transform(dp)

    # REPLACE the image tensor with HF's pixel_values
    # HF: (1, 3, 1008, 1008), RF expects (3, 1008, 1008) stored in images[0].data
    assert pixel_values.shape[1:] == (3, 1008, 1008), pixel_values.shape
    dp.images[0].data = pixel_values[0].clone()

    batch = collate_fn_api(batch=[dp], dict_key="x")["x"]
    return batch


def hf_tokens_via_monkey_patch(rf_model, hf_input_ids, hf_attention_mask):
    """Monkey-patch Roboflow's VETextEncoder.tokenizer so it returns
    HF's pre-tokenized input_ids instead of tokenizing the passed string.

    Roboflow's SimpleTokenizer returns a Tensor of shape (B, 77) with pad 0.
    HF returns (B, 32) with pad 49407. We need to adapt: pad HF's to 77 with
    0's so RF's downstream code handles the sequence the same way.
    """
    from sam3.model.tokenizer_ve import SimpleTokenizer
    orig_call = SimpleTokenizer.__call__

    # HF: input_ids shape (1, 32), pad=49407
    # RF expects: shape (1, 77), pad=0
    # Keep the 3 real tokens [49406, 1929, 49407], pad the rest with 0's
    # But attention_mask tells us which ones are real
    valid_mask = hf_attention_mask[0].bool()
    real_tokens = hf_input_ids[0][valid_mask]  # (n_real,)

    # Build a length-77 tensor
    padded = torch.zeros((1, 77), dtype=torch.long)
    padded[0, :real_tokens.numel()] = real_tokens

    def patched_call(self, texts, context_length=77, **kwargs):
        # Ignore the input list of strings; always return our padded tokens
        # sized to the requested context_length.
        device = next(rf_model.parameters()).device
        out = torch.zeros((1, context_length), dtype=torch.long, device=device)
        n = min(real_tokens.numel(), context_length)
        out[0, :n] = real_tokens[:n].to(device)
        return out

    SimpleTokenizer.__call__ = patched_call
    return orig_call


def run_hf(pixel_values, input_ids, attention_mask, original_size, prompt):
    from transformers import Sam3Model, Sam3Processor
    proc = Sam3Processor.from_pretrained("facebook/sam3", token=os.environ["HF_TOKEN"])
    model = Sam3Model.from_pretrained(
        "facebook/sam3", token=os.environ["HF_TOKEN"],
    ).to("cuda").eval()
    with torch.inference_mode():
        out = model(
            pixel_values=pixel_values.to("cuda"),
            input_ids=input_ids.to("cuda"),
            attention_mask=attention_mask.to("cuda"),
        )
    # Don't post-process here — we want raw outputs
    results = {
        "pred_logits": out.pred_logits.float().cpu(),
        "pred_boxes": out.pred_boxes.float().cpu(),
        "pred_masks": out.pred_masks.float().cpu(),
        "presence_logits": out.presence_logits.float().cpu(),
    }
    del model
    return results


def run_rf(pixel_values, input_ids, attention_mask, original_size, prompt):
    from inference.models.sam3.segment_anything3 import SegmentAnything3

    m = SegmentAnything3(
        model_id="sam3/sam3_final", api_key=os.environ["ROBOFLOW_API_KEY"],
    )
    h, w = original_size
    # Monkey-patch the tokenizer so it returns HF's tokens
    orig_tokenize = hf_tokens_via_monkey_patch(m.model, input_ids, attention_mask)

    batch = build_rf_batch_from_hf_tensors(
        pixel_values, input_ids, attention_mask, prompt, h=h, w=w,
    )
    # Move batch to cuda
    from sam3.model.utils.misc import copy_data_to_device
    batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            raw = m.model(batch)

    # Restore tokenizer
    from sam3.model.tokenizer_ve import SimpleTokenizer
    SimpleTokenizer.__call__ = orig_tokenize

    # raw is a list of length 1 (one decoder trajectory). Each element is a
    # dict; we want the last-layer outputs.
    out = raw[0]
    results = {
        "pred_logits": out["pred_logits"].float().cpu(),
        "pred_boxes": out["pred_boxes"].float().cpu() if "pred_boxes" in out else None,
        "pred_masks": out["pred_masks"].float().cpu(),
        "presence_logits": out.get("presence_logit_dec", torch.tensor([0.0])).float().cpu(),
    }
    return results


def _apply_hf_postprocess(raw, target_size):
    """Run HF's Sam3Processor.post_process_instance_segmentation on a
    raw dict produced by either model. target_size is (h, w) for the
    original image."""
    from transformers import Sam3Processor
    proc = Sam3Processor.from_pretrained("facebook/sam3", token=os.environ["HF_TOKEN"])

    # Normalize shapes: HF pred_logits is (B, Q), RF is (B, Q, 1). Squeeze last.
    pl = raw["pred_logits"]
    if pl.ndim == 3 and pl.shape[-1] == 1:
        pl = pl.squeeze(-1)
    # RF presence_logit_dec may be shape (B, 1) and HF presence_logits is (B, 1)
    prl = raw["presence_logits"]
    if prl.ndim == 1:
        prl = prl.unsqueeze(0)

    # HF expects pred_masks shape (B, Q, H, W). RF gives the same.
    pm = raw["pred_masks"]
    if pm.ndim == 3:
        pm = pm.unsqueeze(0)

    obj = SimpleNamespace(
        pred_logits=pl.to("cuda"),
        pred_boxes=raw["pred_boxes"].to("cuda") if raw["pred_boxes"] is not None else None,
        pred_masks=pm.to("cuda"),
        presence_logits=prl.to("cuda"),
    )
    results = proc.post_process_instance_segmentation(
        obj,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=[target_size],
    )
    return results[0]


def main() -> int:
    IMG = Path(os.environ.get("SAM3_ASSETS", "tests/workflows/integration_tests/execution/assets") + "/dogs.jpg")
    PROMPT = "dog"

    image = Image.open(IMG).convert("RGB")
    H, W = image.size[1], image.size[0]
    print(f"Image: {IMG.name} ({W}x{H}), prompt={PROMPT!r}")

    # Single preprocessing via HF
    from transformers import Sam3Processor
    proc = Sam3Processor.from_pretrained("facebook/sam3", token=os.environ["HF_TOKEN"])
    inputs = proc(images=image, text=PROMPT, return_tensors="pt")
    print(f"HF pixel_values: {tuple(inputs['pixel_values'].shape)}")
    print(f"HF input_ids: {inputs['input_ids'][0][:6].tolist()}...")
    print(f"HF attention_mask sum: {inputs['attention_mask'].sum().item()}")

    print("\n=== Running HF model with HF inputs ===")
    hf_raw = run_hf(inputs["pixel_values"], inputs["input_ids"],
                    inputs["attention_mask"], (H, W), PROMPT)
    print(f"  HF pred_logits std: {hf_raw['pred_logits'].std():.4f} "
          f"range=[{hf_raw['pred_logits'].min():.4f}..{hf_raw['pred_logits'].max():.4f}]")

    import gc
    gc.collect(); torch.cuda.empty_cache()

    print("\n=== Running Roboflow model with HF-preprocessed inputs + HF tokens ===")
    rf_raw = run_rf(inputs["pixel_values"], inputs["input_ids"],
                    inputs["attention_mask"], (H, W), PROMPT)
    print(f"  RF pred_logits std: {rf_raw['pred_logits'].std():.4f} "
          f"range=[{rf_raw['pred_logits'].min():.4f}..{rf_raw['pred_logits'].max():.4f}]")

    gc.collect(); torch.cuda.empty_cache()

    # Raw-output comparison (logit level)
    print("\n=== Raw-output comparison (identical preprocessing) ===")
    for k in ["pred_logits", "pred_boxes", "pred_masks", "presence_logits"]:
        hv = hf_raw[k]; rv = rf_raw[k]
        if hv is None or rv is None:
            print(f"  {k}: one side is None, skipping")
            continue
        # Squeeze trailing singletons for comparison
        hv_s = hv.squeeze(-1) if hv.ndim > 1 and hv.shape[-1] == 1 else hv
        rv_s = rv.squeeze(-1) if rv.ndim > 1 and rv.shape[-1] == 1 else rv
        if hv_s.shape != rv_s.shape:
            print(f"  {k}: SHAPE MISMATCH hf={tuple(hv.shape)} rf={tuple(rv.shape)}")
            continue
        diff = (hv_s - rv_s).abs()
        cos = float((hv_s.flatten() @ rv_s.flatten())
                    / (hv_s.flatten().norm() * rv_s.flatten().norm() + 1e-12))
        shape_str = str(tuple(hv_s.shape))
        print(f"  {k:17s} shape={shape_str:20s} "
              f"max|Δ|={diff.max().item():.4g} "
              f"mean|Δ|={diff.mean().item():.4g} cos={cos:.6f}")

    # Unified post-processing
    print("\n=== Unified post-processing (HF post_process on both) ===")
    hf_results = _apply_hf_postprocess(hf_raw, (H, W))
    rf_results = _apply_hf_postprocess(rf_raw, (H, W))
    print(f"  HF post->HF n={len(hf_results['scores'])}  scores={hf_results['scores'].cpu().tolist()[:5]}")
    print(f"  RF post->HF n={len(rf_results['scores'])}  scores={rf_results['scores'].cpu().tolist()[:5]}")

    # Match masks
    hf_masks = hf_results["masks"].cpu().numpy().astype(np.uint8)
    rf_masks = rf_results["masks"].cpu().numpy().astype(np.uint8)

    def _iou(a, b):
        if a.shape != b.shape: return 0.0
        a = (a > 0).astype(np.uint8); b = (b > 0).astype(np.uint8)
        i = np.logical_and(a, b).sum(); u = np.logical_or(a, b).sum()
        return float(i) / float(u) if u > 0 else 0.0

    if len(hf_masks) and len(rf_masks):
        pairs = [(i, j, _iou(hf_masks[i], rf_masks[j]))
                 for i in range(len(hf_masks)) for j in range(len(rf_masks))]
        pairs.sort(key=lambda p: -p[2])
        used_i, used_j = set(), set()
        print(f"\n  matched pairs (greedy by IoU):")
        for i, j, iou in pairs:
            if i in used_i or j in used_j: continue
            used_i.add(i); used_j.add(j)
            print(f"    hf={i} rf={j} iou={iou:.4f}  "
                  f"hf_score={float(hf_results['scores'][i]):.4f} "
                  f"rf_score={float(rf_results['scores'][j]):.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
