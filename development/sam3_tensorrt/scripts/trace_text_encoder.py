#!/usr/bin/env python3
"""Compare text encoder outputs between HF and RF.

Both should produce [batch, seq, 256] embeddings for "dog" + padding.
Different architectures (CLIP text model + projection in HF,
VETextEncoder with TextTransformer + Linear in RF) but same weights.
"""

from __future__ import annotations

import os
import sys

# requires HF_TOKEN env var
# requires ROBOFLOW_API_KEY env var

import torch


def main() -> int:
    from transformers import Sam3Processor, Sam3Model
    proc = Sam3Processor.from_pretrained("facebook/sam3", token=os.environ["HF_TOKEN"])
    inputs = proc(images=None, text="dog", return_tensors="pt")
    print(f"HF input_ids: {inputs['input_ids'][0].tolist()}")
    print(f"HF attention_mask: {inputs['attention_mask'][0].tolist()}")

    # HF text encoder
    print("\nRunning HF text encoder ...")
    hf = Sam3Model.from_pretrained("facebook/sam3", token=os.environ["HF_TOKEN"]).to("cuda").eval()
    with torch.inference_mode():
        # HF forward: vision_encoder sees image, text_encoder sees text
        # Call text path directly — pull out text_encoder(input_ids, attention_mask)
        te_out = hf.text_encoder(
            input_ids=inputs["input_ids"].to("cuda"),
            attention_mask=inputs["attention_mask"].to("cuda"),
        )
        # Sam3 uses last_hidden_state -> text_projection
        last_hidden = te_out.last_hidden_state  # (1, 32, 512)
        projected = hf.text_projection(last_hidden)  # (1, 32, 256)
        pooler = te_out.pooler_output if hasattr(te_out, "pooler_output") else None
    print(f"HF text last_hidden: {tuple(last_hidden.shape)} mean={last_hidden.mean():.4f} std={last_hidden.std():.4f}")
    print(f"HF text projected:   {tuple(projected.shape)} mean={projected.mean():.4f} std={projected.std():.4f}")
    if pooler is not None:
        print(f"HF text pooler:      {tuple(pooler.shape)} mean={pooler.mean():.4f} std={pooler.std():.4f}")
    # Show first few rows of projected
    print(f"HF projected[0, 0, :5]: {projected[0, 0, :5].cpu().tolist()}")
    print(f"HF projected[0, 1, :5]: {projected[0, 1, :5].cpu().tolist()}")  # "dog"
    print(f"HF projected[0, 2, :5]: {projected[0, 2, :5].cpu().tolist()}")  # EOS

    del hf
    import gc; gc.collect(); torch.cuda.empty_cache()

    # RF text encoder
    print("\nRunning RF text encoder ...")
    from inference.models.sam3.segment_anything3 import SegmentAnything3
    rf = SegmentAnything3(model_id="sam3/sam3_final", api_key=os.environ["ROBOFLOW_API_KEY"])
    lb = rf.model.backbone.language_backbone

    # Monkey-patch to use HF's tokens
    from sam3.model.tokenizer_ve import SimpleTokenizer
    orig_call = SimpleTokenizer.__call__
    def patched(self, texts, context_length=77, **kwargs):
        device = next(rf.model.parameters()).device
        ids = inputs["input_ids"][0]
        mask = inputs["attention_mask"][0]
        real = ids[mask.bool()]
        out = torch.zeros((1, context_length), dtype=torch.long, device=device)
        n = min(real.numel(), context_length)
        out[0, :n] = real[:n].to(device)
        return out
    SimpleTokenizer.__call__ = patched

    try:
        with torch.inference_mode():
            # VETextEncoder.forward signature: (text: Union[List[str], Tuple[Tensor, Tensor, dict]], ...)
            text_attention_mask, text_memory, text_embeds = lb(["dog"], device=torch.device("cuda"))
    finally:
        SimpleTokenizer.__call__ = orig_call

    print(f"RF text_attention_mask: {tuple(text_attention_mask.shape)} sum={text_attention_mask.sum()}")
    print(f"RF text_memory: {tuple(text_memory.shape)} mean={text_memory.mean():.4f} std={text_memory.std():.4f}")
    print(f"RF text_embeds: {tuple(text_embeds.shape)} mean={text_embeds.mean():.4f} std={text_embeds.std():.4f}")

    # text_memory is likely (77, 1, 256), text_embeds is (1, 256)?
    # Compare RF text_memory (aligned first 3 real tokens) to HF projected (first 3)
    if text_memory.ndim == 3 and text_memory.shape[0] == 77:
        rf_real = text_memory[:3, 0, :]   # (3, 256)
    elif text_memory.ndim == 3 and text_memory.shape[1] == 77:
        rf_real = text_memory[0, :3, :]
    else:
        rf_real = None
    if rf_real is not None:
        hf_real = projected[0, :3, :]  # (3, 256)
        print(f"\nHF real tokens[0..2]: shape {tuple(hf_real.shape)}")
        print(f"RF real tokens[0..2]: shape {tuple(rf_real.shape)}")
        d = (hf_real.cpu() - rf_real.cpu()).abs()
        cos = float((hf_real.cpu().flatten() @ rf_real.cpu().flatten()) /
                    (hf_real.cpu().flatten().norm() * rf_real.cpu().flatten().norm() + 1e-12))
        print(f"HF[0..2] vs RF[0..2]:  cos={cos:.6f}  max|Δ|={d.max():.4e}  mean|Δ|={d.mean():.4e}")
        # Per-token
        for t in range(3):
            a = hf_real[t].cpu(); b = rf_real[t].cpu()
            c = float(a @ b / (a.norm() * b.norm() + 1e-12))
            print(f"  token[{t}]: cos={c:.6f}, HF[:5]={a[:5].tolist()}, RF[:5]={b[:5].tolist()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
