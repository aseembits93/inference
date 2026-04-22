#!/usr/bin/env python3
"""Compare text encoder outputs at the 3 REAL token positions.

If HF and RF differ only because of pad positions' influence, then the
3 real-token embeddings may match. If they differ at real tokens too,
the text encoder internals truly differ.
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

    # HF
    hf = Sam3Model.from_pretrained("facebook/sam3", token=os.environ["HF_TOKEN"]).to("cuda").eval()
    with torch.inference_mode():
        te_out = hf.text_encoder(
            input_ids=inputs["input_ids"].to("cuda"),
            attention_mask=inputs["attention_mask"].to("cuda"),
        )
        hf_last = te_out.last_hidden_state  # (1, 32, 1024)
        hf_proj = hf.text_projection(hf_last)  # (1, 32, 256)
    print(f"HF last_hidden (1, 32, 1024):")
    for t in range(4):
        v = hf_last[0, t].float().cpu()
        print(f"  token {t}: mean={v.mean():.4f} std={v.std():.4f}  [:5]={v[:5].tolist()}")
    print(f"HF projected (1, 32, 256):")
    for t in range(4):
        v = hf_proj[0, t].float().cpu()
        print(f"  token {t}: mean={v.mean():.4f} std={v.std():.4f}  [:5]={v[:5].tolist()}")

    del hf; import gc; gc.collect(); torch.cuda.empty_cache()

    # RF — direct call to the TextTransformer encoder
    from inference.models.sam3.segment_anything3 import SegmentAnything3
    rf = SegmentAnything3(model_id="sam3/sam3_final", api_key=os.environ["ROBOFLOW_API_KEY"])
    lb = rf.model.backbone.language_backbone

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
            text_attn_mask, text_memory_resized, text_embeds = lb(["dog"], device=torch.device("cuda"))
    finally:
        SimpleTokenizer.__call__ = orig_call

    # text_memory_resized: (77, 1, 256) post-resizer
    # text_embeds: inputs_embeds.transpose(0, 1) = (77, 1, 1024)
    # but wait, that's supposed to be pre-encoder embeds (token_embedding output)

    print(f"\nRF text_memory_resized: {tuple(text_memory_resized.shape)}")
    for t in range(4):
        v = text_memory_resized[t, 0].float().cpu()
        print(f"  token {t}: mean={v.mean():.4f} std={v.std():.4f}  [:5]={v[:5].tolist()}")
    print(f"\nRF text_embeds (inputs_embeds, pre-encoder): {tuple(text_embeds.shape)}")
    for t in range(4):
        v = text_embeds[t, 0].float().cpu()
        print(f"  token {t}: mean={v.mean():.4f} std={v.std():.4f}  [:5]={v[:5].tolist()}")

    # Also run the encoder manually to get last_hidden
    from sam3.model.tokenizer_ve import SimpleTokenizer as ST
    ST.__call__ = patched
    try:
        with torch.inference_mode():
            tokenized = lb.tokenizer(["dog"], context_length=lb.context_length).to("cuda")
            inputs_embeds = lb.encoder.token_embedding(tokenized)
            _, text_memory = lb.encoder(tokenized)
    finally:
        ST.__call__ = orig_call

    print(f"\nRF raw encoder last_hidden: {tuple(text_memory.shape)}")
    for t in range(4):
        v = text_memory[0, t].float().cpu()
        print(f"  token {t}: mean={v.mean():.4f} std={v.std():.4f}  [:5]={v[:5].tolist()}")

    # Now compare HF vs RF on real tokens (0, 1, 2)
    print("\n=== Comparison on real tokens [0, 1, 2] ===")
    hf_real = hf_last[0, :3].float().cpu()  # (3, 1024)
    rf_real = text_memory[0, :3].float().cpu()  # (3, 1024)
    for t in range(3):
        a, b = hf_real[t], rf_real[t]
        d = (a - b).abs()
        cos = float(a @ b / (a.norm() * b.norm() + 1e-12))
        print(f"  raw token {t}: cos={cos:.6f} max|Δ|={d.max():.4e} mean|Δ|={d.mean():.4e}")

    # Projected (post text_projection / resizer)
    hf_real_p = hf_proj[0, :3].float().cpu()
    rf_real_p = text_memory_resized[:3, 0].float().cpu()
    print("Projected tokens:")
    for t in range(3):
        a, b = hf_real_p[t], rf_real_p[t]
        d = (a - b).abs()
        cos = float(a @ b / (a.norm() * b.norm() + 1e-12))
        print(f"  proj token {t}: cos={cos:.6f} max|Δ|={d.max():.4e} mean|Δ|={d.mean():.4e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
