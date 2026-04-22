#!/usr/bin/env python3
"""Better weight comparison: match tensors by semantic role, not by
sorted-order heuristic.

HF names use `vision_encoder.backbone.layers.N.attention.{q,k,v,o}_proj`
while Roboflow uses `backbone.vision_backbone.trunk.blocks.N.attn.{qkv,proj}`.
Roboflow fuses Q/K/V into a single (3H, H) weight; HF keeps them as three
(H, H) weights. We compare after un-fusing Roboflow's QKV.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

# requires HF_TOKEN env var
# requires ROBOFLOW_API_KEY env var

import torch


def main() -> int:
    print("Loading HF Sam3Model ...")
    from transformers import Sam3Model
    hf = Sam3Model.from_pretrained("facebook/sam3", token=os.environ["HF_TOKEN"]).eval()
    hf_sd = hf.state_dict()

    print("Loading Roboflow SegmentAnything3 ...")
    from inference.models.sam3.segment_anything3 import SegmentAnything3
    rf = SegmentAnything3(model_id="sam3/sam3_final", api_key=os.environ["ROBOFLOW_API_KEY"])
    rf_sd = rf.model.state_dict()

    print(f"\nHF: {len(hf_sd)} tensors, {sum(t.numel() for t in hf_sd.values()) / 1e6:.1f}M params")
    print(f"RF: {len(rf_sd)} tensors, {sum(t.numel() for t in rf_sd.values()) / 1e6:.1f}M params")

    # Look at a few concrete same-role tensor pairs
    # 1. vision_encoder.backbone.layers.0.attention.q_proj.weight (HF, shape [H, H])
    #    vs backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight (RF, shape [3H, H])
    #    Roboflow QKV = torch.cat([Q, K, V], dim=0) by convention
    print("\n--- Block 0 attention weights (unfuse RF's QKV) ---")
    hf_q = hf_sd["vision_encoder.backbone.layers.0.attention.q_proj.weight"]
    hf_k = hf_sd["vision_encoder.backbone.layers.0.attention.k_proj.weight"]
    hf_v = hf_sd["vision_encoder.backbone.layers.0.attention.v_proj.weight"]
    rf_qkv = rf_sd["backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight"]
    H = hf_q.shape[0]
    rf_q, rf_k, rf_v = rf_qkv[:H], rf_qkv[H:2*H], rf_qkv[2*H:]

    def cmp(name, a, b):
        a = a.detach().float().cpu()
        b = b.detach().float().cpu()
        if a.shape != b.shape:
            print(f"  {name}: SHAPE MISMATCH {tuple(a.shape)} vs {tuple(b.shape)}")
            return
        d = (a - b).abs()
        max_d = float(d.max())
        mean_d = float(d.mean())
        rel = mean_d / (a.abs().mean().item() + 1e-12)
        na, nb = a.flatten().norm(), b.flatten().norm()
        cos = float((a.flatten() @ b.flatten()) / (na * nb + 1e-12))
        print(f"  {name:20s}: max_diff={max_d:.6e}  mean_diff={mean_d:.6e}  "
              f"rel={rel:.4e}  cos={cos:.8f}")

    cmp("Q weight", hf_q, rf_q)
    cmp("K weight", hf_k, rf_k)
    cmp("V weight", hf_v, rf_v)

    # Try the other 5 permutations just in case QKV order differs
    print("\n  try other orderings of RF QKV:")
    orderings = [
        ("QKV", (rf_qkv[:H], rf_qkv[H:2*H], rf_qkv[2*H:])),
        ("QVK", (rf_qkv[:H], rf_qkv[2*H:], rf_qkv[H:2*H])),
        ("KQV", (rf_qkv[H:2*H], rf_qkv[:H], rf_qkv[2*H:])),
        ("KVQ", (rf_qkv[H:2*H], rf_qkv[2*H:], rf_qkv[:H])),
        ("VKQ", (rf_qkv[2*H:], rf_qkv[H:2*H], rf_qkv[:H])),
        ("VQK", (rf_qkv[2*H:], rf_qkv[:H], rf_qkv[H:2*H])),
    ]
    for label, (q, k, v) in orderings:
        dq = (hf_q.float().cpu() - q.float().cpu()).abs().mean().item()
        dk = (hf_k.float().cpu() - k.float().cpu()).abs().mean().item()
        dv = (hf_v.float().cpu() - v.float().cpu()).abs().mean().item()
        total = dq + dk + dv
        print(f"    {label}:  |dq|={dq:.4e}  |dk|={dk:.4e}  |dv|={dv:.4e}  total={total:.4e}")

    # 2. output projection
    print("\n--- Block 0 attention output projection ---")
    hf_o = hf_sd["vision_encoder.backbone.layers.0.attention.o_proj.weight"]
    rf_o = rf_sd["backbone.vision_backbone.trunk.blocks.0.attn.proj.weight"]
    cmp("o_proj weight", hf_o, rf_o)

    # 3. layer norm
    print("\n--- Block 0 LayerNorm 1 ---")
    cmp("norm1.weight",
        hf_sd["vision_encoder.backbone.layers.0.layer_norm1.weight"],
        rf_sd["backbone.vision_backbone.trunk.blocks.0.norm1.weight"])
    cmp("norm1.bias",
        hf_sd["vision_encoder.backbone.layers.0.layer_norm1.bias"],
        rf_sd["backbone.vision_backbone.trunk.blocks.0.norm1.bias"])

    # 4. Patch embedding
    print("\n--- Patch embedding projection ---")
    cmp("patch.weight",
        hf_sd["vision_encoder.backbone.embeddings.patch_embeddings.projection.weight"],
        rf_sd["backbone.vision_backbone.trunk.patch_embed.proj.weight"])
    cmp("patch.bias",
        hf_sd["vision_encoder.backbone.embeddings.patch_embeddings.projection.bias"],
        rf_sd["backbone.vision_backbone.trunk.patch_embed.proj.bias"])

    # 5. Position embeddings
    print("\n--- Position embeddings ---")
    hf_pe = hf_sd["vision_encoder.backbone.embeddings.position_embeddings"]
    rf_pe = rf_sd["backbone.vision_backbone.trunk.pos_embed"]
    print(f"  HF shape: {tuple(hf_pe.shape)}, RF shape: {tuple(rf_pe.shape)}")
    # RF has 1 extra token (cls_token); skip first or last token to compare
    if hf_pe.shape[1] + 1 == rf_pe.shape[1]:
        cmp("pos_embed (RF[:, 1:])", hf_pe, rf_pe[:, 1:])
        cmp("pos_embed (RF[:, :-1])", hf_pe, rf_pe[:, :-1])

    # 6. MLP
    print("\n--- Block 0 MLP ---")
    # HF: vision_encoder.backbone.layers.0.mlp.fc1 / fc2
    # RF: backbone.vision_backbone.trunk.blocks.0.mlp.layers.0 / layers.1 (depending on naming)
    print(f"  HF MLP keys: {[k for k in hf_sd if 'layers.0.mlp' in k and 'vision_encoder' in k][:4]}")
    print(f"  RF MLP keys: {[k for k in rf_sd if 'blocks.0.mlp' in k and 'vision_backbone' in k][:4]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
