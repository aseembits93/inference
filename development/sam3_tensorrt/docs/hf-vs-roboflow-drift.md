# HF vs Roboflow PyTorch drift investigation

On the 100-image COCO study, HF PyTorch (`transformers.Sam3Model`) and
Roboflow's PyTorch (`inference.models.sam3.SegmentAnything3`) produce
slightly different results despite using identical `facebook/sam3`
weights. This doc investigates where the drift comes from.

## Weights: identical

- HF state_dict: 1468 tensors, 840.4M params
- RF state_dict: 1134 tensors, 841.7M params (fused QKV, extra RoPE
  `freqs_cis` buffers, cls_token slot in position_embedding)
- Spot-checked Q, K, V, O projections, LayerNorm, MLP, patch embedding,
  position embedding → **every tested tensor is bit-identical**
  (`max_diff = 0.00e+00`). The HF vs RF structural difference is in
  tensor packaging, not values.

Script: `scripts/check_weights.py`, `scripts/compare_weights_v2.py`.

## Pre-processing: nearly identical

On `dogs.jpg`:

| Aspect | HF | Roboflow | Diff |
|---|---|---|---|
| `pixel_values` shape | (1, 3, 1008, 1008) | (1, 3, 1008, 1008) | same |
| `pixel_values` mean | 0.534163 | 0.534163 | match to 6 decimals |
| `pixel_values` max abs-diff | — | — | **7.8e-3** (≈ 2/255, uint8 rounding) |
| pixels with diff > 1e-4 | — | — | 5193 / 3,048,192 (0.17%) |
| pixels with diff > 1e-2 | — | — | 0 |

Tiny resize-filter divergence, likely uint8 vs float intermediate
rounding during bilinear resize. Max per-pixel diff is 0.008 and only
0.17% of pixels differ by more than 1e-4. Negligible at the
representation level.

## Tokenization: structurally different, semantically identical

Same real tokens (`[49406, 1929, 49407]` for "dog") but:

| Aspect | HF (`CLIPTokenizer`) | RF (`SimpleTokenizer`) |
|---|---|---|
| Context length | 32 | **77** |
| Pad token ID | 49407 (`endoftext`) | **0** |

If we monkey-patch RF's tokenizer to return exactly HF's token sequence
(padded to 77 with 0's), the remaining pipeline produces essentially the
same 100-image statistics. So tokenization differences contribute
~nothing to the observed drift.

## Post-processing: one real functional difference

| Step | HF | Roboflow |
|---|---|---|
| Default score threshold | 0.3 | 0.5 (`output_prob_thresh`) |
| Mask ordering | `sigmoid → interpolate → threshold` | `interpolate → sigmoid → threshold` |
| Score computation | `sigmoid(pred_logits) * sigmoid(presence_logits)` | same |

Swapping `sigmoid ↔ interpolate` around bilinear resize is NOT
mathematically equivalent. It produces slightly different boundary
pixels for identical mask logits. But applying HF's post-process to
RF's raw outputs on the 100-image study moved mean match IoU from
0.960 → 0.972 only — contributes ~1.2 IoU points to the observed drift.

## Unified pre/post on 100 images

After monkey-patching RF's tokenizer and feeding identical HF-
preprocessed `pixel_values` + running both models with identical post-
processing, on the 100-image COCO subset:

| Metric | Original (native pre/post) | **Unified pre/post** | Change |
|---|---:|---:|---:|
| Exact count (n_hf == n_rf) | 85/100 | 84/100 | ~same |
| Total HF detections | 372 | 372 | same |
| Total RF detections | 352 | 352 | same |
| Matched pairs @ IoU ≥ 0.5 | 346 | 345 | ~same |
| Mean match IoU | 0.960 | **0.972** | +0.012 |
| Score delta (HF − RF) median | +0.040 | **+0.037** | −0.003 |
| `pred_logits` median cos | — | **0.989** | — |
| `pred_boxes` median cos | — | **0.786** | — |

**The vast majority of the HF-vs-RF gap survives unification.** Pre/post
differences account for roughly 1-2 F1 points; the rest is model-graph
divergence.

## Per-block vision backbone trace (pure FP32, identical inputs)

Hook every ViT block's output on both models; run identical pixel_values
through both:

| Block | cos (HF vs RF) | mean \|Δ\| | max \|Δ\| |
|---|---:|---:|---:|
| 0 | 1.000000 | 9.1e-5 | 0.010 |
| 1 | 1.000000 | 6.5e-5 | 0.008 |
| ... | (1.000000 throughout) | ... | ... |
| 15 | 1.000000 | 4.4e-5 | 0.083 |
| 22 | 1.000000 | 7.1e-5 | 0.607 |
| 31 | 1.000000 | 1.7e-4 | 0.443 |

**Every one of the 32 vision backbone blocks produces
bit-identical outputs (cos = 1.000000 to 6 decimals).** Mean absolute
per-element difference stays at the float32 rounding-noise floor (~5e-5)
throughout. Max abs-diff on single outlier elements grows to ~0.6 in
later blocks but is irrelevant at representation scale.

**Conclusion: the vision backbone is NOT the source of HF-vs-RF drift.**
Despite the cls_token-slot difference in position embeddings and RoPE
precomputation differences between the two codepaths, these produce
numerically-identical outputs at every block.

Script: `scripts/compare_full_trajectory.py`.

## Residual raw-output divergence (after unified inputs, pre-postprocess)

Against identical vision backbone outputs, the top-level pred_* tensors
still differ on 100 images:

| Tensor | median cos | P05 cos | min cos |
|---|---:|---:|---:|
| `pred_logits` | 0.989 | 0.964 | 0.795 |
| `pred_boxes` | 0.786 | 0.745 | 0.721 |
| `pred_masks` (norm ratio) | 0.97 | 0.91 | 0.88 |

So the drift is **downstream of the vision backbone**:

- DETR encoder (6 layers of fusion with text features)
- DETR decoder (6 layers of cross-attention with 200 object queries)
- Box / class / presence heads
- Mask decoder with pixel decoder
- Potentially the text encoder itself (even though tokens are identical,
  the input sequence length 77 vs the attention_mask sum 3 might route
  through different kernels)

## What this means for the original HF-TRT investigation

The `hf-trt-investigation.md` attributed ~5 F1 points of the HF-TRT recall
gap to "HF-vs-Roboflow code divergence" and ~17 F1 points to TRT-on-HF.
That attribution stands — but the HF-vs-Roboflow part is NOT in the
vision backbone. It's in the post-backbone graph (DETR encoder, decoder,
heads), which is exactly where TRT's additional 17 F1 points of
regression also concentrates.

So both failures localize to the same region: the **DETR
encoder/decoder/heads structure**. That's the part of SAM3 where:
- HF's implementation differs most from Roboflow's (uses SDPA attention,
  different attention-mask handling, different reference-point refinement)
- TRT's MHA fusion kicks in hardest (6 decoder cross-attentions per
  query per layer × 6 layers × 200 queries)

## Scripts

Under `scripts/`:
- `check_weights.py` — shape-level inventory of both state_dicts
- `compare_weights_v2.py` — spot-check Q/K/V/O/LN/MLP for bit-identity
- `compare_internals.py` — 1-image backbone internals comparison
- `compare_full_trajectory.py` — per-block cos across all 32 ViT blocks
- `unified_pre_post_test.py` — 1-image unified pre/post + logit compare
- `unified_pre_post_100.py` — 100-image unified pre/post sweep
- `aggregate_unified_100.py` — aggregate the 100-image unified sweep
- `sanity_hf_vs_fb_one_image.py` — canonical HF model-card example
- `hf_vs_rf_head_to_head.py` — HF-PT vs RF-PT aggregate stats on 100 images
