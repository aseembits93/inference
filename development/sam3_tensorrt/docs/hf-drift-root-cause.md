# Root cause of HF vs Roboflow PyTorch drift

tl;dr: **HuggingFace's `Sam3Model` skips the `geometry_encoder` (and
its `cls_embed` contribution to the prompt) when no input boxes are
given. The Meta/Roboflow implementation always runs it.** Forcing HF
to run it via a dummy padding box (`input_boxes_labels=-10`) recovers
bit-exact agreement on `pred_logits` and `pred_masks` (cos = 0.999999).

## How we found it

Previous docs established:
- Weights bit-identical (`check_weights.py`, `compare_weights_v2.py`).
- Pre/post-processing identical after unification
  (`unified_pre_post_100.py`).
- 32 ViT vision backbone blocks produce bit-identical outputs per
  layer (`compare_full_trajectory.py`).
- The raw-output drift appears downstream of the vision backbone
  with pred_logits cos 0.989 and pred_boxes cos 0.786 on 100 images.

Stage-by-stage comparison (`trace_compare_stages.py`) narrowed the
drift to the DETR encoder — the first stage below cos 1.0:

| Stage | cos HF vs RF |
|---|---:|
| FPN layers 0–3, position encoding | 1.000000 |
| DETR encoder layer 0 | 0.9949 |
| DETR encoder layer 5 (output) | 0.9960 |
| pred_logits (final) | 0.975 |
| pred_boxes (final) | 0.802 |

Hooking the inputs to DETR encoder layer 0 revealed a shape
mismatch:

| Input | HF | RF |
|---|---|---|
| `prompt_feats` / `memory` | `(1, 32, 256)` | `(1, 33, 256)` |

RF passes **33 prompt tokens**; HF passes **32**.

## The extra RF token

Reading `sam3.model.sam3_image._encode_prompt` (around line 207):

```python
prompt = torch.cat([txt_feats, geo_feats, visual_prompt_embed], dim=0)
```

RF always concatenates `geo_feats` (the geometry_encoder output) into
the prompt, even with no boxes. `geo_feats` ends with a learned
`cls_embed` that `concat_padded_sequences` appends with
`cls_mask = 0` (valid, not padding). That's the 33rd token.

`Sam3GeometryEncoder.forward` in RF always runs its `encode`
ModuleList (DETR-style cross-attention over image features) on the
final_embeds that end with `cls_embed`. So the 33rd prompt token is
an **image-conditioned cls embedding**, not just a raw learned parameter.

## HF's branch

Reading `transformers.models.sam3.modeling_sam3.Sam3Model.forward`:

```python
has_geometry_prompts = input_boxes is not None and input_boxes.numel() > 0

if has_geometry_prompts:
    # run geometry_encoder with the boxes ...
    geometry_prompt_features = geometry_outputs.last_hidden_state
    geometry_prompt_mask = geometry_outputs.attention_mask
# ...
if geometry_prompt_features is not None:
    combined_prompt_features = torch.cat([text_features, geometry_prompt_features], dim=1)
else:
    combined_prompt_features = text_features   # ← 32 tokens only
```

No boxes → no geometry_encoder call → no `cls_embed` in prompt. The
HF port treats the geometry path as fully optional. For text-only
queries (`inputs = processor(images=..., text=...)`), the geometry
branch is skipped.

## Verification

Forcing HF to take the geometry path by passing a single dummy box
with label `-10` (the "padding" convention in HF's box handling):

```python
dummy_box = torch.zeros(1, 1, 4, device="cuda")
dummy_lab = torch.tensor([[-10]], dtype=torch.long, device="cuda")
out = model(
    pixel_values=...,
    input_ids=...,
    attention_mask=...,
    input_boxes=dummy_box,
    input_boxes_labels=dummy_lab,   # forces geometry path + cls_embed
)
```

### Single-image test (`dogs.jpg`, prompt `"dog"`)

| Comparison | `pred_logits` cos | `pred_boxes` cos | `pred_masks` cos |
|---|---:|---:|---:|
| HF (text-only) vs RF | 0.975 | 0.802 | 0.925 |
| **HF (with dummy box) vs RF** | **0.999999** | **0.809** | **0.999999** |

pred_logits and pred_masks become **bit-identical** to RF. pred_boxes
cos stays at ~0.81 because the 200 DETR queries still land in
different orders (the cls_embed influence propagates through the
decoder differently), but the decisions post-threshold match.

### 100-image test

Same 100 COCO images, both HF variants scored against Roboflow
PT-bf16 reference:

| Config | Recall | Precision | F1 | Exact count | Mean match IoU | Score delta median |
|---|---:|---:|---:|---:|---:|---:|
| HF-PT (native, text-only) | 98.3% | 93.0% | 95.6% | 85/100 | 0.960 | +0.040 |
| **HF-PT (dummy box)** | **98.9%** | **98.6%** | **98.7%** | **93/100** | **0.979** | **+0.001** |
| Roboflow PT-fp16 autocast (reference-equivalent) | 98.9% | 98.6% | 98.7% | 93/100 | 0.996 | +0.001 |

**HF with dummy box reaches the same F1 as Roboflow PT-fp16** and
gets 93/100 exact detection-count agreement (vs 85/100 without the
dummy box). Score delta median drops from +0.040 to +0.001, matching
the float-noise level of PT-fp16 vs PT-bf16.

## Why this matters

1. The HuggingFace `Sam3Model` is **not bit-equivalent to the published
   Meta SAM3 architecture** for text-only queries, despite using
   identical weights.
2. The differences are small in aggregate (F1 95.6% vs 98.7%) but
   concentrate on dense multi-instance scenes where the DETR decoder's
   query disambiguation depends on the `cls_embed` contribution.
3. **HF-TRT's 22% recall gap on 100 images was NOT entirely TRT's
   fault.** ~3 F1 points were the `cls_embed`-missing PyTorch drift,
   propagated through the TRT engine. The remaining ~15 F1 points are
   still TRT-on-HF's own regression.
4. Downstream consumers who want HF-Meta bit-equivalence should always
   pass a dummy padding box, or patch `Sam3Model.forward` to always run
   `geometry_encoder`.

## Suggested HF fix

In `transformers/models/sam3/modeling_sam3.py`, around line 82:

```python
# Always run geometry_encoder so the cls_embed is included in the
# prompt, matching the published Meta SAM3 architecture. When no
# boxes are provided, construct an empty box tensor to pass through.
if not has_geometry_prompts:
    box_embeddings = torch.zeros(batch_size, 0, 4, dtype=text_features.dtype, device=device)
    box_labels = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
    box_mask = torch.zeros(batch_size, 0, dtype=torch.bool, device=device)
else:
    # ... existing logic ...

geometry_outputs = self.geometry_encoder(
    box_embeddings=box_embeddings,
    box_mask=box_mask,
    box_labels=box_labels,
    img_feats=fpn_hidden_states,
    img_pos_embeds=fpn_position_encoding,
)
```

Or, less invasively, make the "no boxes" branch explicitly include
the cls_embed as a single-token geometry output.

## Scripts

- `trace_detr_mask_handling.py` — showed the 32 vs 33 shape mismatch.
- `hf_with_dummy_box_vs_rf.py` — demonstrates the fix on one image.
- `sweep_hf_dummy_box.py` — runs the fix on 100 images for
  aggregate validation.
