# HF-TRT with the cls_embed fix applied — does the TRT gap shrink?

After identifying that HF's text-only path misses the `geometry_encoder`
(and its `cls_embed`) contribution to the prompt (see
[`hf-drift-root-cause.md`](hf-drift-root-cause.md)), the natural next
question is: if we re-export HF with the fix baked in and rebuild the
TRT engine, does the 22% HF-TRT recall gap shrink?

## The fix at export time

[`export_hf_sam3_fixed.py`](../scripts/export_hf_sam3_fixed.py) wraps
`Sam3Model` in a thin module that always passes a dummy padding box
(`input_boxes = zeros(1, 1, 4)`, `input_boxes_labels = [[-10]]`) to
the underlying model. The wrapper exposes the same 3 inputs and 5
outputs as the original export, so the runtime adapter is unchanged.

The sanity check at export time confirms the fix takes effect: the
wrapper's `pred_logits.std()` is 1.009, matching HF-PT-with-dummy-box
rather than HF-PT-native (1.097).

TRT engine built from this ONNX is `sam3_hf_fp16.engine` under
`sam3_hf_onnx_fixed/`. Build requires
`trt.init_libnvinfer_plugins(logger, "")` because the geometry_encoder
uses `RoIAlign` which needs TRT's plugin registry initialized.

## Raw-output comparison on dogs.jpg

HF-TRT-fixed vs HF-PT-with-dummy-box (its correct PT reference):

| Tensor | cos | TRT std | PT std | std ratio |
|---|---:|---:|---:|---:|
| `pred_logits` | 0.962 | 0.84 | 1.01 | 0.83 |
| `pred_boxes` | 0.964 | 0.22 | 0.25 | 0.86 |
| `pred_masks` | 0.875 | 8.4 | 10.2 | 0.82 |
| `presence_logits` | 1.000 | — | — | — |

TRT is **still compressing all outputs by ~17% in std and losing
~4% cos similarity** even against the correct PT reference. This
matches the TRT-on-HF FP16 regression we identified earlier — and the
fix does NOT reduce it.

## 100-image benchmark, all HF variants

All rows compared against Roboflow PT-bf16 as reference:

| Config | Recall | Precision | F1 | Mean match IoU | Score Δ median | Silent fail |
|---|---:|---:|---:|---:|---:|---:|
| HF-PT (native, no cls_embed) | 98.3% | 93.0% | 95.6% | 0.960 | +0.040 | 0 |
| HF-PT (with dummy box, cls_embed fixed) | **98.9%** | **98.6%** | **98.7%** | **0.979** | +0.001 | 0 |
| HF-TRT (original, no cls_embed) | 78.4% | 88.5% | 83.1% | 0.879 | −0.081 | 1 |
| **HF-TRT (with dummy box baked into ONNX)** | **75.3%** | **90.8%** | **82.3%** | **0.890** | **−0.092** | **3** |

## The unexpected result

**HF-TRT with the fix applied has slightly *lower* F1 than without**
(82.3% vs 83.1%). The fix helps PT by +3.1 F1 points; it hurts TRT by
−0.8 F1 points.

Why? The `geometry_encoder` path adds a cls_embed token whose
production requires extra computation: `boxes_pool_project`,
`boxes_pos_enc_project`, `final_proj`, and a DETR-style `encode`
ModuleList that cross-attends over image features. When those run in
FP16 through TRT's graph execution, they accumulate additional
precision error that partially offsets the benefit of having the
cls_embed in the prompt. In PT the computation is numerically clean,
so the fix is a pure win.

The per-class picture confirms this: HF-TRT-fixed's worst classes
(`handbag` 45%, `book` 51%, `suitcase` 52%) overlap the same
multi-instance scenes where the original HF-TRT struggled. The fix
doesn't move the failure mode; it just shifts which queries lose
under the TRT FP16 regression.

## Revised decomposition of the HF-TRT gap

| Source | F1 contribution |
|---|---:|
| HF-PT native → HF-PT with cls_embed fix | +3.1 (from 95.6% to 98.7%) |
| PT → TRT FP16 execution regression | −16.4 (HF-PT-fixed 98.7% → HF-TRT-fixed 82.3%) |
| **Net HF-TRT vs Roboflow-PT** | **−16.4 F1 points** |

Previously we attributed ~5 F1 points to HF-vs-Roboflow PT drift, but
that number included the cls_embed effect that actually benefits PT.
With the fix, PT is ~equal to Roboflow — so **the entire HF-TRT gap
is now TRT's fault, not HF-PT's code path**.

## Implications

1. **Don't bother patching HF to always run geometry_encoder for the
   TRT use case.** The fix recovers PT quality but not TRT quality.
2. The TRT FP16 regression on HF's SAM3 graph is the root-root cause.
   It's not pre/post-processing, not the ONNX shape mismatch, not
   the MHA fusion pattern, not per-layer precision choices. All
   these were ruled out by earlier experiments.
3. The gap appears to be in how TRT lowers the HF-specific attention
   pattern (using `torch.nn.functional.scaled_dot_product_attention`
   via HF's `ALL_ATTENTION_FUNCTIONS["sdpa"]`) into its internal
   kernel. Fixing it probably requires either a different attention
   implementation in HF or an upstream TRT fix.

## Scripts

- [`export_hf_sam3_fixed.py`](../scripts/export_hf_sam3_fixed.py) —
  re-export HF with dummy box baked in
- [`build_hf_sam3_fixed_engine.py`](../scripts/build_hf_sam3_fixed_engine.py)
  — build TRT FP16 engine from fixed ONNX (requires plugin init)
- [`hf_trt_fixed_raw_compare.py`](../scripts/hf_trt_fixed_raw_compare.py)
  — one-image raw-logit comparison
- [`aggregate_hf_fixed_ref.py`](../scripts/aggregate_hf_fixed_ref.py)
  — 100-image aggregate using `hf_pt_dummy_box` as reference
