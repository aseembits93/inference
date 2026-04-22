# Correctness gating

Two gates were used. The mask-IoU gate was the original; the logit-cosine
gate replaced it after the IoU gate proved to be too coarse a signal. The
docs below describe both so the shortcomings of IoU are also recorded.

## Mask IoU gate (initial)

Implemented in [`sam3_correctness_gate.py`](../scripts/sam3_correctness_gate.py) and [`bench_sam3_t4.py`](../scripts/bench_sam3_t4.py).

For each test image + prompt:

1. Run the PyTorch reference and the TRT-patched model end to end.
2. Convert predicted RLE masks back to dense boolean arrays.
3. Greedy 1-to-1 match by descending IoU across ref/test mask pairs whose
   shapes agree.
4. Report mean and min IoU of matched pairs.

Pass condition: **mean IoU >= 0.95 across all 4 images**
(`dogs.jpg/dog`, `car.jpg/car`, `crowd.jpg/person`, `multi-fruit.jpg/fruit`).

### What this catches

Pixel-level disagreement on successfully matched masks. If two runs agree
on "there is a dog here, roughly here," the gate tells you how similar the
boundaries are.

### What this misses

- **Missed/extra detections.** If the reference finds 21 people and the
  test finds 7, the gate only sees the 7 matched pairs and happily passes
  if those 7 overlap well. The 14 missed detections are invisible.
- **Score deltas.** Only mask shape is compared; confidences are not asserted.
- **Threshold-sensitive.** Masks are binary decisions; a logit shift of
  0.01 near the 0.5 threshold flips pixel classes but leaves most of the
  mask unchanged, so IoU stays near 1.0 even when model behavior changed.
- **Amplification failures collapse to "nan".** The FP16 amplification bug
  (see [precision-bug.md](precision-bug.md)) drove every logit above the
  0.5 threshold by ~2.5x, producing zero predicted masks after
  thresholding. The IoU gate reports "nan" / "0 detections, fail" -- true,
  but uninformative.
- **Only 4 images, one prompt per image.** No text-prompt variation, no
  box prompts, no multi-prompt queries, no edge cases.

## Logit cosine + std-ratio gate (improved)

Implemented in [`bench_logit_correctness.py`](../scripts/bench_logit_correctness.py) and summarized by [`final_summary.py`](../scripts/final_summary.py).

For each run, capture the raw `self.model(batch)` output (a dict of
tensors -- `pred_masks`, `pred_logits`, `semantic_seg`, `queries`,
`vision_features`, ...) *before* `PostProcessImage` ever runs. For each
tensor, compute:

- **Cosine similarity** between the flattened run output and the
  PT-bf16 reference.
- **Std ratio** (`test.std() / ref.std()`) -- catches magnitude drift.
- **Max absolute difference**.

### Proposed gate

Across all float output tensors:

- `min(cos) >= 0.95` (tight enough to catch real divergence, loose enough
  to let pure FP32 and FP16 baselines through)
- `0.7 <= std_ratio <= 1.5` (catches amplification/attenuation, including
  the 2.32x amplification of the broken FP16 engine)

### Why this is better

- **Continuous**: regressions surface as gradual drift (cos 0.999 ->
  0.997 -> 0.99) instead of the bimodal mask IoU cliff.
- **Threshold-free**: no dependency on 0.5 mask threshold.
- **Amplification-invariant**: the FP16 amplification that zeroed out
  mask IoU shows up as std ratio 2.32 and cos = -1.0 on
  `presence_logit_dec`.
- **No matching ambiguity**: no greedy matching, no 1-to-1 pairing.

### Observed values on T4

| Config | min cos | worst std ratio | Notes |
|---|---|---|---|
| PT-bf16 (reference) | 1.000 | 1.000 | by definition |
| PT-fp32 | 0.991 | 1.01 | noise floor of precision difference |
| PT-fp16 | 0.990 | 1.01 | noise floor of precision difference |
| TRT `rope_fp32_d10` | 0.995 | 1.01 | closer to bf16 than fp32 is |
| TRT `rope_windowed_d8` | 0.996 | 1.01 | best correct TRT |
| TRT `fp16` (broken) | **-1.000** | **2.32** | easily caught |

## What's not gated

- Robustness over image distribution (only 4 test images).
- Box prompts, multi-prompt queries, cross-prompt NMS paths.
- Regression over release lifetime (no snapshot of reference logits; gate
  is computed live against a fresh PyTorch run each time).

For a production gate, replace the live PyTorch reference with a pinned
snapshot, widen the test set, and add a precision / recall check on
detection counts in addition to the logit comparison.
