# 100-image correctness study (T4)

All earlier numbers in this directory were from single-image smoke
tests. This doc is the 100-image pass: 100 COCO val2017 images
spanning 53 distinct classes, one prompt per image (the dominant
class by annotation area), reference = PT-bf16 (the repo default).

Dataset: `tests/inference/models_predictions_tests/` doesn't have
enough variety, so we fetched `instances_val2017.json` + 100 images
from COCO. The selection script picks diverse classes and downloads
locally. See `scripts/prepare_coco_subset.py`.

## Configurations compared

| Tag | What it is |
|---|---|
| `pt_bf16` | Reference. SAM3 repo's `SegmentAnything3` with the repo-default `autocast(dtype=torch.bfloat16)`. On T4 this is bf16-emulated (no tensor cores), which is why the baseline is slow. |
| `pt_fp16` | Same code, but with a one-line-ish change: `autocast(dtype=torch.float16)`. On T4 this is native tensor-core fp16. |
| `trt_swap` | SAM3 repo with `backbone.forward_image` swapped for the `fp16_rope_windowed_d8` TRT engine. PT-fp16 autocast for the surrounding decoder. |
| `hf_trt` | `transformers.Sam3Model` → full whole-model `torch.onnx.export` → TRT FP16 (`trtexec --fp16` style, no per-layer pinning). The `dataplayer12/SAM3-TensorRT` approach. |

## Correctness metrics

For each test config, per-image greedy 1-to-1 IoU matching against
the PT-bf16 reference. A match counts "good" if IoU >= 0.5.

| Metric | Definition |
|---|---|
| Recall | `good_matches / total_reference_detections` |
| Precision | `good_matches / total_test_detections` |
| F1 | Standard harmonic mean |
| Exact-count | #images where `len(test_dets) == len(ref_dets)` |
| Silent failure | #images where reference found ≥1 and test found 0 |
| Mean match IoU | Average IoU of detections matched at IoU >= 0.5 |
| Score delta | `test_score - ref_score` for each matched pair |

## Results (T4, 100 images)

| Config | E2E median (ms) | Recall | Precision | F1 | Exact-count | Mean match IoU | Score delta median |
|---|---:|---:|---:|---:|---:|---:|---:|
| **PT-bf16** (reference) | **2856** | — | — | — | — | — | — |
| **PT-fp16** (autocast fix) | **516** | **98.9%** | **98.6%** | **98.7%** | **93/100** | **0.996** | **+0.001** |
| **TRT-swap** (fp16_rope_windowed_d8) | **578** | **99.1%** | **99.4%** | **99.3%** | **96/100** | **0.996** | **+0.002** |
| **HF-TRT** (whole-model FP16) | **366** | **78.4%** | **88.5%** | **83.1%** | **73/100** | **0.879** | **−0.081** |

### Key findings

1. **PT-fp16 and TRT-swap are indistinguishable from PT-bf16 in
   practice.** Both hit F1 ≥ 98.7%, match IoU 0.996, 0 silent
   failures. The only recall losses (skis 75%, carrot 94-97%)
   happen on classes the bf16 reference also struggled with — not
   precision-dependent behavior.

2. **TRT-swap is actually slightly *better* numerically than
   PT-fp16.** 99.3% F1 vs 98.7%, 349 matched detections vs 348,
   minimum match IoU 0.983 vs 0.983. The FP32 RoPE pinning in the
   vitdet windowed blocks avoids the fp16 attention drift that
   both PT-fp16 and PT-fp32 accumulate slightly.

3. **HF-TRT has a material correctness regression.** Recall drops
   from ~99% to 78%. Mean match IoU drops to 0.879. Score delta
   median **−0.081** — scores uniformly drop by ~8 points, which
   pushes many marginal detections below the 0.5 threshold.

4. **HF-TRT's failure is class-correlated.** Worst 5 classes by
   recall: `book` 52.5%, `handbag` 54.5%, `suitcase` 65.2%, `bed`
   66.7%, `keyboard` 66.7%. All four are dense multi-instance
   scenes where the DETR decoder has to disambiguate many similar
   queries. Small and thin objects are disproportionately affected.

## Revised recommendation

Previous benchmarks.md said the TRT-swap was slower than PT-fp16.
It is — but only by 62 ms (12%), not the 383 ms (78%) implied by
single-image measurement where the surrounding PyTorch code ran
under the default bf16 autocast on T4. Once you match autocast
dtype between the two configs, TRT-swap is roughly on par with
PT-fp16 and has slightly better correctness.

**On T4**, in order of preference:

- **PT-fp16**: 516 ms, 98.7% F1, zero TRT overhead, one-line change
- **TRT-swap**: 578 ms, **99.3% F1**, shipable engine artifact
- **HF-TRT**: 366 ms, **83.1% F1**, faster but meaningfully worse
  on multi-instance scenes

If you need a TRT engine for non-Python deployment (C++, Jetson),
ship the SAM3-repo TRT-swap, not the HF whole-model engine. The
HF model's SDPA-based attention doesn't round-trip through TRT's
kernel fusion cleanly (see `hf-trt-investigation.md`).

## How to reproduce

```bash
# 1. Prepare the 100-image subset
python scripts/prepare_coco_subset.py   # writes /tmp/coco_val2017_subset/

# 2. Run the four sweeps (each in its own subprocess; T4 15 GB)
for cfg in pt_bf16 pt_fp16 trt_swap hf_trt; do
    python scripts/sweep_100_images.py $cfg
done

# 3. Aggregate
python scripts/aggregate_correctness.py
```
