# SAM3 benchmark summary (T4, 100 images)

Full results from the 100-image COCO subset benchmark, all configs in
one table. Companion to [`100-image-study.md`](100-image-study.md) and
[`hf-trt-investigation.md`](hf-trt-investigation.md); this doc is the
cheat sheet.

**Reference:** Roboflow PT-bf16 (`SegmentAnything3` with repo-default
`autocast(dtype=torch.bfloat16)`).
**Metric:** greedy 1-to-1 IoU matching, "good" match at IoU ≥ 0.5.
**Weights:** all configs use identical `facebook/sam3` weights. HF
and Roboflow ship the same checkpoint; the differences are in
PyTorch code structure (fused vs separate QKV, cls_token handling,
RoPE implementation, attention dispatch).

## Full table

| Config | E2E median (ms) | Speedup vs PT-bf16 | Recall | Precision | F1 | Mean match IoU | Score delta (median) | Silent fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Roboflow PT-bf16** (reference) | 2856 | 1.00× | — | — | — | — | — | — |
| Roboflow PT-fp32 | 1744 | 1.64× | — | — | — | — | — | — |
| **Roboflow PT-fp16** (autocast fix) | **516** | **5.53×** | **98.9%** | **98.6%** | **98.7%** | **0.996** | **+0.001** | **0/100** |
| **TRT-swap** (`fp16_rope_windowed_d8`) | **578** | **4.94×** | **99.1%** | **99.4%** | **99.3%** | **0.996** | **+0.002** | **0/100** |
| HF PyTorch (canonical, no TRT) | 1800 | 1.59× | 98.3% | 93.0% | 95.6% | 0.960 | +0.041 | 0/100 |
| **HF-TRT** (whole-model FP16) | **366** | **7.80×** | **78.4%** | **88.5%** | **83.1%** | **0.879** | **−0.081** | **1/100** |
| HF-TRT decoder-FP32 pinned | 390 | 7.32× | 77.6% | 88.1% | 82.5% | 0.879 | −0.088 | 1/100 |
| HF-TRT non-backbone-FP32 pinned | 547 | 5.22× | 77.6% | 87.8% | 82.4% | 0.879 | −0.088 | 1/100 |
| HF-TRT attn-FP32 pinned | 902 | 3.17× | 77.8% | 88.1% | 82.7% | 0.879 | −0.089 | 1/100 |
| HF-TRT pure FP32 | 1752 | 1.63× | 78.4% | 87.9% | 82.9% | 0.879 | −0.086 | 1/100 |
| HF-TRT shape-inferred ONNX | 362 | 7.89× | 78.1% | 87.0% | 82.3% | 0.879 | −0.087 | 1/100 |
| HF-TRT decoder-nofuse (graphsurgeon) | 360 | 7.93× | 78.4% | 87.9% | 82.9% | 0.879 | −0.087 | 1/100 |
| HF-TRT all-nofuse (graphsurgeon) | 365 | 7.82× | 78.4% | 88.5% | 83.1% | 0.879 | −0.087 | 1/100 |
| HF-TRT (baseline), vs HF-PT as reference | — | — | 74.2% | 88.5% | 80.7% | 0.900 | −0.123 | 2/100 |

## Key numbers to take away

| | |
|---|---:|
| Best shipable option, full correctness | **PT-fp16 autocast — 98.7% F1 at 516 ms** |
| Best TRT engine, full correctness | **TRT-swap — 99.3% F1 at 578 ms** |
| Fastest option (but broken) | HF-TRT — 7.8× speedup but −16 F1 points on 100 images |
| HF-vs-Roboflow implementation drift (both PyTorch) | −3 F1 points, +0.04 score bias |
| TRT-on-HF additional regression (vs HF-PT) | −15 F1 points, −0.12 score compression |

## One-sentence summary per config

- **Roboflow PT-bf16**: the repo default, 2.8 s on T4 because bf16 is
  emulated on Turing (no native tensor cores).
- **Roboflow PT-fp32**: no autocast, 1.7 s, same correctness as
  PT-fp16. Included for completeness.
- **Roboflow PT-fp16**: one-line autocast dtype change, 5.5× faster
  than default, essentially zero correctness cost. Ship this.
- **TRT-swap**: SAM3-repo vision backbone swapped for a TRT engine
  with FP32 RoPE islands, same speed class as PT-fp16, slightly
  better correctness. Ship this if you need a TRT artifact (C++
  server, Jetson).
- **HF PT**: HuggingFace `Sam3Model` in PyTorch, same weights as
  Roboflow but different code path — scores run ~0.04 higher and
  finds ~5% more detections on marginal classes.
- **HF-TRT (and all 7 rescue-attempt variants)**: 22% recall regression
  vs Roboflow reference (17% regression vs HF's own PyTorch), 15% F1
  loss. Neither FP32 pinning (4 variants), shape inference, nor
  onnx-graphsurgeon MHA-break recovers it. The remaining gap is a
  TRT-10.12-specific graph-execution regression for this particular
  ONNX, not a precision issue.

## Practical recommendation (T4)

Pick based on deployment shape:

- **Python server or PyTorch-native pipeline** → use PT-fp16 autocast
  (one-line fix to `segment_anything3.py:535`).
- **Non-Python deployment or need a TRT artifact** → build and ship
  the SAM3-repo TRT-swap engine. Correctness is on par with PT-fp16
  (99.3% F1), latency is 12% higher.
- **Do not ship HF-TRT** in any form. The 22% recall loss concentrates
  on dense multi-instance scenes (`book`, `handbag`, `suitcase`, `bed`),
  which are exactly the scenes users care most about, and none of the
  tried engine-level fixes recover it.

## How to reproduce

See [100-image-study.md](100-image-study.md#how-to-reproduce). Config
aliases (`pt_fp16`, `trt_swap`, `hf_trt`, etc.) match the sweep-file
names under `/tmp/coco_sweep_<alias>.pkl`.
