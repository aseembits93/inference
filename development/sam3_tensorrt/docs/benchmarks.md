# Benchmarks

All numbers are E2E `SegmentAnything3.infer_from_request()` latency on one
image (`dogs.jpg`, prompt `"dog"`), 15 iterations after 3 warmup, `time.perf_counter` around a `torch.cuda.synchronize()`. Correctness is logit cosine similarity vs PT-bf16 (see [correctness.md](correctness.md)).

## NVIDIA L4 (Ada, SM 8.9, 22 GB)

| Config | E2E (ms) | Speedup vs PT-bf16 | min cos | pred_masks cos | Notes |
|---|---|---|---|---|---|
| **PT-bf16** (repo default) | **247** | 1.00x | 1.000 | 1.000 | native bf16 tensor cores |
| PT-fp32 | ~600 | 0.41x | 0.991 | 0.993 | CUDA cores only |
| PT-fp16 | ~250 | ~1.0x | 0.990 | 0.992 | roughly matches bf16 on Ada |
| TRT `fp16` (broken)             | 189 | 1.31x | -1.000 | 0.629 | 0 detections, amplification 2.3x |
| TRT `rope_fp32_d10` | 399 | 0.62x | 0.995 | 0.996 | correct, slower than PT |
| TRT `rope_windowed_d8` | n/a | n/a | n/a | n/a | not rebuilt for L4 in this round |
| TRT `fp16_attn_hard_v2` | 438 | 0.56x | 0.999 | 0.999 | all attn FP32, safest but slowest |

On L4, PyTorch with bf16 autocast is already at the hardware roofline for
this model. Every correct TRT variant is slower than the PT baseline. The
"fast" TRT engine hits the FP16 amplification bug.

## NVIDIA T4 (Turing, SM 7.5, 15 GB)

| Config | E2E (ms) | Speedup vs PT-bf16 | min cos | pred_masks cos | pred_masks std ratio | Notes |
|---|---|---|---|---|---|---|
| **PT-bf16** (repo default) | **2786** | 1.00x | 1.000 | 1.000 | 1.000 | bf16 emulated -- no tensor cores |
| PT-fp32 | 1744 | 1.60x | 0.991 | 0.993 | 1.009 | FP32 CUDA cores |
| **PT-fp16** | **488** | **5.71x** | 0.990 | 0.992 | 1.013 | native fp16 tensor cores |
| TRT `fp16` (broken) | 748 | 3.73x | -1.000 | 0.629 | 0.963 | 0 detections |
| TRT `rope_fp32_d10` | 974 | 2.86x | 0.995 | 0.996 | 0.989 | correct, all blocks RoPE->FP32 |
| **TRT `rope_windowed_d8`** | **870** | **3.20x** | **0.996** | **0.997** | **0.988** | **best correct TRT** |

The huge PT-bf16 number on T4 is misleading: the repo hard-codes
`autocast(dtype=torch.bfloat16)` and T4 has no native BF16 tensor cores,
so every matmul runs through a software emulation path. Once you correct
that by switching to FP16 autocast, PyTorch itself delivers 5.71x on T4
with essentially the same numerical profile as TRT.

## What the numbers tell us

- **TRT's apparent speedup on T4 was largely an artifact of a suboptimal
  PyTorch dtype choice on Turing.** The one-line fix (pick `fp16` on
  pre-Ampere, `bf16` otherwise) beats the TRT engine by 1.78x on T4.
- **On L4**, PT-bf16 is already tight; no correct TRT variant beats it.
- **TRT's role** on these desktop GPUs is marginal. The deployment targets
  where it matters are Jetson (no Flash Attention in PT) and C++ inference
  servers (no Python runtime).

## How to reproduce

On T4:

```bash
# All four in their own subprocesses so T4 15GB doesn't OOM
for cfg in bf16 fp16 fp32 trt; do
  python bench_pt_dtype_comparison.py $cfg
done
python compare_masks.py
```

On L4:

```bash
python bench_sam3_final.py
```

Logit-level correctness (captures raw model output, produces cosine + std
ratio vs PT-bf16):

```bash
for cfg in bf16 fp16 fp32 trt; do
  python bench_logit_correctness.py $cfg
done
python bench_logit_correctness.py compare
python final_summary.py    # combines latency + logit into one table
```
