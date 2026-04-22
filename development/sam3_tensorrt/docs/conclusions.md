# Conclusions and recommendations

After measuring correctness with a logit-level gate and latency across
PT (bf16/fp16/fp32) and TRT (various precision pinning strategies) on
both L4 (Ada, native bf16) and T4 (Turing, emulated bf16), the overall
picture is:

## TL;DR

- The TRT pipeline works and produces numerically correct results.
- On **T4**, the best correct TRT engine delivers **3.2x E2E speedup vs
  the repo default** -- but a one-line PyTorch change delivers **5.7x
  with zero TRT overhead**. TRT should not ship as the T4 answer.
- On **L4**, the PyTorch baseline is already at the hardware roofline
  and no correct TRT variant beats it.
- The TRT pipeline still has a place for deployments where Python + PT
  are not the runtime (C++ servers, Jetson). Not for this repo's
  current deployment shape.

## The real win: change PyTorch dtype selection

[`segment_anything3.py`](../../../inference/models/sam3/segment_anything3.py) at line 535 hardcodes:

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
```

That's correct on Ampere/Ada/Hopper (native bf16 tensor cores) and a
performance trap on Turing (T4/V100, no bf16 tensor cores, falls back
to software emulation).

Proposed change:

```python
def _select_autocast_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    major, minor = torch.cuda.get_device_capability(0)
    # Ampere (SM 8.0) and later have native BF16 tensor cores.
    return torch.bfloat16 if major >= 8 else torch.float16

_DTYPE = _select_autocast_dtype()

# ...

with torch.autocast(device_type="cuda", dtype=_DTYPE):
```

Measured effect:

- T4: 2786 ms -> 488 ms (5.71x faster).
- L4: no change (stays on bf16, the current default, which is already
  optimal there).
- Correctness: logit cosine vs current bf16 baseline = 0.990 on both
  T4 and L4, matching the PT-fp32 noise floor. Mask IoU
  indistinguishable.

This is a two-line change in one file. It supersedes the entire TRT
pipeline for single-GPU Python deployments on T4 / V100.

## When TRT is actually worth keeping

- **Jetson / embedded.** PyTorch on Jetson lacks good attention
  kernels; TRT is the native path. The reference repo
  (`dataplayer12/SAM3-TensorRT`) targets this scenario.
- **C++ inference servers.** Wire SAM3 into a binary that doesn't
  carry the Python runtime.
- **H100 / H200.** Not tested here, but the correct-but-slow engines
  (e.g. `rope_fp32_d10`) should run significantly faster on Hopper's
  FP8 + WGMMA path and may cross over PT baselines there.

## Why the speedup is smaller than claimed in the reference repo

The reference (`dataplayer12/SAM3-TensorRT`) claims to "ship SAM-3
faster" but doesn't publish numbers. Likely reasons the improvement
there looks bigger:

1. Their baseline is HuggingFace `Sam3Model` in PyTorch, which may not
   have the same autocast setup as our model path.
2. They target Jetson, where PT is weaker.
3. No correctness gate, so if FP16 is amplifying outputs, they wouldn't
   necessarily notice on smoke-test images.

## Known limitations of this work

- **Single batch size (1) only.** Engines are built with static shape
  `(1, 3, 1008, 1008)`. Multi-image batches would need dynamic axes or
  a rebuild.
- **Correctness measured on 4 images, one prompt each.** Not a serious
  evaluation -- see [correctness.md](correctness.md#whats-not-gated).
- **No ground-truth comparison.** Gate is "matches PT-bf16", not
  "matches human annotation". A TRT engine that reproduces a degraded
  PT-bf16 output is graded "correct" by this system.
- **Box prompts and multi-prompt NMS paths are not exercised** in the
  gate. They should work (the vision backbone is invariant to prompt
  type) but aren't validated.

## Files to keep if we merge this

Core pipeline:
- `export_sam3_backbone_v2.py` -- the export we'd actually use
  (rotate_half RoPE, cleanest)
- `build_sam3_engine.py` -- the `fp16_rope_windowed` preset
- `sam3_trt_adapter.py` -- the runtime adapter
- `sam3_correctness_gate.py` -- mask IoU gate
- `bench_logit_correctness.py` + `final_summary.py` -- logit gate

Diagnostics worth keeping:
- `diagnose_fp16_divergence.py` -- per-block cosine probe, invaluable
  if the FP16 bug reappears
- `bench_ort_fp16.py` -- proof that ONNX is TRT-independent-correct

## Files that were useful but are probably archive material

Every alternative export and alternative precision build. Worth reading
the commit message / docs but not running again unless you're
investigating the same numerical issue.
