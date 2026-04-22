# The FP16 amplification bug

When any significant portion of SAM3's vitdet backbone runs in FP16 or
BF16 through TensorRT 10.12, the model's outputs are produced with
magnitudes ~2.5x the PyTorch reference. After the 0.5 detection
threshold in `PostProcessImage`, every predicted mask gets zeroed out,
so the model returns zero detections on every image.

Reproduced on:
- NVIDIA L4 (SM 8.9, Ada)
- NVIDIA T4 (SM 7.5, Turing)
- TensorRT 10.12.0.36

## Evidence it is not our ONNX

The same ONNX graph produces correct outputs under ONNX Runtime's
CUDAExecutionProvider:

```
Loading ORT (CUDA)...
ORT CUDA FP16: 294.42ms
  vision_features range: -3.867..3.420   # matches PT-bf16 (-3.78..3.77)
```

vs the TRT engine built from that same ONNX:

```
vision_features range: -9.58..9.40       # amplified ~2.5x
```

Both runtimes consume the same file; the divergence is in TRT's FP16
kernel selection or fusion, not in the ONNX graph.

See [`bench_ort_fp16.py`](../scripts/bench_ort_fp16.py) for the script
that produced these numbers.

## Where it accumulates

The per-block diagnostic in [`diagnose_fp16_divergence.py`](../scripts/diagnose_fp16_divergence.py)
marks each block's `norm2/LayerNormalization` output as an additional
engine output and compares to PyTorch hook outputs at the same points:

| Block | TRT LN output range | PT LN output range | cosine vs PT |
|------:|---:|---:|---:|
| 0 | ±5.1 | ±5.1 | 0.995 |
| 1 | ±7.8 | ±7.6 | 0.977 |
| 4 | ±5.6 | ±5.5 | 0.941 |
| 8 | ±6.8 | ±6.9 | 0.870 |
| 11 | ±6.6 | ±7.5 | 0.767 |
| 15 | ±5.5 | ±6.2 | 0.305 |
| 22 | ±4.4 | ±7.5 | 0.133 |
| 31 | ±32 | ±33 | 0.165 |

So the drift starts from block 0 and accumulates monotonically. Pinning
"only the late blocks" to FP32 isn't sufficient -- blocks 0-10 also
contribute to the accumulated error even though their individual cosine
looks decent.

## What didn't work

Every option short of pinning the actual RoPE math to FP32:

- `BuilderFlag.FP16` alone
- `BuilderFlag.BF16` alone (ran at FP32 anyway, 3x slower)
- Strongly-typed FP16 network (`STRONGLY_TYPED` flag)
- Strongly-typed BF16 network
- `BuilderFlag.OBEY_PRECISION_CONSTRAINTS` without per-layer overrides
- Disabling tactic sources one by one: `CUBLAS`, `CUBLAS_LT`, `CUDNN`,
  `JIT_CONVOLUTIONS`, `EDGE_MASK_CONVOLUTIONS`
- Setting `builder_optimization_level` 0 through 5
- Forcing Softmax to FP32 (alone)
- Forcing LayerNorm to FP32 (alone)
- Forcing all MatMul to FP32
- Forcing all residual Adds to FP32
- Forcing all ElementWise ops to FP32
- Matching PT autocast convention (LN in FP32, everything else FP16)

All of the above either fail the correctness gate, break at build time
for unrelated reasons, or (in the case of global FP32 residuals)
recover correctness but eliminate the speedup.

## What worked

Pinning the RoPE math layers to FP32 via a BFS forward from every
`freqs_cos_*` / `freqs_sin_*` tensor consumer, stopping at
convolutions. Two presets in [`build_sam3_engine.py`](../scripts/build_sam3_engine.py):

- **`fp16_rope_fp32`** (BFS depth 10, applied to all 32 blocks) -- 974 ms
  on T4, mean IoU 0.998.
- **`fp16_rope_windowed`** (BFS depth 8, windowed blocks only; blocks
  7/15/23/31 stay FP16) -- 870 ms on T4, mean IoU 0.998, logit cos 0.996.

The windowed variant is faster *and* slightly more accurate. Leaving the
4 global-attention blocks in FP16 lets TRT match the
`MatMul -> Softmax -> MatMul` pattern and emit a fused `_gemm_mha_v2`
kernel that keeps softmax numerator in FP32 internally. That's
numerically more accurate than our FP32-everywhere approach because
Flash-Attention-style kernels already do the right thing for FP16
attention.

## Suspected root cause

Not proven. Circumstantial evidence:

- The issue is graph-specific (reproduces on both T4 and L4).
- It is not tactic-specific (disabling all tactic sources doesn't fix
  it).
- It is not ONNX-level (ORT runs the same ONNX correctly).
- It is not strong-typing-related (happens with and without the flag).
- It accumulates monotonically through the residual stream of 32
  transformer blocks.

Most likely: TRT 10.12 is choosing an FP16 epilogue for the RoPE
`Slice -> Neg -> Concat -> Mul -> Mul -> Add` pattern that accumulates a
small relative-error bias per block, and the cumulative effect over 32
blocks produces the ~2.5x amplification. The per-layer FP32 pin prevents
TRT from picking that epilogue on the relevant ops.

Upstream reporting of a minimal reproducer would be the clean path
forward but was not completed in this work.
