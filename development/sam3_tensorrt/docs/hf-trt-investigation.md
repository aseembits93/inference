# HF-TRT correctness investigation

The `hf_trt` config — HuggingFace `Sam3Model` exported whole to ONNX,
then `trtexec --fp16` — runs fast (366 ms on T4, the fastest of any
tested config) but only achieves **78.4% recall** on the 100-image
benchmark vs PT-bf16's 99%+.

This doc is the full debug trail. Spoiler: the bug is intrinsic to
TRT's handling of the SDPA attention graph in HF's SAM3, not something
fixable with per-layer precision pinning or ONNX simplification.

## The pattern

HF-TRT systematically:
- Misses ~22% of reference detections
- When it does detect, mean match IoU is 0.879 (vs 0.996 for
  correct configs)
- Confidence scores drop uniformly by ~8 points
- Logit std compresses from PT's ~1.29 to TRT's ~0.74 on the last
  DETR decoder layer
- Failure concentrates on dense multi-instance scenes (book,
  handbag, suitcase, bed, keyboard — all 52-67% recall vs 100% on
  easier classes)

## Hypothesis chain and experiments

### H1: FP16 precision loss in the DETR decoder

Pin the DETR decoder + mask_decoder + box_head + dot_product_scoring
to FP32 via `OBEY_PRECISION_CONSTRAINTS` + per-layer overrides. 1074
layers pinned.

**Result:** 77.6% recall, 0.879 mean IoU, −0.088 score delta.
**Identical to unpinned HF-TRT.** Latency went up 27 ms, correctness
didn't budge.

Script: `scripts/build_hf_sam3_decoder_fp32.py`.

### H2: FP16 precision loss in the DETR encoder, neck, or heads

Extend the pin to everything non-backbone: DETR encoder, vision
neck/FPN, DETR decoder, mask decoder, box_head, dot_product_scoring.
1410 layers pinned, keep only vision_backbone and text_encoder FP16.

**Result:** 77.6% recall, 0.879 mean IoU, −0.088 score delta.
**Identical.** Latency 547 ms.

Script: `scripts/build_hf_sam3_non_backbone_fp32.py`.

### H3: FP16 precision loss in the vision backbone attention

Pin every layer whose name matches
`/sam3/vision_encoder/backbone/layers.<k>/attention/` to FP32.
928 layers across all 32 blocks, analogous to our SAM3-repo
`fp16_attn_hard` preset.

**Result:** 77.8% recall, 0.879 mean IoU, −0.089 score delta.
**Identical.** Latency 902 ms (2.5× slower than baseline HF-TRT).

Script: `scripts/build_hf_sam3_attn_fp32.py`.

### H4: FP16 precision loss anywhere

Build the engine with NO FP16 flag — pure FP32 everywhere.

**Result:** 78.4% recall, 0.879 mean IoU, −0.086 score delta.
**Identical to FP16.** Latency 1752 ms (5× slower).

This is the definitive null result. **FP32 everywhere gives
identical recall to FP16** — the 22% recall gap is NOT precision.

Script: `scripts/build_hf_sam3_fp32.py`.

### H5: The ONNX graph itself is wrong (export bug)

Compare PT vs ORT running the **same ONNX**. If ORT produces
correct outputs, the ONNX is fine and TRT is the bug. If ORT
also produces wrong outputs, the ONNX export is broken.

Run on `dogs.jpg` with prompt `"dog"` (the exact image used at
ONNX export time to eliminate trace-vs-runtime branch mismatches):

| Runner | pred_logits std | Matches PT? |
|---|---:|:---:|
| PT-HF (reference) | 1.0941 | (reference) |
| ORT CPU on our ONNX | 1.0941 | **cosine = 1.00000** |
| TRT FP32 on our ONNX | 0.8661 | cosine = 0.964 |
| TRT FP16 on our ONNX | ~0.866 | cosine ≈ 0.964 |

**The ONNX graph is perfectly correct.** ORT reproduces PT's
output bit-identically. TRT runs the same graph but produces a
systematically-different result.

Note: ORT emits a shape-inference warning when loading our ONNX:

```
[W] Error merging shape info for output
'/sam3/vision_encoder/backbone/embeddings/Concat_output_0'
source:{4} target:{5}. Falling back to lenient merge.
```

ORT handles this leniently and matches PT. TRT might not.

### H6: Shape-annotation inconsistency causes TRT to pick wrong kernels

Run `onnx.shape_inference.infer_shapes_path` to reconcile conflicting
annotations before TRT sees the graph. Output is a clean,
fully-annotated ONNX.

**Result:** 78.1% recall, 0.879 mean IoU, −0.087 score delta.
**Identical to unshape-inferred.** Latency 363 ms (basically
unchanged).

Script: `scripts/shape_infer_hf_onnx.py` →
`scripts/build_hf_sam3_inferred_engine.py`.

### H7: TRT mis-fuses the MHA pattern; break it with onnx-graphsurgeon

Insert `Identity` nodes between every `Softmax` output and its
downstream `MatMul`. Two variants:
- **decoder-only**: 12 Identities in the 6 DETR decoder cross-attentions
- **all**: 50 Identities across vision_backbone + detr_encoder +
  detr_decoder

**Results:**

| Variant | Recall | Match IoU | Score delta | `_gemm_mha_v2` kernels in engine |
|---|---:|---:|---:|---:|
| hf_trt (baseline) | 78.4% | 0.879 | −0.087 | 9 |
| hf_trt_nofuse_decoder | 78.4% | 0.879 | −0.087 | 9 |
| hf_trt_nofuse_all | 78.4% | 0.879 | −0.087 | 9 |

**TRT constant-folds the Identity nodes during build** and still emits
the same 9 `_gemm_mha_v2` fused kernels. The inserted Identity doesn't
change the fusion result OR the correctness numbers.

Either interpretation is possible: (a) TRT's MHA fusion is robust to
Identity insertion, so this approach can't break it; or (b) the MHA
fusion isn't actually the bug, and pattern-breaking would be a
waste of effort.

Scripts: `scripts/break_mha_fusion.py` → `scripts/build_hf_sam3_nofuse.py`.

### H8: The bug is CUDA-kernel-level (affects any SDPA runner)

If the issue is that CUDA's SDPA kernels are genuinely producing
different numbers than CPU SDPA, then **ORT-CUDA** should show the
same bug as TRT, and PT-CUDA would match them (if PT uses the same
underlying kernel).

**Result:**

| Runner | pred_logits std | cos vs PT-CUDA |
|---|---:|---:|
| PT-CUDA (reference) | 1.0941 | (reference) |
| ORT-CPU on our ONNX | 1.0941 | 1.00000 |
| **ORT-CUDA on our ONNX** | **1.0941** | **1.00000** |
| TRT-FP32 on our ONNX | 0.8661 | 0.964 |
| TRT-FP16 on our ONNX | ~0.866 | 0.964 |

**ORT-CUDA matches PT-CUDA bit-exactly.** The bug is not in CUDA
SDPA kernels in general — it is specific to TensorRT 10.12's
execution of this graph.

Script: modify `scripts/bench_ort_fp16.py` to use CUDAExecutionProvider.

## Diagnosis

After eight experiments covering precision, shape inference, fusion
breaking, and cross-runtime comparison:

1. **The ONNX graph is correct.** Both ORT-CPU and ORT-CUDA produce
   bit-identical outputs to PT (cos = 1.00000).
2. **The bug is TRT-specific.** Pure FP32 TRT produces the same wrong
   output as FP16 TRT, ruling out precision.
3. **The bug is not MHA fusion** (tested via Identity-insertion
   graphsurgeon), though it could be MHA-adjacent — the fused
   `_gemm_mha_v2` kernels are the most expensive ops in the engine.
4. **The bug is not localized to any subgraph** — per-layer FP32
   pinning of decoder, decoder+encoder, all attention, or all
   non-backbone produces identical numbers.

Most likely explanation: **TensorRT 10.12 has a bug in its
graph-level attention lowering for this specific ONNX shape**. The
semantics of `F.scaled_dot_product_attention`, when lowered through
`torch.onnx.export(dynamo=False)`, produce an ONNX subgraph that
TRT's build pipeline transforms into a kernel that computes a
subtly different operation. The result is systematic compression of
DETR decoder confidence scores by ~8 points, causing ~22% of
detections to fall below threshold on multi-instance scenes.

## What would actually fix it

Having ruled out the easier options, the remaining paths all require
significant engineering:

1. ~~**Patch the ONNX graph with onnx-graphsurgeon**~~ (tried in H7,
   doesn't work — TRT folds Identity nodes away).
2. **Disable `_gemm_mha_v2` fusion entirely** via tactic source
   restrictions. We tried `config.set_tactic_sources(0)` in our
   SAM3-repo investigation (see `precision-bug.md`); it didn't fix
   the SAM3-repo FP16 bug. The mis-lowering for HF likely happens
   at a graph-transform level below tactic selection.
3. **Rewrite `Sam3Attention` to use explicit `torch.matmul` +
   `softmax`** instead of `F.scaled_dot_product_attention`, then
   re-export. Would produce a different ONNX op shape that TRT
   maps to eager MatMul kernels. Requires forking transformers.
4. **File a TRT bug report** with a minimal repro and wait for a
   fix in a future TRT version. Our `sam3_full.onnx` + the
   PT-CUDA / ORT-CUDA / TRT-FP32 cross-runtime comparison from H5
   and H8 would be a clean report.

Given the 100-image study already shows the SAM3-repo TRT-swap at
99.3% F1 and 578 ms, and PT-fp16 at 98.7% F1 and 516 ms, pursuing
options 2-3 is hard to justify — neither the SAM3-repo route nor
PT-fp16 has this bug, and either is a viable deployment target on
T4 and similar GPUs.

## Related finding: SDPA export in PyTorch 2.10+ is fragile

`torch.onnx.export(dynamo=True)` (the recommended modern path) fails
on this model:

```
ValueError: Cannot view a tensor with shape torch.Size([1, 201, 8, 32])
and strides (51456, 32, 6432, 1) as a tensor with shape (1, 201, 256)!

While executing ... line 397, in forward
    attn_output = attn_output.reshape(...).contiguous()
```

This is in `detr_decoder.vision_cross_attn`. The dynamo exporter is
stricter about stride-incompatible views than TorchScript tracing;
HF's attention code works in eager mode but trips dynamo. Would need
a `.contiguous()` added before the reshape in the HF source to get
a clean dynamo export.

## Scripts for this investigation

Under `scripts/`:
- `build_hf_sam3_decoder_fp32.py`    — H1 (decoder FP32 pin)
- `build_hf_sam3_non_backbone_fp32.py` — H2 (non-backbone FP32 pin)
- `build_hf_sam3_attn_fp32.py`       — H3 (backbone attention FP32)
- `build_hf_sam3_fp32.py`            — H4 (pure FP32)
- `bench_ort_fp16.py` (pre-existing) — H5 (ORT-CPU)
- `shape_infer_hf_onnx.py` + `build_hf_sam3_inferred_engine.py` — H6
- `break_mha_fusion.py` + `build_hf_sam3_nofuse.py` — H7 (graphsurgeon)
- `export_hf_sam3_dynamo.py`         — dynamo export attempt
- `sweep_100_images.py`              — 100-image evaluation driver
- `aggregate_correctness.py`         — per-config recall/precision/IoU
