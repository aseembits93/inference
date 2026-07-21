# Comparison with `dataplayer12/SAM3-TensorRT`

The reference repo ([github.com/dataplayer12/SAM3-TensorRT](https://github.com/dataplayer12/SAM3-TensorRT)) also exports SAM3 to TensorRT. It predates this work and got us unblocked on the ONNX export approach. We took a different path because we have different constraints.

## Scope

| | Reference | This work |
|---|---|---|
| Source model | HuggingFace `facebook/sam3` (`transformers.Sam3Model`) | Roboflow's `sam3/sam3_final` via `SegmentAnything3` |
| What gets exported | Whole model end-to-end | Only `backbone.forward_image` |
| Input signature | `pixel_values`, `input_ids`, `attention_mask` (plain tensors) | `samples` tensor |
| Output | `pred_masks`, `semantic_seg` | dict shaped like `backbone.forward_image()` output |
| Postprocessing | Custom CUDA (`prepost.cu`, ~200 lines) | Reuses the repo's `PostProcessImage` in PyTorch |
| Integration target | C++ standalone app (`sam3_pcs_app`) | `sam3_trt_adapter.patch_sam3_with_trt_backbone()` monkey-patching a live Python model |
| Engine build | `trtexec --fp16 --verbose` | Python builder with per-preset precision pinning |

## RoPE handling

The reference repo's ONNX export is ~15 lines. No RoPE patching is
needed because HuggingFace's `Sam3Model` implementation doesn't use
`torch.view_as_complex` -- the rotation is already real-arithmetic in
HF's PyTorch source.

This repo pulls SAM3 from the `sam3` PyPI package (Meta's original
research code), whose `vitdet.apply_rotary_enc` does use
`view_as_complex` and is not exportable at opset 17. That forced us to
write [`export_sam3_backbone_onnx.py`](../scripts/export_sam3_backbone_onnx.py)
and [`export_sam3_backbone_v2.py`](../scripts/export_sam3_backbone_v2.py)
which mutate `freqs_cis: complex` into `(freqs_cos, freqs_sin): real`
buffers and monkey-patch `_apply_rope` to use real arithmetic.

## Precision strategy

Reference: `trtexec --fp16`, no per-layer overrides documented.

This work: we discovered that global FP16 on this graph produces 2.5x
amplification and zero detections on T4 and L4 (TensorRT 10.12). See
[precision-bug.md](precision-bug.md). Workaround is to pin the RoPE math
layers to FP32 via a BFS from every `freqs_cos`/`freqs_sin` consumer.
The cost is a 17% latency regression vs the broken FP16 baseline.

The reference repo does not publish any correctness validation, so we
can't tell whether the same bug triggers on the HF model they're
exporting or whether it was invisible on their test hardware.

## Correctness

Reference: no correctness gate in the published code. Implicitly, the
benchmark mode (`sam3_pcs_app <dir> <engine> 1`) writes visualizations
without scoring.

This work:
- `sam3_correctness_gate.py` -- mean mask IoU >= 0.95 on 4 test images.
- `bench_logit_correctness.py` + `final_summary.py` -- per-tensor
  cosine similarity and std-ratio comparison against a PyTorch reference.

## Performance

Reference: no published numbers. They include a benchmark harness but
no results.

This work (see [benchmarks.md](benchmarks.md)):
- T4: PT-bf16 baseline 2786 ms, TRT correct 870 ms (3.2x), TRT broken
  748 ms (3.7x but wrong), PT-fp16 with one-line repo change 488 ms
  (5.7x).
- L4: PT-bf16 baseline 247 ms, every correct TRT engine slower.

## What the reference does better

- **Simplicity.** Whole-model export in 15 lines. If we were starting
  clean we would do this.
- **C++ story.** `sam3_pcs_app` is a self-contained standalone binary
  suitable for Jetson / embedded deployment where the Python stack
  isn't available.
- **Pre/post on GPU.** Custom CUDA kernels keep data on-device, avoiding
  D2H/H2D round trips during postprocessing.

## What this work does differently

- **Fits into an existing Python inference server** without ripping out
  the preprocessing and postprocessing paths that are already wired up.
- **Doesn't require upgrading `transformers` or switching model
  providers.** The `sam3` PyPI dependency stays unchanged.
- **Documents a real TRT numerical issue** with a workaround. If you run
  the reference's `trtexec --fp16` flow against the PyPI `sam3` model on
  an Ada / Turing GPU, you'd most likely ship broken inference without
  noticing, because there's no gate.
- **Publishes benchmark numbers.** On the GPUs we tested, TRT is rarely
  the best answer; the recommendation often comes out as "change the
  dtype in PyTorch instead."

## What to take from the reference

- If we were willing to migrate off the `sam3` PyPI package to
  `transformers.Sam3Model`, we could drop the RoPE patch entirely.
- If we were building a non-Python deployment, the C++ / CUDA pre-post
  kernels in `prepost.cu` are the right pattern.
