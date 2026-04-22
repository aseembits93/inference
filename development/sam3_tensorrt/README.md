# SAM3 TensorRT Export

Experimental pipeline that exports SAM3's vision backbone to TensorRT and
hot-swaps it into `SegmentAnything3` at runtime. The rest of the SAM3
inference path (preprocessing, transformer decoder, postprocessing) stays in
PyTorch.

**Status:** works end-to-end with mean mask IoU >= 0.998 vs the PyTorch
reference, but the benchmark numbers argue against shipping this as-is on
most GPUs -- see [docs/conclusions.md](docs/conclusions.md).

## Quick links

- [docs/benchmarks.md](docs/benchmarks.md) -- original single-image
  latency + correctness on T4 and L4.
- [docs/100-image-study.md](docs/100-image-study.md) -- **100-image**
  correctness study across 53 COCO classes, the most reliable numbers
  in this directory.
- [docs/benchmark-summary.md](docs/benchmark-summary.md) -- one-page
  table of every config benchmarked, recall/precision/IoU/speedup.
- [docs/correctness.md](docs/correctness.md) -- how correctness is measured
  (mask IoU gate + logit cosine / std-ratio gate) and why the logit gate is
  the one to trust.
- [docs/precision-bug.md](docs/precision-bug.md) -- the FP16 numerical
  issue in the SAM3-repo path that forced per-layer FP32 pinning in the
  RoPE math.
- [docs/hf-trt-investigation.md](docs/hf-trt-investigation.md) -- why
  the whole-model HF SAM3 TRT engine has 22% lower recall and 8
  experiments that failed to fix it.
- [docs/vs-dataplayer12.md](docs/vs-dataplayer12.md) -- comparison with
  `dataplayer12/SAM3-TensorRT`.
- [docs/conclusions.md](docs/conclusions.md) -- what to ship, what to skip.

## Pipeline

```
PyTorch SAM3 model
   |
   +-- .backbone.forward_image (vision path, ~90% of forward time)
   |         |
   |         v
   |   export_sam3_backbone_onnx.py   # ONNX opset-17, real-arithmetic RoPE patch
   |         |
   |         v
   |   build_sam3_engine.py fp16_rope_windowed    # TRT engine with FP32 RoPE islands
   |         |
   |         v
   |   sam3_trt_adapter.py     # runs engine, returns backbone-shaped dict
   |         |
   |         v
   +-- patch_sam3_with_trt_backbone()   # monkey-patch the live model
```

## Scripts in `scripts/`

Exports:

- `export_sam3_backbone_onnx.py` -- original export, FP32 weights, pair-stack RoPE patch
- `export_sam3_backbone_v2.py` -- rotate_half RoPE formulation, FP32 weights
- `export_sam3_backbone_fp16_native.py` -- strongly-typed FP16 weights in the ONNX
- `export_sam3_backbone_fp16_autocast.py` -- FP16 weights except LayerNorm (mimics PT autocast)
- `export_sam3_backbone_bf16.py` -- strongly-typed BF16 weights in the ONNX

Engine builders:

- `build_sam3_engine.py` -- multi-preset builder (fp16, bf16, various FP32-pinning strategies)
- `build_fp16_native_engine.py` -- strongly-typed FP16 build from fp16_native ONNX
- `build_fp16_autocast_engine.py` -- strongly-typed build from fp16_autocast ONNX

Adapter:

- `sam3_trt_adapter.py` -- `Sam3VisionTRT` runner + `patch_sam3_with_trt_backbone`

Benchmarks:

- `bench_sam3_final.py` -- backbone-only + E2E latency + correctness (L4-friendly)
- `bench_sam3_t4.py` -- same but runs PT and TRT passes in serial (fits in T4 15GB)
- `bench_pt_dtype_comparison.py` -- PT-bf16 vs PT-fp16 vs PT-fp32 vs TRT
- `bench_logit_correctness.py` -- captures raw model logits for cosine / std-ratio comparison
- `bench_ort_fp16.py` -- PyTorch vs ORT-CUDA vs ORT-TRT (used to prove ONNX is correct)

Diagnostics:

- `sam3_correctness_gate.py` -- 4-image mask IoU gate (pass >= 0.95)
- `diagnose_fp16_divergence.py` -- per-block TRT-vs-PT cosine (finds where FP16 diverges)
- `debug_sam3_trt.py` -- raw backbone output cosine similarity vs PT FP32/BF16
- `inspect_engines.py` -- dump I/O dtype/shape per engine
- `profile_engine.py` -- layer-time profiler with name-based bucketing
- `final_summary.py` -- combine latency + logit comparisons into one table
- `compare_masks.py` -- post-hoc mask-IoU comparison between saved runs

## Running the pipeline

All scripts expect `ROBOFLOW_API_KEY` in the environment and `SAM3_ASSETS`
pointing to the directory with test images (defaults to the repo's own
integration-test asset directory).

```bash
export ROBOFLOW_API_KEY=...
export SAM3_ASSETS=tests/workflows/integration_tests/execution/assets

cd development/sam3_tensorrt/scripts

# 1. Export ONNX (real-arithmetic RoPE, ~1.7 GB, requires CUDA + SAM3 weights)
python export_sam3_backbone_v2.py

# 2. Build a TRT engine. Best correctness/speed trade on T4 and L4:
python build_sam3_engine.py fp16_rope_windowed

# 3. Gate on correctness vs PyTorch
python sam3_correctness_gate.py

# 4. Benchmark
python bench_sam3_final.py     # L4 / Ada
python bench_sam3_t4.py        # T4 / Turing (runs PT and TRT serially)
```

## Memory and storage

- ONNX export: ~1.7 GB on disk, runs on CPU for max compatibility
- TRT engine: 870 MB - 1.8 GB depending on precision preset
- Engine + PyTorch SAM3 live simultaneously on the GPU during the adapter
  run; needs ~15 GB VRAM for T4, ~22 GB fits comfortably on L4.
