# GPU Optimization Session - RF-DETR

## Session Info
- **Date:** 2026-04-21
- **Domain:** GPU/CUDA optimization (TensorRT)
- **Target:** RF-DETR (rfdetr-base) .infer() method - heaviest model
- **Branch:** codeflash/optimize
- **Run tag:** 2026-04-21

## Target Identification
Benchmarked three models to find the heaviest:
- **RF-DETR base** (rfdetr-base): 3.83ms ← **HEAVIEST (selected)**
- YOLOv8n (yolov8n-640): 2.47ms
- YOLOv8n-seg (yolov8n-seg-640): 2.42ms

## Baseline Profile (torch.profiler)
**Wall-clock time:** 3.83ms average (20 iterations after 10 warmup)
**Profile metrics (10 iterations):**
- Self CPU time: 31.66ms
- Self CUDA time: 21.44ms
- **cudaStreamSynchronize:** 15.17ms (47.92% CPU time) ← HIGH BLOCKING OVERHEAD

### Top CUDA Bottlenecks
1. **GEMM kernels:** 6.38ms (29.74%) + 3.99ms (18.60%) = 10.37ms combined
   - These are TensorRT-optimized matrix multiplications - already optimal
2. **H2D transfers (Pageable):** 737.66us (3.44%, 70 calls)
   - Using **pageable memory** instead of pinned ← OPTIMIZATION TARGET
3. **aten::copy_:** 813.08us (3.79%, 150 calls)
   - Frequent tensor copies
4. **aten::to:** 737.66us CUDA time (150 calls) + 5.24ms CPU time (16.55%)
   - Type conversions and device transfers ← OPTIMIZATION TARGET

### Pipeline Stage Breakdown
Based on profiler output:
- **Data transfer overhead:** ~737us H2D + copy/to operations
- **Synchronization overhead:** 15.17ms CPU blocking (but masked in E2E)
- **GPU compute:** ~21.4ms CUDA time (dominated by GEMM kernels)

## Optimization Strategy
Focus on **pipeline stages surrounding TensorRT inference**, not the model itself:

### Priority 1: Memory Transfer Optimization
- **Use pinned memory** for H2D transfers (pageable → pinned)
  - Current: 737.66us across 70 H2D calls
  - Expected: 30-50% reduction in transfer time
- **Use non_blocking transfers** where possible

### Priority 2: Reduce Synchronization Overhead
- **Event-based synchronization** instead of blocking stream sync
- **Async pipeline overlap** between preprocessing, inference, postprocessing

### Priority 3: Tensor Operation Consolidation
- Reduce `aten::to` and `aten::copy_` calls
- **Cache tensors** that don't change between inferences
- **Fuse operations** where possible (preprocessing monolith strategy)

### Priority 4: Postprocessing GPU Optimization
- Keep more operations on GPU to avoid D2H/H2D roundtrips
- Use GPU-resident tensors for NMS and filtering operations

## Key Discoveries
1. RF-DETR uses TensorRT engine (detected TRT kernels in profiler)
2. The model's GEMM operations are already highly optimized (TRT compilation)
3. Main optimization surface is **around** the model: transfers, preprocessing, postprocessing
4. High sync overhead (47.92% CPU time) suggests opportunity for async pipelines

## Next Steps
1. Optimize H2D transfers (pinned memory)
2. Reduce synchronization calls
3. Cache reusable tensors
4. Profile per-stage (preprocess, inference, postprocess) to find specific bottlenecks
5. Apply monolithic kernel strategy if micro-optimizations don't register

## Files
- Profile script: `.codeflash/profile_rfdetr.py`
- Trace export: `/tmp/rfdetr_trace.json` (Chrome trace format)
- Model comparison: `/tmp/model_comparison.log`
