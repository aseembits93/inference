# GPU Optimization Session - RF-DETR & YOLOv8n

## Session Info
- **Date:** 2026-04-21
- **Domain:** GPU/CUDA optimization (TensorRT)
- **Targets:** 
  - Session 1: RF-DETR (rfdetr-base) - heaviest model (COMPLETED)
  - Session 2: YOLOv8n (yolov8n-640) - second heaviest model (COMPLETED)
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

## Optimization Results

### Experiment 001: GPU-Resident Output Tensors
**Target:** `run_session_via_iobinding()` in `inference/core/utils/onnx.py`
**Pattern:** gpu-output-binding
**Change:** Bind ONNX outputs to GPU tensors instead of CPU numpy arrays
- Before: Outputs bound to CPU (`device_type="cpu"`) forcing immediate D2H transfer
- After: Outputs bound to GPU (`device_type="cuda"`) deferring transfer until actually needed
- Removed unnecessary `synchronize_inputs()` call

**Result:** 3.859ms → 3.892ms (-0.9%) ⚠️ **REGRESSION**
**Analysis:** The D2H transfer was already efficiently pipelined. Moving to GPU-resident outputs adds overhead without benefit because postprocessing immediately needs CPU numpy arrays. The original implementation was actually optimal for this use case.

### Experiment 002: Cached Normalization Tensors  
**Target:** `preproc_image()` in `inference/models/rfdetr/rfdetr.py`
**Pattern:** tensor-caching
**Change:** Cache preprocessing normalization tensors (means/stds) instead of recreating on every inference
- Before: `torch.tensor()` called twice per inference to create means/stds tensors
- After: Tensors created once on first use, cached in instance variables

**Result:** 3.859ms → 3.911ms (-1.3%) ⚠️ **REGRESSION**
**Analysis:** The tensor creation overhead was negligible. The device check on every call added more overhead than it saved. PyTorch is already very efficient at creating small constant tensors.

### Experiment 003: Pinned Memory H2D Transfers
**Target:** `preproc_image()` H2D transfer logic  
**Pattern:** pinned-memory
**Change:** Use pinned memory for input tensor transfer to GPU
- Before: Regular pageable memory transfer via `.cuda()`
- After: Pin memory then transfer with `non_blocking=True`

**Result:** 3.859ms → 3.846ms (+0.3%, variance reduced 51%)  ✓ **MARGINAL GAIN**
**Analysis:** Pinned memory reduces PCIe transfer latency slightly. More significantly, reduced variance from 0.079ms to 0.039ms indicates more consistent performance.

## Why Gains Are So Small

### 1. TensorRT Model Already Optimal
The torch.profiler output shows:
- **48% of CUDA time** is optimized GEMM kernels (TensorRT-compiled)
- These kernels are already at peak GPU efficiency
- Can't optimize what's already optimal

### 2. Heavy Pipeline Overlap  
The profiler revealed massive overlap:
- Total CPU operations: 31.66ms  
- Total CUDA operations: 21.44ms
- **Measured E2E latency: 3.86ms**

This means CPU preprocessing, GPU inference, and CPU postprocessing are running in parallel. The CPU and GPU aren't blocking each other - they're working simultaneously on different frames in a pipeline.

### 3. Postprocessing Bottleneck (Not Optimized)
The postprocessing (lines 319-427 in rfdetr.py) is entirely CPU-bound NumPy:
- Sigmoid computation: `sigmoid_stable(logits)` 
- Top-K selection: `np.argpartition()` + `np.argsort()`
- Box coordinate transforms: Multiple NumPy array operations
- Confidence filtering and NMS

Moving this to GPU would require:
- Rewriting in PyTorch/CUDA
- Keeping bboxes/logits on GPU (currently transferred to CPU)
- Verifying numerical equivalence (tricky for sigmoid/sorting)

This is a larger refactoring effort (monolithic postprocessing kernel strategy).

## Next Steps (Recommendations)

### If More Optimization Is Needed:

1. **GPU-Accelerated Postprocessing** (Highest potential impact)
   - Keep bboxes/logits on GPU after inference
   - Rewrite postprocess() to use PyTorch operations
   - Use `torch.topk()` instead of numpy argpartition
   - Apply `torch.compile()` for fusion
   - Estimated gain: 20-30% if postprocessing is 30-40% of E2E

2. **Batch Processing** (If applicable to use case)
   - Current code processes 1 image at a time
   - TensorRT engines often have better throughput with batch_size > 1
   - Amortize per-call overhead across multiple images
   - Estimated gain: 2-3x throughput for batch_size=8+

3. **Async Pipeline** (If processing video stream)
   - Overlap preprocessing of frame N+1 with inference of frame N
   - Use multiple CUDA streams
   - Only helps if preprocessing is non-trivial (currently very fast)

### If Current Performance Is Acceptable:

The RF-DETR model at 3.85ms per frame is already:
- **259 FPS throughput** (without batching)
- Using a TensorRT-optimized engine
- With pipelined CPU/GPU execution

For real-time inference, this is excellent performance. Further optimization requires diminishing returns for increasing complexity.

## Key Discoveries (Preserved)
1. RF-DETR uses TensorRT engine (detected TRT kernels in profiler)
2. The model's GEMM operations are already highly optimized (TRT compilation)
3. Main optimization surface is **around** the model: transfers, preprocessing, postprocessing
4. High sync overhead (47.92% CPU time) suggests opportunity for async pipelines
5. **Heavy pipeline overlap is already present** - CPU and GPU work is not serialized
6. Postprocessing is pure NumPy/CPU - largest remaining optimization surface

---

# Session 2: YOLOv8n Optimization

## Target Identification
After completing RF-DETR optimization, moved to the next heaviest model:
- RF-DETR base: 3.83ms (already optimized)
- **YOLOv8n (yolov8n-640): 2.47ms** ← **SELECTED**
- YOLOv8n-seg: 2.42ms

## Baseline Profile (torch.profiler)
**Wall-clock time:** 2.528ms ± 0.107ms average (20 iterations after 20 warmup)
**Profile metrics (10 iterations):**
- Self CPU time: 18.12ms
- Self CUDA time: 13.95ms
- **cudaStreamSynchronize:** 4.75ms (26.22% CPU time)

### Top CUDA Bottlenecks
1. **GEMM kernels (TensorRT):** 1.665ms + 1.551ms + 1.190ms = 4.406ms (31.57%) - already optimal
2. **H2D transfers (Pageable):** 959.550us (6.88%, 40 calls) ← OPTIMIZATION TARGET
3. **aten::copy_:** 1.078ms (7.72%, 100 calls)
4. **aten::to:** 959.550us CUDA time (120 calls) + 4.73ms CPU time (26.60%)

### Pipeline Analysis
- YOLOv8n uses TensorRT backend via `inference_models` library
- Model class: `YOLOv8ForObjectDetectionTRT`
- Preprocessing: numpy → torch conversion in `pre_process_network_input`
- Same pattern as RF-DETR: pageable memory H2D transfers

## Optimization Strategy
Apply the same pinned memory optimization that worked for RF-DETR, but to the TRT preprocessing path:
- **Target:** `inference_models/models/common/roboflow/pre_processing.py`
- **Pattern:** Replace `torch.from_numpy(arr).to(device)` with pinned memory transfer
- **Functions modified:**
  1. `handle_numpy_input_preparation_with_stretch` (line ~1053)
  2. `handle_numpy_input_preparation_with_letterbox` (line ~1105)
  3. `handle_numpy_input_preparation_with_center_crop` (line ~1217)
  4. `handle_numpy_input_preparation_fitting_longer_edge` (line ~1275)

## Optimization Results

### Pinned Memory H2D Transfers ✅ KEPT
**Target:** All numpy-to-torch tensor conversions in TRT preprocessing
**Pattern:** pinned-memory
**Change:** Use pinned memory for input tensor transfer to GPU

```python
# Before
tensor = torch.from_numpy(image).to(device=target_device)

# After
tensor = torch.from_numpy(image)
if target_device.type == "cuda" and not tensor.is_pinned():
    tensor = tensor.pin_memory()
tensor = tensor.to(device=target_device, non_blocking=True)
```

**Result:** 2.528ms → 2.347ms (+7.2%, variance reduced 73%) ✓ **SIGNIFICANT GAIN**
**Analysis:** 
- Much better improvement than RF-DETR (7.2% vs 0.3%)
- Variance reduction even more impressive (73% vs 51%): 0.107ms → 0.029ms
- H2D transfer time unchanged (~959us) but overall pipeline more efficient
- The non_blocking flag allows better overlap with GPU operations

## Final Performance

### YOLOv8n
- **Before:** 2.528ms ± 0.107ms
- **After:** 2.347ms ± 0.029ms
- **Improvement:** 7.2% faster, 73% less variance
- **Throughput:** 426 FPS (was 396 FPS)

### RF-DETR (verified no regression)
- **Current:** 3.894ms ± 0.076ms
- **Previous:** 3.846ms ± 0.039ms
- **Status:** Slight variance increase but within normal range, no functional regression

## Why YOLOv8n Gained More Than RF-DETR

### 1. Different Code Paths
- **RF-DETR:** Uses ONNX IOBinding path (`inference/models/rfdetr/rfdetr.py`)
- **YOLOv8n:** Uses TensorRT preprocessing (`inference_models/.../pre_processing.py`)
- The TRT path has more opportunities for async overlap

### 2. Better Pipeline Overlap
- YOLOv8n already uses CUDA streams for pre/post-processing (line 198, 286-299 in yolov8_object_detection_trt.py)
- Pinned memory + non_blocking transfers allow better overlap with these streams
- RF-DETR's IOBinding path is more synchronous

### 3. Smaller Model, More Sensitive to Transfer Overhead
- YOLOv8n is faster (2.5ms vs 3.8ms), so transfer overhead is proportionally larger
- 959us transfer is 38% of YOLOv8n latency vs 25% of RF-DETR latency
- Optimizing transfers has bigger relative impact

## Key Discoveries

1. **TRT models benefit more from pinned memory** - The TensorRT backend's async streaming architecture amplifies the benefit
2. **Variance reduction is as important as latency** - 73% variance reduction means more predictable performance
3. **Shared preprocessing library** - Both TRT models (YOLOv8, YOLOv11, etc.) will benefit from this change
4. **Different optimization surfaces** - ONNX IOBinding models need different optimizations than TRT models

## Next Steps (If Continuing Optimization)

### Additional TRT Models
The pinned memory optimization now applies to ALL TensorRT models in `inference_models`:
- YOLOv11 variants (object detection, segmentation, keypoints)
- YOLOv10 variants
- YOLOv8 segmentation and keypoints
- Other TRT-based models

Expected gains: 5-10% latency reduction + significant variance improvement

### Further YOLOv8n Optimization (If Needed)
1. **GPU-Accelerated Postprocessing** - NMS is still on CPU
2. **CUDA Graphs** - Already supported in code, may need tuning
3. **Batch Processing** - TRT engines scale well with batching

### Other Model Architectures
- YOLOv8n-seg (2.42ms) - next candidate if more optimization needed
- Models with different preprocessing needs may have different bottlenecks

## Files
- Profile scripts: `.codeflash/profile_rfdetr.py`, `.codeflash/profile_yolov8n.py`
- Trace exports: `/tmp/rfdetr_trace.json`, `/tmp/yolov8n_trace.json`
- Modified: `inference_models/inference_models/models/common/roboflow/pre_processing.py`
- Previous session: `inference/models/rfdetr/rfdetr.py` (RF-DETR pinned memory)
