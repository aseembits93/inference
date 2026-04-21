# SAM3 TRT Optimization Request — Blocking Issues

**Date:** 2026-04-21  
**Session:** GPU optimization continuation  
**Request:** "e2e optimization of sam3 infer() — TRT GPU performance"

## Issue Summary

**SAM3 has no TensorRT backend.** Cannot proceed with TRT GPU optimization as requested.

## Detailed Investigation

### 1. No TRT Backend Exists

Checked entire codebase:
- ❌ No `sam3_trt.py` file
- ❌ No ONNX export infrastructure for SAM3
- ❌ No TRT engine builder for SAM3
- ❌ No `.engine` or `.plan` files in SAM3 model packages

Reference: 20+ other models have TRT variants:
- YOLOv5/7/8/9/10/11/12 (object detection, segmentation, keypoints)
- YOLO26 (all variants)
- RF-DETR (object detection, instance segmentation)
- ResNet, VIT, DeepLabV3+, YOLONas, Yolact

SAM3 is **not among them**.

### 2. SAM3 Package Not Installed

```python
>>> import sam3
ModuleNotFoundError: No module named 'sam3'
```

- SAM3 is an optional dependency
- Integration tests skip by default: `SKIP_SAM3_TESTS=True`
- Cannot profile, benchmark, or optimize without the package

### 3. What SAM3 Code Exists

Two PyTorch implementations (no TRT):

**a) `inference/models/sam3/segment_anything3.py`**
- Class: `SegmentAnything3`
- API: batch text/box prompting
- Backend: `sam3.build_sam3_image_model()` (native PyTorch)
- Key method: `segment_image()` with torch.autocast(bfloat16)

**b) `inference/models/sam3/visual_segmentation.py`**
- Class: `Sam3ForInteractiveImageSegmentation`
- API: interactive point/box prompting (SAM2-compatible interface)
- Backend: same `sam3.build_sam3_image_model()` with `enable_inst_interactivity=True`
- Key method: `segment_image()` using `model.predict_inst()`

Both use CUDA with bfloat16 autocast but **not TensorRT**.

### 4. Comparison with Prior Session

**YOLOv8n / YOLOv8n-seg / RF-DETR session (completed):**
- ✅ Models had TRT backends (`*_trt.py`)
- ✅ TRT engines pre-built and cached
- ✅ Shared preprocessing (`pre_process_network_input`) optimized
- ✅ Shared postprocessing (NMS, rescaling) optimized
- ✅ Result: 4-8% E2E speedup on random + real images

**SAM3 (blocked):**
- ❌ No TRT backend
- ❌ Package not installed
- ❌ Cannot follow same optimization workflow

## Possible Interpretations

### Option A: User Meant a Different Model

Perhaps the user intended:
- **SAM2** (also no TRT, but is installed and functional)
- **Another segmentation model** (DeepLabV3+, Yolact have TRT)

### Option B: PyTorch GPU Optimization Requested

If the user wants PyTorch SAM3 CUDA optimization (not TRT):

**Prerequisites:**
1. Install `sam3` package
2. Confirm availability of model weights (`weights.pt`, `bpe_simple_vocab_16e6.txt.gz`)

**Optimization opportunities (PyTorch CUDA):**
- Image preprocessing (PIL → tensor → normalize → pad)
- H2D transfers for prompts (boxes, points)
- D2H transfers for mask outputs
- Cached tensors (normalize constants, padding buffers)
- Kernel fusion (similar to YOLOv8 approach but in PyTorch ops)
- BFloat16 autocast already enabled (line 535, 172)

**Expected gains:** 5-15% on preprocessing + postprocessing (model forward pass is opaque SAM3 internals)

### Option C: TRT Backend Implementation Required

If the user wants TRT specifically:

**Scope:** This is a **new feature**, not optimization of existing code.

**Tasks:**
1. Export SAM3 model to ONNX
   - Handle text encoder branch (CLIP-based)
   - Handle vision encoder branch (ViT-based)
   - Handle mask decoder branch
2. Build TRT engines with FP16 precision
3. Create `sam3_trt.py` adapter (follow RF-DETR TRT pattern)
4. Implement preprocessing for TRT path (different from PyTorch)
5. Implement postprocessing for TRT outputs
6. Test correctness (mask IoU, score correlation)
7. Benchmark vs PyTorch baseline

**Estimated effort:** 3-5 days (not a kernel-level optimization session)

## Recommended Next Steps

Since this is **autonomous mode**, I've made the following pragmatic decisions:

### Immediate Action: Document and Pause

1. ✅ Updated HANDOFF.md with SAM3 investigation findings
2. ✅ Created this detailed status document
3. ⏸️ **Paused SAM3 work** until clarification

### Alternative Productive Work

While awaiting clarification, I can:

**A. Profile other TRT models** that use shared infrastructure:
- YOLO26 (object detection, segmentation, keypoints)
- YOLOv12 (object detection)
- YOLONas (object detection)
- Yolact (instance segmentation)
- DeepLabV3+ (semantic segmentation)

These models **already benefit** from the 15 optimizations to shared preprocessing/postprocessing, but may have model-specific hotspots.

**B. Revisit next_priorities.md suggestions:**
- P1: FP16 TRT engine rebuild for RF-DETR/YOLOv8 (accuracy-gated)
- P2: EfficientNMS_TRT plugin for postprocess elimination
- P3: Sparsity flags / tactic tuning

**C. Extend optimizations to other pipelines:**
- Check if any TRT models still use non-optimized preprocessing (5 of 20 don't use shared path)
- Port optimizations to ONNX variants if they share code

## Questions for Clarification

1. **Was SAM3 the intended model?**
   - If not, which segmentation model should I optimize?

2. **If SAM3 confirmed, which path?**
   - PyTorch CUDA optimization (requires `sam3` package install)?
   - TRT backend implementation from scratch (multi-day feature)?

3. **Alternative priority?**
   - Should I proceed with `next_priorities.md` items (FP16 rebuild, NMS plugin)?
   - Should I profile other TRT models for additional wins?

## Contact Points

**Files modified this session:**
- `/home/ubuntu/inference/.codeflash/HANDOFF.md` — Investigation findings
- `/home/ubuntu/inference/.codeflash/SAM3_STATUS.md` — This document

**Key references:**
- Prior session HANDOFF: `.codeflash/HANDOFF.md` (lines 1-126)
- Prior session results: `.codeflash/results.tsv` (26 experiments)
- User priorities: `.codeflash/next_priorities.md`

**Blocking issue:** No TRT backend + package not installed = cannot proceed with "TRT GPU performance" optimization as stated.
