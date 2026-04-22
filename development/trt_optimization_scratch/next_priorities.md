# Follow-up priorities (user-directed, 2026-04-21)

User wants kernel-level optimizations next. Prioritize in this order:

## P1: FP16 TensorRT engine rebuild
- Targets: RF-DETR, YOLOv8n, YOLOv8n-seg
- Locate engine build path (inference_models TRT builders). Check current precision; if FP32 or mixed, rebuild with `BuilderFlag.FP16` enabled.
- Gate on accuracy: run a fixed image set through FP32 and FP16 engines, compare top-K detections (IoU >= 0.95 on kept boxes, class-match >= 99%).
- Expected gain: 1.5–2x on GEMM-dominated layers (GEMMs are ~48% of RF-DETR GPU time).
- Also try `BuilderFlag.TF32` on Ampere+ as a free step.
- If the engine is loaded from a pre-built blob without source ONNX, document that as a blocker and skip.

## P2: EfficientNMS_TRT plugin for postprocess
- Target: RF-DETR (NumPy sigmoid + argpartition + argsort postprocess is ~20–30% of E2E per prior profile).
- Graft TensorRT's `EfficientNMS_TRT` plugin into the engine via onnx-graphsurgeon so decode + sigmoid + top-K + NMS run as one GPU op.
- Same accuracy gate as P1.
- Do the equivalent for YOLOv8n (NMS currently on CPU).

## P3: Sparsity / tactic flags
- `BuilderFlag.SPARSE_WEIGHTS` if weights are 2:4 sparse.
- Increase workspace size to unlock fused tactics.
- Only pursue if P1 and P2 didn't plateau.

Keep prior wins committed. Commit each accepted experiment separately on codeflash/optimize.
