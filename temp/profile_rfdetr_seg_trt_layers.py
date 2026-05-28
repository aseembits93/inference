import argparse
from collections import defaultdict

import cv2
import torch
import tensorrt as trt

from inference_models import AutoModel


class LayerProfiler(trt.IProfiler):
    def __init__(self) -> None:
        super().__init__()
        self.rows = []

    def report_layer_time(self, layer_name: str, ms: float) -> None:
        self.rows.append((layer_name, ms))


def load_frame(video_path: str):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read first frame from {video_path}.")
    return frame


def bucket_layer(name: str) -> str:
    if "_gemm_mha_v2" in name:
        return "mha_gemm"
    if "mlp/fc2" in name:
        return "encoder_mlp_fc2"
    if "mlp/fc1" in name:
        return "encoder_mlp_fc1"
    if "attention" in name.lower():
        return "attention_other"
    if "segmentation_head" in name or "mask_embed" in name.lower():
        return "segmentation_head"
    if "/Conv" in name or "projector" in name:
        return "conv"
    if "MatMul" in name or "_myl_Fc_" in name or "Gemm" in name:
        return "other_matmul"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_reference",
        default="/home/ubuntu/inference/vehicles_312px.mp4",
    )
    parser.add_argument("--model_id", default="rfdetr-seg-nano")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--top_n", type=int, default=30)
    args = parser.parse_args()

    frame = load_frame(args.video_reference)
    model = AutoModel.from_pretrained(
        model_id_or_path=args.model_id,
        device=torch.device(args.device),
        backend="trt",
    )
    for _ in range(args.warmup):
        _ = model(frame)

    context = getattr(model, "_execution_context", None)
    if context is None:
        raise RuntimeError("Model does not expose a TensorRT execution context.")

    profiler = LayerProfiler()
    context.profiler = profiler
    context.enqueue_emits_profile = True
    _ = model(frame)

    total_ms = sum(ms for _, ms in profiler.rows)
    print(f"total_ms={total_ms:.4f}")

    per_bucket = defaultdict(float)
    for name, ms in profiler.rows:
        per_bucket[bucket_layer(name)] += ms

    print("\nBuckets:")
    for bucket, ms in sorted(per_bucket.items(), key=lambda item: item[1], reverse=True):
        pct = 100.0 * ms / total_ms if total_ms > 0 else 0.0
        print(f"{ms:.4f} ms\t{pct:5.1f}%\t{bucket}")

    print("\nTop layers:")
    top_rows = sorted(profiler.rows, key=lambda item: item[1], reverse=True)[: args.top_n]
    for name, ms in top_rows:
        print(f"{ms:.4f} ms\t{name}")


if __name__ == "__main__":
    main()
