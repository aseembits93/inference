#!/usr/bin/env python3
"""
Profile YOLO26 TRT inference with torch.profiler to identify optimization targets.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity

# Add inference_models to path
sys.path.insert(0, str(Path(__file__).parent.parent / "inference_models"))


def profile_model(model, model_name: str, num_iterations: int = 10):
    """Profile a model's inference with torch.profiler."""
    print(f"\n{'='*60}")
    print(f"Profiling: {model_name}")
    print(f"{'='*60}\n")

    # Random input image
    img = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)

    # Warmup
    print("Warming up...")
    for _ in range(20):
        _ = model.infer(img)
    torch.cuda.synchronize()

    # Profile
    print(f"Profiling {num_iterations} iterations...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=True,
        record_shapes=True,
        profile_memory=False,
    ) as prof:
        for _ in range(num_iterations):
            _ = model.infer(img)

    # Print results sorted by CUDA time
    print(f"\n{'='*60}")
    print(f"Top 30 operations by CUDA time")
    print(f"{'='*60}\n")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=30,
        max_src_column_width=60,
    ))

    # Print results sorted by CPU time
    print(f"\n{'='*60}")
    print(f"Top 30 operations by CPU time")
    print(f"{'='*60}\n")
    print(prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=30,
        max_src_column_width=60,
    ))

    # Look for memory transfers
    print(f"\n{'='*60}")
    print(f"Memory transfer operations (H2D, D2H, D2D)")
    print(f"{'='*60}\n")
    events = prof.key_averages()
    transfers = [e for e in events if 'copy' in e.key.lower() or 'memcpy' in e.key.lower()]
    if transfers:
        for event in transfers[:20]:
            print(f"{event.key:<60} {event.count:>6}x  CUDA: {event.cuda_time_total/1000:>8.3f}ms")
    else:
        print("No explicit memory transfer operations found in profile.")


def main():
    if len(sys.argv) < 3:
        print("Usage: python profile_yolo26.py <model_type> <engine_path>")
        print()
        print("model_type: det, seg, or pose")
        print("engine_path: path to TRT engine directory")
        print()
        print("Example:")
        print("  python profile_yolo26.py det ~/.cache/roboflow/yolo26_trt_engines/yolo26-det")
        sys.exit(1)

    model_type = sys.argv[1]
    engine_path = sys.argv[2]

    device = torch.device("cuda:0")

    # Load appropriate model class
    if model_type == "det":
        from inference_models.models.yolo26.yolo26_object_detection_trt import (
            YOLO26ForObjectDetectionTRT,
        )
        model_class = YOLO26ForObjectDetectionTRT
        model_name = "YOLO26 Object Detection"
    elif model_type == "seg":
        from inference_models.models.yolo26.yolo26_instance_segmentation_trt import (
            YOLO26ForInstanceSegmentationTRT,
        )
        model_class = YOLO26ForInstanceSegmentationTRT
        model_name = "YOLO26 Instance Segmentation"
    elif model_type == "pose":
        from inference_models.models.yolo26.yolo26_key_points_detection_trt import (
            YOLO26ForKeyPointsDetectionTRT,
        )
        model_class = YOLO26ForKeyPointsDetectionTRT
        model_name = "YOLO26 Keypoints Detection"
    else:
        print(f"ERROR: Unknown model type '{model_type}'. Use 'det', 'seg', or 'pose'.")
        sys.exit(1)

    # Load model
    print(f"Loading {model_name} from {engine_path}...")
    model = model_class.from_pretrained(
        model_name_or_path=engine_path,
        device=device,
        engine_host_code_allowed=True,
    )
    print("Model loaded successfully!\n")

    # Profile
    profile_model(model, model_name, num_iterations=10)

    print(f"\n{'='*60}")
    print("Profile complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
