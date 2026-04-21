#!/usr/bin/env python3
"""
Benchmark YOLO26 TRT to verify that shared preprocessing/postprocessing
optimizations from the YOLOv8/RF-DETR session also benefit YOLO26.
"""

import time
import numpy as np
import torch

# Attempt to import - YOLO26 may not be in path
try:
    from inference_models.models.yolo26.yolo26_object_detection_trt import (
        YOLO26ForObjectDetectionTRT,
    )
    YOLO26_AVAILABLE = True
except ImportError as e:
    print(f"YOLO26 not available: {e}")
    YOLO26_AVAILABLE = False


def benchmark_yolo26(model_path: str, num_iters: int = 200, warmup: int = 50):
    """Benchmark YOLO26 TRT inference E2E."""
    if not YOLO26_AVAILABLE:
        print("YOLO26 TRT not available, skipping.")
        return None

    print(f"Loading YOLO26 from {model_path}")
    device = torch.device("cuda:0")

    try:
        model = YOLO26ForObjectDetectionTRT.from_pretrained(
            model_name_or_path=model_path,
            device=device,
        )
    except Exception as e:
        print(f"Failed to load YOLO26: {e}")
        return None

    # Random 640x640 RGB image
    img = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)

    print(f"Warmup {warmup} iterations...")
    for _ in range(warmup):
        _ = model.infer(img)

    torch.cuda.synchronize()

    print(f"Benchmarking {num_iters} iterations...")
    times = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        _ = model.infer(img)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    print(f"\nYOLO26 E2E (single 640x640 image):")
    print(f"  Mean:   {times.mean():.3f}ms ± {times.std():.3f}ms")
    print(f"  Median: {np.median(times):.3f}ms")
    print(f"  P95:    {np.percentile(times, 95):.3f}ms")
    print(f"  Min:    {times.min():.3f}ms")
    print(f"  Max:    {times.max():.3f}ms")

    return times.mean()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python bench_yolo26.py <model_path>")
        print("Example: python bench_yolo26.py /path/to/yolo26n-640-trt")
        sys.exit(1)

    model_path = sys.argv[1]
    benchmark_yolo26(model_path)
