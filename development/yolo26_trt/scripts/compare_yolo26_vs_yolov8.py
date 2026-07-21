#!/usr/bin/env python3
"""
Compare YOLO26 vs YOLOv8n performance to identify YOLO26-specific overhead.
"""

import sys
import time
import statistics
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "inference_models"))


def quick_benchmark(model, name: str, iterations: int = 100) -> dict:
    """Quick benchmark to get mean/median/stdev."""
    img = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)

    # Warmup
    for _ in range(20):
        _ = model.infer(img)
    torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = model.infer(img)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "name": name,
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "stdev": statistics.stdev(times),
        "min": min(times),
        "max": max(times),
    }


def main():
    print("YOLO26 vs YOLOv8n Comparison")
    print("="*60)

    device = torch.device("cuda:0")

    # Load YOLOv8n (if available from prior sessions)
    yolov8n_path = Path.home() / ".cache" / "roboflow" / "yolov8n_trt"  # adjust path as needed
    yolo26_det_path = Path.home() / ".cache" / "roboflow" / "yolo26_trt_engines" / "yolo26-det"

    results = []

    # Try YOLOv8n
    if yolov8n_path.exists():
        try:
            print("\nBenchmarking YOLOv8n...")
            from inference_models.models.yolov8.yolov8_object_detection_trt import (
                YOLOv8ForObjectDetectionTRT,
            )
            yolov8n = YOLOv8ForObjectDetectionTRT.from_pretrained(
                model_name_or_path=str(yolov8n_path),
                device=device,
                engine_host_code_allowed=True,
            )
            result = quick_benchmark(yolov8n, "YOLOv8n", iterations=100)
            results.append(result)
            print(f"  Mean: {result['mean']:.3f}ms ± {result['stdev']:.3f}ms")
        except Exception as e:
            print(f"  ERROR: {e}")
    else:
        print(f"\nYOLOv8n not found at {yolov8n_path}, skipping comparison")

    # Benchmark YOLO26
    if yolo26_det_path.exists():
        try:
            print("\nBenchmarking YOLO26...")
            from inference_models.models.yolo26.yolo26_object_detection_trt import (
                YOLO26ForObjectDetectionTRT,
            )
            yolo26 = YOLO26ForObjectDetectionTRT.from_pretrained(
                model_name_or_path=str(yolo26_det_path),
                device=device,
                engine_host_code_allowed=True,
            )
            result = quick_benchmark(yolo26, "YOLO26", iterations=100)
            results.append(result)
            print(f"  Mean: {result['mean']:.3f}ms ± {result['stdev']:.3f}ms")
        except Exception as e:
            print(f"  ERROR: {e}")
    else:
        print(f"\nYOLO26 not found at {yolo26_det_path}, waiting for engines to build")

    # Comparison
    if len(results) == 2:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}\n")

        yolov8_result = next(r for r in results if "YOLOv8" in r["name"])
        yolo26_result = next(r for r in results if "YOLO26" in r["name"])

        diff_ms = yolo26_result["mean"] - yolov8_result["mean"]
        diff_pct = (diff_ms / yolov8_result["mean"]) * 100

        print(f"YOLOv8n: {yolov8_result['mean']:.3f}ms")
        print(f"YOLO26:  {yolo26_result['mean']:.3f}ms")
        print(f"Delta:   {diff_ms:+.3f}ms ({diff_pct:+.1f}%)")

        if abs(diff_pct) < 5:
            print(f"\n✓ Performance is within 5% - YOLO26 benefits fully from shared optimizations")
        elif diff_pct > 5:
            print(f"\n⚠ YOLO26 is {diff_pct:.1f}% slower - may have model-specific overhead")
            print("  Investigate: preprocessing differences, postprocessing overhead, engine efficiency")
        else:
            print(f"\n✓ YOLO26 is faster - possibly better engine optimization or simpler postprocessing")

    elif len(results) == 1:
        print("\nOnly one model available for benchmarking. Need both for comparison.")


if __name__ == "__main__":
    main()
