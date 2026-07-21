#!/usr/bin/env python3
"""
Final YOLO26 TRT benchmark using locally-built engines.
Establishes baselines for all three variants and identifies the heaviest one.
"""

import os
import sys
import time
import json
import statistics
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

# Add inference_models to path
sys.path.insert(0, str(Path(__file__).parent.parent / "inference_models"))

from inference_models.models.yolo26.yolo26_object_detection_trt import (
    YOLO26ForObjectDetectionTRT,
)
from inference_models.models.yolo26.yolo26_instance_segmentation_trt import (
    YOLO26ForInstanceSegmentationTRT,
)
from inference_models.models.yolo26.yolo26_key_points_detection_trt import (
    YOLO26ForKeyPointsDetectionTRT,
)


def benchmark_model(
    model,
    model_name: str,
    num_iters: int = 200,
    warmup: int = 50,
    batch_size: int = 1,
) -> Dict:
    """Benchmark a YOLO26 TRT model and return statistics."""
    # Generate random input images
    if batch_size == 1:
        img = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
    else:
        img = [
            np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
            for _ in range(batch_size)
        ]

    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name} (batch={batch_size})")
    print(f"{'='*60}")

    # Warmup
    print(f"Warmup: {warmup} iterations...")
    for _ in range(warmup):
        _ = model.infer(img)
    torch.cuda.synchronize()

    # Benchmark
    print(f"Measuring: {num_iters} iterations...")
    times = []
    for i in range(num_iters):
        t0 = time.perf_counter()
        _ = model.infer(img)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{num_iters}")

    # Statistics
    mean = statistics.mean(times)
    median = statistics.median(times)
    stdev = statistics.stdev(times)
    p95 = np.percentile(times, 95)
    min_time = min(times)
    max_time = max(times)

    print(f"\nResults:")
    print(f"  Mean:   {mean:.3f}ms ± {stdev:.3f}ms")
    print(f"  Median: {median:.3f}ms")
    print(f"  P95:    {p95:.3f}ms")
    print(f"  Min:    {min_time:.3f}ms")
    print(f"  Max:    {max_time:.3f}ms")

    return {
        "model": model_name,
        "batch_size": batch_size,
        "num_iters": num_iters,
        "mean_ms": mean,
        "median_ms": median,
        "stdev_ms": stdev,
        "p95_ms": p95,
        "min_ms": min_time,
        "max_ms": max_time,
        "times": times,
    }


def main():
    print("YOLO26 TRT Baseline Benchmark")
    print("="*60)
    print("Using locally-built TRT engines for L4 GPU (compute 8.9)")
    print()

    device = torch.device("cuda:0")
    engine_base = Path.home() / ".cache" / "roboflow" / "yolo26_trt_engines"

    # Model configurations
    models_config = [
        {
            "name": "yolo26-det",
            "class": YOLO26ForObjectDetectionTRT,
            "path": engine_base / "yolo26-det",
        },
        {
            "name": "yolo26-seg",
            "class": YOLO26ForInstanceSegmentationTRT,
            "path": engine_base / "yolo26-seg",
        },
        {
            "name": "yolo26-pose",
            "class": YOLO26ForKeyPointsDetectionTRT,
            "path": engine_base / "yolo26-pose",
        },
    ]

    all_results = []

    for config in models_config:
        print(f"\n{'#'*60}")
        print(f"# {config['name'].upper()}")
        print(f"{'#'*60}")

        if not config["path"].exists():
            print(f"ERROR: Engine directory not found: {config['path']}")
            continue

        try:
            # Load model
            print(f"\nLoading model from {config['path']}...")
            model = config["class"].from_pretrained(
                model_name_or_path=str(config["path"]),
                device=device,
                engine_host_code_allowed=True,
            )
            print("Model loaded successfully!")

            # Benchmark single image
            result_single = benchmark_model(
                model=model,
                model_name=config["name"],
                num_iters=200,
                warmup=50,
                batch_size=1,
            )
            all_results.append(result_single)

            # Benchmark batch=8
            result_batch = benchmark_model(
                model=model,
                model_name=config["name"],
                num_iters=100,
                warmup=20,
                batch_size=8,
            )
            all_results.append(result_batch)

        except Exception as e:
            print(f"\nERROR benchmarking {config['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    print(f"\n{'='*60}")
    print("BASELINE SUMMARY")
    print(f"{'='*60}\n")

    print(f"{'Model':<20} {'Batch':<8} {'Mean (ms)':<12} {'Median (ms)':<12} {'Stdev (ms)':<12}")
    print("-" * 64)
    for result in all_results:
        print(
            f"{result['model']:<20} "
            f"{result['batch_size']:<8} "
            f"{result['mean_ms']:>10.3f}  "
            f"{result['median_ms']:>10.3f}  "
            f"{result['stdev_ms']:>10.3f}"
        )

    # Identify heaviest model
    single_results = [r for r in all_results if r["batch_size"] == 1]
    if single_results:
        heaviest = max(single_results, key=lambda r: r["mean_ms"])
        print(f"\n{'='*60}")
        print(f"PRIMARY OPTIMIZATION TARGET")
        print(f"{'='*60}")
        print(f"Heaviest model: {heaviest['model']} ({heaviest['mean_ms']:.3f}ms mean)")
        print(f"This will be the primary target for optimization experiments.")

    # Save results
    output_file = Path(__file__).parent / "yolo26_baseline_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")

    # Update results.tsv with baselines
    tsv_file = Path(__file__).parent / "results.tsv"
    with open(tsv_file, 'a') as f:
        for result in single_results:
            f.write(
                f"baseline-{result['model']}\tbaseline\t-\t"
                f"{result['model']}.infer()\t"
                f"{result['mean_ms']:.3f}ms±{result['stdev_ms']:.3f}ms\t-\t-\t"
                f"2026-04-21T23:00:00\t-\n"
            )
    print(f"Baselines appended to: {tsv_file}")

    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print(f"{'='*60}")
    print("1. Profile the heaviest model with torch.profiler")
    print("2. Compare against YOLOv8n baseline to identify YOLO26-specific hotspots")
    print("3. Begin optimization experiments")


if __name__ == "__main__":
    main()
