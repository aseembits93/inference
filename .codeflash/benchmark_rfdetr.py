#!/usr/bin/env python3
"""Benchmark RF-DETR inference with proper warmup and statistics"""
import numpy as np
import statistics
import sys
from inference import get_model

def benchmark_model(model_id="rfdetr-base", iterations=100, warmup=20):
    """Benchmark a model's infer() method"""
    print(f"Loading model: {model_id}")
    model = get_model(model_id=model_id, api_key=None)

    # Create dummy input
    dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        _ = model.infer(dummy_image)

    # Benchmark
    print(f"Benchmarking ({iterations} iterations)...")
    times = []
    import time
    for _ in range(iterations):
        start = time.perf_counter()
        _ = model.infer(dummy_image)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    # Statistics
    mean_time = statistics.mean(times)
    median_time = statistics.median(times)
    stdev_time = statistics.stdev(times) if len(times) > 1 else 0
    min_time = min(times)
    max_time = max(times)

    print(f"\nResults for {model_id}:")
    print(f"  Mean:   {mean_time:.3f}ms")
    print(f"  Median: {median_time:.3f}ms")
    print(f"  Stdev:  {stdev_time:.3f}ms")
    print(f"  Min:    {min_time:.3f}ms")
    print(f"  Max:    {max_time:.3f}ms")

    return {
        'mean': mean_time,
        'median': median_time,
        'stdev': stdev_time,
        'min': min_time,
        'max': max_time
    }

if __name__ == "__main__":
    model_id = sys.argv[1] if len(sys.argv) > 1 else "rfdetr-base"
    iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    results = benchmark_model(model_id, iterations, warmup)
    sys.exit(0)
