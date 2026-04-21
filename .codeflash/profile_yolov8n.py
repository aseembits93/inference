#!/usr/bin/env python3
"""Profile YOLOv8n inference pipeline with torch.profiler"""
import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity
import time

from inference import get_model

# Load model
print("Loading YOLOv8n model...")
model = get_model(model_id="yolov8n-640", api_key=None)

# Create dummy input
dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

# Warmup
print("Warming up (20 iterations)...")
for _ in range(20):
    _ = model.infer(dummy_image)

if torch.cuda.is_available():
    torch.cuda.synchronize()

# Baseline timing
print("\nBaseline timing (20 iterations after warmup)...")
times = []
for _ in range(20):
    start = time.perf_counter()
    _ = model.infer(dummy_image)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()
    times.append((end - start) * 1000)  # convert to ms

mean_time = np.mean(times)
std_time = np.std(times)
print(f"Baseline: {mean_time:.3f}ms ± {std_time:.3f}ms")

print("\nProfiling with torch.profiler...")
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_stack=True,
    record_shapes=True,
    profile_memory=True,
) as prof:
    for _ in range(10):
        _ = model.infer(dummy_image)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

print("\n" + "="*80)
print("Top 30 operations by CUDA time:")
print("="*80)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

print("\n" + "="*80)
print("Top 30 operations by CPU time:")
print("="*80)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

print("\n" + "="*80)
print("Top 20 memory allocations:")
print("="*80)
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))

# Export trace for TensorBoard if needed
prof.export_chrome_trace("/tmp/yolov8n_trace.json")
print("\nTrace exported to /tmp/yolov8n_trace.json")
