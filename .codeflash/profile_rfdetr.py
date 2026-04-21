#!/usr/bin/env python3
"""Profile RF-DETR inference pipeline with torch.profiler"""
import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity

from inference import get_model

# Load model
print("Loading RF-DETR model...")
model = get_model(model_id="rfdetr-base", api_key=None)

# Create dummy input
dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

# Warmup
print("Warming up (20 iterations)...")
for _ in range(20):
    _ = model.infer(dummy_image)

if torch.cuda.is_available():
    torch.cuda.synchronize()

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
prof.export_chrome_trace("/tmp/rfdetr_trace.json")
print("\nTrace exported to /tmp/rfdetr_trace.json")
