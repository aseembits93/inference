#!/usr/bin/env python3
"""Detailed torch.profiler run for deep look at remaining hotspots."""
import sys
import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity
from inference import get_model

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "yolov8n-640"
N_ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 50

model = get_model(model_id=MODEL_ID, api_key=None)
dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

# Warmup
for _ in range(30):
    _ = model.infer(dummy_image)
torch.cuda.synchronize()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=False,
) as prof:
    for _ in range(N_ITERS):
        _ = model.infer(dummy_image)
torch.cuda.synchronize()

print("\n=== Top 30 ops by CPU time (self) ===")
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))

print("\n=== Top 30 ops by CUDA time (self) ===")
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))

# group by op name, compute self cpu and cuda
from collections import defaultdict
stats = defaultdict(lambda: {"cpu": 0, "cuda": 0, "count": 0})
for e in prof.key_averages():
    key = e.key
    stats[key]["cpu"] += e.self_cpu_time_total
    stats[key]["cuda"] += e.self_device_time_total
    stats[key]["count"] += e.count

# top CPU self
print("\n=== Top by CPU self-time (aggregated) ===")
for k, v in sorted(stats.items(), key=lambda x: -x[1]['cpu'])[:20]:
    per_iter = (v['cpu'] / N_ITERS) / 1000  # ms per iteration
    print(f"  {k:50s}  {per_iter:7.3f}ms/iter  count={v['count']:5d}  cuda={(v['cuda']/N_ITERS)/1000:7.3f}ms/iter")
