#!/usr/bin/env python3
"""Profile just the preprocess stage."""
import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity
from inference import get_model

model = get_model(model_id="yolov8n-640", api_key=None)
dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

# warmup
for _ in range(30):
    _ = model.infer(dummy_image)
torch.cuda.synchronize()

# Profile pre_process only
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=False,
) as prof:
    for _ in range(100):
        _ = model.preprocess(dummy_image)
torch.cuda.synchronize()

print("\n=== Top 30 CPU (self) ===")
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))

print("\n=== Top 15 CUDA (self) ===")
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
