#!/usr/bin/env python3
"""Profile postprocess only."""
import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity
from inference import get_model

model = get_model(model_id="yolov8n-640", api_key=None)
dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

for _ in range(30):
    _ = model.infer(dummy_image)
torch.cuda.synchronize()

# Get the intermediate tensors once
pre_out = model.preprocess(dummy_image)
torch.cuda.synchronize()
pred_out = model.predict(pre_out[0])
torch.cuda.synchronize()

# warmup just postprocess
for _ in range(20):
    _ = model.postprocess(pred_out, pre_out[1])
torch.cuda.synchronize()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=False,
) as prof:
    for _ in range(100):
        # Need fresh pred since postprocess modifies in-place
        torch.cuda.synchronize()
        pred_copy = [t.clone() for t in pred_out] if isinstance(pred_out, list) else pred_out.clone()
        _ = model.postprocess(pred_copy, pre_out[1])
torch.cuda.synchronize()

print("\n=== Top 30 CPU (self) ===")
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))
