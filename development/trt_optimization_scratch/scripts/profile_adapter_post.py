#!/usr/bin/env python3
"""Profile adapter postprocess vs model post_process."""
import time
import numpy as np
import torch
from inference import get_model

model = get_model(model_id="yolov8n-640", api_key=None)
dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

for _ in range(30):
    _ = model.infer(dummy_image)
torch.cuda.synchronize()

# Isolate internal post_process vs adapter overhead
inner = model._model

N = 300
pre_out = model.preprocess(dummy_image)
torch.cuda.synchronize()
pred_out = model.predict(pre_out[0])
torch.cuda.synchronize()

# Inner post_process only
inner_times = []
for _ in range(N):
    pred_copy = pred_out.clone() if hasattr(pred_out, 'clone') else [t.clone() for t in pred_out]
    torch.cuda.synchronize()
    s = time.perf_counter()
    _ = inner.post_process(pred_copy, pre_out[1])
    torch.cuda.synchronize()
    inner_times.append((time.perf_counter() - s) * 1000)

# Full adapter postprocess (includes formatting)
adapter_times = []
for _ in range(N):
    pred_copy = pred_out.clone() if hasattr(pred_out, 'clone') else [t.clone() for t in pred_out]
    torch.cuda.synchronize()
    s = time.perf_counter()
    _ = model.postprocess(pred_copy, pre_out[1])
    torch.cuda.synchronize()
    adapter_times.append((time.perf_counter() - s) * 1000)

inner_a = np.array(inner_times)
adapter_a = np.array(adapter_times)
print(f"inner post_process:  {inner_a.mean():.3f}ms (median {np.median(inner_a):.3f}ms)")
print(f"adapter postprocess: {adapter_a.mean():.3f}ms (median {np.median(adapter_a):.3f}ms)")
print(f"adapter overhead:    {(adapter_a.mean() - inner_a.mean()):.3f}ms")
