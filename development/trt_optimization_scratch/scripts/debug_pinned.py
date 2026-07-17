#!/usr/bin/env python3
"""Check what's calling the H2D transfers."""
import numpy as np
import torch
import tracemalloc
from inference import get_model

model = get_model(model_id="yolov8n-640", api_key=None)
dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

# warmup
for _ in range(10):
    _ = model.infer(dummy_image)
torch.cuda.synchronize()

# Hook into torch copy to count
import torch.overrides
orig_to = torch.Tensor.to

h2d_calls = []

def traced_to(self, *args, **kwargs):
    if len(args) > 0 and isinstance(args[0], (torch.device, str)):
        target = args[0]
    else:
        target = kwargs.get("device", None)
    is_h2d = (self.device.type == "cpu"
              and ((isinstance(target, torch.device) and target.type == "cuda")
                   or (isinstance(target, str) and "cuda" in target)))
    if is_h2d:
        import traceback
        stack = traceback.extract_stack(limit=10)
        h2d_calls.append((tuple(self.shape), self.dtype, self.is_pinned(), stack[-3:]))
    return orig_to(self, *args, **kwargs)

torch.Tensor.to = traced_to

# one call
_ = model.infer(dummy_image)
torch.cuda.synchronize()

torch.Tensor.to = orig_to

print(f"H2D .to() calls for one infer: {len(h2d_calls)}")
for shape, dtype, is_pinned, stack in h2d_calls:
    print(f"  shape={shape} dtype={dtype} pinned={is_pinned}")
    for frame in stack:
        print(f"    {frame.filename}:{frame.lineno}: {frame.name}")
    print()
