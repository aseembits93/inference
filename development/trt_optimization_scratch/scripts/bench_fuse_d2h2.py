#!/usr/bin/env python3
"""Micro-bench: fair comparison of D2H strategies."""
import time
import numpy as np
import torch

device = torch.device("cuda:0")
torch.randn(16, device=device)
torch.cuda.synchronize()

N = 5000

# Pre-create tensors to avoid measuring gen cost
tensors = []
for _ in range(N):
    n = 10
    xyxy = torch.rand(n, 4, device=device) * 640
    conf = torch.rand(n, device=device)
    class_id = torch.randint(0, 80, (n,), dtype=torch.int32, device=device)
    tensors.append((xyxy, conf, class_id))
torch.cuda.synchronize()

# Approach A: 3 separate D2H (the current impl)
def approach_a():
    for xyxy, conf, class_id in tensors:
        xyxy_np = xyxy.detach().cpu().numpy()
        conf_np = conf.detach().cpu().numpy()
        cls_np = class_id.detach().cpu().numpy()

# Approach B: 1 fused D2H
def approach_b():
    for xyxy, conf, class_id in tensors:
        packed = torch.cat([
            xyxy,
            conf.unsqueeze(1),
            class_id.float().unsqueeze(1),
        ], dim=1)
        packed_np = packed.detach().cpu().numpy()
        xyxy_np = packed_np[:, :4]
        conf_np = packed_np[:, 4]
        cls_np = packed_np[:, 5].astype(np.int32)

# Approach D: Use Tensor.to(non_blocking=True, ... host) then single sync
def approach_d():
    # Batch all D2H in a stream, then sync once at end
    results = []
    for xyxy, conf, class_id in tensors:
        xyxy_cpu = xyxy.to('cpu', non_blocking=True)
        conf_cpu = conf.to('cpu', non_blocking=True)
        cls_cpu = class_id.to('cpu', non_blocking=True)
        results.append((xyxy_cpu, conf_cpu, cls_cpu))
    torch.cuda.synchronize()

# warmup
for _ in range(5):
    approach_a()
for _ in range(5):
    approach_b()
torch.cuda.synchronize()

def bench(name, fn):
    torch.cuda.synchronize()
    s = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    total = (time.perf_counter() - s) * 1000
    print(f"{name}: {total/N*1000:.2f}us/call (total {total:.1f}ms for {N})")

bench("A 3x D2H        ", approach_a)
bench("B fused D2H     ", approach_b)
bench("D async+sync    ", approach_d)
