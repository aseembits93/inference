#!/usr/bin/env python3
"""Micro-bench: 3 separate D2H vs 1 fused D2H."""
import time
import numpy as np
import torch

device = torch.device("cuda:0")
torch.randn(16, device=device)
torch.cuda.synchronize()

N = 5000

# Simulate typical post-NMS output
def make():
    n = 10  # few detections
    xyxy = torch.rand(n, 4, device=device) * 640
    conf = torch.rand(n, device=device)
    class_id = torch.randint(0, 80, (n,), dtype=torch.int32, device=device)
    return xyxy, conf, class_id

# Approach A: 3 separate D2H
def approach_a():
    xyxy, conf, class_id = make()
    xyxy_np = xyxy.detach().cpu().numpy()
    conf_np = conf.detach().cpu().numpy()
    cls_np = class_id.detach().cpu().numpy()

# Approach B: 1 fused D2H via concatenation
def approach_b():
    xyxy, conf, class_id = make()
    # pack: (n, 4+1+1) all float32
    packed = torch.cat([
        xyxy,
        conf.unsqueeze(1),
        class_id.float().unsqueeze(1),
    ], dim=1)
    packed_np = packed.detach().cpu().numpy()
    xyxy_np = packed_np[:, :4]
    conf_np = packed_np[:, 4]
    cls_np = packed_np[:, 5].astype(np.int32)

# Approach C: non_blocking with pinned buffer
pinned_buf = torch.empty((100, 6), dtype=torch.float32, pin_memory=True)
def approach_c():
    xyxy, conf, class_id = make()
    n = xyxy.shape[0]
    packed = torch.cat([xyxy, conf.unsqueeze(1), class_id.float().unsqueeze(1)], dim=1)
    buf_slice = pinned_buf[:n]
    buf_slice.copy_(packed, non_blocking=True)
    torch.cuda.synchronize()
    packed_np = buf_slice.numpy()

# warmup
for _ in range(100):
    approach_a(); approach_b(); approach_c()
torch.cuda.synchronize()

def bench(name, fn):
    torch.cuda.synchronize()
    s = time.perf_counter()
    for _ in range(N):
        fn()
    torch.cuda.synchronize()
    total = (time.perf_counter() - s) * 1000
    print(f"{name}: {total/N*1000:.2f}us/call")

bench("A 3x D2H       ", approach_a)
bench("B fused D2H    ", approach_b)
bench("C pinned D2H   ", approach_c)
