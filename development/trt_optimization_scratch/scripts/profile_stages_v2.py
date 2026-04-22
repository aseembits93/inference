#!/usr/bin/env python3
"""Stage-level profiling of .infer() for TRT models."""
import sys
import time

import numpy as np
import torch

from inference import get_model

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "yolov8n-640"
N_WARMUP = int(sys.argv[2]) if len(sys.argv) > 2 else 50
N_ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 200

print(f"=== Loading {MODEL_ID} ===")
model = get_model(model_id=MODEL_ID, api_key=None)

# Inspect model object
print(f"Model type: {type(model).__name__}")

# Generic image; real model uses letterbox/stretch so use 640x640 as a common size
dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

# Warmup
print(f"Warmup ({N_WARMUP} iters)...")
for _ in range(N_WARMUP):
    _ = model.infer(dummy_image)
torch.cuda.synchronize()

# E2E timing
print(f"E2E timing ({N_ITERS} iters)...")
times = []
for _ in range(N_ITERS):
    torch.cuda.synchronize()
    start = time.perf_counter()
    _ = model.infer(dummy_image)
    torch.cuda.synchronize()
    end = time.perf_counter()
    times.append((end - start) * 1000)

times = np.array(times)
print(f"E2E: mean={times.mean():.3f}ms std={times.std():.3f}ms "
      f"median={np.median(times):.3f}ms p10={np.percentile(times, 10):.3f}ms "
      f"p90={np.percentile(times, 90):.3f}ms")

# Now stage-level breakdown by calling internals
# Find pre_process / forward / post_process or _predict / _make_prediction structure
print("\n=== Stage breakdown ===")

# Check whether this model has pre_process/forward/post_process (new inference_models API)
from inference.models.utils import ROBOFLOW_MODEL_TYPES
has_new_api = hasattr(model, "pre_process") and hasattr(model, "forward") and hasattr(model, "post_process")

if has_new_api:
    print("Using pre_process/forward/post_process API")
    # Pre-bind methods
    pre_process = model.pre_process
    forward = model.forward
    post_process = model.post_process

    pre_times, fwd_times, post_times = [], [], []
    for _ in range(N_ITERS):
        torch.cuda.synchronize()
        s1 = time.perf_counter()
        pre_out = pre_process(dummy_image)
        torch.cuda.synchronize()
        s2 = time.perf_counter()
        fwd_out = forward(pre_out[0])
        torch.cuda.synchronize()
        s3 = time.perf_counter()
        _ = post_process(fwd_out, pre_out[1])
        torch.cuda.synchronize()
        s4 = time.perf_counter()
        pre_times.append((s2 - s1) * 1000)
        fwd_times.append((s3 - s2) * 1000)
        post_times.append((s4 - s3) * 1000)
    pre_a = np.array(pre_times)
    fwd_a = np.array(fwd_times)
    post_a = np.array(post_times)
    print(f"pre_process: {pre_a.mean():.3f}ms ± {pre_a.std():.3f}ms "
          f"(median {np.median(pre_a):.3f}ms)")
    print(f"forward:     {fwd_a.mean():.3f}ms ± {fwd_a.std():.3f}ms "
          f"(median {np.median(fwd_a):.3f}ms)")
    print(f"post_process:{post_a.mean():.3f}ms ± {post_a.std():.3f}ms "
          f"(median {np.median(post_a):.3f}ms)")
    print(f"sum-of-stages: {(pre_a.mean() + fwd_a.mean() + post_a.mean()):.3f}ms")
else:
    print("No pre_process/forward/post_process, inspecting attributes:")
    print([a for a in dir(model) if not a.startswith('_')][:40])
