#!/usr/bin/env python3
"""Stage-level profiling using adapter's preprocess/predict/postprocess."""
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
print(f"Model type: {type(model).__name__}")
print(f"Inner model: {type(model._model).__name__ if hasattr(model, '_model') else 'N/A'}")

dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

print(f"Warmup ({N_WARMUP})")
for _ in range(N_WARMUP):
    _ = model.infer(dummy_image)
torch.cuda.synchronize()

# E2E
times = []
for _ in range(N_ITERS):
    torch.cuda.synchronize()
    s = time.perf_counter()
    _ = model.infer(dummy_image)
    torch.cuda.synchronize()
    times.append((time.perf_counter() - s) * 1000)
e2e = np.array(times)
print(f"E2E: {e2e.mean():.3f}ms ± {e2e.std():.3f}ms median={np.median(e2e):.3f}ms")

# Use adapter-level stages
pre_times, pred_times, post_times = [], [], []
for _ in range(N_ITERS):
    torch.cuda.synchronize()
    s1 = time.perf_counter()
    out_pre = model.preprocess(dummy_image)
    torch.cuda.synchronize()
    s2 = time.perf_counter()
    out_pred = model.predict(out_pre[0])
    torch.cuda.synchronize()
    s3 = time.perf_counter()
    _ = model.postprocess(out_pred, out_pre[1])
    torch.cuda.synchronize()
    s4 = time.perf_counter()
    pre_times.append((s2 - s1) * 1000)
    pred_times.append((s3 - s2) * 1000)
    post_times.append((s4 - s3) * 1000)

pre_a = np.array(pre_times)
pred_a = np.array(pred_times)
post_a = np.array(post_times)
print(f"preprocess:  {pre_a.mean():.3f}ms ± {pre_a.std():.3f}ms (median {np.median(pre_a):.3f}ms)")
print(f"predict:     {pred_a.mean():.3f}ms ± {pred_a.std():.3f}ms (median {np.median(pred_a):.3f}ms)")
print(f"postprocess: {post_a.mean():.3f}ms ± {post_a.std():.3f}ms (median {np.median(post_a):.3f}ms)")
print(f"sum:         {(pre_a.mean()+pred_a.mean()+post_a.mean()):.3f}ms")
print(f"overhead vs E2E: {(e2e.mean() - (pre_a.mean()+pred_a.mean()+post_a.mean())):.3f}ms")
