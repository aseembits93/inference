#!/usr/bin/env python3
"""Profile batch inference (realistic benchmark pattern)."""
import sys
import time
import cv2
import numpy as np
import torch
from inference import get_model

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "yolov8n-640"
BATCH_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 8
N_WARMUP = 20
N_ITERS = 100

model = get_model(model_id=MODEL_ID, api_key=None)

# Use real images if available
import glob
asset_imgs = glob.glob("tests/inference/models_predictions_tests/assets/*.jpg")
real_imgs = []
for p in asset_imgs[:BATCH_SIZE]:
    im = cv2.imread(p)
    if im is not None:
        real_imgs.append(im)

# Pad to batch size with first image
while len(real_imgs) < BATCH_SIZE:
    real_imgs.append(real_imgs[0])
real_imgs = real_imgs[:BATCH_SIZE]

print(f"Using {len(real_imgs)} real images, shapes: {[i.shape for i in real_imgs]}")

# warmup
for _ in range(N_WARMUP):
    _ = model.infer(real_imgs)
torch.cuda.synchronize()

times = []
for _ in range(N_ITERS):
    torch.cuda.synchronize()
    s = time.perf_counter()
    _ = model.infer(real_imgs)
    torch.cuda.synchronize()
    times.append((time.perf_counter() - s) * 1000)

t = np.array(times)
print(f"Batch={BATCH_SIZE} E2E: {t.mean():.3f}ms ± {t.std():.3f}ms median={np.median(t):.3f}ms")
print(f"Per-image: {t.mean()/BATCH_SIZE:.3f}ms")
