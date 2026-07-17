#!/usr/bin/env python3
"""Correctness check using real image with detections."""
import sys
import cv2
import numpy as np
import torch
from inference import get_model

IMAGE_PATH = "tests/inference/models_predictions_tests/assets/person_image.jpg"

MODEL_IDS = sys.argv[1:] if len(sys.argv) > 1 else ["yolov8n-640", "rfdetr-base", "yolov8n-seg-640"]

img = cv2.imread(IMAGE_PATH)
print(f"Image shape: {img.shape}")

for MODEL_ID in MODEL_IDS:
    print(f"\n=== {MODEL_ID} ===")
    model = get_model(model_id=MODEL_ID, api_key=None)

    # Run 3 times and confirm identical output
    results = []
    for _ in range(3):
        out = model.infer(img)
        torch.cuda.synchronize()
        resp = out[0] if isinstance(out, list) else out
        preds = resp.predictions
        results.append([(round(p.confidence, 4), p.class_id, round(p.x, 2), round(p.y, 2)) for p in preds])

    print(f"  n_predictions per run: {[len(r) for r in results]}")
    print(f"  run1: {results[0][:5]}")
    print(f"  run2: {results[1][:5]}")
    print(f"  all match: {results[0] == results[1] == results[2]}")
