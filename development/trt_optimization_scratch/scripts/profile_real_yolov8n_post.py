#!/usr/bin/env python3
"""Detailed profile of YOLOv8n postprocess with real image (with detections)."""
import cv2
import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity
from inference import get_model

model = get_model(model_id="yolov8n-640", api_key=None)
img = cv2.imread("tests/inference/models_predictions_tests/assets/person_image.jpg")

for _ in range(30):
    _ = model.infer(img)
torch.cuda.synchronize()

inner = model._model
pre_out = model.preprocess(img)
torch.cuda.synchronize()
pred_out = model.predict(pre_out[0])
torch.cuda.synchronize()

# Profile the full postprocess (adapter + inner)
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=False,
) as prof:
    for _ in range(100):
        pred_copy = pred_out.clone() if hasattr(pred_out, 'clone') else [t.clone() for t in pred_out]
        _ = model.postprocess(pred_copy, pre_out[1])
torch.cuda.synchronize()

print("\n=== Top 30 CPU (self) ===")
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))
