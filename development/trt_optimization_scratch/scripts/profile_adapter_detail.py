#!/usr/bin/env python3
"""Profile adapter postprocess internal breakdown."""
import time
import cv2
import numpy as np
import torch
from inference import get_model

IMAGE_PATH = "tests/inference/models_predictions_tests/assets/person_image.jpg"

model = get_model(model_id="yolov8n-640", api_key=None)
img = cv2.imread(IMAGE_PATH)

for _ in range(20):
    _ = model.infer(img)
torch.cuda.synchronize()

# Take apart the adapter postprocess
inner = model._model
pre_out = model.preprocess(img)
torch.cuda.synchronize()
pred_out = model.predict(pre_out[0])
torch.cuda.synchronize()

N = 500

# inner postprocess time
inner_times = []
for _ in range(N):
    pred_copy = pred_out.clone() if hasattr(pred_out, 'clone') else [t.clone() for t in pred_out]
    torch.cuda.synchronize()
    s = time.perf_counter()
    inner_out = inner.post_process(pred_copy, pre_out[1])
    torch.cuda.synchronize()
    inner_times.append((time.perf_counter() - s) * 1000)

# Now the rest (adapter iteration)
# We have inner_out (list of Detections)
# Re-run full adapter postprocess
full_times = []
cpu_transfer_times = []
iter_times = []
for _ in range(N):
    pred_copy = pred_out.clone() if hasattr(pred_out, 'clone') else [t.clone() for t in pred_out]
    torch.cuda.synchronize()
    s = time.perf_counter()

    inner_det = inner.post_process(pred_copy, pre_out[1])
    torch.cuda.synchronize()
    s2 = time.perf_counter()

    # CPU transfer
    xyxys = [d.xyxy.detach().cpu().numpy() for d in inner_det]
    confs = [d.confidence.detach().cpu().numpy() for d in inner_det]
    cids = [d.class_id.detach().cpu().numpy() for d in inner_det]
    torch.cuda.synchronize()
    s3 = time.perf_counter()

    # Python iteration + object construction
    from inference.core.entities.responses.inference import ObjectDetectionPrediction
    for xyxy, conf, cids_arr in zip(xyxys, confs, cids):
        predictions = []
        for (x1, y1, x2, y2), c, cid in zip(xyxy, conf, cids_arr):
            cx = (float(x1) + float(x2)) / 2.0
            cy = (float(y1) + float(y2)) / 2.0
            w = float(x2) - float(x1)
            h = float(y2) - float(y1)
            cid_int = int(cid)
            predictions.append(ObjectDetectionPrediction(
                x=cx, y=cy, width=w, height=h,
                confidence=float(c),
                class_id=cid_int,
                **{"class": model.class_names[cid_int]},
            ))
    s4 = time.perf_counter()

    full_times.append((s4 - s) * 1000)
    cpu_transfer_times.append((s3 - s2) * 1000)
    iter_times.append((s4 - s3) * 1000)

inner_a = np.array(inner_times)
cpu_a = np.array(cpu_transfer_times)
iter_a = np.array(iter_times)
full_a = np.array(full_times)

print(f"inner post_process: {inner_a.mean():.3f}ms (median {np.median(inner_a):.3f})")
print(f"D2H transfer:       {cpu_a.mean():.3f}ms")
print(f"Python iteration:   {iter_a.mean():.3f}ms")
print(f"total adapter:      {full_a.mean():.3f}ms")
print(f"n_predictions: {len(xyxy)}")
