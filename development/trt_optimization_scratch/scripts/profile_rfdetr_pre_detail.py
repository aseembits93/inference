#!/usr/bin/env python3
"""Find where RF-DETR preprocess time goes."""
import time
import numpy as np
import torch
from inference import get_model

model = get_model(model_id="rfdetr-base", api_key=None)
dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

for _ in range(20):
    _ = model.infer(dummy_image)
torch.cuda.synchronize()

# Inner model
inner = model._model

N = 500
full_times = []
pre_bare_times = []
for _ in range(N):
    torch.cuda.synchronize()
    s = time.perf_counter()
    _ = model.preprocess(dummy_image)
    torch.cuda.synchronize()
    full_times.append((time.perf_counter() - s) * 1000)

# Now just the inner pre_process (no adapter wrapper)
for _ in range(N):
    torch.cuda.synchronize()
    s = time.perf_counter()
    _ = inner.pre_process(dummy_image)
    torch.cuda.synchronize()
    pre_bare_times.append((time.perf_counter() - s) * 1000)

# Just the shared preprocess handler (no model wrapper)
from inference_models.models.common.roboflow.pre_processing import pre_process_network_input

device = torch.device("cuda:0")
handler_times = []
for _ in range(N):
    torch.cuda.synchronize()
    s = time.perf_counter()
    _ = pre_process_network_input(
        images=dummy_image,
        image_pre_processing=inner._inference_config.image_pre_processing,
        network_input=inner._inference_config.network_input,
        target_device=device,
    )
    torch.cuda.synchronize()
    handler_times.append((time.perf_counter() - s) * 1000)

# And just the raw cv2.resize + torch upload + affine
import cv2
from inference_models.models.common.roboflow.pre_processing import _numpy_to_device_via_pinned_buffer, _maybe_apply_scale_and_normalize

net_input = inner._inference_config.network_input

raw_times = []
with torch.inference_mode():
    for _ in range(N):
        torch.cuda.synchronize()
        s = time.perf_counter()
        # resize
        resized = cv2.resize(dummy_image, (560, 560))
        # upload
        t = _numpy_to_device_via_pinned_buffer(resized, device)
        # permute, unsqueeze
        t = t.unsqueeze(0).permute(0, 3, 1, 2)
        # swap if needed
        t = t[:, [2, 1, 0], :, :]
        # scaling + normalize
        t = _maybe_apply_scale_and_normalize(t, net_input.scaling_factor, net_input.normalization)
        torch.cuda.synchronize()
        raw_times.append((time.perf_counter() - s) * 1000)

def summary(name, times):
    a = np.array(times)
    print(f"{name}: mean={a.mean():.3f}ms median={np.median(a):.3f}ms std={a.std():.3f}ms")

summary("Adapter.preprocess   ", full_times)
summary("Inner.pre_process    ", pre_bare_times)
summary("Shared pre_process_   ", handler_times)
summary("Raw pipeline         ", raw_times)
