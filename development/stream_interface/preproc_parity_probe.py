"""Probe the preprocessed tensor produced by the Triton kernel vs
the F.interpolate reference path, pixel-by-pixel.

Loads one frame, runs both preprocessors, and prints:
  - max abs diff
  - count of mismatched fp32 elements
  - first mismatch location + value pair
"""
import argparse
import os

os.environ.setdefault(
    "ONNXRUNTIME_EXECUTION_PROVIDERS",
    "[TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider]",
)

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def pytorch_reference(frame_bgr: np.ndarray, target_h: int, target_w: int,
                      means, stds) -> torch.Tensor:
    """Mirror handle_tensor_input_preparation_with_stretch + the
    scaling_factor=255 + functional.normalize steps exactly."""
    t = torch.from_numpy(frame_bgr).cuda()
    t = t.permute(2, 0, 1).contiguous().unsqueeze(0)   # (1,3,H,W) uint8 BGR
    t = t.float()                                      # fp32 (0..255) BGR
    t = F.interpolate(t, size=(target_h, target_w), mode="bilinear")
    t = t[:, [2, 1, 0], :, :]                          # BGR -> RGB
    t = t / 255.0                                      # scaling_factor=255
    mean = torch.tensor(means, device=t.device).view(1, 3, 1, 1)
    std = torch.tensor(stds, device=t.device).view(1, 3, 1, 1)
    t = (t - mean) / std
    return t.contiguous()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_reference", required=True)
    ap.add_argument("--target_h", type=int, default=512)
    ap.add_argument("--target_w", type=int, default=512)
    ap.add_argument("--frame", type=int, default=0)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video_reference)
    for _ in range(args.frame + 1):
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("could not read frame")
    cap.release()
    print(f"frame shape {frame.shape} dtype {frame.dtype}")

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    ref = pytorch_reference(frame, args.target_h, args.target_w, means, stds)

    from inference_models.models.rfdetr.triton_preprocess import (
        triton_preprocess_rfdetr_stretch,
    )

    src_gpu = torch.from_numpy(frame).cuda()  # HWC uint8 BGR
    tri = triton_preprocess_rfdetr_stretch(
        src_gpu,
        target_h=args.target_h,
        target_w=args.target_w,
        means=means,
        stds=stds,
    )

    diff = (ref - tri).abs()
    print(f"ref shape {tuple(ref.shape)} dtype {ref.dtype}")
    print(f"tri shape {tuple(tri.shape)} dtype {tri.dtype}")
    print(f"max abs diff   : {diff.max().item():.3e}")
    print(f"mean abs diff  : {diff.mean().item():.3e}")
    print(f"n mismatched   : {(diff > 0).sum().item()}/{diff.numel()} "
          f"({100.0 * (diff > 0).sum().item() / diff.numel():.2f}%)")
    # Per-channel stats
    for c, name in enumerate(("R", "G", "B")):
        d = diff[0, c]
        print(f"  ch {name}: max {d.max().item():.3e} mean {d.mean().item():.3e} "
              f"n>0 {(d>0).sum().item()}/{d.numel()}")
    # First mismatch
    flat = diff.flatten()
    idx = (flat > 0).nonzero()
    if idx.numel():
        k = idx[0].item()
        print(f"first mismatch flat_idx={k} ref={ref.flatten()[k].item():.10f} "
              f"tri={tri.flatten()[k].item():.10f}")


if __name__ == "__main__":
    main()
