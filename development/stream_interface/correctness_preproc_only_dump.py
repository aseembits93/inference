"""Isolate the preproc change.

Both runs use the *identical* forward/post_process pipeline on the
native TRT engine; the only difference is the function that fills the
preprocessed (1, 3, H, W) fp32 tensor:

  --preproc ref    : torch.from_numpy(bgr).cuda() -> unsqueeze -> permute
                     -> float -> F.interpolate -> BGR->RGB -> /255
                     -> (x - mean) / std
  --preproc triton : triton_preprocess_rfdetr_stretch(bgr_gpu, ...)

We allocate a fresh (1,3,H,W) fp32 tensor per frame for both paths so
neither inherits the fast-path's in-place-into-TRT-input-buffer trick;
TRT does its own DtoD into its captured input either way, so that
channel is identical between runs.

RFDETR_USE_TRITON_PREPROC is always false (we bypass the fast-path
eligibility entirely by handing a preprocessed tensor to forward()).
RFDETR_TRITON_FULLPOSTPROC is always false too. CUDA-graphs are left
at default.
"""
import argparse
import hashlib
import json
import os

_ALL_BACKENDS = {
    "torch",
    "torch-script",
    "onnx",
    "trt",
    "hugging-face",
    "ultralytics",
    "mediapipe",
    "custom",
}
os.environ.setdefault(
    "ONNXRUNTIME_EXECUTION_PROVIDERS",
    "[TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider]",
)
os.environ["DISABLED_INFERENCE_MODELS_BACKENDS"] = ",".join(
    sorted(_ALL_BACKENDS - {"trt"})
)
# Make sure the fast-path init code in the model doesn't even set up
# its preproc-into-TRT buffer — we want to explicitly provide our own
# tensor to forward() every frame, so both runs take the same code path.
os.environ.setdefault("RFDETR_USE_TRITON_PREPROC", "false")
os.environ.setdefault("RFDETR_TRITON_FULLPOSTPROC", "false")

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from inference_models import AutoModel
from inference_models.entities import ImageDimensions
from inference_models.models.common.roboflow.model_packages import (
    PreProcessingMetadata,
    StaticCropOffset,
)
from inference_models.models.rfdetr.triton_preprocess import (
    triton_preprocess_rfdetr_stretch,
)


def preproc_ref(frame_bgr: np.ndarray, target_h: int, target_w: int,
                means, stds) -> torch.Tensor:
    """F.interpolate path — equivalent to
    handle_tensor_input_preparation_with_stretch + scaling + normalize."""
    t = torch.from_numpy(frame_bgr).cuda()
    t = t.permute(2, 0, 1).contiguous().unsqueeze(0)
    t = t.float()
    t = F.interpolate(t, size=(target_h, target_w), mode="bilinear")
    t = t[:, [2, 1, 0], :, :]
    t = t / 255.0
    mean = torch.tensor(means, device=t.device).view(1, 3, 1, 1)
    std = torch.tensor(stds, device=t.device).view(1, 3, 1, 1)
    t = (t - mean) / std
    return t.contiguous()


def preproc_triton(frame_bgr: np.ndarray, target_h: int, target_w: int,
                   means, stds) -> torch.Tensor:
    src = torch.from_numpy(frame_bgr).cuda()  # HWC uint8 BGR
    # Allocate fresh every call (no _trt_reuse_as_input_buffer marker)
    # so the TRT graph-replay path behaves identically to the ref case.
    out = torch.empty((1, 3, target_h, target_w), dtype=torch.float32,
                      device=src.device)
    return triton_preprocess_rfdetr_stretch(
        src, target_h=target_h, target_w=target_w,
        means=means, stds=stds, out=out,
    )


def _det_record(xyxy, conf, class_id, mask) -> dict:
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    if mask is None:
        mask_md5 = None
    else:
        m = np.ascontiguousarray(np.asarray(mask).astype(np.bool_))
        mask_md5 = hashlib.md5(m.tobytes()).hexdigest()
    return {
        "xyxy": [x1, y1, x2, y2],
        "conf": float(conf),
        "class_id": int(class_id),
        "mask_md5": mask_md5,
    }


def _canonical_key(d: dict):
    return (*d["xyxy"], d["conf"], d["class_id"], d["mask_md5"] or "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_reference", required=True)
    ap.add_argument("--model_id", default="rfdetr-seg-nano")
    ap.add_argument("--confidence", type=float, default=0.4)
    ap.add_argument("--dump_path", required=True)
    ap.add_argument(
        "--preproc",
        choices=("ref", "triton"),
        required=True,
    )
    ap.add_argument("--max_frames", type=int, default=0)
    args = ap.parse_args()

    print(f"loading {args.model_id} ...", flush=True)
    model = AutoModel.from_pretrained(args.model_id)

    ni = model._inference_config.network_input
    target_h = ni.training_input_size.height
    target_w = ni.training_input_size.width
    means, stds = ni.normalization

    preproc_fn = preproc_ref if args.preproc == "ref" else preproc_triton

    cap = cv2.VideoCapture(args.video_reference)
    assert cap.isOpened(), f"cannot open {args.video_reference}"

    frame_idx = 0
    with open(args.dump_path, "w") as fh:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if args.max_frames and frame_idx >= args.max_frames:
                break

            tensor = preproc_fn(frame_bgr, target_h, target_w, means, stds)

            orig_h, orig_w = frame_bgr.shape[:2]
            orig_size = ImageDimensions(height=orig_h, width=orig_w)
            target_size = ImageDimensions(height=target_h, width=target_w)
            metadata = PreProcessingMetadata(
                pad_left=0,
                pad_top=0,
                pad_right=0,
                pad_bottom=0,
                original_size=orig_size,
                size_after_pre_processing=orig_size,
                inference_size=target_size,
                scale_width=target_w / orig_w,
                scale_height=target_h / orig_h,
                static_crop_offset=StaticCropOffset(
                    offset_x=0, offset_y=0,
                    crop_width=orig_w, crop_height=orig_h,
                ),
            )

            raw = model.forward(tensor)
            dets_list = model.post_process(
                raw, [metadata], confidence=args.confidence,
            )
            det = dets_list[0] if isinstance(dets_list, list) else dets_list

            records = []
            if det is not None and len(det.xyxy):
                xyxy = det.xyxy.cpu().numpy()
                conf = det.confidence.cpu().numpy()
                cls = det.class_id.cpu().numpy()
                mask = det.mask.cpu().numpy()
                for i in range(xyxy.shape[0]):
                    records.append(_det_record(xyxy[i], conf[i], cls[i], mask[i]))
            records.sort(key=_canonical_key)
            fh.write(json.dumps({"frame": frame_idx, "dets": records}) + "\n")
            frame_idx += 1

    cap.release()
    print(f"frames_dumped={frame_idx} path={args.dump_path}")


if __name__ == "__main__":
    main()
