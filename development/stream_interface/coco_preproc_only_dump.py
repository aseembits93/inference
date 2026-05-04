"""Run rfdetr-seg-nano over COCO val2017 with isolated preproc.

Preproc function is either:
  --preproc ref    : cast uint8 BGR -> float -> F.interpolate -> BGR->RGB
                     -> /255 -> (x - mean) / std
  --preproc triton : triton_preprocess_rfdetr_stretch (fused kernel)

forward() and post_process() are identical across runs. Emits two
outputs for each run:

  <dump_prefix>.jsonl   per-image detection digest for pairwise diff:
                        {"image_id", "file_name", "dets": [...]}
                        det = {xyxy, conf, class_id, mask_md5}

  <dump_prefix>.json    COCO detections format for pycocotools:
                        [{"image_id", "category_id", "bbox": [x,y,w,h],
                          "score", "segmentation": RLE}]

COCO category_id = model class_id + 1 (model outputs 0..89, the "90"
slots include 11 duplicated placeholders for removed COCO categories
like street sign / hat / shoe; the +1 mapping lines them up with the
actual COCO category ids used in instances_val2017.json).
"""
import argparse
import hashlib
import json
import os
import time

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
os.environ.setdefault("RFDETR_USE_TRITON_PREPROC", "false")
os.environ.setdefault("RFDETR_TRITON_FULLPOSTPROC", "false")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_utils

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
    src = torch.from_numpy(frame_bgr).cuda()
    out = torch.empty((1, 3, target_h, target_w), dtype=torch.float32,
                      device=src.device)
    return triton_preprocess_rfdetr_stretch(
        src, target_h=target_h, target_w=target_w,
        means=means, stds=stds, out=out,
    )


def _canonical_key(d: dict):
    return (*d["xyxy"], d["conf"], d["class_id"], d["mask_md5"] or "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--annotations_json", required=True)
    ap.add_argument("--model_id", default="rfdetr-seg-nano")
    ap.add_argument("--confidence", type=float, default=0.05,
                    help="low threshold so COCO eval sees the full PR curve")
    ap.add_argument("--dump_prefix", required=True,
                    help="paths <prefix>.jsonl and <prefix>.json are written")
    ap.add_argument("--preproc", choices=("ref", "triton"), required=True)
    ap.add_argument("--max_images", type=int, default=0)
    args = ap.parse_args()

    with open(args.annotations_json) as f:
        coco = json.load(f)
    images = coco["images"]
    if args.max_images:
        images = images[: args.max_images]
    print(f"loaded {len(images)} images from {args.annotations_json}",
          flush=True)

    print(f"loading {args.model_id} ...", flush=True)
    model = AutoModel.from_pretrained(args.model_id)
    ni = model._inference_config.network_input
    target_h = ni.training_input_size.height
    target_w = ni.training_input_size.width
    means, stds = ni.normalization

    preproc_fn = preproc_ref if args.preproc == "ref" else preproc_triton

    coco_dets = []
    jsonl_path = args.dump_prefix + ".jsonl"
    coco_path = args.dump_prefix + ".json"
    t0 = time.time()
    n_dets_emitted = 0
    n_warn_bad_read = 0
    with open(jsonl_path, "w") as fh:
        for idx, img_meta in enumerate(images):
            path = os.path.join(args.images_dir, img_meta["file_name"])
            frame_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                n_warn_bad_read += 1
                fh.write(json.dumps({"image_id": img_meta["id"],
                                     "file_name": img_meta["file_name"],
                                     "dets": None}) + "\n")
                continue

            orig_h, orig_w = frame_bgr.shape[:2]
            tensor = preproc_fn(frame_bgr, target_h, target_w, means, stds)

            metadata = PreProcessingMetadata(
                pad_left=0, pad_top=0, pad_right=0, pad_bottom=0,
                original_size=ImageDimensions(height=orig_h, width=orig_w),
                size_after_pre_processing=ImageDimensions(
                    height=orig_h, width=orig_w),
                inference_size=ImageDimensions(height=target_h, width=target_w),
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
            # post_process runs on a dedicated stream; the default stream (on
            # which .cpu() below copies) hasn't waited for it, so without
            # this sync we read garbage that may collide with the next
            # forward() overwriting graph-owned output buffers.
            torch.cuda.synchronize()
            det = dets_list[0] if isinstance(dets_list, list) else dets_list

            records = []
            if det is not None and len(det.xyxy):
                xyxy = det.xyxy.cpu().numpy()
                conf = det.confidence.cpu().numpy()
                cls = det.class_id.cpu().numpy()
                mask = det.mask.cpu().numpy().astype(np.bool_)
                for i in range(xyxy.shape[0]):
                    x1, y1, x2, y2 = (float(v) for v in xyxy[i])
                    mi = np.ascontiguousarray(mask[i])
                    md5 = hashlib.md5(mi.tobytes()).hexdigest()
                    records.append({
                        "xyxy": [x1, y1, x2, y2],
                        "conf": float(conf[i]),
                        "class_id": int(cls[i]),
                        "mask_md5": md5,
                    })
                    rle = mask_utils.encode(np.asfortranarray(mi))
                    rle["counts"] = rle["counts"].decode("ascii")
                    coco_dets.append({
                        "image_id": int(img_meta["id"]),
                        "category_id": int(cls[i]) + 1,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(conf[i]),
                        "segmentation": rle,
                    })
                    n_dets_emitted += 1
            records.sort(key=_canonical_key)
            fh.write(json.dumps({"image_id": int(img_meta["id"]),
                                 "file_name": img_meta["file_name"],
                                 "dets": records}) + "\n")
            if (idx + 1) % 500 == 0:
                dt = time.time() - t0
                print(f"  [{idx+1}/{len(images)}] elapsed={dt:.1f}s "
                      f"dets={n_dets_emitted} "
                      f"rate={(idx+1)/dt:.1f} im/s",
                      flush=True)

    with open(coco_path, "w") as f:
        json.dump(coco_dets, f)
    dt = time.time() - t0
    print(f"done: {len(images)} images, {n_dets_emitted} detections, "
          f"{n_warn_bad_read} bad reads, {dt:.1f}s ({len(images)/dt:.1f} im/s)")
    print(f"  jsonl: {jsonl_path}")
    print(f"  coco:  {coco_path}")


if __name__ == "__main__":
    main()
