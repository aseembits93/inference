"""Drive RFDetrForInstanceSegmentationTRT directly (no InferencePipeline,
no workflow engine) and dump per-frame detection digests.

The inference_models adapter passes numpy straight through to
`pre_process`, and `pre_process` dispatches on *input type*: numpy frames
take the cv2.resize path; torch.Tensor frames take
`handle_tensor_input_preparation_with_stretch`, which uses
`torch.nn.functional.interpolate` — equivalent to the legacy
`USE_PYTORCH_FOR_PREPROCESSING=true` path.

--input_mode numpy   : numpy BGR HWC uint8 frames   (cv2 resize baseline)
--input_mode tensor  : torch.Tensor CHW uint8 frames (F.interpolate baseline)

The Triton fast-path only triggers on numpy inputs, so setting
RFDETR_USE_TRITON_PREPROC=true with --input_mode tensor falls back to the
reference PyTorch path (deliberate — that's how we compare Triton vs the
interpolate reference).
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

import cv2
import numpy as np
import torch

from inference_models import AutoModel


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_reference", required=True)
    parser.add_argument("--model_id", default="rfdetr-seg-nano")
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--dump_path", required=True)
    parser.add_argument(
        "--input_mode",
        choices=("numpy", "tensor"),
        default="numpy",
        help="numpy: cv2.resize baseline; tensor: F.interpolate baseline",
    )
    parser.add_argument("--max_frames", type=int, default=0)
    args = parser.parse_args()

    print(f"loading {args.model_id} ...", flush=True)
    model = AutoModel.from_pretrained(args.model_id)

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

            if args.input_mode == "numpy":
                inp = frame_bgr
            else:
                # HWC uint8 BGR -> CHW uint8 tensor on GPU; the adapter's
                # tensor-preprocess handler normalizes+BGR->RGB+interpolates.
                t = torch.from_numpy(frame_bgr).cuda()  # HWC uint8
                t = t.permute(2, 0, 1).contiguous()     # CHW uint8
                inp = t

            # Force BGR for both modes: the numpy preproc path defaults to
            # BGR but the tensor preproc path defaults to RGB, and cv2 hands
            # us BGR, so we must name it explicitly or the tensor baseline
            # silently swaps R<->B.
            dets_list = model.infer(
                inp, confidence=args.confidence,
                input_color_format="bgr",
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
            fh.write(
                json.dumps({"frame": frame_idx, "dets": records}) + "\n"
            )
            frame_idx += 1

    cap.release()
    print(f"frames_dumped={frame_idx} path={args.dump_path}")


if __name__ == "__main__":
    main()
