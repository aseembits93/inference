"""Dump per-frame detection digests for the rfdetr-seg-nano TRT workflow.

Produces one JSONL line per frame, where each line is:
    {"frame": int, "dets": [{"xyxy": [x1,y1,x2,y2],
                              "conf": float,
                              "class_id": int,
                              "mask_md5": str}, ...]}

Dets are sorted deterministically (by xyxy then conf then class_id) so two
runs with different compute paths but identical semantic output produce
byte-identical JSONL.

Intended to be run twice — once with RFDETR_TRITON_FULLPOSTPROC=true, once
with =false — and the two dump files compared.
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


def _select_backend_from_argv() -> str:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--backend", choices=("trt", "onnx", "torch"), default="trt")
    args, _ = pre.parse_known_args()
    return args.backend


_BACKEND = _select_backend_from_argv()
os.environ.setdefault(
    "ONNXRUNTIME_EXECUTION_PROVIDERS",
    "[TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider]",
)
os.environ["DISABLED_INFERENCE_MODELS_BACKENDS"] = ",".join(
    sorted(_ALL_BACKENDS - {_BACKEND})
)

import numpy as np
import supervision as sv

from inference import InferencePipeline


def build_workflow(model_id: str, confidence: float) -> dict:
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": "roboflow_core/roboflow_instance_segmentation_model@v3",
                "name": "segmentation",
                "images": "$inputs.image",
                "model_id": model_id,
                "confidence_mode": "custom",
                "custom_confidence": confidence,
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "predictions",
                "selector": "$steps.segmentation.predictions",
            },
        ],
    }


FRAME_IDX = 0
DUMP_FH = None


def _det_record(xyxy, conf, class_id, mask) -> dict:
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    if mask is None:
        mask_md5 = None
    else:
        m = np.ascontiguousarray(mask.astype(np.bool_))
        mask_md5 = hashlib.md5(m.tobytes()).hexdigest()
    return {
        "xyxy": [x1, y1, x2, y2],
        "conf": float(conf),
        "class_id": int(class_id),
        "mask_md5": mask_md5,
    }


def _canonical_key(d: dict):
    # Sort key: xyxy first (spatial), then conf, then class id.
    return (*d["xyxy"], d["conf"], d["class_id"], d["mask_md5"] or "")


def sink(preds_list, _video_frames) -> None:
    global FRAME_IDX, DUMP_FH
    del _video_frames
    if not isinstance(preds_list, list):
        preds_list = [preds_list]
    for pred in preds_list:
        if pred is None:
            DUMP_FH.write(json.dumps({"frame": FRAME_IDX, "dets": None}) + "\n")
            FRAME_IDX += 1
            continue
        det = pred.get("predictions") if isinstance(pred, dict) else None
        records = []
        if isinstance(det, sv.Detections):
            masks = det.mask
            for i in range(len(det)):
                records.append(
                    _det_record(
                        det.xyxy[i],
                        det.confidence[i] if det.confidence is not None else 0.0,
                        det.class_id[i] if det.class_id is not None else -1,
                        masks[i] if masks is not None else None,
                    )
                )
        records.sort(key=_canonical_key)
        DUMP_FH.write(
            json.dumps({"frame": FRAME_IDX, "dets": records}) + "\n"
        )
        FRAME_IDX += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_reference", required=True)
    parser.add_argument("--model_id", default="rfdetr-seg-nano")
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument(
        "--backend",
        choices=("trt", "onnx", "torch"),
        default="trt",
    )
    parser.add_argument("--dump_path", required=True)
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="if >0, stop after this many frames",
    )
    args = parser.parse_args()

    global DUMP_FH
    DUMP_FH = open(args.dump_path, "w")

    pipeline = InferencePipeline.init_with_workflow(
        video_reference=args.video_reference,
        workflow_specification=build_workflow(args.model_id, args.confidence),
        on_prediction=sink,
    )
    pipeline.start()
    pipeline.join()
    DUMP_FH.flush()
    DUMP_FH.close()
    print(f"frames_dumped={FRAME_IDX} path={args.dump_path}")


if __name__ == "__main__":
    main()
