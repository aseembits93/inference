"""Cross-checkout parity harness for the RF-DETR workflow video benchmark.

Mirrors `development/stream_interface/rfdetr_nano_seg_trt_workflow.py` on a
single video source and compares `main` against the current working tree with:

  RFDETR_TRITON_POSTPROC=true
  INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED=true
  RFDETR_PIPELINE_DEPTH=2

Because the workflow demo file does not exist on `main`, this script executes
from the current checkout and uses `--repo-root` bootstrapping to import either
target checkout's code.
"""

import argparse
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, Iterator, Optional


SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SELF = Path(__file__).resolve()
VIDEO_REFERENCE = os.environ.get(
    "WORKFLOW_PARITY_VIDEO", str(SCRIPT_REPO_ROOT / "vehicles_312px.mp4")
)
MODEL_ID = os.environ.get("WORKFLOW_PARITY_MODEL_ID", "rfdetr-seg-nano")
BASE_MODEL_ID = os.environ.get("WORKFLOW_PARITY_BASE_MODEL_ID", MODEL_ID)
CANDIDATE_MODEL_ID = os.environ.get("WORKFLOW_PARITY_CANDIDATE_MODEL_ID", MODEL_ID)
CONFIDENCE = float(os.environ.get("WORKFLOW_PARITY_CONFIDENCE", "0.4"))
BOX_DRIFT_PX = int(os.environ.get("WORKFLOW_PARITY_BOX_DRIFT_PX", "5"))
BACKEND = "trt"
OUT_BASE = "/tmp/workflow_parity_base.pkl"
OUT_CANDIDATE = "/tmp/workflow_parity_candidate.pkl"
ALL_FLAGS = {
    "RFDETR_TRITON_POSTPROC": "true",
    "INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED": "true",
    "RFDETR_PIPELINE_DEPTH": "2",
}
TRT_ONLY_DISABLED_BACKENDS = "torch,torch-script,onnx,hugging-face,ultralytics,custom"


def _iter_pickles(path: str) -> Iterator[dict]:
    with open(path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                return


def _bootstrap_repo_root(repo_root: str) -> Path:
    repo_path = Path(repo_root).resolve()
    os.chdir(repo_path)
    search_roots = [repo_path, repo_path / "inference_models"]
    for search_root in reversed(search_roots):
        search_root_str = str(search_root)
        if search_root_str in sys.path:
            sys.path.remove(search_root_str)
        if search_root.exists():
            sys.path.insert(0, search_root_str)
    return repo_path


def _git_output(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=str(repo_root),
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def _safe_git_output(repo_root: Path, *args: str, default: str = "<unknown>") -> str:
    try:
        return _git_output(repo_root, *args)
    except subprocess.CalledProcessError:
        return default


def _current_branch_label() -> str:
    branch = _safe_git_output(SCRIPT_REPO_ROOT, "rev-parse", "--abbrev-ref", "HEAD")
    return f"{branch} (working-tree)"


def _sanitize_ref(ref: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", ref).strip("-") or "ref"


def _remove_worktree(worktree_root: Path) -> None:
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(worktree_root)],
        cwd=str(SCRIPT_REPO_ROOT),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    shutil.rmtree(worktree_root, ignore_errors=True)


def _materialize_target(ref: str) -> Dict[str, object]:
    normalized = ref.lower()
    if normalized in {"working-tree", "worktree", "current"}:
        return {
            "ref": ref,
            "label": _current_branch_label(),
            "repo_root": SCRIPT_REPO_ROOT,
            "cleanup": None,
        }
    worktree_root = Path(
        tempfile.mkdtemp(prefix=f"workflow-parity-{_sanitize_ref(ref)}-")
    )
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree_root), ref],
        cwd=str(SCRIPT_REPO_ROOT),
        check=True,
    )
    return {
        "ref": ref,
        "label": ref,
        "repo_root": worktree_root,
        "cleanup": lambda: _remove_worktree(worktree_root),
    }


def _build_workflow(model_id: str, confidence: float) -> dict:
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


def _run_signature(repo_root: Path) -> dict:
    return {
        "git_head": _safe_git_output(repo_root, "rev-parse", "--short", "HEAD"),
        "git_describe": _safe_git_output(
            repo_root, "describe", "--always", "--dirty", "--broken"
        ),
    }


def _pack_masks(polygons, image_height: int, image_width: int):
    import cv2
    import numpy as np

    if not polygons:
        return None, None
    masks = np.zeros((len(polygons), image_height, image_width), dtype=np.uint8)
    for idx, polygon in enumerate(polygons):
        if not polygon:
            continue
        points = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(masks[idx], [points], color=1)
    packed = np.packbits(masks.reshape(len(polygons), -1), axis=1)
    return packed, (image_height, image_width)


def _unpack_masks(record):
    import numpy as np

    if record["mask_packed"] is None:
        return None
    n_masks = len(record["mask_packed"])
    image_height, image_width = record["mask_shape"]
    flat = np.unpackbits(
        record["mask_packed"], axis=1, count=image_height * image_width
    )
    return flat.reshape(n_masks, image_height, image_width).astype(bool)


def _record_from_prediction(frame_id: int, video_frame, workflow_prediction: dict) -> dict:
    import numpy as np

    payload = workflow_prediction["predictions"]
    frame_predictions = payload["predictions"]
    image = getattr(video_frame, "image", None)
    if image is None:
        raise ValueError(f"VideoFrame for frame_id={frame_id} does not carry image data.")
    image_height, image_width = image.shape[:2]

    if not frame_predictions:
        return {
            "_kind": "rec",
            "frame_id": frame_id,
            "xyxy": None,
            "conf": None,
            "cls": None,
            "polys": None,
            "mask_packed": None,
            "mask_shape": (image_height, image_width),
            "n_predictions": 0,
        }

    xyxy = np.empty((len(frame_predictions), 4), dtype=np.float32)
    conf = np.empty((len(frame_predictions),), dtype=np.float32)
    cls = np.empty((len(frame_predictions),), dtype=np.int32)
    polys = []

    for idx, prediction in enumerate(frame_predictions):
        width = float(prediction["width"])
        height = float(prediction["height"])
        x_center = float(prediction["x"])
        y_center = float(prediction["y"])
        xyxy[idx] = (
            x_center - (width / 2.0),
            y_center - (height / 2.0),
            x_center + (width / 2.0),
            y_center + (height / 2.0),
        )
        conf[idx] = float(prediction["confidence"])
        cls[idx] = int(prediction["class_id"])
        polygon = [
            (int(round(point["x"])), int(round(point["y"])))
            for point in prediction.get("points", [])
        ]
        polys.append(polygon)

    packed_masks, mask_shape = _pack_masks(
        polygons=polys, image_height=image_height, image_width=image_width
    )
    return {
        "_kind": "rec",
        "frame_id": frame_id,
        "xyxy": xyxy,
        "conf": conf,
        "cls": cls,
        "polys": polys,
        "mask_packed": packed_masks,
        "mask_shape": mask_shape,
        "n_predictions": len(frame_predictions),
    }


def _expected_preproc_calls(run_meta: dict, n_frames: int) -> int:
    if run_meta.get("fast_path_enabled") and run_meta.get("triton_preproc_available"):
        return n_frames
    return 0


def _expected_postproc_calls(run_meta: dict, n_frames: int) -> int:
    if run_meta.get("triton_postproc_ready"):
        return n_frames
    return 0


class _WorkflowRecorder:
    def __init__(self, progress_every: int = 50):
        self.records = []
        self.errors = []
        self.first_non_empty_frame_id = None
        self.progress_every = progress_every
        self.lock = threading.Lock()
        self.start_time = None

    def sink(self, predictions, video_frame) -> None:
        try:
            frame_id = int(getattr(video_frame, "frame_id"))
            record = _record_from_prediction(
                frame_id=frame_id,
                video_frame=video_frame,
                workflow_prediction=predictions,
            )
        except Exception as error:  # pragma: no cover - surfaced by the run
            with self.lock:
                self.errors.append(error)
            return

        with self.lock:
            if self.start_time is None:
                self.start_time = time.perf_counter()
            self.records.append(record)
            if (
                self.first_non_empty_frame_id is None
                and record["n_predictions"] > 0
            ):
                self.first_non_empty_frame_id = frame_id
            if len(self.records) % self.progress_every == 0:
                elapsed = time.perf_counter() - self.start_time
                print(
                    f"  [workflow] callbacks={len(self.records)} "
                    f"last_frame_id={frame_id} elapsed={elapsed:.0f}s",
                    flush=True,
                )


def do_run(out_path: str, repo_root: str, label: Optional[str]) -> None:
    repo_path = _bootstrap_repo_root(repo_root)
    os.environ.setdefault(
        "DISABLED_INFERENCE_MODELS_BACKENDS",
        TRT_ONLY_DISABLED_BACKENDS,
    )

    video_path = Path(VIDEO_REFERENCE)
    if not video_path.exists():
        raise FileNotFoundError(f"Missing video reference: {video_path}")

    from inference import InferencePipeline
    import inference_models.models.rfdetr.common as common_mod
    import inference_models.models.rfdetr.rfdetr_instance_segmentation_trt as trt_mod

    preproc_calls = {"count": 0}
    postproc_calls = {"count": 0}
    combined_graph_stack = {"depth": 0}

    original_preproc = getattr(trt_mod, "triton_preprocess_rfdetr_stretch", None)
    if original_preproc is not None:

        def counting_preproc(*args, **kwargs):
            if combined_graph_stack["depth"] == 0:
                preproc_calls["count"] += 1
            return original_preproc(*args, **kwargs)

        trt_mod.triton_preprocess_rfdetr_stretch = counting_preproc

    original_postproc = getattr(common_mod, "rfdetr_triton_postproc", None)
    if original_postproc is not None:

        def counting_postproc(*args, **kwargs):
            postproc_calls["count"] += 1
            return original_postproc(*args, **kwargs)

        common_mod.rfdetr_triton_postproc = counting_postproc

    original_combined_postproc = getattr(
        trt_mod.RFDetrForInstanceSegmentationTRT,
        "_maybe_forward_async_combined_dense_graph",
        None,
    )
    if original_combined_postproc is not None:

        def counting_combined_postproc(self, *args, **kwargs):
            pre_processed_images = (
                args[0] if args else kwargs.get("pre_processed_images")
            )
            combined_graph_stack["depth"] += 1
            try:
                fut = original_combined_postproc(self, *args, **kwargs)
            finally:
                combined_graph_stack["depth"] -= 1
            if fut is not None:
                postproc_calls["count"] += 1
                if bool(
                    getattr(pre_processed_images, "_trt_preprocess_deferred", False)
                ):
                    preproc_calls["count"] += 1
            return fut

        trt_mod.RFDetrForInstanceSegmentationTRT._maybe_forward_async_combined_dense_graph = (  # type: ignore[attr-defined]
            counting_combined_postproc
        )

    recorder = _WorkflowRecorder()
    signature = _run_signature(repo_path)
    resolved_label = label or _safe_git_output(
        repo_path, "rev-parse", "--abbrev-ref", "HEAD", default=repo_path.name
    )

    print(
        "[run] "
        f"label={resolved_label} repo_root={repo_path} head={signature['git_head']} "
        f"video={video_path.name} backend={BACKEND} "
        f"RFDETR_TRITON_POSTPROC={os.environ.get('RFDETR_TRITON_POSTPROC', '<unset>')} "
        f"INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED="
        f"{os.environ.get('INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED', '<unset>')} "
        f"RFDETR_PIPELINE_DEPTH={os.environ.get('RFDETR_PIPELINE_DEPTH', '<unset>')} "
        f"fast_path_enabled={getattr(trt_mod, '_FAST_PATH_ENABLED', False)} "
        f"triton_preproc_available={getattr(trt_mod, '_TRITON_AVAILABLE', False)} "
        f"triton_postproc_ready={getattr(common_mod, '_TRITON_POSTPROC_READY', False)}",
        flush=True,
    )

    pipeline = InferencePipeline.init_with_workflow(
        video_reference=str(video_path),
        workflow_specification=_build_workflow(MODEL_ID, CONFIDENCE),
        on_prediction=recorder.sink,
        serialize_results=True,
    )
    t0 = time.perf_counter()
    pipeline.start()
    pipeline.join()
    elapsed = time.perf_counter() - t0

    if recorder.errors:
        raise recorder.errors[0]

    frame_ids = [record["frame_id"] for record in recorder.records]
    header = {
        "_kind": "header",
        "label": resolved_label,
        "repo_root": str(repo_path),
        "git_head": signature["git_head"],
        "git_describe": signature["git_describe"],
        "video_reference": str(video_path),
        "model_id": MODEL_ID,
        "confidence": CONFIDENCE,
        "backend": BACKEND,
        "flags": dict(ALL_FLAGS),
        "fast_path_enabled": bool(getattr(trt_mod, "_FAST_PATH_ENABLED", False)),
        "triton_preproc_available": bool(
            getattr(trt_mod, "_TRITON_AVAILABLE", False)
        ),
        "triton_postproc_ready": bool(
            getattr(common_mod, "_TRITON_POSTPROC_READY", False)
        ),
        "first_non_empty_frame_id": recorder.first_non_empty_frame_id,
    }
    footer = {
        "_kind": "footer",
        "label": resolved_label,
        "n_frames": len(recorder.records),
        "frame_id_min": min(frame_ids) if frame_ids else None,
        "frame_id_max": max(frame_ids) if frame_ids else None,
        "preproc_calls": preproc_calls["count"],
        "postproc_calls": postproc_calls["count"],
        "elapsed_s": elapsed,
    }

    with open(out_path, "wb") as f:
        pickle.dump(header, f)
        for record in recorder.records:
            pickle.dump(record, f)
        pickle.dump(footer, f)

    print(
        "[run] "
        f"label={resolved_label} callbacks={len(recorder.records)} "
        f"first_non_empty_frame_id={recorder.first_non_empty_frame_id} "
        f"preproc_calls={preproc_calls['count']} "
        f"postproc_calls={postproc_calls['count']} "
        f"elapsed={elapsed:.2f}s saved -> {out_path}",
        flush=True,
    )


def iou_box(left, right) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    iw = max(0, x1 - x0)
    ih = max(0, y1 - y0)
    inter = iw * ih
    area_left = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    area_right = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = area_left + area_right - inter
    return inter / union if union > 0 else 0.0


def _max_box_drift(left, right) -> float:
    return float(max(abs(float(l) - float(r)) for l, r in zip(left, right)))


def _class_box_matches(base_classes, base_boxes, candidate_classes, candidate_boxes):
    pairs = []
    for base_idx in range(len(base_boxes)):
        for candidate_idx in range(len(candidate_boxes)):
            if int(base_classes[base_idx]) != int(candidate_classes[candidate_idx]):
                continue
            drift = _max_box_drift(base_boxes[base_idx], candidate_boxes[candidate_idx])
            if drift <= BOX_DRIFT_PX:
                pairs.append((drift, base_idx, candidate_idx))

    matches = []
    used_base = set()
    used_candidate = set()
    for drift, base_idx, candidate_idx in sorted(pairs):
        if base_idx in used_base or candidate_idx in used_candidate:
            continue
        used_base.add(base_idx)
        used_candidate.add(candidate_idx)
        matches.append((base_idx, candidate_idx, drift))
    return matches


def _compare_records(base_record: dict, candidate_record: dict) -> dict:
    import numpy as np

    n_base = 0 if base_record["xyxy"] is None else len(base_record["xyxy"])
    n_candidate = 0 if candidate_record["xyxy"] is None else len(candidate_record["xyxy"])

    result = {
        "base_dets": n_base,
        "candidate_dets": n_candidate,
        "matched": 0,
        "relaxed_matched": 0,
        "count_mismatch": int(n_base != n_candidate),
        "class_disagree": 0,
        "box_ious": [],
        "relaxed_box_drifts": [],
        "score_deltas": [],
        "mask_ious": [],
        "pixel_identical": 0,
        "polygon_identical": 0,
        "frame_exact_match": False,
        "frame_relaxed_match": False,
    }

    if n_base == 0 and n_candidate == 0:
        result["frame_exact_match"] = True
        return result

    base_boxes = base_record["xyxy"] if n_base else np.zeros((0, 4), dtype=float)
    candidate_boxes = (
        candidate_record["xyxy"] if n_candidate else np.zeros((0, 4), dtype=float)
    )
    base_scores = base_record["conf"] if n_base else np.zeros(0, dtype=float)
    candidate_scores = (
        candidate_record["conf"] if n_candidate else np.zeros(0, dtype=float)
    )
    base_classes = base_record["cls"] if n_base else np.zeros(0, dtype=np.int32)
    candidate_classes = (
        candidate_record["cls"] if n_candidate else np.zeros(0, dtype=np.int32)
    )
    base_masks = _unpack_masks(base_record) if n_base else None
    candidate_masks = _unpack_masks(candidate_record) if n_candidate else None
    base_polys = base_record["polys"] or []
    candidate_polys = candidate_record["polys"] or []

    relaxed_matches = _class_box_matches(
        base_classes=base_classes,
        base_boxes=base_boxes,
        candidate_classes=candidate_classes,
        candidate_boxes=candidate_boxes,
    )
    result["relaxed_matched"] = len(relaxed_matches)
    result["relaxed_box_drifts"] = [drift for _, _, drift in relaxed_matches]
    result["frame_relaxed_match"] = n_base == n_candidate == len(relaxed_matches)

    used = set()
    for candidate_idx in range(n_candidate):
        best_base_idx = -1
        best_iou = 0.5
        for base_idx in range(n_base):
            if base_idx in used:
                continue
            box_iou = iou_box(base_boxes[base_idx], candidate_boxes[candidate_idx])
            if box_iou > best_iou:
                best_iou = box_iou
                best_base_idx = base_idx
        if best_base_idx < 0:
            continue
        used.add(best_base_idx)
        result["matched"] += 1
        result["box_ious"].append(best_iou)
        result["score_deltas"].append(
            abs(
                float(base_scores[best_base_idx])
                - float(candidate_scores[candidate_idx])
            )
        )
        if int(base_classes[best_base_idx]) != int(candidate_classes[candidate_idx]):
            result["class_disagree"] += 1
        if base_polys[best_base_idx] == candidate_polys[candidate_idx]:
            result["polygon_identical"] += 1
        if base_masks is not None and candidate_masks is not None:
            base_mask = base_masks[best_base_idx]
            candidate_mask = candidate_masks[candidate_idx]
            inter = int((base_mask & candidate_mask).sum())
            union = int((base_mask | candidate_mask).sum())
            mask_iou = 1.0 if union == 0 else float(inter) / float(union)
            result["mask_ious"].append(mask_iou)
            if np.array_equal(base_mask, candidate_mask):
                result["pixel_identical"] += 1

    result["frame_exact_match"] = (
        n_base == n_candidate
        and result["matched"] == n_base
        and result["class_disagree"] == 0
        and result["pixel_identical"] == n_base
        and result["polygon_identical"] == n_base
    )
    return result


def _compare_shift(
    base_header: dict,
    candidate_header: dict,
    base_records: dict,
    candidate_records: dict,
    base_footer: dict,
    candidate_footer: dict,
    shift: int,
) -> dict:
    import numpy as np

    base_frame_ids = sorted(base_records)
    totals = {
        "shift": shift,
        "aligned_frames": 0,
        "missing_candidate_frames": 0,
        "base_total_dets": 0,
        "candidate_total_dets": 0,
        "matched": 0,
        "relaxed_matched": 0,
        "count_mismatch_frames": 0,
        "class_disagree": 0,
        "frame_exact_matches": 0,
        "frame_relaxed_matches": 0,
        "empty_nonempty_frame_mismatch": 0,
        "box_ious": [],
        "relaxed_box_drifts": [],
        "score_deltas": [],
        "mask_ious": [],
        "pixel_identical": 0,
        "polygon_identical": 0,
        "example_mismatches": [],
    }

    for base_frame_id in base_frame_ids:
        candidate_frame_id = base_frame_id + shift
        candidate_record = candidate_records.get(candidate_frame_id)
        if candidate_record is None:
            totals["missing_candidate_frames"] += 1
            continue
        base_record = base_records[base_frame_id]
        frame_metrics = _compare_records(base_record, candidate_record)
        totals["aligned_frames"] += 1
        totals["base_total_dets"] += frame_metrics["base_dets"]
        totals["candidate_total_dets"] += frame_metrics["candidate_dets"]
        totals["matched"] += frame_metrics["matched"]
        totals["relaxed_matched"] += frame_metrics["relaxed_matched"]
        totals["count_mismatch_frames"] += frame_metrics["count_mismatch"]
        totals["class_disagree"] += frame_metrics["class_disagree"]
        if frame_metrics["frame_exact_match"]:
            totals["frame_exact_matches"] += 1
        if frame_metrics["frame_relaxed_match"]:
            totals["frame_relaxed_matches"] += 1
        if (frame_metrics["base_dets"] == 0) != (frame_metrics["candidate_dets"] == 0):
            totals["empty_nonempty_frame_mismatch"] += 1
        totals["box_ious"].extend(frame_metrics["box_ious"])
        totals["relaxed_box_drifts"].extend(frame_metrics["relaxed_box_drifts"])
        totals["score_deltas"].extend(frame_metrics["score_deltas"])
        totals["mask_ious"].extend(frame_metrics["mask_ious"])
        totals["pixel_identical"] += frame_metrics["pixel_identical"]
        totals["polygon_identical"] += frame_metrics["polygon_identical"]
        if (
            not frame_metrics["frame_exact_match"]
            and len(totals["example_mismatches"]) < 5
        ):
            totals["example_mismatches"].append(
                {
                    "base_frame_id": base_frame_id,
                    "candidate_frame_id": candidate_frame_id,
                    "base_dets": frame_metrics["base_dets"],
                    "candidate_dets": frame_metrics["candidate_dets"],
                    "matched": frame_metrics["matched"],
                    "relaxed_matched": frame_metrics["relaxed_matched"],
                    "class_disagree": frame_metrics["class_disagree"],
                    "mask_iou_min": (
                        min(frame_metrics["mask_ious"])
                        if frame_metrics["mask_ious"]
                        else None
                    ),
                }
            )

    totals["mean_box_iou"] = (
        float(np.mean(totals["box_ious"])) if totals["box_ious"] else None
    )
    totals["mean_relaxed_box_drift"] = (
        float(np.mean(totals["relaxed_box_drifts"]))
        if totals["relaxed_box_drifts"]
        else None
    )
    totals["max_relaxed_box_drift"] = (
        float(np.max(totals["relaxed_box_drifts"]))
        if totals["relaxed_box_drifts"]
        else None
    )
    totals["mean_score_delta"] = (
        float(np.mean(totals["score_deltas"])) if totals["score_deltas"] else None
    )
    totals["max_score_delta"] = (
        float(np.max(totals["score_deltas"])) if totals["score_deltas"] else None
    )
    totals["mean_mask_iou"] = (
        float(np.mean(totals["mask_ious"])) if totals["mask_ious"] else None
    )
    totals["min_mask_iou"] = (
        float(np.min(totals["mask_ious"])) if totals["mask_ious"] else None
    )
    totals["expected_base_preproc"] = _expected_preproc_calls(
        base_header, base_footer["n_frames"]
    )
    totals["expected_candidate_preproc"] = _expected_preproc_calls(
        candidate_header, candidate_footer["n_frames"]
    )
    totals["expected_base_postproc"] = _expected_postproc_calls(
        base_header, base_footer["n_frames"]
    )
    totals["expected_candidate_postproc"] = _expected_postproc_calls(
        candidate_header, candidate_footer["n_frames"]
    )
    return totals


def do_compare(base_path: str, candidate_path: str) -> None:
    base_iter = _iter_pickles(base_path)
    candidate_iter = _iter_pickles(candidate_path)

    base_header = next(base_iter)
    candidate_header = next(candidate_iter)
    if base_header.get("_kind") != "header" or candidate_header.get("_kind") != "header":
        raise ValueError("Malformed workflow parity pickle: missing header")

    base_records = {}
    candidate_records = {}
    base_footer = None
    candidate_footer = None

    for obj in base_iter:
        if obj.get("_kind") == "footer":
            base_footer = obj
            break
        base_records[obj["frame_id"]] = obj
    for obj in candidate_iter:
        if obj.get("_kind") == "footer":
            candidate_footer = obj
            break
        candidate_records[obj["frame_id"]] = obj

    if base_footer is None or candidate_footer is None:
        raise ValueError("Malformed workflow parity pickle: missing footer")

    shift_results = [
        _compare_shift(
            base_header=base_header,
            candidate_header=candidate_header,
            base_records=base_records,
            candidate_records=candidate_records,
            base_footer=base_footer,
            candidate_footer=candidate_footer,
            shift=shift,
        )
        for shift in (-2, -1, 0, 1, 2)
    ]
    best_shift = max(
        shift_results,
        key=lambda result: (
            result["frame_relaxed_matches"],
            result["relaxed_matched"],
            result["frame_exact_matches"],
            result["matched"],
            -(result["count_mismatch_frames"] + result["missing_candidate_frames"]),
        ),
    )

    print()
    print(
        "==== workflow parity: "
        f"{base_header['label']} vs {candidate_header['label']} "
        f"({Path(base_header['video_reference']).name}) ===="
    )
    print(
        f"  base repo                    : {base_header['label']} "
        f"@ {base_header['git_describe']}"
    )
    print(
        f"  candidate repo               : {candidate_header['label']} "
        f"@ {candidate_header['git_describe']}"
    )
    print(
        f"  base callbacks               : {base_footer['n_frames']} "
        f"(first_non_empty={base_header['first_non_empty_frame_id']})"
    )
    print(
        f"  candidate callbacks          : {candidate_footer['n_frames']} "
        f"(first_non_empty={candidate_header['first_non_empty_frame_id']})"
    )
    print(
        f"  preproc calls (base/cand)    : "
        f"{base_footer['preproc_calls']} / {candidate_footer['preproc_calls']}"
    )
    print(
        f"  postproc calls (base/cand)   : "
        f"{base_footer['postproc_calls']} / {candidate_footer['postproc_calls']}"
    )
    print(
        f"  best shift                   : {best_shift['shift']:+d} "
        f"(relaxed_frame_matches={best_shift['frame_relaxed_matches']}, "
        f"frame_exact_matches={best_shift['frame_exact_matches']})"
    )

    for result in shift_results:
        print()
        print(
            f"  -- shift {result['shift']:+d} "
            f"(candidate_frame_id = base_frame_id + {result['shift']}) --"
        )
        print(
            f"     aligned / missing         : "
            f"{result['aligned_frames']} / {result['missing_candidate_frames']}"
        )
        print(
            f"     frame exact matches       : {result['frame_exact_matches']}"
        )
        print(
            f"     relaxed frame matches     : {result['frame_relaxed_matches']} "
            f"(cls exact, box drift <= {BOX_DRIFT_PX}px)"
        )
        print(
            f"     dets base / candidate     : "
            f"{result['base_total_dets']} / {result['candidate_total_dets']}"
        )
        print(
            f"     matched (IoU>0.5)         : {result['matched']}"
        )
        print(
            f"     relaxed matched           : {result['relaxed_matched']}"
        )
        print(
            f"     count-mismatch frames     : {result['count_mismatch_frames']}"
        )
        print(
            f"     empty/non-empty mismatch  : {result['empty_nonempty_frame_mismatch']}"
        )
        print(
            f"     class disagreements       : {result['class_disagree']}"
        )
        if result["mean_box_iou"] is not None:
            print(
                f"     mean box IoU              : {result['mean_box_iou']:.6f}"
            )
        if result["mean_relaxed_box_drift"] is not None:
            print(
                f"     mean / max box drift      : "
                f"{result['mean_relaxed_box_drift']:.3f} / "
                f"{result['max_relaxed_box_drift']:.3f}"
            )
        if result["mean_score_delta"] is not None:
            print(
                f"     mean / max |Δscore|       : "
                f"{result['mean_score_delta']:.3e} / {result['max_score_delta']:.3e}"
            )
        if result["mean_mask_iou"] is not None:
            print(
                f"     mean / min mask IoU       : "
                f"{result['mean_mask_iou']:.6f} / {result['min_mask_iou']:.6f}"
            )
        if result["matched"] > 0:
            print(
                f"     pixel-identical masks     : "
                f"{result['pixel_identical']}/{result['matched']}"
            )
            print(
                f"     polygon-identical         : "
                f"{result['polygon_identical']}/{result['matched']}"
            )
        if result["example_mismatches"]:
            print(f"     example mismatches        : {result['example_mismatches']}")

    print()
    base_preproc_ok = (
        base_footer["preproc_calls"] == best_shift["expected_base_preproc"]
    )
    candidate_preproc_ok = (
        candidate_footer["preproc_calls"] == best_shift["expected_candidate_preproc"]
    )
    base_postproc_ok = (
        base_footer["postproc_calls"] == best_shift["expected_base_postproc"]
    )
    candidate_postproc_ok = (
        candidate_footer["postproc_calls"] == best_shift["expected_candidate_postproc"]
    )
    print(
        f"  {'[PASS]' if base_preproc_ok else '[FAIL]'} "
        f"base preproc calls     -> {base_footer['preproc_calls']}/"
        f"{best_shift['expected_base_preproc']}"
    )
    print(
        f"  {'[PASS]' if candidate_preproc_ok else '[FAIL]'} "
        f"candidate preproc calls -> {candidate_footer['preproc_calls']}/"
        f"{best_shift['expected_candidate_preproc']}"
    )
    print(
        f"  {'[PASS]' if base_postproc_ok else '[FAIL]'} "
        f"base postproc calls    -> {base_footer['postproc_calls']}/"
        f"{best_shift['expected_base_postproc']}"
    )
    print(
        f"  {'[PASS]' if candidate_postproc_ok else '[FAIL]'} "
        f"candidate postproc calls -> {candidate_footer['postproc_calls']}/"
        f"{best_shift['expected_candidate_postproc']}"
    )


def _run_child(repo_root: Path, label: str, out_path: str, model_id: str) -> None:
    env = os.environ.copy()
    env.update(ALL_FLAGS)
    env["WORKFLOW_PARITY_MODEL_ID"] = model_id
    env.setdefault("ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES", "true")
    env.setdefault("DISABLED_INFERENCE_MODELS_BACKENDS", TRT_ONLY_DISABLED_BACKENDS)
    print(
        "\n---- child ----\n"
        f"  label={label}\n"
        f"  repo_root={repo_root}\n"
        f"  out={out_path}\n"
        f"  video={Path(VIDEO_REFERENCE).name}\n"
        f"  model_id={model_id}\n"
        f"  flags={ALL_FLAGS}",
        flush=True,
    )
    subprocess.run(
        [
            PY,
            str(SELF),
            "--mode",
            "run",
            "--repo-root",
            str(repo_root),
            "--label",
            label,
            "--out",
            out_path,
        ],
        cwd=str(SCRIPT_REPO_ROOT),
        env=env,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("driver", "run", "compare"), default="driver")
    parser.add_argument("--out")
    parser.add_argument("--repo-root")
    parser.add_argument("--label")
    parser.add_argument("--base", default=OUT_BASE)
    parser.add_argument("--candidate", default=OUT_CANDIDATE)
    parser.add_argument("--base-ref", default="main")
    parser.add_argument("--candidate-ref", default="working-tree")
    parser.add_argument("--keep-worktrees", action="store_true")
    args = parser.parse_args()

    if args.mode == "run":
        if not args.out:
            raise ValueError("--out is required in --mode run")
        do_run(
            out_path=args.out,
            repo_root=args.repo_root or str(SCRIPT_REPO_ROOT),
            label=args.label,
        )
        return

    if args.mode == "compare":
        do_compare(args.base, args.candidate)
        return

    base_target = _materialize_target(args.base_ref)
    candidate_target = _materialize_target(args.candidate_ref)
    cleanup_callbacks = []
    for target in (base_target, candidate_target):
        cleanup = target["cleanup"]
        if callable(cleanup):
            cleanup_callbacks.append(cleanup)

    try:
        _run_child(
            repo_root=Path(base_target["repo_root"]),
            label=str(base_target["label"]),
            out_path=args.base,
            model_id=BASE_MODEL_ID,
        )
        _run_child(
            repo_root=Path(candidate_target["repo_root"]),
            label=str(candidate_target["label"]),
            out_path=args.candidate,
            model_id=CANDIDATE_MODEL_ID,
        )
    finally:
        if not args.keep_worktrees:
            for cleanup in reversed(cleanup_callbacks):
                cleanup()

    do_compare(args.base, args.candidate)


if __name__ == "__main__":
    main()
