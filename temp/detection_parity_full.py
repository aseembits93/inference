"""Cross-checkout RF-DETR segmentation parity with all fast-path flags on.

Default driver mode compares `main` against the current working tree by:
  1. materializing `main` in a temporary git worktree,
  2. running this script twice in subprocesses, once per checkout root,
  3. forcing:
       RFDETR_TRITON_POSTPROC=true
       INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED=true
       RFDETR_PIPELINE_DEPTH=2
  4. recording per-image detections from
     `InferenceModelsInstanceSegmentationAdapter`,
  5. comparing boxes / scores / masks in lockstep.

Because `temp/detection_parity_full.py` does not exist on `main`, child
processes always execute the current script file and use `--repo-root` to point
imports at the checkout whose code should be exercised.

Usage:
  python temp/detection_parity_full.py
  python temp/detection_parity_full.py --base-ref main --candidate-ref working-tree
  python temp/detection_parity_full.py --mode run --repo-root /tmp/inference-main --label main --out /tmp/base.pkl
  python temp/detection_parity_full.py --mode compare --base /tmp/base.pkl --candidate /tmp/candidate.pkl
"""

import argparse
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Iterator, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_REPO_ROOT = REPO_ROOT
MODEL_ID = os.environ.get("PARITY_MODEL_ID", "rfdetr-seg-nano")
CONFIDENCE = 0.4
PY = sys.executable
SELF = Path(__file__).resolve()
OUT_BASE = "/tmp/det_parity_full_base.pkl"
OUT_CANDIDATE = "/tmp/det_parity_full_candidate.pkl"
MAX_IMAGES = int(os.environ.get("PARITY_MAX_IMAGES", "1500"))
ALL_FLAGS = {
    "RFDETR_TRITON_POSTPROC": "true",
    "INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED": "true",
    "RFDETR_PIPELINE_DEPTH": "2",
}
TRT_PACKAGE_SUFFIXES = (
    "-orin-trt-package",
    "-trt-package",
)
TRT_PACKAGE_REQUIRED_FILES = (
    "model_config.json",
    "class_names.txt",
    "inference_config.json",
)


def _resolve_coco_dir(value: Optional[str]) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    return (REPO_ROOT / "coco" / "val2017").resolve()


COCO = _resolve_coco_dir(os.environ.get("PARITY_COCO_DIR"))


def _is_trt_package(package_dir: Path) -> bool:
    if not package_dir.is_dir():
        return False
    if not all((package_dir / filename).exists() for filename in TRT_PACKAGE_REQUIRED_FILES):
        return False
    if not any(
        (package_dir / filename).exists()
        for filename in ("engine.plan", "weights.onnx")
    ):
        return False

    model_config_path = package_dir / "model_config.json"
    try:
        model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return model_config.get("backend_type") == "trt"


def _iter_local_model_packages(model_id: str) -> Iterator[Path]:
    explicit_model_path = os.environ.get("PARITY_MODEL_PATH")
    if explicit_model_path:
        yield Path(explicit_model_path).expanduser()

    package_names = [f"{model_id}{suffix}" for suffix in TRT_PACKAGE_SUFFIXES]
    search_roots = (
        Path.cwd(),
        REPO_ROOT,
        Path(tempfile.gettempdir()),
    )
    for search_root in search_roots:
        for package_name in package_names:
            yield search_root / package_name


def _iter_cached_model_packages(model_id: str) -> Iterator[Path]:
    inference_home = os.environ.get("INFERENCE_HOME")
    if not inference_home:
        return

    models_cache_root = Path(inference_home).expanduser() / "models-cache"
    if not models_cache_root.exists():
        return

    for model_root in sorted(models_cache_root.glob(f"{model_id}-*")):
        for package_dir in sorted(model_root.glob("*")):
            yield package_dir


def _resolve_model_reference(model_id: str) -> str:
    for package_dir in _iter_local_model_packages(model_id):
        if _is_trt_package(package_dir):
            return str(package_dir.resolve())

    for package_dir in _iter_cached_model_packages(model_id):
        if _is_trt_package(package_dir):
            return str(package_dir.resolve())

    return model_id


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
    _prioritize_local_packages(repo_path)
    return repo_path


def _repo_import_roots(repo_root: Path) -> list[Path]:
    return [
        repo_root,
        repo_root / "inference_models",
    ]


def _prioritize_local_packages(repo_root: Path) -> None:
    search_roots = _repo_import_roots(repo_root)
    for search_root in reversed(search_roots):
        search_root_str = str(search_root)
        if search_root_str in sys.path:
            sys.path.remove(search_root_str)
        if search_root.exists():
            sys.path.insert(0, search_root_str)

    # Force subsequent imports to come from the selected checkout rather than
    # any already-imported site-packages copy in the current interpreter.
    for module_name in list(sys.modules):
        if module_name == "inference" or module_name.startswith("inference."):
            sys.modules.pop(module_name, None)
        if module_name == "inference_models" or module_name.startswith(
            "inference_models."
        ):
            sys.modules.pop(module_name, None)


def _child_pythonpath(repo_root: Path, existing_pythonpath: Optional[str]) -> str:
    entries = [str(path) for path in _repo_import_roots(repo_root) if path.exists()]
    if existing_pythonpath:
        entries.append(existing_pythonpath)
    return os.pathsep.join(entries)


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
        tempfile.mkdtemp(prefix=f"det-parity-{_sanitize_ref(ref)}-")
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


def _remove_worktree(worktree_root: Path) -> None:
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(worktree_root)],
        cwd=str(SCRIPT_REPO_ROOT),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    shutil.rmtree(worktree_root, ignore_errors=True)


def _normalized_rle(rle: dict) -> dict:
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": list(rle["size"]), "counts": counts}


def _rle_for_coco_iou(rle: dict) -> dict:
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.encode("ascii")
    return {"size": list(rle["size"]), "counts": counts}


def _rles_equal(left: dict, right: dict) -> bool:
    left_norm = _rle_for_coco_iou(left)
    right_norm = _rle_for_coco_iou(right)
    return (
        left_norm["size"] == right_norm["size"]
        and left_norm["counts"] == right_norm["counts"]
    )


def _rle_iou(left: dict, right: dict) -> float:
    from pycocotools import mask as mask_utils

    left_norm = _rle_for_coco_iou(left)
    right_norm = _rle_for_coco_iou(right)
    return float(mask_utils.iou([left_norm], [right_norm], [False])[0, 0])


def _record_from_response(path: str, response) -> dict:
    import numpy as np

    predictions = response.predictions
    if not predictions:
        return {
            "_kind": "rec",
            "path": path,
            "xyxy": None,
            "conf": None,
            "cls": None,
            "rles": None,
        }

    xyxy = np.empty((len(predictions), 4), dtype=np.float32)
    conf = np.empty((len(predictions),), dtype=np.float32)
    cls = np.empty((len(predictions),), dtype=np.int32)
    rles = []

    for idx, pred in enumerate(predictions):
        x1 = float(pred.x) - (float(pred.width) / 2.0)
        y1 = float(pred.y) - (float(pred.height) / 2.0)
        x2 = float(pred.x) + (float(pred.width) / 2.0)
        y2 = float(pred.y) + (float(pred.height) / 2.0)
        xyxy[idx] = (x1, y1, x2, y2)
        conf[idx] = float(pred.confidence)
        cls[idx] = int(pred.class_id)
        rle = getattr(pred, "rle", None)
        if rle is None:
            raise ValueError(
                "Expected response_mask_format='rle' to produce RLE predictions."
            )
        rles.append(_normalized_rle(rle))

    return {
        "_kind": "rec",
        "path": path,
        "xyxy": xyxy,
        "conf": conf,
        "cls": cls,
        "rles": rles,
    }


def _expected_preproc_calls(run_meta: dict, n_images: int) -> int:
    if run_meta.get("fast_path_enabled") and run_meta.get("triton_preproc_available"):
        return n_images
    return 0


def _expected_postproc_calls(run_meta: dict, n_images: int) -> int:
    if run_meta.get("triton_postproc_ready"):
        return n_images
    return 0


def _run_signature(repo_root: Path) -> dict:
    return {
        "git_head": _safe_git_output(repo_root, "rev-parse", "--short", "HEAD"),
        "git_describe": _safe_git_output(
            repo_root, "describe", "--always", "--dirty", "--broken"
        ),
    }


def do_run(out_path: str, repo_root: str, label: Optional[str]) -> None:
    repo_path = _bootstrap_repo_root(repo_root)
    os.environ.setdefault(
        "DISABLED_INFERENCE_MODELS_BACKENDS",
        "torch,torch-script,onnx,hugging-face,ultralytics,custom",
    )
    model_reference = _resolve_model_reference(MODEL_ID)
    if os.path.exists(model_reference):
        os.environ.setdefault(
            "ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES",
            "true",
        )
    if not COCO.exists():
        raise FileNotFoundError(f"Missing COCO directory: {COCO}")

    import cv2

    from inference.core.models import inference_models_adapters as adapter_mod
    from inference.core.models.inference_models_adapters import (
        InferenceModelsInstanceSegmentationAdapter,
    )
    import inference_models.models.rfdetr.common as common_mod
    import inference_models.models.rfdetr.rfdetr_instance_segmentation_trt as trt_mod

    pipeline_priming = getattr(adapter_mod, "_PIPELINE_PRIMING", None)

    preproc_calls = {"count": 0}
    postproc_calls = {"count": 0}

    original_preproc = getattr(trt_mod, "triton_preprocess_rfdetr_stretch", None)
    if original_preproc is not None:

        def counting_preproc(*args, **kwargs):
            preproc_calls["count"] += 1
            return original_preproc(*args, **kwargs)

        trt_mod.triton_preprocess_rfdetr_stretch = counting_preproc

    original_postproc = getattr(common_mod, "rfdetr_triton_postproc", None)
    if original_postproc is not None:

        def counting_postproc(*args, **kwargs):
            postproc_calls["count"] += 1
            return original_postproc(*args, **kwargs)

        common_mod.rfdetr_triton_postproc = counting_postproc

    model = InferenceModelsInstanceSegmentationAdapter(model_reference)
    pipeline_depth = getattr(model, "_pipeline_depth", 1)
    resolved_label = label or _safe_git_output(
        repo_path, "rev-parse", "--abbrev-ref", "HEAD", default=repo_path.name
    )
    signature = _run_signature(repo_path)

    print(
        "[run] "
        f"label={resolved_label} repo_root={repo_path} head={signature['git_head']} "
        f"model_reference={model_reference} "
        f"RFDETR_TRITON_POSTPROC={os.environ.get('RFDETR_TRITON_POSTPROC', '<unset>')} "
        f"INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED="
        f"{os.environ.get('INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED', '<unset>')} "
        f"RFDETR_PIPELINE_DEPTH={os.environ.get('RFDETR_PIPELINE_DEPTH', '<unset>')} "
        f"adapter_pipeline_depth={pipeline_depth} "
        f"fast_path_enabled={getattr(trt_mod, '_FAST_PATH_ENABLED', False)} "
        f"triton_preproc_available={getattr(trt_mod, '_TRITON_AVAILABLE', False)} "
        f"triton_postproc_ready={getattr(common_mod, '_TRITON_POSTPROC_READY', False)}",
        flush=True,
    )

    header = {
        "_kind": "header",
        "label": resolved_label,
        "repo_root": str(repo_path),
        "model_id": MODEL_ID,
        "model_reference": model_reference,
        "confidence": CONFIDENCE,
        "flags": dict(ALL_FLAGS),
        "git_head": signature["git_head"],
        "git_describe": signature["git_describe"],
        "adapter_pipeline_depth": pipeline_depth,
        "fast_path_enabled": bool(getattr(trt_mod, "_FAST_PATH_ENABLED", False)),
        "triton_preproc_available": bool(
            getattr(trt_mod, "_TRITON_AVAILABLE", False)
        ),
        "triton_postproc_ready": bool(
            getattr(common_mod, "_TRITON_POSTPROC_READY", False)
        ),
        "max_images": MAX_IMAGES,
    }

    paths = sorted(COCO.glob("*.jpg"))[:MAX_IMAGES]
    pending_paths: Deque[str] = deque()
    n_records = 0
    t0 = time.perf_counter()

    with open(out_path, "wb") as f:
        pickle.dump(header, f)
        for idx, image_path in enumerate(paths):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue

            preprocessed, metadata = model.preprocess(
                image,
                confidence=CONFIDENCE,
                response_mask_format="rle",
            )
            prediction_handle = model.predict(
                preprocessed,
                confidence=CONFIDENCE,
                response_mask_format="rle",
            )
            responses = model.postprocess(
                prediction_handle,
                metadata,
                confidence=CONFIDENCE,
                response_mask_format="rle",
            )
            pending_paths.append(str(image_path))

            is_priming = (
                pipeline_priming is not None and prediction_handle is pipeline_priming
            )
            if not is_priming:
                if len(responses) != 1:
                    raise ValueError(
                        f"Expected one response for {image_path}, got {len(responses)}"
                    )
                record = _record_from_response(pending_paths.popleft(), responses[0])
                pickle.dump(record, f)
                n_records += 1

            if (idx + 1) % 250 == 0:
                print(
                    f"  [{resolved_label}] {idx + 1}/{len(paths)} "
                    f"records={n_records} ({time.perf_counter() - t0:.0f}s)",
                    flush=True,
                )

        flush_responses = model.flush() if hasattr(model, "flush") else []
        for response in flush_responses:
            if not pending_paths:
                raise ValueError("flush() returned a response but no pending path")
            record = _record_from_response(pending_paths.popleft(), response)
            pickle.dump(record, f)
            n_records += 1

        if pending_paths:
            raise ValueError(
                f"Unflushed pending paths remain for {resolved_label}: "
                f"{list(pending_paths)[:3]}"
            )

        footer = {
            "_kind": "footer",
            "label": resolved_label,
            "n_records": n_records,
            "preproc_calls": preproc_calls["count"],
            "postproc_calls": postproc_calls["count"],
            "elapsed_s": time.perf_counter() - t0,
        }
        pickle.dump(footer, f)

    print(
        "[run] "
        f"label={resolved_label} records={n_records} "
        f"preproc_calls={preproc_calls['count']} "
        f"postproc_calls={postproc_calls['count']} "
        f"saved -> {out_path}",
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


def do_compare(base_path: str, candidate_path: str) -> None:
    import numpy as np

    base_iter = _iter_pickles(base_path)
    candidate_iter = _iter_pickles(candidate_path)

    base_header = next(base_iter)
    candidate_header = next(candidate_iter)
    if base_header.get("_kind") != "header" or candidate_header.get("_kind") != "header":
        raise ValueError("Malformed parity pickle: missing header")

    tot_base = tot_candidate = matched = 0
    class_disagree = count_mismatch = pixel_identical = 0
    box_ious = []
    score_deltas = []
    mask_ious = []
    n_images = 0
    base_footer = None
    candidate_footer = None

    for base_record, candidate_record in zip(base_iter, candidate_iter):
        if (
            base_record.get("_kind") == "footer"
            or candidate_record.get("_kind") == "footer"
        ):
            base_footer = base_record
            candidate_footer = candidate_record
            break

        if base_record["path"] != candidate_record["path"]:
            raise AssertionError((base_record["path"], candidate_record["path"]))

        n_images += 1
        n_base = 0 if base_record["xyxy"] is None else len(base_record["xyxy"])
        n_candidate = (
            0 if candidate_record["xyxy"] is None else len(candidate_record["xyxy"])
        )
        tot_base += n_base
        tot_candidate += n_candidate

        if n_base != n_candidate:
            count_mismatch += 1
        if n_base == 0 and n_candidate == 0:
            continue

        base_boxes = base_record["xyxy"] if n_base else np.zeros((0, 4), dtype=float)
        candidate_boxes = (
            candidate_record["xyxy"] if n_candidate else np.zeros((0, 4), dtype=float)
        )
        base_scores = base_record["conf"] if n_base else np.zeros(0, dtype=float)
        candidate_scores = (
            candidate_record["conf"] if n_candidate else np.zeros(0, dtype=float)
        )
        base_classes = (
            base_record["cls"] if n_base else np.zeros(0, dtype=np.int32)
        )
        candidate_classes = (
            candidate_record["cls"] if n_candidate else np.zeros(0, dtype=np.int32)
        )
        base_rles = base_record["rles"] or []
        candidate_rles = candidate_record["rles"] or []

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
            matched += 1
            box_ious.append(best_iou)
            score_deltas.append(
                abs(float(base_scores[best_base_idx]) - float(candidate_scores[candidate_idx]))
            )
            if int(base_classes[best_base_idx]) != int(candidate_classes[candidate_idx]):
                class_disagree += 1

            if base_rles and candidate_rles:
                base_rle = base_rles[best_base_idx]
                candidate_rle = candidate_rles[candidate_idx]
                mask_ious.append(_rle_iou(base_rle, candidate_rle))
                if _rles_equal(base_rle, candidate_rle):
                    pixel_identical += 1

    if base_footer is None:
        for obj in base_iter:
            if obj.get("_kind") == "footer":
                base_footer = obj
                break
    if candidate_footer is None:
        for obj in candidate_iter:
            if obj.get("_kind") == "footer":
                candidate_footer = obj
                break
    if base_footer is None or candidate_footer is None:
        raise ValueError("Malformed parity pickle: missing footer")

    expected_base_preproc = _expected_preproc_calls(
        base_header, base_footer["n_records"]
    )
    expected_candidate_preproc = _expected_preproc_calls(
        candidate_header, candidate_footer["n_records"]
    )
    expected_base_postproc = _expected_postproc_calls(
        base_header, base_footer["n_records"]
    )
    expected_candidate_postproc = _expected_postproc_calls(
        candidate_header, candidate_footer["n_records"]
    )

    print()
    print(
        "==== parity: "
        f"{base_header['label']} vs {candidate_header['label']} "
        f"({n_images} images, model={base_header['model_id']}) ===="
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
        f"  pipeline depth (base/cand)   : "
        f"{base_header['adapter_pipeline_depth']} / "
        f"{candidate_header['adapter_pipeline_depth']}"
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
        f"  records base / candidate     : "
        f"{base_footer['n_records']} / {candidate_footer['n_records']}"
    )
    print(
        f"  dets base / candidate        : {tot_base} / {tot_candidate}"
    )
    print(
        f"  matched (IoU>0.5)            : {matched} "
        f"({100 * matched / max(1, tot_base):.2f}% of base)"
    )
    print(f"  count-mismatch images        : {count_mismatch}")
    print(f"  class-id disagreements       : {class_disagree}")
    if box_ious:
        print(f"  mean box IoU                 : {np.mean(box_ious):.6f}")
    if score_deltas:
        print(
            f"  mean / max |Δscore|          : "
            f"{np.mean(score_deltas):.3e} / {np.max(score_deltas):.3e}"
        )
    if mask_ious:
        mask_iou_array = np.array(mask_ious)
        print(
            f"  mean / min mask IoU          : "
            f"{mask_iou_array.mean():.6f} / {mask_iou_array.min():.6f}"
        )
        print(
            f"  pixel-identical masks        : "
            f"{pixel_identical}/{len(mask_ious)}"
        )

    print()
    base_preproc_ok = base_footer["preproc_calls"] == expected_base_preproc
    candidate_preproc_ok = (
        candidate_footer["preproc_calls"] == expected_candidate_preproc
    )
    base_postproc_ok = base_footer["postproc_calls"] == expected_base_postproc
    candidate_postproc_ok = (
        candidate_footer["postproc_calls"] == expected_candidate_postproc
    )
    candidate_pipeline_ok = candidate_header["adapter_pipeline_depth"] == int(
        ALL_FLAGS["RFDETR_PIPELINE_DEPTH"]
    )

    print(
        f"  {'[PASS]' if base_preproc_ok else '[FAIL]'} "
        f"base preproc calls     -> {base_footer['preproc_calls']}/"
        f"{expected_base_preproc}"
    )
    print(
        f"  {'[PASS]' if candidate_preproc_ok else '[FAIL]'} "
        f"candidate preproc calls -> {candidate_footer['preproc_calls']}/"
        f"{expected_candidate_preproc}"
    )
    print(
        f"  {'[PASS]' if base_postproc_ok else '[FAIL]'} "
        f"base postproc calls    -> {base_footer['postproc_calls']}/"
        f"{expected_base_postproc}"
    )
    print(
        f"  {'[PASS]' if candidate_postproc_ok else '[FAIL]'} "
        f"candidate postproc calls -> {candidate_footer['postproc_calls']}/"
        f"{expected_candidate_postproc}"
    )
    print(
        f"  {'[PASS]' if candidate_pipeline_ok else '[FAIL]'} "
        f"candidate pipeline depth -> {candidate_header['adapter_pipeline_depth']}/"
        f"{ALL_FLAGS['RFDETR_PIPELINE_DEPTH']}"
    )


def _run_child(repo_root: Path, label: str, out_path: str) -> None:
    env = os.environ.copy()
    env.update(ALL_FLAGS)
    env["PYTHONPATH"] = _child_pythonpath(
        repo_root=repo_root,
        existing_pythonpath=env.get("PYTHONPATH"),
    )
    print(
        "\n---- child ----\n"
        f"  label={label}\n"
        f"  repo_root={repo_root}\n"
        f"  out={out_path}\n"
        f"  flags={ALL_FLAGS}\n"
        f"  PYTHONPATH={env['PYTHONPATH']}",
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
        )
        _run_child(
            repo_root=Path(candidate_target["repo_root"]),
            label=str(candidate_target["label"]),
            out_path=args.candidate,
        )
    finally:
        if not args.keep_worktrees:
            for cleanup in reversed(cleanup_callbacks):
                cleanup()

    do_compare(args.base, args.candidate)


if __name__ == "__main__":
    main()
