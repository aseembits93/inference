"""Runs correctness_dump.py twice with one RFDETR_* flag flipped, then
diffs the two dump files. The flag to vary is chosen via --flag:

  --flag fullpostproc  (default)  vary RFDETR_TRITON_FULLPOSTPROC with
                                  preproc + CUDA graphs both on
  --flag preproc                  vary RFDETR_USE_TRITON_PREPROC  with
                                  fullpostproc off + CUDA graphs on

Per-frame diff report (exact and tolerant):
  - frame count mismatch -> fatal
  - per-frame det-count mismatch -> report
  - per-frame det index mismatch after canonical sort -> report
       xyxy (exact, they're rounded int in both paths)
       class_id (exact)
       conf (abs diff; report max)
       mask_md5 (exact)

Exits non-zero if any semantic difference is found beyond a very small
tolerance on conf (the Triton path uses tl.exp for the sigmoid, so tiny
fp rounding is possible on extreme logits; the commit claims 0-diff).
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = str(REPO / ".venv" / "bin" / "python3")
DUMP = str(REPO / "development" / "stream_interface" / "correctness_dump.py")


def run_dump(env_overrides: dict, video: str, model_id: str, confidence: float,
             dump_path: str, max_frames: int, timeout_s: int) -> int:
    env = os.environ.copy()
    env.update(env_overrides)
    cmd = [
        PY,
        DUMP,
        "--video_reference", video,
        "--model_id", model_id,
        "--confidence", str(confidence),
        "--backend", "trt",
        "--dump_path", dump_path,
    ]
    if max_frames > 0:
        cmd += ["--max_frames", str(max_frames)]
    print(f"[run] env={env_overrides} -> {dump_path}", flush=True)
    # cwd is intentionally not REPO: sys.path[0]=='' would then resolve
    # `import inference_models` to the outer namespace-package dir and hide
    # the editable install. Run from REPO/development so the editable finder
    # wins.
    proc = subprocess.run(
        cmd, env=env, cwd=str(REPO / "development"), timeout=timeout_s,
    )
    return proc.returncode


def load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    u = aa + bb - inter
    return inter / u if u > 0 else 0.0


def _pair_dets(da: list[dict], db: list[dict], iou_tol: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy best-IoU pairing restricted to same class_id. Returns
    (pairs, unmatched_a, unmatched_b)."""
    pairs: list[tuple[int, int]] = []
    used_b = set()
    unmatched_a: list[int] = []
    # Sort by conf descending so the highest-confidence survivors get dibs.
    order = sorted(range(len(da)), key=lambda i: -da[i]["conf"])
    for i in order:
        best_j = -1
        best_iou = iou_tol
        for j in range(len(db)):
            if j in used_b:
                continue
            if db[j]["class_id"] != da[i]["class_id"]:
                continue
            iou = _iou(da[i]["xyxy"], db[j]["xyxy"])
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0:
            pairs.append((i, best_j))
            used_b.add(best_j)
        else:
            unmatched_a.append(i)
    unmatched_b = [j for j in range(len(db)) if j not in used_b]
    return pairs, unmatched_a, unmatched_b


def diff_dumps(a_path: str, b_path: str, conf_tol: float, iou_tol: float) -> int:
    A = load_jsonl(a_path)
    B = load_jsonl(b_path)
    if len(A) != len(B):
        print(f"FATAL: frame count {len(A)} vs {len(B)}")
        return 2
    total_dets_a = 0
    total_dets_b = 0
    matched_pairs = 0
    unmatched_a_total = 0
    unmatched_b_total = 0
    frames_with_mismatch = 0
    max_conf_delta = 0.0
    max_box_edge_px = 0.0
    mask_mismatches = 0
    first_examples: list[str] = []

    for i, (a, b) in enumerate(zip(A, B)):
        da = a["dets"] or []
        db = b["dets"] or []
        total_dets_a += len(da)
        total_dets_b += len(db)
        pairs, una, unb = _pair_dets(da, db, iou_tol)
        matched_pairs += len(pairs)
        unmatched_a_total += len(una)
        unmatched_b_total += len(unb)
        frame_has_mismatch = bool(una or unb)
        for ai, bj in pairs:
            ra = da[ai]; rb = db[bj]
            edge_delta = max(abs(ra["xyxy"][k] - rb["xyxy"][k]) for k in range(4))
            if edge_delta > max_box_edge_px:
                max_box_edge_px = edge_delta
            d = abs(ra["conf"] - rb["conf"])
            if d > max_conf_delta:
                max_conf_delta = d
            if ra["mask_md5"] != rb["mask_md5"]:
                mask_mismatches += 1
                frame_has_mismatch = True
                if len(first_examples) < 5:
                    first_examples.append(
                        f"frame {i}: matched pair A[{ai}]/B[{bj}] mask_md5 differs"
                    )
            if d > conf_tol:
                frame_has_mismatch = True
                if len(first_examples) < 5:
                    first_examples.append(
                        f"frame {i}: pair A[{ai}]/B[{bj}] conf {ra['conf']:.4f} vs {rb['conf']:.4f} delta={d:.2e}"
                    )
        if una and len(first_examples) < 5:
            ai = una[0]; ra = da[ai]
            first_examples.append(
                f"frame {i}: A[{ai}] unmatched xyxy={ra['xyxy']} conf={ra['conf']:.3f} cls={ra['class_id']}"
            )
        if unb and len(first_examples) < 5:
            bj = unb[0]; rb = db[bj]
            first_examples.append(
                f"frame {i}: B[{bj}] unmatched xyxy={rb['xyxy']} conf={rb['conf']:.3f} cls={rb['class_id']}"
            )
        if frame_has_mismatch:
            frames_with_mismatch += 1

    print("\n==================== DIFF REPORT ====================")
    print(f"frames              : {len(A)}")
    print(f"total dets (a / b)  : {total_dets_a} / {total_dets_b}")
    print(f"matched pairs       : {matched_pairs}")
    print(f"unmatched A / B     : {unmatched_a_total} / {unmatched_b_total}")
    print(f"frames w/ mismatch  : {frames_with_mismatch}")
    print(f"mask_md5 mismatches : {mask_mismatches} (of {matched_pairs} matched)")
    print(f"max conf delta      : {max_conf_delta:.3e} (tol={conf_tol:.0e})")
    print(f"max box edge delta  : {max_box_edge_px:.1f} px")
    if first_examples:
        print("first 5 examples:")
        for ex in first_examples:
            print("  -", ex)
    print("=====================================================")
    any_fail = (
        unmatched_a_total or unmatched_b_total
        or mask_mismatches or max_conf_delta > conf_tol
    )
    return 1 if any_fail else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_reference", default=str(REPO / "vehicles_312px.mp4"))
    ap.add_argument("--model_id", default="rfdetr-seg-nano")
    ap.add_argument("--confidence", type=float, default=0.4)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--conf_tol", type=float, default=1e-6)
    ap.add_argument(
        "--iou_tol",
        type=float,
        default=0.5,
        help="min IoU to pair detections across runs",
    )
    ap.add_argument("--dump_dir", default=str(REPO))
    ap.add_argument(
        "--diff_only",
        action="store_true",
        help="skip the runs; just diff existing dumps in --dump_dir",
    )
    ap.add_argument(
        "--flag",
        choices=("fullpostproc", "preproc"),
        default="fullpostproc",
        help="which RFDETR_* flag to toggle off vs on",
    )
    args = ap.parse_args()

    # Axes locked in every run. The --flag-specific section below pins the
    # peer flag (so only one axis varies).
    base_env = {
        "ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND": "true",
        # disable optional model-types whose deps are not installed in this venv
        "QWEN_3_5_ENABLED": "false",
        "QWEN_3_ENABLED": "false",
        "QWEN_2_5_ENABLED": "false",
        "PALIGEMMA_ENABLED": "false",
        "FLORENCE2_ENABLED": "false",
        "CORE_MODEL_SAM_ENABLED": "false",
        "CORE_MODEL_SAM2_ENABLED": "false",
        "CORE_MODEL_SAM3_ENABLED": "false",
        "CORE_MODEL_GAZE_ENABLED": "false",
        "CORE_MODEL_CLIP_ENABLED": "false",
        "CORE_MODEL_OWLV2_ENABLED": "false",
        "CORE_MODEL_PE_ENABLED": "false",
        "CORE_MODEL_DOCTR_ENABLED": "false",
        "CORE_MODEL_EASYOCR_ENABLED": "false",
        "CORE_MODEL_TROCR_ENABLED": "false",
        "CORE_MODEL_GROUNDINGDINO_ENABLED": "false",
        "CORE_MODEL_YOLO_WORLD_ENABLED": "false",
        "SMOLVLM2_ENABLED": "false",
        "DEPTH_ESTIMATION_ENABLED": "false",
        "MOONDREAM2_ENABLED": "false",
        "GLM_OCR_ENABLED": "false",
        "SAM3_3D_OBJECTS_ENABLED": "false",
    }

    if args.flag == "fullpostproc":
        flag_name = "RFDETR_TRITON_FULLPOSTPROC"
        base_env["RFDETR_USE_TRITON_PREPROC"] = "true"
        out_stem = "correctness_fullpost"
    else:  # preproc
        flag_name = "RFDETR_USE_TRITON_PREPROC"
        base_env["RFDETR_TRITON_FULLPOSTPROC"] = "false"
        out_stem = "correctness_preproc"

    dump_dir = Path(args.dump_dir)
    a_path = str(dump_dir / f"{out_stem}_off.jsonl")
    b_path = str(dump_dir / f"{out_stem}_on.jsonl")

    env_off = dict(base_env); env_off[flag_name] = "false"
    env_on = dict(base_env); env_on[flag_name] = "true"

    if not args.diff_only:
        rc = run_dump(env_off, args.video_reference, args.model_id, args.confidence,
                      a_path, args.max_frames, args.timeout)
        if rc != 0:
            print(f"FATAL: off-run exited {rc}", file=sys.stderr)
            return 3
        rc = run_dump(env_on, args.video_reference, args.model_id, args.confidence,
                      b_path, args.max_frames, args.timeout)
        if rc != 0:
            print(f"FATAL: on-run exited {rc}", file=sys.stderr)
            return 3

    return diff_dumps(a_path, b_path, args.conf_tol, args.iou_tol)


if __name__ == "__main__":
    raise SystemExit(main())
