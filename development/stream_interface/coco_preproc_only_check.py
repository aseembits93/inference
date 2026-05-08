"""Drives the COCO preproc-only study end-to-end.

1. Runs coco_preproc_only_dump.py once per selected preproc (default:
   ref + cv2 + triton).
2. Diffs per-image detection digests with greedy IoU pairing for each
   pair of runs involving triton (triton-vs-ref and triton-vs-cv2).
3. Computes pycocotools bbox + segm mAP for every run and prints a
   side-by-side table with deltas vs the F.interpolate reference.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = str(REPO / ".venv" / "bin" / "python3")
DUMP = str(REPO / "development" / "stream_interface" / "coco_preproc_only_dump.py")

sys.path.insert(0, str(REPO / "development" / "stream_interface"))
from correctness_check import _pair_dets  # noqa: E402


def _env_common() -> dict:
    return {
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


def run_dump(preproc: str, images_dir: str, annotations_json: str,
             model_id: str, confidence: float, dump_prefix: str,
             max_images: int, timeout_s: int) -> int:
    env = os.environ.copy()
    env.update(_env_common())
    cmd = [
        PY, DUMP,
        "--images_dir", images_dir,
        "--annotations_json", annotations_json,
        "--model_id", model_id,
        "--confidence", str(confidence),
        "--dump_prefix", dump_prefix,
        "--preproc", preproc,
    ]
    if max_images > 0:
        cmd += ["--max_images", str(max_images)]
    print(f"[run] preproc={preproc} -> {dump_prefix}.{{jsonl,json}}",
          flush=True)
    proc = subprocess.run(cmd, env=env, cwd=str(REPO / "development"),
                          timeout=timeout_s)
    return proc.returncode


def load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def diff_detection_dumps(a_path: str, b_path: str, conf_tol: float,
                         iou_tol: float, diff_conf_threshold: float,
                         label: str = "") -> None:
    """The dumps include dets down to confidence 0.05 (for COCO mAP), but
    the detection diff only makes sense at a realistic threshold — the
    0.05..0.4 tail is dominated by degenerate boxes that pair arbitrarily
    across runs. `diff_conf_threshold` filters both sides before pairing."""
    A = load_jsonl(a_path)
    B = load_jsonl(b_path)
    assert len(A) == len(B), f"image count mismatch {len(A)} vs {len(B)}"
    by_id_a = {r["image_id"]: r for r in A}
    by_id_b = {r["image_id"]: r for r in B}
    assert set(by_id_a) == set(by_id_b), "image_id sets differ"

    total_a = total_b = 0
    matched = 0
    unmatched_a = unmatched_b = 0
    mask_mm = 0
    conf_deltas = []
    edge_deltas = []
    frames_clean = 0

    for iid in by_id_a:
        da = [d for d in (by_id_a[iid]["dets"] or [])
              if d["conf"] >= diff_conf_threshold]
        db = [d for d in (by_id_b[iid]["dets"] or [])
              if d["conf"] >= diff_conf_threshold]
        total_a += len(da)
        total_b += len(db)
        pairs, una, unb = _pair_dets(da, db, iou_tol)
        matched += len(pairs)
        unmatched_a += len(una)
        unmatched_b += len(unb)
        clean = not una and not unb
        for ai, bj in pairs:
            ra, rb = da[ai], db[bj]
            d_edge = max(abs(ra["xyxy"][k] - rb["xyxy"][k]) for k in range(4))
            edge_deltas.append(d_edge)
            d_conf = abs(ra["conf"] - rb["conf"])
            conf_deltas.append(d_conf)
            if ra["mask_md5"] != rb["mask_md5"]:
                mask_mm += 1
                clean = False
            elif d_conf > conf_tol or d_edge > 0:
                clean = False
        if clean:
            frames_clean += 1

    def _pct(arr, q):
        arr.sort()
        return arr[int(len(arr) * q)] if arr else 0.0

    header = "DETECTION DIFF" if not label else f"DETECTION DIFF [{label}]"
    print(f"\n==================== {header} ====================")
    print(f"a / b                 : {a_path}  vs  {b_path}")
    print(f"images                : {len(A)}")
    print(f"filter conf >=        : {diff_conf_threshold}")
    print(f"images fully clean    : {frames_clean}")
    print(f"total dets (a / b)    : {total_a} / {total_b}")
    print(f"matched pairs         : {matched}")
    print(f"unmatched A / B       : {unmatched_a} / {unmatched_b}")
    print(f"mask_md5 mismatches   : {mask_mm} / {matched} "
          f"({100*mask_mm/matched if matched else 0:.1f}%)")
    if conf_deltas:
        print(f"conf delta mean       : {sum(conf_deltas)/len(conf_deltas):.3e}")
        print(f"conf delta p50/p95/p99: "
              f"{_pct(conf_deltas,0.5):.3e} / "
              f"{_pct(conf_deltas,0.95):.3e} / "
              f"{_pct(conf_deltas,0.99):.3e}")
        print(f"conf delta max        : {max(conf_deltas):.3e}")
        print(f"box edge delta max    : {max(edge_deltas):.1f} px")
    print("========================================================")


def coco_eval(detections_json: str, annotations_json: str,
              label: str) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    print(f"\n==================== COCO mAP [{label}] ====================")
    coco_gt = COCO(annotations_json)
    # Restrict GT to the images actually evaluated (safe if all 5000).
    with open(detections_json) as f:
        dets = json.load(f)
    img_ids_with_dets = sorted({d["image_id"] for d in dets})
    out = {}
    for iouType in ("bbox", "segm"):
        print(f"-- {iouType} --")
        coco_dt = coco_gt.loadRes(detections_json)
        e = COCOeval(coco_gt, coco_dt, iouType=iouType)
        e.params.imgIds = img_ids_with_dets
        e.evaluate()
        e.accumulate()
        e.summarize()
        stats = list(map(float, e.stats))
        out[iouType] = {
            "AP_50_95": stats[0], "AP_50": stats[1], "AP_75": stats[2],
            "AP_S": stats[3], "AP_M": stats[4], "AP_L": stats[5],
            "AR_1": stats[6], "AR_10": stats[7], "AR_100": stats[8],
            "AR_S": stats[9], "AR_M": stats[10], "AR_L": stats[11],
        }
    print("=============================================================")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir",
                    default=str(REPO / "coco" / "val2017"))
    ap.add_argument("--annotations_json",
                    default=str(REPO / "coco" / "annotations"
                                / "instances_val2017.json"))
    ap.add_argument("--model_id", default="rfdetr-seg-nano")
    ap.add_argument("--confidence", type=float, default=0.05)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--max_images", type=int, default=0)
    ap.add_argument("--conf_tol", type=float, default=1e-4)
    ap.add_argument("--iou_tol", type=float, default=0.5)
    ap.add_argument("--diff_conf_threshold", type=float, default=0.4,
                    help="min conf for a detection to enter the diff pass")
    ap.add_argument("--dump_dir", default=str(REPO))
    ap.add_argument("--diff_only", action="store_true")
    ap.add_argument(
        "--preprocs",
        default="ref,cv2,triton",
        help="comma list of preproc variants to run/diff "
             "(subset of ref,cv2,triton). The 'triton' run is the "
             "subject under test; diffs are produced for each other "
             "variant vs triton.",
    )
    args = ap.parse_args()

    variants = [v.strip() for v in args.preprocs.split(",") if v.strip()]
    valid = {"ref", "cv2", "triton"}
    bad = [v for v in variants if v not in valid]
    if bad:
        print(f"FATAL: unknown preproc(s) {bad}; valid={sorted(valid)}",
              file=sys.stderr)
        return 2
    if "triton" not in variants:
        print("FATAL: --preprocs must include 'triton' (the subject "
              "under test)", file=sys.stderr)
        return 2

    dump_dir = Path(args.dump_dir)
    prefixes = {v: str(dump_dir / f"coco_preproc_{v}") for v in variants}

    if not args.diff_only:
        t0 = time.time()
        for v in variants:
            rc = run_dump(v, args.images_dir, args.annotations_json,
                          args.model_id, args.confidence, prefixes[v],
                          args.max_images, args.timeout)
            if rc != 0:
                print(f"FATAL: {v} exit {rc}", file=sys.stderr)
                return 3
        print(f"[bench] {len(variants)} dumps: {time.time()-t0:.1f}s",
              flush=True)

    # Diff every non-triton variant against triton. Pairings are directed
    # (a=reference, b=triton) so "unmatched A" means dets present only in
    # the reference path.
    for v in variants:
        if v == "triton":
            continue
        diff_detection_dumps(
            prefixes[v] + ".jsonl", prefixes["triton"] + ".jsonl",
            args.conf_tol, args.iou_tol, args.diff_conf_threshold,
            label=f"{v} vs triton",
        )

    # mAP for every variant.
    maps = {
        v: coco_eval(prefixes[v] + ".json", args.annotations_json, v)
        for v in variants
    }

    # Pick the baseline for the delta column: 'ref' if present, else the
    # first non-triton variant, else triton itself (degenerate but valid).
    baseline = "ref" if "ref" in maps else next(
        (v for v in variants if v != "triton"), "triton")

    print("\n==================== SIDE-BY-SIDE ====================")
    header = f"{'metric':<20}"
    for v in variants:
        header += f" {v:>10}"
    for v in variants:
        if v != baseline:
            header += f" {('d.'+v[:6]):>10}"
    print(header)
    for t in ("bbox", "segm"):
        for k in ("AP_50_95", "AP_50", "AP_75", "AP_S", "AP_M", "AP_L"):
            row = f"{t+'.'+k:<20}"
            for v in variants:
                row += f" {maps[v][t][k]:>10.4f}"
            for v in variants:
                if v != baseline:
                    row += f" {maps[v][t][k] - maps[baseline][t][k]:>+10.4f}"
            print(row)
    print(f"(delta is <variant> - {baseline})")
    print("======================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
