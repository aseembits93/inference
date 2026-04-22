#!/usr/bin/env python3
"""Numerical correctness at the logit level, not the mask level.

Runs SAM3 forward pass under four configurations and captures the raw
model output (pre-threshold, pre-RLE) for comparison:

 - PT-bf16 (repo default)
 - PT-fp16 (PT autocast fp16)
 - PT-fp32 (no autocast)
 - TRT    (best correct engine)

For each tensor in the output dict, reports max-abs diff, mean-abs diff,
relative error, and cosine similarity vs the PT-bf16 reference. Also
reports the magnitude ratio (tst.std / ref.std) — this was what would
have flagged the FP16 amplification bug immediately.
"""

from __future__ import annotations

import base64
import gc
import os
import sys
from pathlib import Path

if "ROBOFLOW_API_KEY" not in os.environ:
    raise SystemExit("Set ROBOFLOW_API_KEY env var before running this script.")


import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

IMAGE = Path(os.environ.get("SAM3_ASSETS", "tests/workflows/integration_tests/execution/assets")) / "dogs.jpg"
PROMPT = "dog"
ENGINE = Path(
    "./sam3_onnx_exports/"
    "sam3_vision_backbone_fp16_rope_windowed_d8.engine"
)


def _patch_autocast(dtype_str: str | None):
    """Monkey-patch torch.autocast to use a chosen dtype, BEFORE model load."""
    if dtype_str is None:
        return
    import torch.amp.autocast_mode as m
    orig_init = m.autocast.__init__
    if dtype_str == "fp32":
        def new_init(self, device_type, dtype=None, enabled=True, cache_enabled=None):
            orig_init(self, device_type=device_type, dtype=torch.float32,
                      enabled=False, cache_enabled=cache_enabled)
    else:
        want = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_str]
        def new_init(self, device_type, dtype=None, enabled=True, cache_enabled=None):
            orig_init(self, device_type=device_type, dtype=want,
                      enabled=enabled, cache_enabled=cache_enabled)
    m.autocast.__init__ = new_init


def run(which: str):
    """which in {bf16, fp16, fp32, trt}. Returns a dict of flat float32
    tensors representing the raw model output (logits etc)."""
    if which == "bf16":
        _patch_autocast(None)  # repo default is bf16
    elif which in ("fp16", "fp32"):
        _patch_autocast(which)
    elif which == "trt":
        _patch_autocast(None)
    else:
        raise ValueError(which)

    from inference.models.sam3.segment_anything3 import (
        SegmentAnything3, _build_text_query,
    )
    from inference.core.utils.image_utils import load_image_rgb
    from sam3.train.data.collator import collate_fn_api
    from sam3.train.data.sam3_image_dataset import Datapoint as Sam3Datapoint
    from sam3.train.data.sam3_image_dataset import Image as Sam3ImageDP
    from sam3.model.utils.misc import copy_data_to_device

    m = SegmentAnything3(
        model_id="sam3/sam3_final", api_key=os.environ["ROBOFLOW_API_KEY"],
    )

    if which == "trt":
        from sam3_trt_adapter import patch_sam3_with_trt_backbone
        patch_sam3_with_trt_backbone(m.model, ENGINE)

    # Build the exact batch the model normally sees
    img_b64 = base64.b64encode(IMAGE.read_bytes()).decode()
    np_image = load_image_rgb({"type": "base64", "value": img_b64})
    h, w = np_image.shape[:2]
    pil = Image.fromarray(np_image)

    dp = Sam3Datapoint(find_queries=[], images=[Sam3ImageDP(data=pil, objects=[], size=(h, w))])
    dp.find_queries.append(_build_text_query(coco_id=0, h=h, w=w, text=PROMPT))
    dp = m.transform(dp)
    batch = collate_fn_api(batch=[dp], dict_key="dummy")["dummy"]
    batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = m.model(batch)

    # Flatten the output dict into (name -> float32 cpu tensor)
    flat = {}

    def _walk(prefix, obj):
        if isinstance(obj, torch.Tensor):
            flat[prefix] = obj.float().detach().cpu().numpy()
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _walk(f"{prefix}/{k}", v)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                _walk(f"{prefix}[{i}]", v)

    _walk("out", output)
    del m
    gc.collect(); torch.cuda.empty_cache()
    return flat


def compare(ref: dict, tst: dict, label: str):
    keys = sorted(ref.keys())
    print(f"\n--- {label} vs PT-bf16 ---")
    print(f"{'tensor':50s} {'shape':18s} {'max|Δ|':>9s} {'mean|Δ|':>9s} "
          f"{'rel_err':>9s} {'cos':>7s} {'std_ratio':>10s}")
    worst_cos = 1.0
    worst_ratio = 1.0
    for k in keys:
        if k not in tst:
            print(f"{k:50s} MISSING IN TEST")
            continue
        a = ref[k]; b = tst[k]
        if a.shape != b.shape:
            print(f"{k:50s} SHAPE MISMATCH {a.shape} vs {b.shape}")
            continue
        if a.dtype.kind not in "f":
            continue
        af = a.flatten().astype(np.float64)
        bf = b.flatten().astype(np.float64)
        diff = np.abs(af - bf)
        max_d = float(diff.max())
        mean_d = float(diff.mean())
        ref_mag = float(np.abs(af).mean()) + 1e-12
        rel = mean_d / ref_mag
        na = np.linalg.norm(af); nb = np.linalg.norm(bf)
        cos = float(af @ bf / (na * nb + 1e-12)) if na > 0 and nb > 0 else float("nan")
        std_ratio = float((np.std(bf) + 1e-12) / (np.std(af) + 1e-12))
        worst_cos = min(worst_cos, cos)
        worst_ratio = worst_ratio if abs(np.log(std_ratio)) < abs(np.log(worst_ratio or 1)) else std_ratio
        print(f"{k:50s} {str(a.shape):18s} "
              f"{max_d:9.4g} {mean_d:9.4g} {rel:9.4g} {cos:7.4f} {std_ratio:10.4f}")
    print(f"  worst cosine: {worst_cos:.6f}")


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        print("Run each pass in its own subprocess (T4 memory limit).")
        print("  python bench_logit_correctness.py bf16")
        print("  python bench_logit_correctness.py fp16")
        print("  python bench_logit_correctness.py fp32")
        print("  python bench_logit_correctness.py trt")
        print("  python bench_logit_correctness.py compare")
        return 1

    if which == "compare":
        ref = dict(np.load(f"{os.environ.get('SAM3_BENCH_DIR', '/tmp')}/sam3_logits_bf16.npz"))
        for tag in ["fp16", "fp32", "trt"]:
            p = Path(f"{os.environ.get('SAM3_BENCH_DIR', '/tmp')}/sam3_logits_{tag}.npz")
            if not p.exists():
                print(f"{tag}: missing, skip")
                continue
            tst = dict(np.load(p))
            compare(ref, tst, f"PT-{tag}" if tag != "trt" else "TRT")
        return 0

    flat = run(which)
    out = Path(f"{os.environ.get('SAM3_BENCH_DIR', '/tmp')}/sam3_logits_{which}.npz")
    np.savez(out, **flat)
    print(f"saved {len(flat)} tensors to {out}")
    for k in sorted(flat.keys()):
        t = flat[k]
        print(f"  {k:50s} shape={t.shape} dtype={t.dtype} "
              f"mean={t.mean():.4g} std={t.std():.4g} "
              f"min={t.min():.4g} max={t.max():.4g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
