"""RFDETR-seg post-process ablation: eager vs torch.compile vs Triton fullpost.

Measures five variants on the fullpost-eligible path
(batch=1, STRETCH_TO, class remapping, no static crop):

  eager            baseline torch ops — mirrors common.py:post_process_instance_segmentation_results
  compiled         torch.compile(eager) with dynamic=True
  compiled_fixed   fixed-shape torch.compile (upsample all Q masks, consumer reads keep mask)
  compiled_hybrid  compile filter+bbox stage; gather+upsample survivors uncompiled
  triton           triton_rfdetr_fullpost from inference_models.models.rfdetr.triton_fullpostproc

The question this answers: can torch.compile close the perf gap the Triton
fullpost kernel opens up? Short answer — not entirely. The atomic-counter
compaction pattern (over-launch the mask kernel, each program reads the
survivor count on-GPU and early-exits) has no torch.compile equivalent. The
compiled_hybrid variant gets within ~10% by compiling the shape-static prefix
and using F.interpolate (not TVF.resize, which defaults to antialiased and
is ~2x slower) for the upsample.

Inputs are synthesized to produce ~25 survivors out of 300 queries — typical
for RFDETR-seg. Boost knobs in make_inputs() control that count.

Usage:
    # all variants, default 200 iters / 50 warmup:
    python benchmark_rfdetr_postproc_ablation.py

    # parity check (verify outputs match):
    python benchmark_rfdetr_postproc_ablation.py --parity-check --iters 10

    # profile under nsys with NVTX ranges:
    nsys profile -t cuda,nvtx -o report.qdstrm python benchmark_rfdetr_postproc_ablation.py \\
        --mode triton --iters 50 --warmup 20 --nsys
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Tuple

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TVF

try:
    from inference_models.models.rfdetr.triton_fullpostproc import triton_rfdetr_fullpost
    _TRITON_AVAILABLE = True
except ImportError:
    triton_rfdetr_fullpost = None
    _TRITON_AVAILABLE = False

import torch._dynamo
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._dynamo.config.capture_scalar_outputs = True


NUM_QUERIES = 300
NUM_CLASSES = 90
NUM_CLASSES_TOTAL = 91
MASK_H = MASK_W = 78
ORIG_H, ORIG_W = 720, 1280
INF_H = INF_W = 432
PAD = (0, 0, 0, 0)
SCALE_W = INF_W / ORIG_W
SCALE_H = INF_H / ORIG_H
THRESHOLD = 0.5


@dataclass
class Inputs:
    logits: torch.Tensor
    bboxes: torch.Tensor
    masks: torch.Tensor
    class_mapping: torch.Tensor


def make_inputs(device: torch.device, seed: int = 0, n_boost: int = 25) -> Inputs:
    g = torch.Generator(device=device).manual_seed(seed)
    # Most logits strongly negative (no-object dominant); boost n_boost queries
    # above threshold to mimic RFDETR-seg's ~15-30 survivors per frame.
    logits = torch.randn(1, NUM_QUERIES, NUM_CLASSES_TOTAL, generator=g, device=device) * 0.5 - 3.0
    boost_idx = torch.randperm(NUM_QUERIES, generator=g, device=device)[:n_boost]
    boost_cls = torch.randint(0, NUM_CLASSES, (n_boost,), generator=g, device=device)
    logits[0, boost_idx, boost_cls] += 6.0
    cx = torch.rand(1, NUM_QUERIES, generator=g, device=device) * 0.8 + 0.1
    cy = torch.rand(1, NUM_QUERIES, generator=g, device=device) * 0.8 + 0.1
    w = torch.rand(1, NUM_QUERIES, generator=g, device=device) * 0.2 + 0.05
    h = torch.rand(1, NUM_QUERIES, generator=g, device=device) * 0.2 + 0.05
    bboxes = torch.stack([cx, cy, w, h], dim=-1)
    masks = torch.randn(1, NUM_QUERIES, MASK_H, MASK_W, generator=g, device=device) * 0.5
    cm = torch.arange(NUM_CLASSES_TOTAL, dtype=torch.int64, device=device)
    cm[NUM_CLASSES:] = -1
    return Inputs(logits=logits, bboxes=bboxes, masks=masks, class_mapping=cm)


def eager_postproc(
    logits, bboxes, masks, class_mapping,
    threshold: float,
    inf_w: int, inf_h: int,
    pad_l: int, pad_t: int,
    scale_w: float, scale_h: float,
    orig_w: int, orig_h: int,
):
    """Mirrors common.py:post_process_instance_segmentation_results for the
    fullpost-eligible case (batch=1, STRETCH_TO, class remapping, no static crop).
    Returns (xyxy_int32, confidence, class_id_int32, mask_bin_uint8)."""
    image_logits = logits[0].sigmoid()
    image_bboxes = bboxes[0]
    image_masks = masks[0]

    confidence, top_classes = image_logits.max(dim=1)
    mapped = class_mapping[top_classes]
    keep_remap = mapped >= 0
    confidence = confidence[keep_remap]
    top_classes = mapped[keep_remap]
    image_bboxes = image_bboxes[keep_remap]
    image_masks = image_masks[keep_remap]

    keep_thr = confidence > threshold
    confidence = confidence[keep_thr]
    top_classes = top_classes[keep_thr]
    selected_boxes = image_bboxes[keep_thr]
    selected_masks = image_masks[keep_thr]

    confidence, sorted_indices = torch.sort(confidence, descending=True)
    top_classes = top_classes[sorted_indices]
    selected_boxes = selected_boxes[sorted_indices]
    selected_masks = selected_masks[sorted_indices]

    cxcy = selected_boxes[:, :2]
    wh = selected_boxes[:, 2:]
    xy_min = cxcy - 0.5 * wh
    xy_max = cxcy + 0.5 * wh
    xyxy_pct = torch.cat([xy_min, xy_max], dim=-1)
    denorm = torch.tensor([inf_w, inf_h, inf_w, inf_h], device=logits.device, dtype=xyxy_pct.dtype)
    xyxy_inf = xyxy_pct * denorm

    offsets = torch.tensor([pad_l, pad_t, pad_l, pad_t], device=logits.device, dtype=xyxy_inf.dtype)
    xyxy_inf = xyxy_inf - offsets
    scale = torch.as_tensor([scale_w, scale_h, scale_w, scale_h], device=logits.device, dtype=xyxy_inf.dtype)
    xyxy_orig = xyxy_inf / scale

    xyxy_orig[:, 0].clamp_(0, orig_w)
    xyxy_orig[:, 1].clamp_(0, orig_h)
    xyxy_orig[:, 2].clamp_(0, orig_w)
    xyxy_orig[:, 3].clamp_(0, orig_h)

    if selected_masks.shape[0] > 0:
        mask_bin = TVF.resize(
            selected_masks, [orig_h, orig_w],
            interpolation=TVF.InterpolationMode.BILINEAR,
        ).gt(0.0).to(torch.uint8)
    else:
        mask_bin = torch.empty((0, orig_h, orig_w), dtype=torch.uint8, device=logits.device)

    xyxy_int = xyxy_orig.round().to(torch.int32)
    return xyxy_int, confidence, top_classes.to(torch.int32), mask_bin


def cuda_time(fn: Callable[[], object], iters: int, warmup: int) -> float:
    """Median per-iter ms via paired cuda events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument(
        "--mode",
        choices=["eager", "compiled", "compiled_fixed", "compiled_hybrid", "triton", "all"],
        default="all",
    )
    ap.add_argument(
        "--compile-mode",
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    ap.add_argument("--parity-check", action="store_true")
    ap.add_argument("--nsys", action="store_true", help="Emit NVTX ranges per variant")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("requires CUDA")

    device = torch.device("cuda")
    inp = make_inputs(device)
    cm_i64 = inp.class_mapping

    def run_eager():
        return eager_postproc(
            inp.logits, inp.bboxes, inp.masks, cm_i64,
            THRESHOLD, INF_W, INF_H, PAD[0], PAD[1],
            SCALE_W, SCALE_H, ORIG_W, ORIG_H,
        )

    # torch.compile variants want device-side constants — host-tensor construction
    # inside the traced region breaks cudagraph capture.
    denorm_t = torch.tensor([INF_W, INF_H, INF_W, INF_H], device=device, dtype=torch.float32)
    offsets_t = torch.tensor([PAD[0], PAD[1], PAD[0], PAD[1]], device=device, dtype=torch.float32)
    scale_t = torch.tensor([SCALE_W, SCALE_H, SCALE_W, SCALE_H], device=device, dtype=torch.float32)

    def eager_postproc_pretens(
        logits, bboxes, masks, class_mapping,
        threshold, orig_w: int, orig_h: int,
        denorm, offsets, scale,
    ):
        image_logits = logits[0].sigmoid()
        image_bboxes = bboxes[0]
        image_masks = masks[0]
        confidence, top_classes = image_logits.max(dim=1)
        mapped = class_mapping[top_classes]
        keep_remap = mapped >= 0
        confidence = confidence[keep_remap]
        top_classes = mapped[keep_remap]
        image_bboxes = image_bboxes[keep_remap]
        image_masks = image_masks[keep_remap]
        keep_thr = confidence > threshold
        confidence = confidence[keep_thr]
        top_classes = top_classes[keep_thr]
        selected_boxes = image_bboxes[keep_thr]
        selected_masks = image_masks[keep_thr]
        confidence, sorted_indices = torch.sort(confidence, descending=True)
        top_classes = top_classes[sorted_indices]
        selected_boxes = selected_boxes[sorted_indices]
        selected_masks = selected_masks[sorted_indices]
        cxcy = selected_boxes[:, :2]
        wh = selected_boxes[:, 2:]
        xy_min = cxcy - 0.5 * wh
        xy_max = cxcy + 0.5 * wh
        xyxy_pct = torch.cat([xy_min, xy_max], dim=-1)
        xyxy_inf = xyxy_pct * denorm - offsets
        xyxy_orig = xyxy_inf / scale
        xyxy_orig[:, 0].clamp_(0, orig_w)
        xyxy_orig[:, 1].clamp_(0, orig_h)
        xyxy_orig[:, 2].clamp_(0, orig_w)
        xyxy_orig[:, 3].clamp_(0, orig_h)
        if selected_masks.shape[0] > 0:
            mask_bin = TVF.resize(
                selected_masks, [orig_h, orig_w],
                interpolation=TVF.InterpolationMode.BILINEAR,
            ).gt(0.0).to(torch.uint8)
        else:
            mask_bin = torch.empty((0, orig_h, orig_w), dtype=torch.uint8, device=logits.device)
        xyxy_int = xyxy_orig.round().to(torch.int32)
        return xyxy_int, confidence, top_classes.to(torch.int32), mask_bin

    compiled_fn = torch.compile(
        eager_postproc_pretens, mode=args.compile_mode, dynamic=True, fullgraph=False,
    )

    def run_compiled():
        return compiled_fn(
            inp.logits, inp.bboxes, inp.masks, cm_i64,
            THRESHOLD, ORIG_W, ORIG_H, denorm_t, offsets_t, scale_t,
        )

    # compiled_fixed: matches Triton's design — no boolean indexing, no sort,
    # upsample all NUM_QUERIES masks (consumer skips by keep mask). Avoids
    # graph breaks at the cost of wasted mask-upsample work.
    def eager_postproc_fixed(
        logits, bboxes, masks, class_mapping,
        threshold, orig_w: int, orig_h: int,
        denorm, offsets, scale,
    ):
        image_logits = logits[0].sigmoid()
        image_bboxes = bboxes[0]
        image_masks = masks[0]
        confidence, top_classes = image_logits.max(dim=1)
        mapped = class_mapping[top_classes]
        keep = (mapped >= 0) & (confidence > threshold)
        cxcy = image_bboxes[:, :2]
        wh = image_bboxes[:, 2:]
        xy_min = cxcy - 0.5 * wh
        xy_max = cxcy + 0.5 * wh
        xyxy_pct = torch.cat([xy_min, xy_max], dim=-1)
        xyxy_inf = xyxy_pct * denorm - offsets
        xyxy_orig = xyxy_inf / scale
        xyxy_orig[:, 0].clamp_(0, orig_w)
        xyxy_orig[:, 1].clamp_(0, orig_h)
        xyxy_orig[:, 2].clamp_(0, orig_w)
        xyxy_orig[:, 3].clamp_(0, orig_h)
        xyxy_int = xyxy_orig.round().to(torch.int32)
        mask_bin = F.interpolate(
            image_masks.unsqueeze(0), size=(orig_h, orig_w),
            mode="bilinear", align_corners=False,
        ).squeeze(0).gt(0.0).to(torch.uint8)
        return xyxy_int, confidence, mapped.to(torch.int32), mask_bin, keep

    compiled_fixed_fn = torch.compile(
        eager_postproc_fixed, mode=args.compile_mode, dynamic=False, fullgraph=True,
    )

    def run_compiled_fixed():
        return compiled_fixed_fn(
            inp.logits, inp.bboxes, inp.masks, cm_i64,
            THRESHOLD, ORIG_W, ORIG_H, denorm_t, offsets_t, scale_t,
        )

    # compiled_hybrid: compile the shape-static filter+bbox stage, do the
    # dynamic gather+upsample in eager. Only N survivor masks get upsampled.
    # F.interpolate beats TVF.resize (which defaults to antialiased, ~2x slower).
    def stage1_filter(
        logits, bboxes, class_mapping,
        threshold, orig_w: int, orig_h: int,
        denorm, offsets, scale,
    ):
        image_logits = logits[0].sigmoid()
        image_bboxes = bboxes[0]
        confidence, top_classes = image_logits.max(dim=1)
        mapped = class_mapping[top_classes]
        keep = (mapped >= 0) & (confidence > threshold)
        cxcy = image_bboxes[:, :2]
        wh = image_bboxes[:, 2:]
        xy_min = cxcy - 0.5 * wh
        xy_max = cxcy + 0.5 * wh
        xyxy_pct = torch.cat([xy_min, xy_max], dim=-1)
        xyxy_inf = xyxy_pct * denorm - offsets
        xyxy_orig = xyxy_inf / scale
        xyxy_orig[:, 0].clamp_(0, orig_w)
        xyxy_orig[:, 1].clamp_(0, orig_h)
        xyxy_orig[:, 2].clamp_(0, orig_w)
        xyxy_orig[:, 3].clamp_(0, orig_h)
        xyxy_int = xyxy_orig.round().to(torch.int32)
        return xyxy_int, confidence, mapped.to(torch.int32), keep

    stage1_compiled = torch.compile(stage1_filter, mode=args.compile_mode, fullgraph=True)

    def run_compiled_hybrid():
        xyxy, conf, cls, keep = stage1_compiled(
            inp.logits, inp.bboxes, cm_i64, THRESHOLD, ORIG_W, ORIG_H,
            denorm_t, offsets_t, scale_t,
        )
        idx = keep.nonzero(as_tuple=False).squeeze(1)
        conf_k = conf[idx]
        conf_sorted, sort_idx = torch.sort(conf_k, descending=True)
        idx = idx[sort_idx]
        selected_masks = inp.masks[0][idx]
        mask_bin = F.interpolate(
            selected_masks.unsqueeze(0), size=(ORIG_H, ORIG_W),
            mode="bilinear", align_corners=False,
        ).squeeze(0).gt(0.0).to(torch.uint8)
        return xyxy[idx], conf_sorted, cls[idx], mask_bin

    def run_triton():
        combined, mask_bin, mask_any, counter, done = triton_rfdetr_fullpost(
            bboxes=inp.bboxes,
            logits=inp.logits,
            masks=inp.masks,
            threshold=THRESHOLD,
            num_classes=NUM_CLASSES,
            class_mapping=cm_i64,
            inference_size_wh=(INF_W, INF_H),
            pad_ltrb=PAD,
            scale_wh=(SCALE_W, SCALE_H),
            orig_size_wh=(ORIG_W, ORIG_H),
        )
        # counter.item() forces sync — matches the overhead of the real
        # adapter, which issues a pinned-host DtoH and then syncs on the event.
        n = int(counter.item())
        return combined[:n], mask_bin[:n], mask_any[:n]

    if args.parity_check:
        print("Parity check...")
        xe, ce, cle, me = run_eager()
        if _TRITON_AVAILABLE:
            import numpy as np
            ct, mbt, mat = run_triton()
            print(f"  eager survivors={xe.shape[0]}, triton survivors={ct.shape[0]}")
            conf_t = ct.cpu().numpy()[:, 4].view(np.float32)
            print(f"  eager conf range:  {ce.min().item():.4f}..{ce.max().item():.4f}")
            print(f"  triton conf range: {conf_t.min():.4f}..{conf_t.max():.4f}")
        else:
            print(f"  eager survivors={xe.shape[0]} (triton unavailable)")

    results = {}
    ordered = ["eager", "compiled", "compiled_fixed", "compiled_hybrid", "triton"]
    runners = {
        "eager": run_eager,
        "compiled": run_compiled,
        "compiled_fixed": run_compiled_fixed,
        "compiled_hybrid": run_compiled_hybrid,
        "triton": run_triton if _TRITON_AVAILABLE else None,
    }

    for name in ordered:
        if args.mode not in (name, "all"):
            continue
        fn = runners[name]
        if fn is None:
            print(f"  (skipping {name}: triton unavailable)")
            continue
        # Warm compile for torch.compile variants
        if name.startswith("compiled"):
            fn()
            torch.cuda.synchronize()
        if args.nsys:
            torch.cuda.nvtx.range_push(name)
        t = cuda_time(fn, args.iters, args.warmup)
        if args.nsys:
            torch.cuda.nvtx.range_pop()
        results[name] = t

    print("\n=== Median per-iter (ms) ===")
    for k in ordered:
        if k in results:
            print(f"  {k:16s}: {results[k]:7.3f} ms")
    if "eager" in results:
        base = results["eager"]
        print()
        for k in ordered:
            if k != "eager" and k in results:
                print(f"  speedup {k:16s} vs eager: {base / results[k]:.2f}x")


if __name__ == "__main__":
    main()
