"""Fused Triton kernel for RF-DETR post-processing first stage.

Replaces the sequence
  sigmoid(logits) -> argmax(class) -> named-filter -> confidence-threshold
with a single kernel launch. The remaining ops (sort, gather by index, bbox
denorm, mask alignment) stay in torch.

Per-query: logits row has `num_classes_total` entries (num_classes + optional
"no-object" slot at the end). The kernel computes, for each query:
  conf[q]     = max_c sigmoid(logits[q, c])
  top_cls[q]  = argmax_c logits[q, c]
  keep[q]     = (top_cls[q] < num_classes) & (conf[q] > threshold[top_cls[q]])
"""
from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _rfdetr_conf_filter_kernel(
        logits_ptr,
        threshold_ptr,
        scalar_threshold,
        class_map_ptr,  # (num_classes_total,) int, maps raw class -> remapped id; -1 = drop
        conf_out_ptr,
        top_class_out_ptr,
        keep_out_ptr,
        num_queries,
        num_classes,
        num_classes_total,
        logits_stride_q,
        PER_CLASS: tl.constexpr,
        HAS_REMAPPING: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= num_queries:
            return
        offs_c = tl.arange(0, BLOCK_C)
        mask_c = offs_c < num_classes_total
        logits_row = tl.load(
            logits_ptr + pid * logits_stride_q + offs_c,
            mask=mask_c,
            other=-float("inf"),
        )
        max_val = tl.max(logits_row, axis=0)
        BIG = 1 << 30
        is_max = logits_row == max_val
        idx_or_big = tl.where(is_max & mask_c, offs_c, BIG)
        raw_c = tl.min(idx_or_big, axis=0)
        if HAS_REMAPPING:
            top_c = tl.load(class_map_ptr + raw_c)
            valid = top_c >= 0
        else:
            top_c = raw_c
            valid = raw_c < num_classes
        abs_max = tl.abs(max_val)
        z = tl.exp(-abs_max)
        sig_pos = 1.0 / (1.0 + z)
        sig_neg = z / (1.0 + z)
        conf = tl.where(max_val >= 0.0, sig_pos, sig_neg)
        if PER_CLASS:
            safe_c = tl.where(valid, top_c, 0)
            thr = tl.load(threshold_ptr + safe_c)
        else:
            thr = scalar_threshold
        keep = valid & (conf > thr)
        tl.store(conf_out_ptr + pid, conf)
        tl.store(top_class_out_ptr + pid, top_c)
        tl.store(keep_out_ptr + pid, keep.to(tl.int8))


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def triton_rfdetr_conf_filter(
    logits: torch.Tensor,
    threshold: "torch.Tensor | float",
    num_classes: int,
    class_mapping: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused sigmoid + argmax + named/remap-filter + confidence-threshold.

    Args:
        logits: (num_queries, num_classes_total) CUDA tensor for a single image.
        threshold: scalar float or (num_classes,) CUDA tensor (per remapped id).
        num_classes: number of named classes (when no remapping: drop rows
            where argmax >= num_classes).
        class_mapping: optional (num_classes_total,) int32/int64 CUDA tensor:
            raw_class -> remapped_id, or -1 to drop.

    Returns:
        (conf, top_class, keep) each of length num_queries.
    """
    assert logits.is_cuda and logits.ndim == 2
    num_queries, num_classes_total = logits.shape
    logits_c = logits.contiguous()

    device = logits.device
    conf = torch.empty((num_queries,), dtype=torch.float32, device=device)
    top_c = torch.empty((num_queries,), dtype=torch.int32, device=device)
    keep = torch.empty((num_queries,), dtype=torch.int8, device=device)

    if isinstance(threshold, torch.Tensor):
        per_class = True
        thr_tensor = threshold.contiguous()
        scalar_thr = 0.0
    else:
        per_class = False
        thr_tensor = torch.empty((1,), dtype=torch.float32, device=device)
        scalar_thr = float(threshold)

    if class_mapping is not None:
        has_remap = True
        cmap = class_mapping.to(dtype=torch.int32, device=device).contiguous()
    else:
        has_remap = False
        cmap = torch.empty((1,), dtype=torch.int32, device=device)

    BLOCK_C = max(32, _next_pow2(num_classes_total))
    _rfdetr_conf_filter_kernel[(num_queries,)](
        logits_c,
        thr_tensor,
        scalar_thr,
        cmap,
        conf,
        top_c,
        keep,
        num_queries,
        num_classes,
        num_classes_total,
        logits_c.stride(0),
        PER_CLASS=1 if per_class else 0,
        HAS_REMAPPING=1 if has_remap else 0,
        BLOCK_C=BLOCK_C,
    )
    return conf, top_c, keep.bool()
