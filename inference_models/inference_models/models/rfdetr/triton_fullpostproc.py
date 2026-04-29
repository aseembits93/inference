"""Experimental: full RF-DETR instance-segmentation post-processing in Triton.

Fuses the entire post-TRT chain into two Triton kernels of fixed grid:

  _rfdetr_fullpost_filter_kernel  (grid = num_queries)
    sigmoid(logits) -> argmax-over-classes -> class remap -> conf > threshold
    -> xywh -> xyxy -> multiply by inference_size -> subtract padding
    -> divide by scale -> clip to orig image bounds
    Emits padded fixed-shape outputs (num_queries rows). Rows that don't pass
    the filter get `keep=False`; downstream consumers skip them.

  _rfdetr_fullpost_mask_kernel  (grid = num_queries * tile_y * tile_x)
    Inverse-letterbox bilinear upsample masks 78x78 -> orig_h x orig_w,
    threshold > 0, emit as uint8. Skips work when keep[q] is False.

Design notes
------------
* Everything is fixed-shape in/out. No compaction, no sort, no variable
  grid. Downstream Python handles selection by `keep`.
* This replaces the torch.sort + gather + align_instance_segmentation_results
  chain for the common case of no static_crop, STRETCH_TO resize, class
  remapping available. Falls back to the existing path otherwise.
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
    def _rfdetr_fullpost_filter_kernel(
        logits_ptr,          # (num_queries, num_classes_total) fp16/fp32
        bboxes_ptr,          # (num_queries, 4) fp32, normalized cxcywh
        threshold_ptr,       # scalar or (num_remapped,) fp32
        class_map_ptr,       # (num_classes_total,) int32; -1 means drop
        # compact outputs (pre-sized to num_queries; host reads counter to slice)
        xyxy_out_ptr,        # (num_queries, 4) int32
        conf_out_ptr,        # (num_queries,) fp32
        class_out_ptr,       # (num_queries,) int32
        survivor_idx_out_ptr,# (num_queries,) int32 — original query id of each survivor
        mask_any_out_ptr,    # (num_queries,) int32 — zeroed at compact slot; mask kernel atomic_maxes up to 1
        counter_ptr,         # (1,) int32 — atomic counter; host reads to get n_survivors
        # static scalars
        num_queries,
        num_classes_total,
        inference_w,
        inference_h,
        pad_left,
        pad_top,
        inv_scale_w,
        inv_scale_h,
        orig_w,
        orig_h,
        logits_stride_q,
        bboxes_stride_q,
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
            valid = raw_c < num_classes_total

        abs_max = tl.abs(max_val)
        z = tl.exp(-abs_max)
        sig_pos = 1.0 / (1.0 + z)
        sig_neg = z / (1.0 + z)
        conf = tl.where(max_val >= 0.0, sig_pos, sig_neg)

        if PER_CLASS:
            safe_c = tl.where(valid, top_c, 0)
            thr = tl.load(threshold_ptr + safe_c)
        else:
            thr = tl.load(threshold_ptr)
        keep = valid & (conf > thr)

        # Early exit for filtered queries — don't compute boxes, don't
        # reserve a slot. Cheap (rest of kernel is ~8 FLOPS + 4 stores).
        if not keep:
            return

        cx = tl.load(bboxes_ptr + pid * bboxes_stride_q + 0) * inference_w
        cy = tl.load(bboxes_ptr + pid * bboxes_stride_q + 1) * inference_h
        w_half = tl.load(bboxes_ptr + pid * bboxes_stride_q + 2) * inference_w * 0.5
        h_half = tl.load(bboxes_ptr + pid * bboxes_stride_q + 3) * inference_h * 0.5

        x1 = cx - w_half - pad_left
        y1 = cy - h_half - pad_top
        x2 = cx + w_half - pad_left
        y2 = cy + h_half - pad_top

        x1 = x1 * inv_scale_w
        y1 = y1 * inv_scale_h
        x2 = x2 * inv_scale_w
        y2 = y2 * inv_scale_h

        x1 = tl.maximum(tl.minimum(x1, orig_w), 0.0)
        y1 = tl.maximum(tl.minimum(y1, orig_h), 0.0)
        x2 = tl.maximum(tl.minimum(x2, orig_w), 0.0)
        y2 = tl.maximum(tl.minimum(y2, orig_h), 0.0)

        x1_i = tl.floor(x1 + 0.5).to(tl.int32)
        y1_i = tl.floor(y1 + 0.5).to(tl.int32)
        x2_i = tl.floor(x2 + 0.5).to(tl.int32)
        y2_i = tl.floor(y2 + 0.5).to(tl.int32)

        # Reserve a compact slot via atomic-add. Order is non-deterministic
        # across survivors but downstream doesn't require query-order.
        slot = tl.atomic_add(counter_ptr, 1)

        tl.store(xyxy_out_ptr + slot * 4 + 0, x1_i)
        tl.store(xyxy_out_ptr + slot * 4 + 1, y1_i)
        tl.store(xyxy_out_ptr + slot * 4 + 2, x2_i)
        tl.store(xyxy_out_ptr + slot * 4 + 3, y2_i)
        tl.store(conf_out_ptr + slot, conf)
        tl.store(class_out_ptr + slot, top_c)
        tl.store(survivor_idx_out_ptr + slot, pid.to(tl.int32))
        # Initialize mask_any[slot] to 0 here. Later the mask kernel's
        # tile-level atomic_max will raise it to 1 if any pixel passes the
        # threshold. The filter kernel writes `slot` monotonically (atomic_add
        # returns unique values), and the entire filter kernel completes
        # before the mask kernel is launched (serialized on the stream), so
        # this store always lands first.
        tl.store(mask_any_out_ptr + slot, 0)


    @triton.jit
    def _rfdetr_fullpost_mask_kernel_compact(
        masks_ptr,           # (num_queries, mask_h, mask_w) fp32
        survivor_idx_ptr,    # (n_survivors,) int32 — indices into num_queries
        counter_ptr,         # (1,) int32 — n_survivors; used for GPU-side early exit
        out_ptr,             # (n_survivors, orig_h, orig_w) uint8 — compact binary mask
        mask_any_ptr,        # (n_survivors,) int32 — 1 if any pixel survives threshold, 0 else
        mask_h,
        mask_w,
        orig_h,
        orig_w,
        # Scale from orig -> mask coords (covers the whole mask span,
        # since STRETCH_TO has no letterbox-crop at mask resolution).
        mask_scale_y,        # mask_h / orig_h
        mask_scale_x,
        masks_stride_q,
        masks_stride_h,
        out_stride_s,
        out_stride_h,
        BLOCK_H: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        s = tl.program_id(0)  # survivor index in [0, num_queries) — over-launched
        tile_y = tl.program_id(1)
        tile_x = tl.program_id(2)

        # GPU-side early exit: filter kernel atomic-added into counter; any
        # program with s >= counter has no corresponding survivor.
        n_survivors = tl.load(counter_ptr)
        if s >= n_survivors:
            return

        # Look up the source query slot for the bilinear gather.
        q = tl.load(survivor_idx_ptr + s)

        offs_y = tile_y * BLOCK_H + tl.arange(0, BLOCK_H)
        offs_x = tile_x * BLOCK_W + tl.arange(0, BLOCK_W)
        mask_yy = offs_y < orig_h
        mask_xx = offs_x < orig_w
        m_outbox = mask_yy[:, None] & mask_xx[None, :]

        # Inverse map orig pixel -> source mask coord, pixel-center bilinear.
        src_y_f = (offs_y.to(tl.float32) + 0.5) * mask_scale_y - 0.5
        src_x_f = (offs_x.to(tl.float32) + 0.5) * mask_scale_x - 0.5
        src_y_2d = src_y_f[:, None]
        src_x_2d = src_x_f[None, :]

        y0 = tl.floor(src_y_2d).to(tl.int32)
        x0 = tl.floor(src_x_2d).to(tl.int32)
        y1 = y0 + 1
        x1 = x0 + 1
        dy = src_y_2d - y0.to(tl.float32)
        dx = src_x_2d - x0.to(tl.float32)

        y0c = tl.maximum(tl.minimum(y0, mask_h - 1), 0)
        y1c = tl.maximum(tl.minimum(y1, mask_h - 1), 0)
        x0c = tl.maximum(tl.minimum(x0, mask_w - 1), 0)
        x1c = tl.maximum(tl.minimum(x1, mask_w - 1), 0)

        base = q * masks_stride_q

        p00 = tl.load(masks_ptr + base + y0c * masks_stride_h + x0c, mask=m_outbox, other=0.0)
        p01 = tl.load(masks_ptr + base + y0c * masks_stride_h + x1c, mask=m_outbox, other=0.0)
        p10 = tl.load(masks_ptr + base + y1c * masks_stride_h + x0c, mask=m_outbox, other=0.0)
        p11 = tl.load(masks_ptr + base + y1c * masks_stride_h + x1c, mask=m_outbox, other=0.0)

        w_tl = (1.0 - dy) * (1.0 - dx)
        w_tr = (1.0 - dy) * dx
        w_bl = dy * (1.0 - dx)
        w_br = dy * dx
        val = p00 * w_tl + p01 * w_tr + p10 * w_bl + p11 * w_br
        bin_val = (val > 0.0).to(tl.int8)

        out_offsets = offs_y[:, None] * out_stride_h + offs_x[None, :]
        # Write to compact row s (not q).
        tl.store(out_ptr + s * out_stride_s + out_offsets, bin_val, mask=m_outbox)

        # Tile-level reduction of any-true within the bool tile, then a single
        # atomic-max into mask_any[s]. Saves a separate torch.any reduction
        # downstream. Atomic max preserves the 0/1 semantic across tiles.
        tile_any = tl.max(bin_val.to(tl.int32), axis=0)
        tile_any2 = tl.max(tile_any, axis=0)
        tl.atomic_max(mask_any_ptr + s, tile_any2)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


# Cache small supporting tensors so we don't incur HtoD for them per frame.
_THRESHOLD_CACHE: dict = {}
_EMPTY_INT32 = torch.empty((1,), dtype=torch.int32)
_MASK_BIN_BUFFER_CACHE: dict = {}


def _get_mask_bin_buffer(
    capacity: int, orig_h: int, orig_w: int, device: torch.device
) -> torch.Tensor:
    """Return a reusable (capacity, orig_h, orig_w) uint8 mask buffer.

    Avoids a per-frame torch.empty kernel for the biggest allocation in the
    post-process path (capacity * H * W bytes — ~10 MB at 100*240*426).
    We return the full buffer; the caller views [:n_survivors]. Rows beyond
    n_survivors may contain stale data from prior frames; the caller must
    only read the slice it sizes via the atomic counter.
    """
    key = (capacity, orig_h, orig_w, device)
    buf = _MASK_BIN_BUFFER_CACHE.get(key)
    if buf is None:
        buf = torch.empty(
            (capacity, orig_h, orig_w), dtype=torch.uint8, device=device
        )
        _MASK_BIN_BUFFER_CACHE[key] = buf
    return buf


def _prepare_threshold(threshold, device: torch.device, num_classes: int):
    """Return (threshold_tensor_on_device, per_class_flag), caching the tensor
    form of scalar thresholds so we don't ship a 4-byte HtoD every frame."""
    if isinstance(threshold, torch.Tensor):
        t = threshold
        if t.dtype != torch.float32 or t.device != device or not t.is_contiguous():
            t = t.to(dtype=torch.float32, device=device).contiguous()
        return t, True
    key = (float(threshold), device)
    cached = _THRESHOLD_CACHE.get(key)
    if cached is None:
        cached = torch.tensor([float(threshold)], dtype=torch.float32, device=device)
        _THRESHOLD_CACHE[key] = cached
    return cached, False


def triton_rfdetr_fullpost(
    bboxes: torch.Tensor,                  # (B=1, num_queries, 4) fp32 normalized cxcywh
    logits: torch.Tensor,                  # (B=1, num_queries, num_classes_total) fp32/fp16
    masks: torch.Tensor,                   # (B=1, num_queries, mask_h, mask_w) fp32
    threshold: "torch.Tensor | float",
    num_classes: int,
    class_mapping: Optional[torch.Tensor],
    inference_size_wh: Tuple[int, int],    # (W, H) of the inference image
    pad_ltrb: Tuple[int, int, int, int],   # (left, top, right, bottom) in inference coords
    scale_wh: Tuple[float, float],         # (scale_w, scale_h) = eff_w/orig_w, eff_h/orig_h
    orig_size_wh: Tuple[int, int],         # (W, H) of the original image
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full RF-DETR post-process fused into two Triton launches. Returns
    already-compacted (n_survivors,)-shaped outputs.

    Returns:
        xyxy_int:  (n_survivors, 4) int32, rounded + clipped to orig image.
        conf:      (n_survivors,) fp32
        class_id:  (n_survivors,) int32 (remapped ids)
        mask_bin:  (n_survivors, orig_h, orig_w) uint8 — already-binary
        mask_any:  (n_survivors,) bool — True if mask has any non-zero pixel
    """
    assert TRITON_AVAILABLE, "triton not available"
    assert bboxes.is_cuda and logits.is_cuda and masks.is_cuda
    assert bboxes.shape[0] == 1 and logits.shape[0] == 1 and masks.shape[0] == 1, "batch=1 only"

    device = bboxes.device
    num_queries, num_classes_total = logits.shape[1], logits.shape[2]
    _, _, mask_h, mask_w = masks.shape

    # Flatten batch dim — these views are contiguous when batch=1 and the
    # tensor came straight from TRT engine outputs, so .contiguous() is a
    # no-op in the common case. Still call it to be defensive; torch skips
    # the kernel launch when the view is already contiguous.
    logits_2d = logits[0] if logits[0].is_contiguous() else logits[0].contiguous()
    bboxes_2d = bboxes[0] if bboxes[0].is_contiguous() else bboxes[0].contiguous()
    masks_3d = masks[0] if masks[0].is_contiguous() else masks[0].contiguous()

    # Compact outputs, pre-sized to max (num_queries). Host reads the counter
    # to know how many slots are populated.
    xyxy = torch.empty((num_queries, 4), dtype=torch.int32, device=device)
    conf = torch.empty((num_queries,), dtype=torch.float32, device=device)
    cls_id = torch.empty((num_queries,), dtype=torch.int32, device=device)
    survivor_idx = torch.empty((num_queries,), dtype=torch.int32, device=device)
    # mask_any is written (not read) by the filter kernel at each survivor's
    # slot -> 0. Mask kernel later atomic_maxes up to 1. No pre-zero needed.
    mask_any = torch.empty((num_queries,), dtype=torch.int32, device=device)
    # Atomic counter — must be zeroed each call since the filter kernel
    # atomic_adds into it.
    counter = torch.zeros((1,), dtype=torch.int32, device=device)

    thr_tensor, per_class = _prepare_threshold(threshold, device, num_classes)

    if class_mapping is not None:
        has_remap = True
        cmap = class_mapping if (
            class_mapping.dtype == torch.int32
            and class_mapping.device == device
            and class_mapping.is_contiguous()
        ) else class_mapping.to(dtype=torch.int32, device=device).contiguous()
    else:
        has_remap = False
        cmap = _EMPTY_INT32.to(device, non_blocking=True)

    inf_w, inf_h = inference_size_wh
    pad_l, pad_t, _, _ = pad_ltrb
    sw, sh = scale_wh
    orig_w, orig_h = orig_size_wh

    BLOCK_C = max(32, _next_pow2(num_classes_total))
    _rfdetr_fullpost_filter_kernel[(num_queries,)](
        logits_2d,
        bboxes_2d,
        thr_tensor,
        cmap,
        xyxy,
        conf,
        cls_id,
        survivor_idx,
        mask_any,
        counter,
        num_queries,
        num_classes_total,
        int(inf_w),
        int(inf_h),
        int(pad_l),
        int(pad_t),
        float(1.0 / sw),
        float(1.0 / sh),
        int(orig_w),
        int(orig_h),
        logits_2d.stride(0),
        bboxes_2d.stride(0),
        PER_CLASS=1 if per_class else 0,
        HAS_REMAPPING=1 if has_remap else 0,
        BLOCK_C=BLOCK_C,
    )

    # Launch the mask kernel with max grid (num_queries). Each program
    # checks counter[0] on GPU and early-exits if its s index is out of
    # range. This lets us skip a CPU-blocking counter.item() between the
    # two kernel launches — both get queued to the stream immediately.
    mask_bin_full = _get_mask_bin_buffer(num_queries, orig_h, orig_w, device)

    BLOCK_H = 16
    BLOCK_W = 16
    grid = (
        num_queries,
        (orig_h + BLOCK_H - 1) // BLOCK_H,
        (orig_w + BLOCK_W - 1) // BLOCK_W,
    )
    _rfdetr_fullpost_mask_kernel_compact[grid](
        masks_3d,
        survivor_idx,
        counter,
        mask_bin_full,
        mask_any,
        int(mask_h),
        int(mask_w),
        int(orig_h),
        int(orig_w),
        float(mask_h / orig_h),
        float(mask_w / orig_w),
        masks_3d.stride(0),
        masks_3d.stride(1),
        mask_bin_full.stride(0),
        mask_bin_full.stride(1),
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )

    # Now read the counter. The sync happens after mask-kernel launch, not
    # between it and the filter, so the two kernels pipeline on the GPU
    # while the host waits.
    n_survivors = int(counter.item())

    if n_survivors == 0:
        return (
            xyxy[:0],
            conf[:0],
            cls_id[:0],
            torch.empty((0, orig_h, orig_w), dtype=torch.uint8, device=device),
            torch.empty((0,), dtype=torch.bool, device=device),
        )

    return (
        xyxy[:n_survivors],
        conf[:n_survivors],
        cls_id[:n_survivors],
        mask_bin_full[:n_survivors],
        mask_any[:n_survivors].bool(),
    )
