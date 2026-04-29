"""Fused Triton preprocessing kernel for RF-DETR.

Combines letterbox resize (bilinear) + BGR->RGB + normalize + CHW/NCHW
layout in a single CUDA kernel launch.

Reference (torch/numpy) pipeline this replaces:
    from_numpy(uint8 HWC BGR).cuda()
        -> permute(HWC -> CHW)
        -> .contiguous().float() / 255
        -> subtract means / divide stds
        -> interpolate (bilinear resize keeping aspect ratio)
        -> pad with grey 114 (letterbox)
        -> BGR -> RGB channel swap
        -> unsqueeze(0)

Each of those torch ops launches at least one CUDA kernel; fusing them
eliminates that overhead for small images where per-launch cost dominates.
"""

from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dep
    triton = None
    tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _rfdetr_preprocess_kernel(
        src_ptr,
        dst_ptr,
        src_h,
        src_w,
        src_stride_h,
        src_stride_w,
        target_h,
        target_w,
        scale,
        pad_x,
        pad_y,
        mean_r,
        mean_g,
        mean_b,
        std_r,
        std_g,
        std_b,
        pad_r_norm,
        pad_g_norm,
        pad_b_norm,
        dst_stride_c,
        dst_stride_h,
        BLOCK_H: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        """One kernel per (tile_y, tile_x) over the target image; writes all
        3 channels in a single pass.

        Src layout: uint8 HWC BGR, strides are (src_stride_h, src_stride_w, 1).
        Dst layout: fp32 (1, 3, target_h, target_w) contiguous CHW RGB.
        """
        pid_y = tl.program_id(0)
        pid_x = tl.program_id(1)

        offs_y = pid_y * BLOCK_H + tl.arange(0, BLOCK_H)
        offs_x = pid_x * BLOCK_W + tl.arange(0, BLOCK_W)
        mask_y = offs_y < target_h
        mask_x = offs_x < target_w
        mask = mask_y[:, None] & mask_x[None, :]

        # Inverse letterbox: map output pixel -> source pixel coordinate.
        src_y_f = (offs_y.to(tl.float32) + 0.5 - pad_y) / scale - 0.5
        src_x_f = (offs_x.to(tl.float32) + 0.5 - pad_x) / scale - 0.5

        src_y_f_2d = src_y_f[:, None]
        src_x_f_2d = src_x_f[None, :]

        y0 = tl.floor(src_y_f_2d).to(tl.int32)
        x0 = tl.floor(src_x_f_2d).to(tl.int32)
        y1 = y0 + 1
        x1 = x0 + 1

        dy = src_y_f_2d - y0.to(tl.float32)
        dx = src_x_f_2d - x0.to(tl.float32)

        # Clamp to source bounds for the gather; we mask the fully-out-of-bounds
        # tiles at the end (pad region).
        y0c = tl.maximum(tl.minimum(y0, src_h - 1), 0)
        y1c = tl.maximum(tl.minimum(y1, src_h - 1), 0)
        x0c = tl.maximum(tl.minimum(x0, src_w - 1), 0)
        x1c = tl.maximum(tl.minimum(x1, src_w - 1), 0)

        # Output-pixel-in-pad-region iff the *center* maps outside the resized
        # image footprint. Using src_y_f_2d (post-center shift) is fine since
        # we compare against [-0.5, src_h - 0.5].
        in_bounds = (
            (src_y_f_2d >= -0.5)
            & (src_y_f_2d <= src_h.to(tl.float32) - 0.5)
            & (src_x_f_2d >= -0.5)
            & (src_x_f_2d <= src_w.to(tl.float32) - 0.5)
        )

        # Gather 4 corners for all three channels. Source is HWC BGR
        # (channel 0 = B, channel 1 = G, channel 2 = R); output CHW RGB.
        # Triton doesn't support nested function defs, so we inline the
        # gather for each channel.
        base_00 = y0c * src_stride_h + x0c * src_stride_w
        base_01 = y0c * src_stride_h + x1c * src_stride_w
        base_10 = y1c * src_stride_h + x0c * src_stride_w
        base_11 = y1c * src_stride_h + x1c * src_stride_w

        w_tl = (1.0 - dy) * (1.0 - dx)
        w_tr = (1.0 - dy) * dx
        w_bl = dy * (1.0 - dx)
        w_br = dy * dx

        # Channel 0 (B)
        p00_b = tl.load(src_ptr + base_00 + 0, mask=mask, other=0).to(tl.float32)
        p01_b = tl.load(src_ptr + base_01 + 0, mask=mask, other=0).to(tl.float32)
        p10_b = tl.load(src_ptr + base_10 + 0, mask=mask, other=0).to(tl.float32)
        p11_b = tl.load(src_ptr + base_11 + 0, mask=mask, other=0).to(tl.float32)
        b_val = p00_b * w_tl + p01_b * w_tr + p10_b * w_bl + p11_b * w_br

        # Channel 1 (G)
        p00_g = tl.load(src_ptr + base_00 + 1, mask=mask, other=0).to(tl.float32)
        p01_g = tl.load(src_ptr + base_01 + 1, mask=mask, other=0).to(tl.float32)
        p10_g = tl.load(src_ptr + base_10 + 1, mask=mask, other=0).to(tl.float32)
        p11_g = tl.load(src_ptr + base_11 + 1, mask=mask, other=0).to(tl.float32)
        g_val = p00_g * w_tl + p01_g * w_tr + p10_g * w_bl + p11_g * w_br

        # Channel 2 (R)
        p00_r = tl.load(src_ptr + base_00 + 2, mask=mask, other=0).to(tl.float32)
        p01_r = tl.load(src_ptr + base_01 + 2, mask=mask, other=0).to(tl.float32)
        p10_r = tl.load(src_ptr + base_10 + 2, mask=mask, other=0).to(tl.float32)
        p11_r = tl.load(src_ptr + base_11 + 2, mask=mask, other=0).to(tl.float32)
        r_val = p00_r * w_tl + p01_r * w_tr + p10_r * w_bl + p11_r * w_br

        r_norm = (r_val / 255.0 - mean_r) / std_r
        g_norm = (g_val / 255.0 - mean_g) / std_g
        b_norm = (b_val / 255.0 - mean_b) / std_b

        r_out = tl.where(in_bounds, r_norm, pad_r_norm)
        g_out = tl.where(in_bounds, g_norm, pad_g_norm)
        b_out = tl.where(in_bounds, b_norm, pad_b_norm)

        out_row_offsets = offs_y[:, None] * dst_stride_h + offs_x[None, :]
        tl.store(dst_ptr + 0 * dst_stride_c + out_row_offsets, r_out, mask=mask)
        tl.store(dst_ptr + 1 * dst_stride_c + out_row_offsets, g_out, mask=mask)
        tl.store(dst_ptr + 2 * dst_stride_c + out_row_offsets, b_out, mask=mask)


def triton_preprocess_rfdetr(
    src: torch.Tensor,
    target_h: int,
    target_w: int,
    means: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    stds: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    pad_color: int = 114,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused preprocess: uint8 HWC BGR -> fp32 (1,3,H,W) CHW RGB normalized
    with letterbox.

    Args:
        src: uint8 tensor of shape (H, W, 3) on CUDA, BGR channel order
            (as produced by cv2.imread / VideoFrame.image).
        target_h, target_w: output spatial dims.
        means, stds: imagenet normalization in RGB order.
        pad_color: uint8 value used in each channel for letterbox padding
            (applied *before* normalization). Default 114 matches RF-DETR
            "Fit (grey edges) in".
        out: optional preallocated fp32 tensor of shape (1, 3, target_h,
            target_w) to write into.

    Returns:
        fp32 tensor of shape (1, 3, target_h, target_w) on the same device
        as src.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "triton is not installed; cannot run triton_preprocess_rfdetr. "
            "Install the optional 'triton-preproc' extra."
        )
    if not src.is_cuda:
        raise ValueError(
            f"triton_preprocess_rfdetr requires a CUDA tensor, got device={src.device}"
        )
    if src.dtype != torch.uint8:
        raise ValueError(
            f"triton_preprocess_rfdetr expects uint8 input, got dtype={src.dtype}"
        )
    if src.ndim != 3 or src.shape[2] != 3:
        raise ValueError(
            f"triton_preprocess_rfdetr expects HWC 3-channel input, got shape={tuple(src.shape)}"
        )

    src = src.contiguous()
    src_h, src_w = int(src.shape[0]), int(src.shape[1])
    # HWC strides: contiguous => (W*3, 3, 1). We pass (row_stride, col_stride)
    # in elements (uint8).
    src_stride_h = int(src.stride(0))
    src_stride_w = int(src.stride(1))

    scale = min(target_h / src_h, target_w / src_w)
    scaled_h = int(src_h * scale)
    scaled_w = int(src_w * scale)
    pad_x = (target_w - scaled_w) / 2.0
    pad_y = (target_h - scaled_h) / 2.0

    if out is None:
        out = torch.empty(
            (1, 3, target_h, target_w), dtype=torch.float32, device=src.device
        )
    else:
        if tuple(out.shape) != (1, 3, target_h, target_w):
            raise ValueError(
                f"out has shape {tuple(out.shape)}, expected (1, 3, {target_h}, {target_w})"
            )
        if out.dtype != torch.float32:
            raise ValueError(f"out must be float32, got {out.dtype}")
        if not out.is_cuda or out.device != src.device:
            raise ValueError("out must be a CUDA tensor on the same device as src")

    # (1,3,H,W) contiguous => per-channel plane stride is H*W, row stride is W.
    dst_stride_c = target_h * target_w
    dst_stride_h = target_w

    pad_norm_r = (pad_color / 255.0 - means[0]) / stds[0]
    pad_norm_g = (pad_color / 255.0 - means[1]) / stds[1]
    pad_norm_b = (pad_color / 255.0 - means[2]) / stds[2]

    BLOCK_H = 16
    BLOCK_W = 16
    grid = (
        (target_h + BLOCK_H - 1) // BLOCK_H,
        (target_w + BLOCK_W - 1) // BLOCK_W,
    )
    _rfdetr_preprocess_kernel[grid](
        src,
        out,
        src_h,
        src_w,
        src_stride_h,
        src_stride_w,
        target_h,
        target_w,
        float(scale),
        float(pad_x),
        float(pad_y),
        float(means[0]),
        float(means[1]),
        float(means[2]),
        float(stds[0]),
        float(stds[1]),
        float(stds[2]),
        float(pad_norm_r),
        float(pad_norm_g),
        float(pad_norm_b),
        dst_stride_c,
        dst_stride_h,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )
    return out
