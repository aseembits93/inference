"""Fused Triton preprocessing kernel for RF-DETR seg (stretch-to resize).

Replaces: cv2.resize(BGR) -> torch.from_numpy.to(cuda) -> unsqueeze -> permute
-> BGR->RGB fancy index -> /255 -> normalize. Eight+ CUDA launches on a
tiny 312x312 tensor collapse into one Triton launch.
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
    def _rfdetr_stretch_preprocess_kernel(
        src_ptr,
        dst_ptr,
        src_h,
        src_w,
        src_stride_h,
        src_stride_w,
        target_h,
        target_w,
        scale_y,
        scale_x,
        mean_r,
        mean_g,
        mean_b,
        std_r,
        std_g,
        std_b,
        dst_stride_c,
        dst_stride_h,
        BLOCK_H: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        """Stretch-to resize + BGR->RGB + normalize, fused.

        Src: uint8 HWC BGR.
        Dst: fp32 (1, 3, target_h, target_w) CHW RGB, normalized to
        (pixel / 255 - mean) / std.

        Arithmetic order matches the PyTorch reference exactly:
            fp32(pixel) -> bilinear -> /255 -> -mean -> /std
        so fp32 output is bit-equal to
            F.interpolate(t.float()) / 255  -> (x - mean) / std.
        The `inv_std_c_255 + offset` fusion used earlier was off by
        1 ULP in the last place; fp16 engines can round those ULPs to
        different values and flip downstream top-1 decisions.
        """
        pid_y = tl.program_id(0)
        pid_x = tl.program_id(1)

        offs_y = pid_y * BLOCK_H + tl.arange(0, BLOCK_H)
        offs_x = pid_x * BLOCK_W + tl.arange(0, BLOCK_W)
        mask_y = offs_y < target_h
        mask_x = offs_x < target_w
        mask = mask_y[:, None] & mask_x[None, :]

        # Pixel-center bilinear sampling (align_corners=False).
        src_y_f = (offs_y.to(tl.float32) + 0.5) * scale_y - 0.5
        src_x_f = (offs_x.to(tl.float32) + 0.5) * scale_x - 0.5

        src_y_f_2d = src_y_f[:, None]
        src_x_f_2d = src_x_f[None, :]

        y0 = tl.floor(src_y_f_2d).to(tl.int32)
        x0 = tl.floor(src_x_f_2d).to(tl.int32)
        y1 = y0 + 1
        x1 = x0 + 1

        dy = src_y_f_2d - y0.to(tl.float32)
        dx = src_x_f_2d - x0.to(tl.float32)

        y0c = tl.maximum(tl.minimum(y0, src_h - 1), 0)
        y1c = tl.maximum(tl.minimum(y1, src_h - 1), 0)
        x0c = tl.maximum(tl.minimum(x0, src_w - 1), 0)
        x1c = tl.maximum(tl.minimum(x1, src_w - 1), 0)

        base_00 = y0c * src_stride_h + x0c * src_stride_w
        base_01 = y0c * src_stride_h + x1c * src_stride_w
        base_10 = y1c * src_stride_h + x0c * src_stride_w
        base_11 = y1c * src_stride_h + x1c * src_stride_w

        w_tl = (1.0 - dy) * (1.0 - dx)
        w_tr = (1.0 - dy) * dx
        w_bl = dy * (1.0 - dx)
        w_br = dy * dx

        # BGR source: channel 0=B, 1=G, 2=R. Output order is RGB.
        p00_b = tl.load(src_ptr + base_00 + 0, mask=mask, other=0).to(tl.float32)
        p01_b = tl.load(src_ptr + base_01 + 0, mask=mask, other=0).to(tl.float32)
        p10_b = tl.load(src_ptr + base_10 + 0, mask=mask, other=0).to(tl.float32)
        p11_b = tl.load(src_ptr + base_11 + 0, mask=mask, other=0).to(tl.float32)
        b_val = p00_b * w_tl + p01_b * w_tr + p10_b * w_bl + p11_b * w_br

        p00_g = tl.load(src_ptr + base_00 + 1, mask=mask, other=0).to(tl.float32)
        p01_g = tl.load(src_ptr + base_01 + 1, mask=mask, other=0).to(tl.float32)
        p10_g = tl.load(src_ptr + base_10 + 1, mask=mask, other=0).to(tl.float32)
        p11_g = tl.load(src_ptr + base_11 + 1, mask=mask, other=0).to(tl.float32)
        g_val = p00_g * w_tl + p01_g * w_tr + p10_g * w_bl + p11_g * w_br

        p00_r = tl.load(src_ptr + base_00 + 2, mask=mask, other=0).to(tl.float32)
        p01_r = tl.load(src_ptr + base_01 + 2, mask=mask, other=0).to(tl.float32)
        p10_r = tl.load(src_ptr + base_10 + 2, mask=mask, other=0).to(tl.float32)
        p11_r = tl.load(src_ptr + base_11 + 2, mask=mask, other=0).to(tl.float32)
        r_val = p00_r * w_tl + p01_r * w_tr + p10_r * w_bl + p11_r * w_br

        # Match PyTorch's `(x / 255 - mean) / std` op order exactly.
        inv_255 = 1.0 / 255.0
        r_out = (r_val * inv_255 - mean_r) / std_r
        g_out = (g_val * inv_255 - mean_g) / std_g
        b_out = (b_val * inv_255 - mean_b) / std_b

        out_row_offsets = offs_y[:, None] * dst_stride_h + offs_x[None, :]
        tl.store(dst_ptr + 0 * dst_stride_c + out_row_offsets, r_out, mask=mask)
        tl.store(dst_ptr + 1 * dst_stride_c + out_row_offsets, g_out, mask=mask)
        tl.store(dst_ptr + 2 * dst_stride_c + out_row_offsets, b_out, mask=mask)


def triton_preprocess_rfdetr_stretch(
    src: torch.Tensor,
    target_h: int,
    target_w: int,
    means: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    stds: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused preprocess: uint8 HWC BGR -> fp32 (1,3,target_h,target_w) CHW RGB
    normalized (stretch-to resize, no padding)."""
    if not TRITON_AVAILABLE:
        raise RuntimeError("triton is not installed")
    if not src.is_cuda:
        raise ValueError(f"expected CUDA tensor, got device={src.device}")
    if src.dtype != torch.uint8:
        raise ValueError(f"expected uint8, got {src.dtype}")
    if src.ndim != 3 or src.shape[2] != 3:
        raise ValueError(f"expected HWC 3-channel, got shape={tuple(src.shape)}")

    src = src.contiguous()
    src_h, src_w = int(src.shape[0]), int(src.shape[1])
    src_stride_h = int(src.stride(0))
    src_stride_w = int(src.stride(1))

    scale_y = src_h / target_h
    scale_x = src_w / target_w

    if out is None:
        out = torch.empty(
            (1, 3, target_h, target_w), dtype=torch.float32, device=src.device
        )
    else:
        if tuple(out.shape) != (1, 3, target_h, target_w):
            raise ValueError(
                f"out has shape {tuple(out.shape)}, expected (1, 3, {target_h}, {target_w})"
            )
        if out.dtype != torch.float32 or not out.is_cuda:
            raise ValueError("out must be fp32 CUDA tensor")

    dst_stride_c = target_h * target_w
    dst_stride_h = target_w

    BLOCK_H = 16
    BLOCK_W = 16
    grid = (
        (target_h + BLOCK_H - 1) // BLOCK_H,
        (target_w + BLOCK_W - 1) // BLOCK_W,
    )
    _rfdetr_stretch_preprocess_kernel[grid](
        src,
        out,
        src_h,
        src_w,
        src_stride_h,
        src_stride_w,
        target_h,
        target_w,
        float(scale_y),
        float(scale_x),
        float(means[0]),
        float(means[1]),
        float(means[2]),
        float(stds[0]),
        float(stds[1]),
        float(stds[2]),
        dst_stride_c,
        dst_stride_h,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )
    return out
