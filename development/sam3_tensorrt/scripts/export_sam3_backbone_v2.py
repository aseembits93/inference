#!/usr/bin/env python3
"""Export SAM3 vision backbone (v2): RoPE via rotate_half formulation.

The previous export used a pair-interleaved rotation (xr, xi = x[..., 0::2],
x[..., 1::2]) implemented via reshape-to-pairs + stack + flatten. That
sequence produced correct outputs in PyTorch but TRT's optimizer
miscomputes the rotation in FP16/BF16, yielding magnitudes ~2.5× off.

This version replaces the rotation with a single Cat, avoiding the
reshape-to-pairs trick:

  Given freqs as interleaved `(c0 s0 c1 s1 ...)` buffers `freqs_cos`,
  `freqs_sin` (both shape [L, head_dim/2]), and an input `x` of shape
  [..., L, head_dim] laid out as `(r0 i0 r1 i1 ...)`, the rotated output
  at pair k is:

      out[2k]   = x[2k] * c[k] - x[2k+1] * s[k]
      out[2k+1] = x[2k] * s[k] + x[2k+1] * c[k]

  Build a cos/sin vector of length head_dim by repeat_interleave(2):
      cos_full[2k] = cos_full[2k+1] = c[k]
      sin_full[2k] = sin_full[2k+1] = s[k]

  Then:
      x_rot[2k]   = -x[2k+1]  (i.e. swap & negate: rotate by 90°)
      x_rot[2k+1] =  x[2k]

      out = x * cos_full + x_rot * sin_full

  This is the same math expressed with only elementwise ops along the last
  dim (no reshape to (..., head_dim//2, 2) nor stack + flatten). TRT can
  more easily keep this in FP32 without losing semantics, and the entire
  rotation compiles to a couple of FP16-safe elementwise kernels.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if "ROBOFLOW_API_KEY" not in os.environ:
    raise SystemExit("Set ROBOFLOW_API_KEY env var before running this script.")


import torch
import torch.nn as nn

EXPORT_DIR = Path("./sam3_onnx_exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_SIZE = 1008
OPSET = 17


def _apply_rotary_enc_rotate_half(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos_full: torch.Tensor,
    freqs_sin_full: torch.Tensor,
    repeat_freqs_k: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate-half formulation. Elementwise along last dim only.

    cos_full / sin_full have the same shape as q/k's last dim (head_dim).
    The rotation is:  out = q * cos_full + rotate_pairs(q) * sin_full
    where rotate_pairs interleaves (-q[1::2], q[0::2]) back along the last
    dim: equivalent to multiplying by i in complex space.
    """
    def rotate_pairs(x):
        # x: [..., head_dim]  (head_dim is even). Split into even/odd indices
        # along the last dim and recombine as (-odd, even) interleaved.
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        # Interleave (-x_odd, x_even) along the last dim. Do this with a
        # stack-along-last + flatten. Because the two halves have the same
        # shape, this stack-flatten behaves predictably in ONNX.
        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)

    # Broadcast freqs to match q/k ndim (add leading singleton dims)
    while freqs_cos_full.ndim < xq.ndim:
        freqs_cos_full = freqs_cos_full.unsqueeze(0)
        freqs_sin_full = freqs_sin_full.unsqueeze(0)

    xq_out = xq * freqs_cos_full + rotate_pairs(xq) * freqs_sin_full

    if xk.shape[-2] == 0:
        return xq_out, xk
    if repeat_freqs_k:
        r = xk.shape[-2] // xq.shape[-2]
        freqs_cos_full = freqs_cos_full.repeat_interleave(r, dim=-2)
        freqs_sin_full = freqs_sin_full.repeat_interleave(r, dim=-2)
    xk_out = xk * freqs_cos_full + rotate_pairs(xk) * freqs_sin_full
    return xq_out, xk_out


def patch_vitdet_rope_v2(model: nn.Module) -> int:
    """Replace complex freqs_cis with (cos_full, sin_full) interleaved buffers."""
    count = 0
    for _, module in model.named_modules():
        fc = getattr(module, "freqs_cis", None)
        if isinstance(fc, torch.Tensor) and torch.is_complex(fc):
            # freqs_cis has shape [..., head_dim//2]; interleave cos/sin so
            # the last dim matches head_dim.
            c = fc.real  # [..., head_dim/2]
            s = fc.imag
            cos_full = torch.repeat_interleave(c, 2, dim=-1).contiguous()
            sin_full = torch.repeat_interleave(s, 2, dim=-1).contiguous()
            del module._buffers["freqs_cis"]
            module.register_buffer("freqs_cos_full", cos_full)
            module.register_buffer("freqs_sin_full", sin_full)

            def _apply_rope_v2(self, q, k, *args, **kwargs):  # noqa: D401
                if not getattr(self, "use_rope", True):
                    return q, k
                return _apply_rotary_enc_rotate_half(
                    q, k, self.freqs_cos_full, self.freqs_sin_full
                )

            module._apply_rope = _apply_rope_v2.__get__(module, module.__class__)
            count += 1
    return count


class Sam3VisionWrapper(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, samples: torch.Tensor):
        out = self.backbone.forward_image(samples)
        return (
            out["vision_features"],
            out["vision_pos_enc"][0],
            out["vision_pos_enc"][1],
            out["vision_pos_enc"][2],
            out["backbone_fpn"][0],
            out["backbone_fpn"][1],
            out["backbone_fpn"][2],
        )


def main() -> int:
    from inference.models.sam3.segment_anything3 import SegmentAnything3

    print("Loading SAM3 ...", flush=True)
    rf = SegmentAnything3(
        model_id="sam3/sam3_final", api_key=os.environ["ROBOFLOW_API_KEY"],
    )
    backbone = rf.model.backbone.eval()

    # Build a fresh reference output BEFORE patching
    dummy_ref = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device="cuda", dtype=torch.float32)
    with torch.inference_mode():
        ref_out = backbone.forward_image(dummy_ref)
    ref_vf = ref_out["vision_features"].clone()
    ref_fpn1 = ref_out["backbone_fpn"][1].clone()

    n = patch_vitdet_rope_v2(backbone)
    print(f"Patched {n} RoPE modules with rotate_half variant")

    # Parity check vs original PyTorch
    with torch.inference_mode():
        pat_out = backbone.forward_image(dummy_ref)
    vf = pat_out["vision_features"]
    fpn1 = pat_out["backbone_fpn"][1]
    def cos(a, b):
        a = a.float().flatten(); b = b.float().flatten()
        return (a @ b / (a.norm() * b.norm() + 1e-12)).item()
    print(f"Parity: vision_features cos = {cos(ref_vf, vf):.6f}")
    print(f"Parity: backbone_fpn[1] cos = {cos(ref_fpn1, fpn1):.6f}")
    print(f"  ref range: {ref_vf.float().min():.3f} .. {ref_vf.float().max():.3f}")
    print(f"  pat range: {vf.float().min():.3f} .. {vf.float().max():.3f}")

    wrap = Sam3VisionWrapper(backbone).eval()

    onnx_path = EXPORT_DIR / "sam3_vision_backbone_v2.onnx"
    print(f"\nExporting to {onnx_path} (opset {OPSET}) ...")
    output_names = [
        "vision_features",
        "vision_pos_enc_0", "vision_pos_enc_1", "vision_pos_enc_2",
        "backbone_fpn_0", "backbone_fpn_1", "backbone_fpn_2",
    ]
    with torch.inference_mode():
        torch.onnx.export(
            wrap,
            (dummy_ref,),
            str(onnx_path),
            input_names=["samples"],
            output_names=output_names,
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
            verbose=False,
        )
    print(f"Exported: {onnx_path}  ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
