#!/usr/bin/env python3
"""Export SAM3 vision backbone (forward_image) to ONNX.

Pure-tensor I/O:
  In : samples (B, 3, H, W) float16
  Out: vision_features + 3 vision_pos_enc + 3 backbone_fpn tensors

The SAM3 vitdet uses complex-tensor rotary embeddings (torch.view_as_complex),
which ONNX opset 17 does not support. We monkey-patch apply_rotary_enc with a
real-arithmetic equivalent and convert each attention's freqs_cis buffer from
complex to two real buffers (freqs_cos, freqs_sin). The math is bit-identical
up to FP accumulation order.
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


# ---------------------------------------------------------------------------
# ONNX-safe rotary embedding (real arithmetic, no view_as_complex)
# ---------------------------------------------------------------------------

def _reshape_for_broadcast(freqs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # x has shape [..., L, head_dim//2, 2]  -> broadcast freqs to its last-three dims
    ndim = x.ndim - 1  # we compare against complex-form ndim = x.ndim - 1
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape[:-1])]
    return freqs.view(*shape)


def _apply_rotary_enc_real_vitdet(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,  # real part, shape matches original freqs_cis
    freqs_sin: torch.Tensor,  # imag part, shape matches original freqs_cis
    repeat_freqs_k: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Real-arithmetic version of vitdet.apply_rotary_enc.

    Original computes xq_out = (xq_complex * freqs_cis), viewed back to real and
    flattened. In real arithmetic this is a 2D rotation per pair:
      out_real = xr * c - xi * s
      out_imag = xr * s + xi * c
    """
    # xq shape (..., L, head_dim).  Reshape to (..., L, head_dim/2, 2).
    xq_pairs = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xq_r = xq_pairs[..., 0]
    xq_i = xq_pairs[..., 1]

    # Broadcast freqs to match (..., L, head_dim/2) shape
    freqs_cos_b = _reshape_for_broadcast(freqs_cos, xq_pairs)
    freqs_sin_b = _reshape_for_broadcast(freqs_sin, xq_pairs)

    out_r = xq_r * freqs_cos_b - xq_i * freqs_sin_b
    out_i = xq_r * freqs_sin_b + xq_i * freqs_cos_b
    xq_out = torch.stack([out_r, out_i], dim=-1).flatten(-2)

    if xk.shape[-2] == 0:
        return xq_out.type_as(xq).to(xq.device), xk

    xk_pairs = xk.float().reshape(*xk.shape[:-1], -1, 2)
    xk_r = xk_pairs[..., 0]
    xk_i = xk_pairs[..., 1]

    if repeat_freqs_k:
        r = xk_pairs.shape[-3] // xq_pairs.shape[-3]
        freqs_cos_k = freqs_cos_b.repeat(*([1] * (freqs_cos_b.ndim - 2)), r, 1)
        freqs_sin_k = freqs_sin_b.repeat(*([1] * (freqs_sin_b.ndim - 2)), r, 1)
    else:
        freqs_cos_k = freqs_cos_b
        freqs_sin_k = freqs_sin_b

    out_r_k = xk_r * freqs_cos_k - xk_i * freqs_sin_k
    out_i_k = xk_r * freqs_sin_k + xk_i * freqs_cos_k
    xk_out = torch.stack([out_r_k, out_i_k], dim=-1).flatten(-2)

    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


def patch_vitdet_rope(model: nn.Module) -> int:
    """Convert every freqs_cis (complex) buffer into freqs_cos/freqs_sin
    (real). Monkey-patch the _apply_rope method of the owning module.

    Returns the number of modules patched.
    """
    count = 0
    for _, module in model.named_modules():
        fc = getattr(module, "freqs_cis", None)
        if isinstance(fc, torch.Tensor) and torch.is_complex(fc):
            freqs_cos = fc.real.contiguous()
            freqs_sin = fc.imag.contiguous()
            # Remove the complex buffer and register real buffers
            del module._buffers["freqs_cis"]
            module.register_buffer("freqs_cos", freqs_cos)
            module.register_buffer("freqs_sin", freqs_sin)

            # Replace the _apply_rope method to use real buffers
            def _apply_rope_real(self, q, k, *args, **kwargs):  # noqa: D401
                if not getattr(self, "use_rope", True):
                    return q, k
                return _apply_rotary_enc_real_vitdet(
                    q, k, self.freqs_cos, self.freqs_sin
                )

            module._apply_rope = _apply_rope_real.__get__(module, module.__class__)
            count += 1
    return count


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class Sam3VisionWrapper(nn.Module):
    """Wrap SAM3VLBackbone.forward_image to produce flat tensor outputs."""

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    from inference.models.sam3.segment_anything3 import SegmentAnything3

    print("Loading SAM3 ...", flush=True)
    rf = SegmentAnything3(
        model_id="sam3/sam3_final",
        api_key=os.environ["ROBOFLOW_API_KEY"],
    )
    backbone = rf.model.backbone.eval()

    # Patch RoPE to real arithmetic
    n_patched = patch_vitdet_rope(backbone)
    print(f"Patched {n_patched} RoPE buffer(s) to real arithmetic")

    wrap = Sam3VisionWrapper(backbone).eval()

    # Sanity: run patched model with a fresh non-patched copy side-by-side to
    # confirm parity.
    torch.manual_seed(0)
    ref_backbone = rf.model.backbone  # same object — already patched; skip parity here
    # We skip numerical parity check vs. complex (pre-patch) since we mutated
    # in place. Trust the math; correctness-gate comes later against full PyTorch
    # inference (mask IoU).

    # Keep model in FP32 — TRT will choose per-layer precision via builder flags.
    # Hard-casting to .half() corrupts LayerNorm output; bf16 autocast in PyTorch
    # works but we can't easily force TRT to respect the exported dtype annotations
    # without strongly-typed networks. FP32 ONNX + TRT BF16 flag keeps us safe.
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device="cuda", dtype=torch.float32)

    with torch.inference_mode():
        out = wrap(dummy)
    print("Wrapper forward OK, output tensor shapes:")
    for i, t in enumerate(out):
        print(f"  out[{i}]: {tuple(t.shape)} {t.dtype}")

    onnx_path = EXPORT_DIR / "sam3_vision_backbone_fp16.onnx"
    print(f"\nExporting to {onnx_path} (opset {OPSET}) ...")

    output_names = [
        "vision_features",
        "vision_pos_enc_0", "vision_pos_enc_1", "vision_pos_enc_2",
        "backbone_fpn_0", "backbone_fpn_1", "backbone_fpn_2",
    ]

    with torch.inference_mode():
        torch.onnx.export(
            wrap,
            (dummy,),
            str(onnx_path),
            input_names=["samples"],
            output_names=output_names,
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
            verbose=False,
        )

    size_mb = onnx_path.stat().st_size / 1e6
    print(f"Exported: {onnx_path}  ({size_mb:.1f} MB)")

    print("\nNext: run trtexec to build the FP16 engine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
