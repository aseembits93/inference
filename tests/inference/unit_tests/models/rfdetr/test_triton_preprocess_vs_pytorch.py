"""Parity tests: triton_preprocess_rfdetr (inference.models.rfdetr.triton_preprocess)
vs the USE_PYTORCH_FOR_PREPROCESSING code path in inference/models/rfdetr/rfdetr.py.

The Triton fast path in ``_try_triton_preprocess`` is a replacement for the
torch-preprocessing chain gated by USE_PYTORCH_FOR_PREPROCESSING=True. These
tests nail down the relationship between the two paths.

Scope and caveats
-----------------

1. The ``triton_preprocess_rfdetr`` kernel is LETTERBOX-ONLY — it always
   resizes while preserving aspect ratio and pads the short axis with
   ``pad_color`` (default 114 for RF-DETR's "Fit (grey edges) in"). The
   production fast-path gate in rfdetr.py only calls this kernel when
   ``resize_method in ("Fit (grey edges) in", "Stretch to")``; if the user
   configured Stretch but the image aspect differs from target, the kernel
   still letterboxes — so the live Stretch call is only exercised when the
   source aspect already matches the target.

2. The reference pytorch path (rfdetr.py:189-297, USE_PYTORCH_FOR_PREPROCESSING
   branch) has three notable divergences from the kernel:
     (a) It normalizes BGR-ordered channels using RGB-ordered means/stds,
         THEN swaps BGR->RGB. Net effect: the R channel of the output ends
         up normalized with B's mean/std, and vice versa. Only G is
         self-consistent. See ``test_documented_pytorch_path_channel_mixup``.
     (b) It pads the letterbox region with raw 114.0 in the already-normalized
         tensor, instead of using a per-channel normalized grey. See
         ``test_documented_pytorch_path_pad_region``.
     (c) When (target_dim - scaled_dim) is odd, the pytorch letterbox floor-
         divides the pad (top = //2, bottom = remainder), so the content is
         shifted by half a pixel relative to the kernel's half-pixel-aware
         centering (which uses pad_y = (target_h - scaled_h) / 2.0 as a
         float in the inverse bilinear). Content tests therefore use source
         shapes that yield an even pad; half-pixel cases are covered by
         ``test_letterbox_half_pixel_case_diverges_predictably``.

   The Triton kernel does none of (a), (b), (c) — it applies the correct
   BGR->RGB swap before normalization, pads with the normalized per-channel
   grey, and keeps pad_y as a float so the content is sub-pixel-centered.
   We treat the pytorch quirks as bugs-in-production and compare the kernel
   against a "what the pytorch path was trying to express" reference.

Runs only on CUDA + triton hosts.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():  # pragma: no cover - host-dependent
    pytest.skip("CUDA not available", allow_module_level=True)

import torch.nn.functional as F  # noqa: E402

from inference.models.rfdetr.triton_preprocess import (  # noqa: E402
    triton_preprocess_rfdetr,
)

_MEANS = (0.485, 0.456, 0.406)
_STDS = (0.229, 0.224, 0.225)
_PAD_COLOR = 114


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


def _corrected_pytorch_letterbox(
    frame_bgr_hwc_uint8: np.ndarray,
    target_h: int,
    target_w: int,
    pad_color: int,
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """What the USE_PYTORCH_FOR_PREPROCESSING letterbox path *should* produce
    if it applied BGR->RGB before normalization and normalized the pad color
    per-channel. Returns (fp32 (1,3,H,W) cuda tensor, (top,bottom,left,right)).

    Equivalent to the Triton kernel's output on the content region, and to
    ``(pad_color/255 - mean)/std`` per RGB channel on the pad region.
    """
    src_h, src_w = frame_bgr_hwc_uint8.shape[:2]
    # Upload and convert to float; match the torch preproc dtype/layout.
    tensor = torch.from_numpy(np.ascontiguousarray(frame_bgr_hwc_uint8)).cuda()
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).contiguous().float()  # (1,3,H,W) BGR
    # BGR -> RGB BEFORE normalize so means/stds apply to the correct channels.
    tensor = tensor[:, [2, 1, 0], :, :]
    tensor = tensor / 255.0
    means = torch.tensor(_MEANS, device=tensor.device).view(3, 1, 1)
    stds = torch.tensor(_STDS, device=tensor.device).view(3, 1, 1)
    tensor = (tensor - means) / stds

    # resize_image_keeping_aspect_ratio — preserve aspect, fit into target.
    img_ratio = src_w / src_h
    desired_ratio = target_w / target_h
    if img_ratio >= desired_ratio:
        new_width = target_w
        new_height = int(target_w / img_ratio)
    else:
        new_height = target_h
        new_width = int(target_h * img_ratio)
    tensor = F.interpolate(tensor, size=(new_height, new_width), mode="bilinear")

    top = (target_h - new_height) // 2
    bottom = target_h - new_height - top
    left = (target_w - new_width) // 2
    right = target_w - new_width - left

    # Pad per-channel with the normalized grey value. torch.nn.functional.pad
    # only accepts a scalar, so pad with zero and overwrite.
    padded = torch.empty(
        (1, 3, target_h, target_w), dtype=torch.float32, device=tensor.device,
    )
    for c in range(3):
        padded[0, c].fill_((pad_color / 255.0 - _MEANS[c]) / _STDS[c])
    padded[:, :, top : top + new_height, left : left + new_width] = tensor
    return padded, (top, bottom, left, right)


def _production_pytorch_letterbox(
    frame_bgr_hwc_uint8: np.ndarray,
    target_h: int,
    target_w: int,
    pad_color: int,
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """Bit-exact reproduction of rfdetr.py:189-297 under
    USE_PYTORCH_FOR_PREPROCESSING=True + resize_method='Fit (grey edges) in'.

    This is the buggy reference — we use it only in tests that document the
    bug, not for parity against the kernel.
    """
    src_h, src_w = frame_bgr_hwc_uint8.shape[:2]
    tensor = torch.from_numpy(np.ascontiguousarray(frame_bgr_hwc_uint8)).cuda()
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).contiguous().float()
    tensor = tensor / 255.0
    means = torch.tensor(_MEANS, device=tensor.device).view(3, 1, 1)
    stds = torch.tensor(_STDS, device=tensor.device).view(3, 1, 1)
    tensor = (tensor - means) / stds

    img_ratio = src_w / src_h
    desired_ratio = target_w / target_h
    if img_ratio >= desired_ratio:
        new_width = target_w
        new_height = int(target_w / img_ratio)
    else:
        new_height = target_h
        new_width = int(target_h * img_ratio)
    tensor = F.interpolate(tensor, size=(new_height, new_width), mode="bilinear")

    top = (target_h - new_height) // 2
    bottom = target_h - new_height - top
    left = (target_w - new_width) // 2
    right = target_w - new_width - left
    # F.pad(..., value=pad_color) — single scalar applied across all channels
    # in the already-normalized tensor.
    tensor = F.pad(tensor, (left, right, top, bottom), "constant", pad_color)
    tensor = tensor[:, [2, 1, 0], :, :]
    return tensor, (top, bottom, left, right)


# ---------------------------------------------------------------------------
# Parity vs the CORRECTED pytorch letterbox path — content region.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    # Source shapes chosen so (target_dim - scaled_dim) is even on both axes —
    # keeps the half-pixel-centering divergence (quirk (c) above) out of the
    # content-region comparison. See _find_integer_pad_sizes helper above for
    # how to discover these.
    "src_h, src_w, target_h, target_w",
    [
        (432, 864, 432, 432),    # landscape, pad_y=108 integer
        (432, 648, 432, 432),    # landscape, pad_y=72
        (864, 432, 432, 432),    # portrait, pad_x=108
        (480, 360, 432, 432),    # portrait, pad_x=54
        (432, 432, 432, 432),    # square, no pad
    ],
)
def test_letterbox_content_matches_corrected_pytorch_path(
    src_h, src_w, target_h, target_w,
):
    """Inside the letterbox content rectangle, the Triton kernel and the
    correctly-written pytorch equivalent must agree to ~1 LSB of uint8 noise."""
    rng = np.random.default_rng(seed=2)
    frame = rng.integers(0, 256, size=(src_h, src_w, 3), dtype=np.uint8)

    ref, (top, bottom, left, right) = _corrected_pytorch_letterbox(
        frame, target_h, target_w, pad_color=_PAD_COLOR,
    )
    src_gpu = torch.from_numpy(frame).cuda()
    triton_out = triton_preprocess_rfdetr(
        src_gpu, target_h, target_w, _MEANS, _STDS, pad_color=_PAD_COLOR,
    )

    content = (
        slice(None), slice(None),
        slice(top, target_h - bottom),
        slice(left, target_w - right),
    )
    # 1 uint8 LSB after normalization, using the most sensitive std.
    atol = 1.0 / 255.0 / min(_STDS)
    torch.testing.assert_close(
        triton_out[content], ref[content], rtol=0.0, atol=atol,
    )


def test_letterbox_solid_color_content_matches_exactly():
    """Uniform input: bilinear is trivial, only normalize+channel-swap math
    runs. Kernel and corrected reference should agree to fp32 noise."""
    frame = np.full((200, 600, 3), (30, 60, 90), dtype=np.uint8)  # BGR
    ref, (top, bottom, left, right) = _corrected_pytorch_letterbox(
        frame, target_h=256, target_w=256, pad_color=_PAD_COLOR,
    )
    src_gpu = torch.from_numpy(frame).cuda()
    triton_out = triton_preprocess_rfdetr(
        src_gpu, 256, 256, _MEANS, _STDS, pad_color=_PAD_COLOR,
    )
    content = (
        slice(None), slice(None),
        slice(top, 256 - bottom),
        slice(left, 256 - right),
    )
    torch.testing.assert_close(
        triton_out[content], ref[content], rtol=0.0, atol=1e-5,
    )


def test_letterbox_pad_region_uses_normalized_grey():
    """Pad region: kernel fills each RGB channel with
    ``(pad_color/255 - mean[c]) / std[c]``."""
    frame = np.random.default_rng(3).integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)
    src_gpu = torch.from_numpy(frame).cuda()
    out = triton_preprocess_rfdetr(
        src_gpu, 432, 432, _MEANS, _STDS, pad_color=_PAD_COLOR,
    )
    # For this aspect, pad lands on top/bottom. Sample row 0.
    for c in range(3):
        expected = (_PAD_COLOR / 255.0 - _MEANS[c]) / _STDS[c]
        got = out[0, c, 0, out.shape[-1] // 2].item()
        assert abs(got - expected) < 1e-4, (
            f"channel {c} pad pixel: expected {expected:.6f}, got {got:.6f}"
        )


# ---------------------------------------------------------------------------
# Documentation tests — pin down the ways the production pytorch path
# diverges from the kernel, so any future fix is a visible diff.
# ---------------------------------------------------------------------------


def test_documented_pytorch_path_channel_mixup():
    """rfdetr.py under USE_PYTORCH_FOR_PREPROCESSING normalizes BGR-ordered
    channels with RGB-ordered means/stds, then swaps BGR->RGB at the end.
    Net: output channel 0 (R) gets B-pixel data normalized with means[2]/stds[2];
         output channel 2 (B) gets R-pixel data normalized with means[0]/stds[0];
         output channel 1 (G) is self-consistent.
    """
    # Solid uniform image so interpolation is identity — only the channel
    # pipeline matters.
    b, g, r = 30, 60, 90
    frame = np.full((300, 300, 3), (b, g, r), dtype=np.uint8)  # BGR
    ref, _ = _production_pytorch_letterbox(
        frame, 256, 256, pad_color=_PAD_COLOR,
    )
    center = ref[0, :, 128, 128].tolist()

    # Output channel 0: pre-swap BGR channel 2 (R data) normalized with
    # means[0]/stds[0] which are RGB-R's but applied to R-data — coincidence
    # that R_params on R_data works out.
    # Wait — let's re-derive. Pre-swap channel order is [B, G, R].
    # Normalize with means.view(3,1,1): channel 0 (B data) uses means[0] (R param),
    # channel 1 (G data) uses means[1] (G param), channel 2 (R data) uses
    # means[2] (B param). Then swap [2,1,0] puts that into output:
    # out[0] = normalized_R_with_B_params
    # out[1] = normalized_G_with_G_params  (correct)
    # out[2] = normalized_B_with_R_params
    expected_out0 = (r / 255.0 - _MEANS[2]) / _STDS[2]   # R pixel, B params
    expected_out1 = (g / 255.0 - _MEANS[1]) / _STDS[1]   # G pixel, G params
    expected_out2 = (b / 255.0 - _MEANS[0]) / _STDS[0]   # B pixel, R params

    assert abs(center[0] - expected_out0) < 1e-4, center
    assert abs(center[1] - expected_out1) < 1e-4, center
    assert abs(center[2] - expected_out2) < 1e-4, center


def test_letterbox_half_pixel_case_diverges_predictably():
    """When (target_dim - scaled_dim) is odd, the two paths pad differently:

      - pytorch path: top=floor/2, bottom=target-scaled-top — asymmetric by 1px
      - triton kernel: pad_y = (target - scaled)/2.0 as a float in inverse map

    So content row 0 in the pytorch ref samples src_y at 0.5 (half-pixel
    center) while the kernel samples at 0.5 - 0.5 = 0.0 with dy=0.5. Picking
    a row well into the interior (far from any rounding boundary) should
    still roughly agree, but the edge rows can differ by a large amount.

    This test verifies we see that half-pixel drift on a case designed to
    isolate it (src 720x1280 -> 432x432, scaled 243x432, odd-pad 189).
    """
    rng = np.random.default_rng(seed=7)
    frame = rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)
    ref, (top, _, _, _) = _corrected_pytorch_letterbox(
        frame, 432, 432, pad_color=_PAD_COLOR,
    )
    src_gpu = torch.from_numpy(frame).cuda()
    out = triton_preprocess_rfdetr(
        src_gpu, 432, 432, _MEANS, _STDS, pad_color=_PAD_COLOR,
    )
    # Interior content row — far enough from the top edge that the two
    # different vertical phases blend to similar values.
    interior_row = top + 100
    interior_diff = (out[0, :, interior_row, :] - ref[0, :, interior_row, :]).abs().max()
    edge_diff = (out[0, :, top, :] - ref[0, :, top, :]).abs().max()
    # The edge row diff is always much larger than interior because the
    # kernel samples src row 0 with weight 1 while pytorch samples row 0
    # with weight 0.5 (blended with row 1).
    assert edge_diff > 0.5, (
        f"expected edge-row diff > 0.5 on odd-pad case, got {edge_diff.item():.3f}"
    )
    # Interior rows should also differ but much less.
    assert interior_diff < edge_diff, (
        f"interior {interior_diff.item():.3f} should be < edge {edge_diff.item():.3f}"
    )


def test_documented_pytorch_path_pad_region():
    """The production pytorch path fills the letterbox pad with raw 114.0
    (in the already-normalized tensor). The kernel uses per-channel
    normalized grey. Assertion keeps this documented so a future unification
    is visible in the diff."""
    frame = np.random.default_rng(4).integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)
    ref, (top, _, _, _) = _production_pytorch_letterbox(
        frame, 432, 432, pad_color=_PAD_COLOR,
    )
    assert top > 0
    # Pad row, any column.
    pad_row_values = ref[0, :, 0, 216]
    torch.testing.assert_close(
        pad_row_values,
        torch.full((3,), float(_PAD_COLOR), device=pad_row_values.device),
        rtol=0.0, atol=0.0,
    )


# ---------------------------------------------------------------------------
# Kernel-level sanity
# ---------------------------------------------------------------------------


def test_kernel_is_deterministic_across_calls():
    frame = np.random.default_rng(5).integers(0, 256, size=(360, 540, 3), dtype=np.uint8)
    src_gpu = torch.from_numpy(frame).cuda()
    a = triton_preprocess_rfdetr(src_gpu, 312, 312, _MEANS, _STDS, pad_color=_PAD_COLOR).clone()
    b = triton_preprocess_rfdetr(src_gpu, 312, 312, _MEANS, _STDS, pad_color=_PAD_COLOR).clone()
    torch.testing.assert_close(a, b, rtol=0.0, atol=0.0)
