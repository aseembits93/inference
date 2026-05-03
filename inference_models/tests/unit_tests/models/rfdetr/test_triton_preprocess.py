"""Numerical parity between the Triton preprocess kernels and a reference
torch implementation.

Only runs on CUDA+Triton hosts; skipped cleanly otherwise so CPU CI does not
fail on these files.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

if not torch.cuda.is_available():  # pragma: no cover - host-dependent
    pytest.skip("CUDA not available", allow_module_level=True)

cv2 = pytest.importorskip("cv2")

from inference_models.entities import ImageDimensions  # noqa: E402
from inference_models.models.common.roboflow.model_packages import (  # noqa: E402
    ColorMode,
    ImagePreProcessing,
    NetworkInputDefinition,
    ResizeMode,
    TrainingInputSize,
)
from inference_models.models.rfdetr.pre_processing import (  # noqa: E402
    pre_process_network_input,
)
from inference_models.models.rfdetr.triton_preprocess import (  # noqa: E402
    triton_preprocess_rfdetr_stretch,
)


_MEANS = (0.485, 0.456, 0.406)
_STDS = (0.229, 0.224, 0.225)


def _reference_stretch(
    frame_bgr_hwc_uint8: np.ndarray,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """Torch reference for the STRETCH_TO Triton kernel.

    Pipeline: resize (bilinear, align-corners=False, pixel-centers) ->
    BGR->RGB -> /255 -> (x - mean) / std -> (1, 3, H, W) fp32 on cuda.
    """
    tensor = torch.from_numpy(frame_bgr_hwc_uint8).to(
        device="cuda", dtype=torch.float32
    )  # (H, W, 3) BGR
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W) BGR
    tensor = torch.nn.functional.interpolate(
        tensor,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )
    # BGR -> RGB
    tensor = tensor[:, [2, 1, 0], :, :]
    tensor = tensor / 255.0
    mean = torch.tensor(_MEANS, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(_STDS, device="cuda").view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor


@pytest.mark.parametrize(
    "src_h, src_w, target_h, target_w",
    [
        (312, 312, 312, 312),   # same-size, no resize
        (720, 1280, 312, 312),  # 1080p-ish down to 312
        (200, 400, 312, 312),   # aspect-stretched upscale
    ],
)
def test_triton_stretch_matches_reference(src_h, src_w, target_h, target_w):
    rng = np.random.default_rng(seed=0)
    frame = rng.integers(0, 256, size=(src_h, src_w, 3), dtype=np.uint8)

    src_gpu = torch.from_numpy(frame).to("cuda")
    out = triton_preprocess_rfdetr_stretch(
        src_gpu,
        target_h=target_h,
        target_w=target_w,
        means=_MEANS,
        stds=_STDS,
    )
    ref = _reference_stretch(frame, target_h, target_w)

    # Tolerance reflects a 1-LSB difference in the uint8 gather path
    # post-normalization: 1/255 / min(std). min(std) = 0.224 gives ~0.0175.
    assert out.shape == (1, 3, target_h, target_w)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref, rtol=0.0, atol=1.0 / 255.0 / min(_STDS))


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_triton_stretch_rejects_cpu_tensor():
    frame = torch.zeros((32, 32, 3), dtype=torch.uint8)  # cpu
    with pytest.raises(ValueError, match="expected CUDA tensor"):
        triton_preprocess_rfdetr_stretch(frame, target_h=16, target_w=16)


def test_triton_stretch_rejects_wrong_dtype():
    frame = torch.zeros((32, 32, 3), dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="expected uint8"):
        triton_preprocess_rfdetr_stretch(frame, target_h=16, target_w=16)


def test_triton_stretch_rejects_wrong_shape():
    frame = torch.zeros((3, 32, 32), dtype=torch.uint8, device="cuda")  # CHW, not HWC
    with pytest.raises(ValueError, match="expected HWC 3-channel"):
        triton_preprocess_rfdetr_stretch(frame, target_h=16, target_w=16)


def test_triton_stretch_rejects_mismatched_out_shape():
    frame = torch.zeros((32, 32, 3), dtype=torch.uint8, device="cuda")
    out = torch.empty((1, 3, 8, 8), dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match=r"out has shape"):
        triton_preprocess_rfdetr_stretch(frame, target_h=16, target_w=16, out=out)


def test_triton_stretch_rejects_out_wrong_dtype():
    frame = torch.zeros((32, 32, 3), dtype=torch.uint8, device="cuda")
    out = torch.empty((1, 3, 16, 16), dtype=torch.float16, device="cuda")
    with pytest.raises(ValueError, match="out must be fp32 CUDA tensor"):
        triton_preprocess_rfdetr_stretch(frame, target_h=16, target_w=16, out=out)


# ---------------------------------------------------------------------------
# Output-tensor reuse (the real adapter passes a persistent buffer in)
# ---------------------------------------------------------------------------


def test_triton_stretch_writes_into_provided_out_buffer():
    rng = np.random.default_rng(seed=1)
    frame = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    src_gpu = torch.from_numpy(frame).to("cuda")

    out_buf = torch.zeros((1, 3, 32, 32), dtype=torch.float32, device="cuda")
    returned = triton_preprocess_rfdetr_stretch(
        src_gpu, target_h=32, target_w=32, means=_MEANS, stds=_STDS, out=out_buf,
    )
    # Must return the exact same storage — no allocation when out is given.
    assert returned.data_ptr() == out_buf.data_ptr()
    # And the buffer must have been populated.
    assert not torch.all(out_buf == 0)


# ---------------------------------------------------------------------------
# Parity with the RF-DETR STRETCH_TO wrapper — the only resize mode the
# Triton fast path currently supports. Uses the same cv2 resize + BGR->RGB +
# normalize that the non-Triton adapter path feeds to TRT, so the output
# must land within a small tolerance.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "src_h, src_w, target_h, target_w",
    [
        (312, 312, 312, 312),
        (720, 1280, 312, 312),
        (200, 400, 312, 312),
    ],
)
def test_triton_stretch_matches_pre_process_network_input(
    src_h, src_w, target_h, target_w,
):
    """End-to-end parity: Triton kernel vs the wrapper the model actually
    calls when RFDETR_USE_TRITON_PREPROC is off.

    The wrapper (`pre_process_network_input` with STRETCH_TO) does
    cv2.resize -> (H,W,C)->(1,C,H,W) -> BGR->RGB (via color_mode flip) ->
    /255 -> normalize. The kernel fuses all of that. Expected outputs equal
    modulo cv2-vs-triton bilinear rounding and the 1-LSB uint8 tolerance.
    """
    rng = np.random.default_rng(seed=42)
    # BGR frame — what cv2.imread / gst output gives us on the hot path.
    frame_bgr = rng.integers(0, 256, size=(src_h, src_w, 3), dtype=np.uint8)

    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=target_h, width=target_w),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,          # model wants RGB
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
        scaling_factor=255.0,
        normalization=(list(_MEANS), list(_STDS)),
    )
    # Base wrapper sees a BGR frame and flips to RGB because input_color_format
    # defaults to BGR via ImagePreProcessing; the wrapper handles the flip.
    ref_tensor, _ = pre_process_network_input(
        images=frame_bgr,
        image_pre_processing=ImagePreProcessing(),
        network_input=network_input,
        target_device=torch.device("cuda"),
    )

    src_gpu = torch.from_numpy(frame_bgr).to("cuda")
    triton_tensor = triton_preprocess_rfdetr_stretch(
        src_gpu,
        target_h=target_h,
        target_w=target_w,
        means=_MEANS,
        stds=_STDS,
    )

    assert triton_tensor.shape == ref_tensor.shape == (1, 3, target_h, target_w)
    assert triton_tensor.dtype == ref_tensor.dtype == torch.float32
    # cv2.resize uses INTER_LINEAR with 5-bit fixed-point coefficients that
    # round slightly differently than the Triton kernel's explicit fp32
    # pixel-center bilinear. Empirically diverges by up to ~1.5 LSBs in
    # normalized space on large downscales (720x1280 -> 312x312). A 2-LSB
    # tolerance gives headroom without hiding real regressions.
    atol = 2.0 / 255.0 / min(_STDS)
    torch.testing.assert_close(triton_tensor, ref_tensor, rtol=0.0, atol=atol)


def test_triton_stretch_matches_pre_process_on_solid_color_image():
    """Solid-color image: every output pixel must equal the analytic
    normalization of that color. No interpolation ambiguity."""
    target_h = target_w = 64
    bgr_color = np.array([30, 60, 90], dtype=np.uint8)  # B=30, G=60, R=90
    frame_bgr = np.broadcast_to(bgr_color, (128, 256, 3)).copy()

    src_gpu = torch.from_numpy(frame_bgr).to("cuda")
    out = triton_preprocess_rfdetr_stretch(
        src_gpu,
        target_h=target_h,
        target_w=target_w,
        means=_MEANS,
        stds=_STDS,
    )

    # After BGR->RGB the channels are (R=90, G=60, B=30).
    expected_r = (90 / 255.0 - _MEANS[0]) / _STDS[0]
    expected_g = (60 / 255.0 - _MEANS[1]) / _STDS[1]
    expected_b = (30 / 255.0 - _MEANS[2]) / _STDS[2]

    torch.testing.assert_close(
        out[0, 0], torch.full((target_h, target_w), expected_r, device="cuda"),
        rtol=0.0, atol=1e-5,
    )
    torch.testing.assert_close(
        out[0, 1], torch.full((target_h, target_w), expected_g, device="cuda"),
        rtol=0.0, atol=1e-5,
    )
    torch.testing.assert_close(
        out[0, 2], torch.full((target_h, target_w), expected_b, device="cuda"),
        rtol=0.0, atol=1e-5,
    )
