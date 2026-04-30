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
