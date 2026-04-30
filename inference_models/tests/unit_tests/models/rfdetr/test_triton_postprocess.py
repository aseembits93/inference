"""Numerical parity between `triton_rfdetr_conf_filter` and a torch reference.

Covers three regimes:
  - scalar threshold, no class remapping (drops DETR no-object rows)
  - scalar threshold, with class remapping (-1 drops)
  - per-class threshold, with class remapping

Only runs on CUDA+Triton hosts.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

if not torch.cuda.is_available():  # pragma: no cover - host-dependent
    pytest.skip("CUDA not available", allow_module_level=True)

from inference_models.models.rfdetr.triton_postprocess import (  # noqa: E402
    triton_rfdetr_conf_filter,
)


def _reference(
    logits: torch.Tensor,
    threshold,
    num_classes: int,
    class_mapping: torch.Tensor | None,
):
    """Matches the kernel docstring: conf = max sigmoid, top_c = argmax,
    keep = (remapped class valid) & (conf > threshold[class])."""
    sig = torch.sigmoid(logits)
    conf, top_c = sig.max(dim=1)
    if class_mapping is not None:
        remapped = class_mapping.to(dtype=torch.int64, device=logits.device)[top_c]
        valid = remapped >= 0
        top_out = torch.where(valid, remapped, torch.zeros_like(remapped))
    else:
        valid = top_c < num_classes
        top_out = top_c.to(dtype=torch.int64)
    if isinstance(threshold, torch.Tensor):
        safe_c = torch.where(valid, top_out, torch.zeros_like(top_out))
        thr = threshold.to(device=logits.device, dtype=conf.dtype)[safe_c]
    else:
        thr = float(threshold)
    keep = valid & (conf > thr)
    return conf, top_out, keep


def test_scalar_threshold_no_remapping_drops_noobject_rows():
    torch.manual_seed(0)
    num_queries = 64
    num_classes = 5
    num_classes_total = num_classes + 1  # trailing no-object slot
    logits = torch.randn(num_queries, num_classes_total, device="cuda")

    conf, top_c, keep = triton_rfdetr_conf_filter(
        logits, threshold=0.2, num_classes=num_classes, class_mapping=None
    )
    ref_conf, ref_top, ref_keep = _reference(
        logits, 0.2, num_classes, class_mapping=None
    )

    torch.testing.assert_close(conf, ref_conf, rtol=0.0, atol=1e-6)
    assert torch.equal(top_c.long(), ref_top)
    assert torch.equal(keep, ref_keep)


def test_scalar_threshold_with_class_remapping():
    torch.manual_seed(1)
    num_queries = 64
    num_classes = 3
    num_classes_total = 6
    logits = torch.randn(num_queries, num_classes_total, device="cuda")
    # Map classes 0,1,2,3,4,5 -> -1, 0, -1, 1, 2, -1 (keep 1/3/4 as 0/1/2).
    class_mapping = torch.tensor([-1, 0, -1, 1, 2, -1], dtype=torch.int32, device="cuda")

    conf, top_c, keep = triton_rfdetr_conf_filter(
        logits, threshold=0.3, num_classes=num_classes, class_mapping=class_mapping
    )
    ref_conf, ref_top, ref_keep = _reference(
        logits, 0.3, num_classes, class_mapping=class_mapping
    )

    torch.testing.assert_close(conf, ref_conf, rtol=0.0, atol=1e-6)
    # top_c for dropped rows is whatever the mapping produced; only compare kept.
    assert torch.equal(keep, ref_keep)
    assert torch.equal(top_c[keep].long(), ref_top[keep])


def test_per_class_threshold_with_class_remapping():
    torch.manual_seed(2)
    num_queries = 32
    num_classes = 3
    num_classes_total = 5
    logits = torch.randn(num_queries, num_classes_total, device="cuda")
    class_mapping = torch.tensor([0, 1, -1, -1, 2], dtype=torch.int32, device="cuda")
    # One threshold per remapped class id.
    threshold = torch.tensor([0.1, 0.5, 0.8], dtype=torch.float32, device="cuda")

    conf, top_c, keep = triton_rfdetr_conf_filter(
        logits, threshold=threshold, num_classes=num_classes, class_mapping=class_mapping
    )
    ref_conf, ref_top, ref_keep = _reference(
        logits, threshold, num_classes, class_mapping=class_mapping
    )

    torch.testing.assert_close(conf, ref_conf, rtol=0.0, atol=1e-6)
    assert torch.equal(keep, ref_keep)
    assert torch.equal(top_c[keep].long(), ref_top[keep])
