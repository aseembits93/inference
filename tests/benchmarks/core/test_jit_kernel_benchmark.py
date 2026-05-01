"""Per-kernel equivalence + micro-benchmarks for the numba JIT kernels in
``inference.core.utils.jit_kernels``.

Each kernel is exercised twice:

1. ``test_equivalence_*`` — asserts the JIT-backed public API produces the
   same output as a reference pure-NumPy implementation (inlined below so
   correctness stays pinned even if the shipped code drifts).
2. ``test_speedup_*`` — runs both implementations under ``time.perf_counter``
   and prints ``speedup = numpy_time / jit_time`` so a reviewer can see the
   per-kernel win without needing ``pytest-benchmark`` installed.

The JIT kernels pay a one-time compile cost on first call. Every speedup test
calls the JIT function once as warmup before timing.
"""

from __future__ import annotations

import time
from typing import Callable, Tuple

import numpy as np
import pytest

from inference.core.nms import non_max_suppression_fast
from inference.core.utils.postprocess import (
    clip_boxes_coordinates,
    clip_keypoints_coordinates,
    crop_mask,
    scale_bboxes,
    shift_bboxes,
    shift_keypoints,
    sigmoid,
    stretch_keypoints,
    undo_image_padding_for_predicted_boxes,
    undo_image_padding_for_predicted_keypoints,
)


# ---------------------------------------------------------------------------
# Reference pure-NumPy implementations (the code as it was before numba).
# Kept verbatim so equivalence tests don't silently pass if both sides regress.
# ---------------------------------------------------------------------------


def _ref_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _ref_crop_mask(masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
    r = np.arange(w, dtype=x1.dtype)[None, None, :]
    c = np.arange(h, dtype=x1.dtype)[None, :, None]
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def _ref_scale_bboxes(bboxes, scale_x, scale_y):
    bboxes = bboxes.copy()
    bboxes[:, 0] *= scale_x
    bboxes[:, 2] *= scale_x
    bboxes[:, 1] *= scale_y
    bboxes[:, 3] *= scale_y
    return bboxes


def _ref_shift_bboxes(bboxes, shift_x, shift_y):
    bboxes = bboxes.copy()
    bboxes[:, 0] += shift_x
    bboxes[:, 2] += shift_x
    bboxes[:, 1] += shift_y
    bboxes[:, 3] += shift_y
    return bboxes


def _ref_clip_bboxes(bboxes, origin_shape):
    bboxes = bboxes.copy()
    bboxes[:, 0] = np.round(np.clip(bboxes[:, 0], 0, origin_shape[1]))
    bboxes[:, 2] = np.round(np.clip(bboxes[:, 2], 0, origin_shape[1]))
    bboxes[:, 1] = np.round(np.clip(bboxes[:, 1], 0, origin_shape[0]))
    bboxes[:, 3] = np.round(np.clip(bboxes[:, 3], 0, origin_shape[0]))
    return bboxes


def _ref_undo_padding_bboxes(bboxes, infer_shape, origin_shape):
    scale = min(infer_shape[0] / origin_shape[0], infer_shape[1] / origin_shape[1])
    inter_h = round(origin_shape[0] * scale)
    inter_w = round(origin_shape[1] * scale)
    pad_x = (infer_shape[1] - inter_w) / 2
    pad_y = (infer_shape[0] - inter_h) / 2
    bboxes = _ref_shift_bboxes(bboxes, -pad_x, -pad_y)
    bboxes /= scale
    return bboxes


def _ref_stretch_keypoints(keypoints, infer_shape, origin_shape):
    keypoints = keypoints.copy()
    sw = origin_shape[1] / infer_shape[1]
    sh = origin_shape[0] / infer_shape[0]
    for k in range(keypoints.shape[1] // 3):
        keypoints[:, k * 3] *= sw
        keypoints[:, k * 3 + 1] *= sh
    return keypoints


def _ref_shift_keypoints(keypoints, sx, sy):
    keypoints = keypoints.copy()
    for k in range(keypoints.shape[1] // 3):
        keypoints[:, k * 3] += sx
        keypoints[:, k * 3 + 1] += sy
    return keypoints


def _ref_clip_keypoints(keypoints, origin_shape):
    keypoints = keypoints.copy()
    for k in range(keypoints.shape[1] // 3):
        keypoints[:, k * 3] = np.round(
            np.clip(keypoints[:, k * 3], 0, origin_shape[1])
        )
        keypoints[:, k * 3 + 1] = np.round(
            np.clip(keypoints[:, k * 3 + 1], 0, origin_shape[0])
        )
    return keypoints


def _ref_undo_padding_keypoints(keypoints, infer_shape, origin_shape):
    keypoints = keypoints.copy()
    scale = min(infer_shape[0] / origin_shape[0], infer_shape[1] / origin_shape[1])
    inter_w = int(origin_shape[1] * scale)
    inter_h = int(origin_shape[0] * scale)
    pad_x = (infer_shape[1] - inter_w) / 2
    pad_y = (infer_shape[0] - inter_h) / 2
    for k in range(keypoints.shape[1] // 3):
        keypoints[:, k * 3] -= pad_x
        keypoints[:, k * 3] /= scale
        keypoints[:, k * 3 + 1] -= pad_y
        keypoints[:, k * 3 + 1] /= scale
    return keypoints


def _ref_nms_fast(boxes: np.ndarray, overlap_thresh: float):
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    conf = boxes[:, 4]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(conf)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0]))
        )
    return boxes[pick].astype("float")


# ---------------------------------------------------------------------------
# Timing helper.
# ---------------------------------------------------------------------------


def _time_and_report(
    name: str,
    jit_call: Callable[[], object],
    ref_call: Callable[[], object],
    rounds: int = 20,
    capsys=None,
) -> Tuple[float, float]:
    # Warmup — JIT compile happens here, not inside timing loop.
    jit_call()
    ref_call()

    def bench(fn):
        best = float("inf")
        for _ in range(rounds):
            t0 = time.perf_counter()
            fn()
            dt = time.perf_counter() - t0
            if dt < best:
                best = dt
        return best

    t_jit = bench(jit_call)
    t_ref = bench(ref_call)
    speedup = t_ref / t_jit if t_jit > 0 else float("inf")
    print(
        f"\n[{name}] jit={t_jit*1e6:9.2f}us  numpy={t_ref*1e6:9.2f}us"
        f"  speedup={speedup:5.2f}x"
    )
    return t_jit, t_ref


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------


def _make_nms_input(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # 6 columns: x1, y1, x2, y2, conf, class — matching w_np_non_max_suppression.
    boxes = np.empty((n, 6), dtype=np.float64)
    cx = rng.uniform(0, 640, n)
    cy = rng.uniform(0, 640, n)
    wh = rng.uniform(10, 80, (n, 2))
    boxes[:, 0] = cx - wh[:, 0] / 2
    boxes[:, 1] = cy - wh[:, 1] / 2
    boxes[:, 2] = cx + wh[:, 0] / 2
    boxes[:, 3] = cy + wh[:, 1] / 2
    boxes[:, 4] = rng.uniform(0, 1, n)
    boxes[:, 5] = 0
    # Caller sorts descending by confidence before handing to non_max_suppression_fast.
    return boxes[np.argsort(-boxes[:, 4])]


@pytest.mark.parametrize("n", [100, 1000, 3000])
def test_equivalence_nms(n):
    boxes = _make_nms_input(n)
    out_jit = non_max_suppression_fast(boxes.copy(), 0.45)
    out_ref = _ref_nms_fast(boxes.copy(), 0.45)
    assert len(out_jit) == len(out_ref)
    # The JIT implementation uses strict > (matching the original); with
    # distinct random confidences there are no ties, so results must be
    # identical row-wise.
    np.testing.assert_allclose(np.asarray(out_jit), np.asarray(out_ref), atol=0)


@pytest.mark.parametrize("n", [100, 1000, 3000])
def test_speedup_nms(n, capsys):
    boxes = _make_nms_input(n)
    _time_and_report(
        f"nms n={n}",
        jit_call=lambda: non_max_suppression_fast(boxes.copy(), 0.45),
        ref_call=lambda: _ref_nms_fast(boxes.copy(), 0.45),
        rounds=10 if n >= 3000 else 25,
    )


# ---------------------------------------------------------------------------
# sigmoid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(100, 160 * 160), (300, 160 * 160)])
def test_equivalence_sigmoid(shape):
    rng = np.random.default_rng(1)
    x = rng.standard_normal(shape).astype(np.float64) * 5
    # sigmoid() is in-place on its input now — copy for each side.
    out_jit = sigmoid(x.copy())
    out_ref = _ref_sigmoid(x.copy())
    np.testing.assert_allclose(out_jit, out_ref, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("shape", [(100, 160 * 160), (300, 160 * 160)])
def test_speedup_sigmoid(shape, capsys):
    rng = np.random.default_rng(1)
    x = rng.standard_normal(shape).astype(np.float64) * 5
    _time_and_report(
        f"sigmoid shape={shape}",
        jit_call=lambda: sigmoid(x.copy()),
        ref_call=lambda: _ref_sigmoid(x.copy()),
    )


# ---------------------------------------------------------------------------
# crop_mask
# ---------------------------------------------------------------------------


def _make_crop_mask_input(n: int, size: int, seed: int = 2):
    rng = np.random.default_rng(seed)
    masks = rng.random((n, size, size)).astype(np.float32)
    boxes = np.empty((n, 4), dtype=np.float32)
    boxes[:, 0] = rng.uniform(0, size * 0.4, n)
    boxes[:, 1] = rng.uniform(0, size * 0.4, n)
    boxes[:, 2] = boxes[:, 0] + rng.uniform(size * 0.3, size * 0.6, n)
    boxes[:, 3] = boxes[:, 1] + rng.uniform(size * 0.3, size * 0.6, n)
    return masks, boxes


@pytest.mark.parametrize("n,size", [(20, 160), (50, 320)])
def test_equivalence_crop_mask(n, size):
    masks, boxes = _make_crop_mask_input(n, size)
    out_jit = crop_mask(masks.copy(), boxes.copy())
    out_ref = _ref_crop_mask(masks.astype(np.float64), boxes.astype(np.float64))
    np.testing.assert_allclose(out_jit, out_ref, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("n,size", [(20, 160), (50, 320)])
def test_speedup_crop_mask(n, size, capsys):
    masks, boxes = _make_crop_mask_input(n, size)
    _time_and_report(
        f"crop_mask n={n} size={size}",
        jit_call=lambda: crop_mask(masks.copy(), boxes.copy()),
        ref_call=lambda: _ref_crop_mask(
            masks.astype(np.float64), boxes.astype(np.float64)
        ),
    )


# ---------------------------------------------------------------------------
# bbox transforms
# ---------------------------------------------------------------------------


def _make_bboxes(n: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bb = np.empty((n, 4), dtype=np.float64)
    bb[:, 0] = rng.uniform(-50, 600, n)
    bb[:, 1] = rng.uniform(-50, 600, n)
    bb[:, 2] = bb[:, 0] + rng.uniform(5, 100, n)
    bb[:, 3] = bb[:, 1] + rng.uniform(5, 100, n)
    return bb


@pytest.mark.parametrize("n", [100, 1000])
def test_equivalence_scale_bboxes(n):
    bb = _make_bboxes(n)
    out_jit = scale_bboxes(bb.copy(), 0.7, 1.3)
    out_ref = _ref_scale_bboxes(bb, 0.7, 1.3)
    np.testing.assert_allclose(out_jit, out_ref)


@pytest.mark.parametrize("n", [100, 1000])
def test_equivalence_shift_bboxes(n):
    bb = _make_bboxes(n)
    out_jit = shift_bboxes(bb.copy(), -12.5, 3.0)
    out_ref = _ref_shift_bboxes(bb, -12.5, 3.0)
    np.testing.assert_allclose(out_jit, out_ref)


@pytest.mark.parametrize("n", [100, 1000])
def test_equivalence_clip_bboxes(n):
    bb = _make_bboxes(n)
    origin_shape = (480, 640)
    out_jit = clip_boxes_coordinates(bb.copy(), origin_shape)
    out_ref = _ref_clip_bboxes(bb, origin_shape)
    np.testing.assert_allclose(out_jit, out_ref)


@pytest.mark.parametrize("n", [100, 1000])
def test_equivalence_undo_padding_bboxes(n):
    bb = _make_bboxes(n)
    infer_shape = (640, 640)
    origin_shape = (480, 640)
    out_jit = undo_image_padding_for_predicted_boxes(bb.copy(), infer_shape, origin_shape)
    out_ref = _ref_undo_padding_bboxes(bb, infer_shape, origin_shape)
    np.testing.assert_allclose(out_jit, out_ref)


@pytest.mark.parametrize("n", [1000])
def test_speedup_bbox_transforms(n, capsys):
    bb = _make_bboxes(n)
    _time_and_report(
        f"scale_bboxes n={n}",
        jit_call=lambda: scale_bboxes(bb.copy(), 0.7, 1.3),
        ref_call=lambda: _ref_scale_bboxes(bb, 0.7, 1.3),
    )
    _time_and_report(
        f"shift_bboxes n={n}",
        jit_call=lambda: shift_bboxes(bb.copy(), -12.5, 3.0),
        ref_call=lambda: _ref_shift_bboxes(bb, -12.5, 3.0),
    )
    _time_and_report(
        f"clip_bboxes n={n}",
        jit_call=lambda: clip_boxes_coordinates(bb.copy(), (480, 640)),
        ref_call=lambda: _ref_clip_bboxes(bb, (480, 640)),
    )


# ---------------------------------------------------------------------------
# keypoint transforms
# ---------------------------------------------------------------------------


def _make_keypoints(n: int, k: int = 17, seed: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    kp = np.empty((n, k * 3), dtype=np.float64)
    for i in range(k):
        kp[:, i * 3] = rng.uniform(-10, 650, n)
        kp[:, i * 3 + 1] = rng.uniform(-10, 650, n)
        kp[:, i * 3 + 2] = rng.uniform(0, 1, n)
    return kp


@pytest.mark.parametrize("n", [100, 500])
def test_equivalence_stretch_keypoints(n):
    kp = _make_keypoints(n)
    infer_shape = (640, 640)
    origin_shape = (480, 720)
    out_jit = stretch_keypoints(kp.copy(), infer_shape, origin_shape)
    out_ref = _ref_stretch_keypoints(kp, infer_shape, origin_shape)
    np.testing.assert_allclose(out_jit, out_ref)


@pytest.mark.parametrize("n", [100, 500])
def test_equivalence_shift_keypoints(n):
    kp = _make_keypoints(n)
    out_jit = shift_keypoints(kp.copy(), 5.0, -2.0)
    out_ref = _ref_shift_keypoints(kp, 5.0, -2.0)
    np.testing.assert_allclose(out_jit, out_ref)


@pytest.mark.parametrize("n", [100, 500])
def test_equivalence_clip_keypoints(n):
    kp = _make_keypoints(n)
    out_jit = clip_keypoints_coordinates(kp.copy(), (480, 640))
    out_ref = _ref_clip_keypoints(kp, (480, 640))
    np.testing.assert_allclose(out_jit, out_ref)


@pytest.mark.parametrize("n", [100, 500])
def test_equivalence_undo_padding_keypoints(n):
    kp = _make_keypoints(n)
    out_jit = undo_image_padding_for_predicted_keypoints(
        kp.copy(), (640, 640), (480, 640)
    )
    out_ref = _ref_undo_padding_keypoints(kp, (640, 640), (480, 640))
    np.testing.assert_allclose(out_jit, out_ref)


@pytest.mark.parametrize("n", [500])
def test_speedup_keypoint_transforms(n, capsys):
    kp = _make_keypoints(n)
    _time_and_report(
        f"stretch_keypoints n={n}",
        jit_call=lambda: stretch_keypoints(kp.copy(), (640, 640), (480, 720)),
        ref_call=lambda: _ref_stretch_keypoints(kp, (640, 640), (480, 720)),
    )
    _time_and_report(
        f"clip_keypoints n={n}",
        jit_call=lambda: clip_keypoints_coordinates(kp.copy(), (480, 640)),
        ref_call=lambda: _ref_clip_keypoints(kp, (480, 640)),
    )
