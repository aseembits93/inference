"""Numba-JIT-compiled kernels for hot post-processing paths.

Keep kernels pure: only scalars + contiguous NumPy arrays in/out, no Python
containers, no OpenCV/Torch. Callers in inference.core.nms and
inference.core.utils.postprocess delegate here while keeping their public
signatures unchanged.
"""

import numpy as np
from numba import njit, prange


@njit(cache=True)
def nms_indices(boxes: np.ndarray, overlap_thresh: float) -> np.ndarray:
    """Malisiewicz-style NMS.

    ``boxes`` is (N, >=5) sorted *descending* by confidence — row 0 is the
    highest-confidence candidate, matching what ``w_np_non_max_suppression``
    passes in after ``np.argsort(-np_detections[:, 4])``. Only the first four
    columns (x1, y1, x2, y2) are read. Returns row indices of the picks in
    descending-confidence order. Uses the original formula
    ``overlap = intersection / area_of_candidate`` (not IoU against the pick),
    preserving behavior bit-for-bit.
    """
    n = boxes.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)

    area = np.empty(n, dtype=np.float64)
    for i in range(n):
        area[i] = (boxes[i, 2] - boxes[i, 0] + 1.0) * (boxes[i, 3] - boxes[i, 1] + 1.0)

    active = np.ones(n, dtype=np.bool_)
    pick = np.empty(n, dtype=np.int64)
    n_picked = 0

    for k in range(n):
        if not active[k]:
            continue
        pick[n_picked] = k
        n_picked += 1

        xk1 = boxes[k, 0]
        yk1 = boxes[k, 1]
        xk2 = boxes[k, 2]
        yk2 = boxes[k, 3]

        for j in range(k + 1, n):
            if not active[j]:
                continue
            x1j = boxes[j, 0]
            y1j = boxes[j, 1]
            x2j = boxes[j, 2]
            y2j = boxes[j, 3]

            xx1 = xk1 if xk1 > x1j else x1j
            yy1 = yk1 if yk1 > y1j else y1j
            xx2 = xk2 if xk2 < x2j else x2j
            yy2 = yk2 if yk2 < y2j else y2j

            w = xx2 - xx1 + 1.0
            if w <= 0.0:
                continue
            h = yy2 - yy1 + 1.0
            if h <= 0.0:
                continue

            overlap = (w * h) / area[j]
            if overlap > overlap_thresh:
                active[j] = False

    return pick[:n_picked]


@njit(cache=True, parallel=True, fastmath=True)
def sigmoid_inplace(x: np.ndarray) -> np.ndarray:
    """Element-wise sigmoid on a contiguous 1-D view. Writes in place and
    returns the same buffer for caller convenience."""
    flat = x.ravel()
    n = flat.shape[0]
    for i in prange(n):
        flat[i] = 1.0 / (1.0 + np.exp(-flat[i]))
    return x


@njit(cache=True, parallel=True)
def crop_mask_kernel(masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Zero out mask entries outside their corresponding bounding box.

    masks: (n, h, w) float. boxes: (n, 4) in (x1, y1, x2, y2) order, in the
    same coordinate frame as the mask pixel grid. Matches the semantics of
    the original ``crop_mask`` in postprocess.py:
        mask *= (col >= x1) & (col < x2) & (row >= y1) & (row < y2)
    """
    n, h, w = masks.shape
    for i in prange(n):
        x1 = boxes[i, 0]
        y1 = boxes[i, 1]
        x2 = boxes[i, 2]
        y2 = boxes[i, 3]
        for r in range(h):
            row_in = (r >= y1) and (r < y2)
            if not row_in:
                for c in range(w):
                    masks[i, r, c] = 0
                continue
            for c in range(w):
                if (c >= x1) and (c < x2):
                    continue
                masks[i, r, c] = 0
    return masks


@njit(cache=True)
def scale_bboxes_kernel(
    bboxes: np.ndarray, scale_x: float, scale_y: float
) -> np.ndarray:
    n = bboxes.shape[0]
    for i in range(n):
        bboxes[i, 0] *= scale_x
        bboxes[i, 1] *= scale_y
        bboxes[i, 2] *= scale_x
        bboxes[i, 3] *= scale_y
    return bboxes


@njit(cache=True)
def shift_bboxes_kernel(
    bboxes: np.ndarray, shift_x: float, shift_y: float
) -> np.ndarray:
    n = bboxes.shape[0]
    for i in range(n):
        bboxes[i, 0] += shift_x
        bboxes[i, 2] += shift_x
        bboxes[i, 1] += shift_y
        bboxes[i, 3] += shift_y
    return bboxes


@njit(cache=True)
def clip_and_round_bboxes_kernel(
    bboxes: np.ndarray, max_x: float, max_y: float
) -> np.ndarray:
    """Clip to [0, max_*] and round to nearest integer (banker's rounding, as
    numpy does). Column 0/2 -> x (max_x), 1/3 -> y (max_y)."""
    n = bboxes.shape[0]
    for i in range(n):
        v = bboxes[i, 0]
        if v < 0.0:
            v = 0.0
        elif v > max_x:
            v = max_x
        bboxes[i, 0] = np.round(v)

        v = bboxes[i, 2]
        if v < 0.0:
            v = 0.0
        elif v > max_x:
            v = max_x
        bboxes[i, 2] = np.round(v)

        v = bboxes[i, 1]
        if v < 0.0:
            v = 0.0
        elif v > max_y:
            v = max_y
        bboxes[i, 1] = np.round(v)

        v = bboxes[i, 3]
        if v < 0.0:
            v = 0.0
        elif v > max_y:
            v = max_y
        bboxes[i, 3] = np.round(v)
    return bboxes


@njit(cache=True)
def undo_padding_bboxes_kernel(
    bboxes: np.ndarray, pad_x: float, pad_y: float, scale: float
) -> np.ndarray:
    """(bboxes - pad) / scale, applied in the same axis pattern as
    ``shift_bboxes`` followed by scalar division."""
    inv = 1.0 / scale
    n = bboxes.shape[0]
    for i in range(n):
        bboxes[i, 0] = (bboxes[i, 0] - pad_x) * inv
        bboxes[i, 2] = (bboxes[i, 2] - pad_x) * inv
        bboxes[i, 1] = (bboxes[i, 1] - pad_y) * inv
        bboxes[i, 3] = (bboxes[i, 3] - pad_y) * inv
    return bboxes


@njit(cache=True)
def scale_keypoints_kernel(
    keypoints: np.ndarray, scale_x: float, scale_y: float
) -> np.ndarray:
    """Keypoints are (N, K*3) where each triplet is (x, y, conf)."""
    n_rows = keypoints.shape[0]
    n_cols = keypoints.shape[1]
    n_triplets = n_cols // 3
    for i in range(n_rows):
        for k in range(n_triplets):
            base = k * 3
            keypoints[i, base] *= scale_x
            keypoints[i, base + 1] *= scale_y
    return keypoints


@njit(cache=True)
def shift_keypoints_kernel(
    keypoints: np.ndarray, shift_x: float, shift_y: float
) -> np.ndarray:
    n_rows = keypoints.shape[0]
    n_cols = keypoints.shape[1]
    n_triplets = n_cols // 3
    for i in range(n_rows):
        for k in range(n_triplets):
            base = k * 3
            keypoints[i, base] += shift_x
            keypoints[i, base + 1] += shift_y
    return keypoints


@njit(cache=True)
def undo_padding_keypoints_kernel(
    keypoints: np.ndarray, pad_x: float, pad_y: float, scale: float
) -> np.ndarray:
    inv = 1.0 / scale
    n_rows = keypoints.shape[0]
    n_cols = keypoints.shape[1]
    n_triplets = n_cols // 3
    for i in range(n_rows):
        for k in range(n_triplets):
            base = k * 3
            keypoints[i, base] = (keypoints[i, base] - pad_x) * inv
            keypoints[i, base + 1] = (keypoints[i, base + 1] - pad_y) * inv
    return keypoints


@njit(cache=True)
def clip_and_round_keypoints_kernel(
    keypoints: np.ndarray, max_x: float, max_y: float
) -> np.ndarray:
    n_rows = keypoints.shape[0]
    n_cols = keypoints.shape[1]
    n_triplets = n_cols // 3
    for i in range(n_rows):
        for k in range(n_triplets):
            base = k * 3
            x = keypoints[i, base]
            if x < 0.0:
                x = 0.0
            elif x > max_x:
                x = max_x
            keypoints[i, base] = np.round(x)

            y = keypoints[i, base + 1]
            if y < 0.0:
                y = 0.0
            elif y > max_y:
                y = max_y
            keypoints[i, base + 1] = np.round(y)
    return keypoints
