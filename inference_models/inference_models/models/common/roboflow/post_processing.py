from typing import List, Literal, Tuple

import torch
import torchvision
from torchvision.transforms import functional

from inference_models.entities import ImageDimensions
from inference_models.models.common.roboflow.model_packages import (
    PreProcessingMetadata,
    StaticCropOffset,
)


def run_nms_for_object_detection(
    output: torch.Tensor,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    max_detections: int = 100,
    class_agnostic: bool = False,
    box_format: Literal["xywh", "xyxy"] = "xywh",
) -> List[torch.Tensor]:
    bs = output.shape[0]
    boxes = output[:, :4, :]
    scores = output[:, 4:, :]
    results = []
    for b in range(bs):
        # Combine transpose & max for efficiency
        class_scores = scores[b]  # (80, 8400)
        class_conf, class_ids = class_scores.max(0)  # (8400,), (8400,)
        mask = class_conf > conf_thresh
        if not torch.any(mask):
            results.append(torch.zeros((0, 6), device=output.device))
            continue
        # Share one ``nonzero`` across the three filtered tensors so we pay
        # the index-materialization cost (and its D2H scalar for size) only
        # once instead of three times (PyTorch's boolean advanced indexing
        # runs ``nonzero`` internally per call).
        idx = mask.nonzero(as_tuple=True)[0]
        bboxes = boxes[b].index_select(1, idx).T  # (num, 4)
        class_conf = class_conf.index_select(0, idx)
        class_ids = class_ids.index_select(0, idx)
        if box_format == "xywh":
            # Vectorized [x, y, w, h] -> [x1, y1, x2, y2]
            xy = bboxes[:, :2]
            wh = bboxes[:, 2:]
            half_wh = wh / 2
            xyxy = torch.cat((xy - half_wh, xy + half_wh), 1)
        else:
            xyxy = bboxes
        # Class-agnostic NMS -> use dummy class ids
        nms_class_ids = torch.zeros_like(class_ids) if class_agnostic else class_ids
        # NMS and limiting max detections
        keep = torchvision.ops.batched_nms(xyxy, class_conf, nms_class_ids, iou_thresh)
        if keep.numel() > max_detections:
            keep = keep[:max_detections]
        # Concatenate [xyxy | conf | cls] once and gather with a single
        # ``index_select`` rather than three separate ``[keep]`` indexings.
        packed = torch.cat(
            (
                xyxy,
                class_conf.unsqueeze(1),
                class_ids.unsqueeze(1).float(),
            ),
            1,
        )  # (num, 6) — [x1, y1, x2, y2, conf, cls]
        detections = packed.index_select(0, keep)
        results.append(detections)
    return results


def post_process_nms_fused_model_output(
    output: torch.Tensor,
    conf_thresh: float = 0.25,
) -> List[torch.Tensor]:
    bs = output.shape[0]
    nms_results = []
    for batch_element_id in range(bs):
        batch_element_result = output[batch_element_id]
        batch_element_result = batch_element_result[
            batch_element_result[:, 4] >= conf_thresh
        ]
        nms_results.append(batch_element_result)
    return nms_results


def run_nms_for_instance_segmentation(
    output: torch.Tensor,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    max_detections: int = 100,
    class_agnostic: bool = False,
    box_format: Literal["xywh", "xyxy"] = "xywh",
) -> List[torch.Tensor]:
    bs = output.shape[0]
    boxes = output[:, :4, :]  # (N, 4, 8400)
    scores = output[:, 4:-32, :]  # (N, 80, 8400)
    masks = output[:, -32:, :]
    results = []

    for b in range(bs):
        bboxes = boxes[b].T  # (8400, 4)
        class_scores = scores[b].T  # (8400, 80)
        box_masks = masks[b].T
        class_conf, class_ids = class_scores.max(1)  # (8400,), (8400,)
        mask = class_conf > conf_thresh
        if mask.sum() == 0:
            results.append(torch.zeros((0, 38), device=output.device))
            continue
        # Share one ``nonzero`` across the four filtered tensors.
        idx = mask.nonzero(as_tuple=True)[0]
        bboxes = bboxes.index_select(0, idx)
        class_conf = class_conf.index_select(0, idx)
        class_ids = class_ids.index_select(0, idx)
        box_masks = box_masks.index_select(0, idx)
        if box_format == "xywh":
            # Vectorized [x, y, w, h] -> [x1, y1, x2, y2]
            xy = bboxes[:, :2]
            wh = bboxes[:, 2:]
            half_wh = wh / 2
            xyxy = torch.cat((xy - half_wh, xy + half_wh), 1)
        else:
            xyxy = bboxes
        # Class-agnostic NMS -> use dummy class ids
        nms_class_ids = torch.zeros_like(class_ids) if class_agnostic else class_ids
        keep = torchvision.ops.batched_nms(xyxy, class_conf, nms_class_ids, iou_thresh)
        keep = keep[:max_detections]
        # Pre-concatenate and do a single index_select rather than four
        # separate ``[keep]`` indexings.
        packed = torch.cat(
            [
                xyxy,
                class_conf.unsqueeze(1),
                class_ids.unsqueeze(1).float(),
                box_masks,
            ],
            dim=1,
        )  # [x1, y1, x2, y2, conf, cls, mask_coeffs(32)]
        detections = packed.index_select(0, keep)
        results.append(detections)
    return results


def run_nms_for_key_points_detection(
    output: torch.Tensor,
    num_classes: int,
    key_points_slots_in_prediction: int,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    max_detections: int = 100,
    class_agnostic: bool = False,
) -> List[torch.Tensor]:
    bs = output.shape[0]
    boxes = output[:, :4, :]
    scores = output[:, 4 : 4 + num_classes, :]
    key_points = output[:, 4 + num_classes :, :]
    results = []
    for b in range(bs):
        class_scores = scores[b]
        class_conf, class_ids = class_scores.max(0)
        mask = class_conf > conf_thresh
        if not torch.any(mask):
            results.append(
                torch.zeros(
                    (0, 6 + key_points_slots_in_prediction * 3), device=output.device
                )
            )
            continue
        # Share one ``nonzero`` across the four filtered tensors.
        idx = mask.nonzero(as_tuple=True)[0]
        bboxes = boxes[b].index_select(1, idx).T
        image_key_points = key_points[b].index_select(1, idx).T
        class_conf = class_conf.index_select(0, idx)
        class_ids = class_ids.index_select(0, idx)
        xy = bboxes[:, :2]
        wh = bboxes[:, 2:]
        half_wh = wh / 2
        xyxy = torch.cat((xy - half_wh, xy + half_wh), 1)
        # Class-agnostic NMS -> use dummy class ids
        nms_class_ids = torch.zeros_like(class_ids) if class_agnostic else class_ids
        # NMS and limiting max detections
        keep = torchvision.ops.batched_nms(xyxy, class_conf, nms_class_ids, iou_thresh)
        if keep.numel() > max_detections:
            keep = keep[:max_detections]
        # Pre-concatenate and do a single index_select rather than four
        # separate ``[keep]`` indexings.
        packed = torch.cat(
            (
                xyxy,
                class_conf.unsqueeze(1),
                class_ids.unsqueeze(1).float(),
                image_key_points,
            ),
            1,
        )  # [x1, y1, x2, y2, conf, cls, keypoints....]
        detections = packed.index_select(0, keep)
        results.append(detections)
    return results


def rescale_detections(
    detections: List[torch.Tensor], images_metadata: List[PreProcessingMetadata]
) -> List[torch.Tensor]:
    for image_detections, metadata in zip(detections, images_metadata):
        _ = rescale_image_detections(
            image_detections=image_detections, image_metadata=metadata
        )
    return detections


def rescale_image_detections(
    image_detections: torch.Tensor,
    image_metadata: PreProcessingMetadata,
) -> torch.Tensor:
    # in-place processing — operate with Python scalars on strided views of
    # the xyxy columns so we avoid allocating small per-call CUDA tensors
    # (each ``torch.as_tensor([...], device='cuda')`` is an H2D transfer
    # plus a tensor allocation). The x-coords (columns 0 and 2) share the
    # same pad/scale, as do the y-coords (columns 1 and 3).
    image_detections[:, 0:4:2].sub_(image_metadata.pad_left).div_(
        image_metadata.scale_width
    )
    image_detections[:, 1:4:2].sub_(image_metadata.pad_top).div_(
        image_metadata.scale_height
    )
    if (
        image_metadata.static_crop_offset.offset_x != 0
        or image_metadata.static_crop_offset.offset_y != 0
    ):
        image_detections[:, 0:4:2].add_(image_metadata.static_crop_offset.offset_x)
        image_detections[:, 1:4:2].add_(image_metadata.static_crop_offset.offset_y)
    return image_detections


def rescale_key_points_detections(
    detections: List[torch.Tensor],
    images_metadata: List[PreProcessingMetadata],
    num_classes: int,
    key_points_slots_in_prediction: int,
) -> List[torch.Tensor]:
    for image_detections, metadata in zip(detections, images_metadata):
        pad_left = metadata.pad_left
        pad_top = metadata.pad_top
        scale_w = metadata.scale_width
        scale_h = metadata.scale_height
        # xyxy columns: [x1, y1, x2, y2]
        image_detections[:, 0:4:2].sub_(pad_left).div_(scale_w)
        image_detections[:, 1:4:2].sub_(pad_top).div_(scale_h)
        # key points triples: (x, y, conf) repeated ``key_points_slots_in_prediction`` times
        # starting at column 6. The x/y columns follow a stride-3 pattern; conf
        # is left untouched.
        image_detections[:, 6::3].sub_(pad_left).div_(scale_w)
        image_detections[:, 7::3].sub_(pad_top).div_(scale_h)
        sc_x = metadata.static_crop_offset.offset_x
        sc_y = metadata.static_crop_offset.offset_y
        if sc_x != 0 or sc_y != 0:
            image_detections[:, 6::3].add_(sc_x)
            image_detections[:, 7::3].add_(sc_y)
            image_detections[:, 0:4:2].add_(sc_x)
            image_detections[:, 1:4:2].add_(sc_y)
    return detections


def preprocess_segmentation_masks(
    protos: torch.Tensor,
    masks_in: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("chw,nc->nhw", protos, masks_in)


import threading as _threading

# Cache ``torch.arange`` indices per (size, device) so repeated mask-crop
# operations on the same mask resolution don't reallocate them. Each
# ``torch.arange`` otherwise allocates a fresh CUDA tensor and issues a
# kernel launch — 17us/call overhead for YOLOv8n-seg.
_arange_cache = _threading.local()


def _get_cached_arange(n: int, device: torch.device) -> torch.Tensor:
    """Return a 1D cached ``torch.arange(n)`` tensor on ``device``."""
    cache = getattr(_arange_cache, "tensors", None)
    if cache is None:
        cache = {}
        _arange_cache.tensors = cache
    key = (n, device)
    t = cache.get(key)
    if t is None:
        t = torch.arange(n, device=device)
        cache[key] = t
    return t


def crop_masks_to_boxes(
    boxes: torch.Tensor,
    masks: torch.Tensor,
    scaling: float = 0.25,
) -> torch.Tensor:
    n, h, w = masks.shape
    scaled_boxes = torch.round(boxes * scaling)
    x1, y1, x2, y2 = (
        scaled_boxes[:, 0][:, None, None],
        scaled_boxes[:, 1][:, None, None],
        scaled_boxes[:, 2][:, None, None],
        scaled_boxes[:, 3][:, None, None],
    )
    rows = _get_cached_arange(w, masks.device)[None, None, :]  # shape: [1, 1, w]
    cols = _get_cached_arange(h, masks.device)[None, :, None]  # shape: [1, h, 1]
    crop_mask = (rows >= x1) & (rows < x2) & (cols >= y1) & (cols < y2)
    return masks * crop_mask


def align_instance_segmentation_results(
    image_bboxes: torch.Tensor,
    masks: torch.Tensor,
    padding: Tuple[int, int, int, int],
    scale_width: float,
    scale_height: float,
    original_size: ImageDimensions,
    size_after_pre_processing: ImageDimensions,
    inference_size: ImageDimensions,
    static_crop_offset: StaticCropOffset,
    binarization_threshold: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if image_bboxes.shape[0] == 0:
        empty_masks = torch.empty(
            size=(0, size_after_pre_processing.height, size_after_pre_processing.width),
            dtype=torch.bool,
            device=image_bboxes.device,
        )
        return image_bboxes, empty_masks
    pad_left, pad_top, pad_right, pad_bottom = padding
    # Strided scalar ops avoid allocating small CUDA offset/scale tensors
    # per call (see rescale_image_detections for the same pattern).
    image_bboxes[:, 0:4:2].sub_(pad_left).div_(scale_width)
    image_bboxes[:, 1:4:2].sub_(pad_top).div_(scale_height)
    n, mh, mw = masks.shape
    mask_h_scale = mh / inference_size.height
    mask_w_scale = mw / inference_size.width
    mask_pad_top, mask_pad_bottom, mask_pad_left, mask_pad_right = (
        round(mask_h_scale * pad_top),
        round(mask_h_scale * pad_bottom),
        round(mask_w_scale * pad_left),
        round(mask_w_scale * pad_right),
    )
    if (
        mask_pad_top < 0
        or mask_pad_bottom < 0
        or mask_pad_left < 0
        or mask_pad_right < 0
    ):
        masks = torch.nn.functional.pad(
            masks,
            (
                abs(min(mask_pad_left, 0)),
                abs(min(mask_pad_right, 0)),
                abs(min(mask_pad_top, 0)),
                abs(min(mask_pad_bottom, 0)),
            ),
            "constant",
            0,
        )
        padded_mask_offset_top = max(mask_pad_top, 0)
        padded_mask_offset_bottom = max(mask_pad_bottom, 0)
        padded_mask_offset_left = max(mask_pad_left, 0)
        padded_mask_offset_right = max(mask_pad_right, 0)
        masks = masks[
            :,
            padded_mask_offset_top : masks.shape[1] - padded_mask_offset_bottom,
            padded_mask_offset_left : masks.shape[2] - padded_mask_offset_right,
        ]
    else:
        masks = masks[
            :, mask_pad_top : mh - mask_pad_bottom, mask_pad_left : mw - mask_pad_right
        ]
    masks = (
        functional.resize(
            masks,
            [size_after_pre_processing.height, size_after_pre_processing.width],
            interpolation=functional.InterpolationMode.BILINEAR,
        )
        .gt_(binarization_threshold)
        .to(dtype=torch.bool)
    )
    if static_crop_offset.offset_x > 0 or static_crop_offset.offset_y > 0:
        mask_canvas = torch.zeros(
            (
                masks.shape[0],
                original_size.height,
                original_size.width,
            ),
            dtype=torch.bool,
            device=masks.device,
        )
        mask_canvas[
            :,
            static_crop_offset.offset_y : static_crop_offset.offset_y + masks.shape[1],
            static_crop_offset.offset_x : static_crop_offset.offset_x + masks.shape[2],
        ] = masks
        image_bboxes[:, 0:4:2].add_(static_crop_offset.offset_x)
        image_bboxes[:, 1:4:2].add_(static_crop_offset.offset_y)
        masks = mask_canvas
    return image_bboxes, masks
