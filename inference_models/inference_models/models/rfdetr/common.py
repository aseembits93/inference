"""Shared post-processing entry point for RF-DETR (detection + segmentation).

Fast paths
----------
Three dispatch tiers, gated by env vars (see
``inference_models/docs/how-to/environment-variables.md#rf-detr``):

1. ``RFDETR_TRITON_FULLPOSTPROC=1`` — full fused post-TRT pipeline (two
   Triton kernels: filter + mask upsample). Only eligible for batch=1,
   ``STRETCH_TO``-equivalent resize, no static crop, and with class remapping;
   falls through otherwise. See ``_fullpost_eligible`` below.
2. ``RFDETR_TRITON_POSTPROC=1`` — fused sigmoid+argmax+confidence filter on
   GPU; remaining ops stay in torch.
3. Default — torch reference path.

InstanceDetections side-channel contract (fullpost path only)
-------------------------------------------------------------
When the full-postproc path fires it returns an ``InstanceDetections`` whose
public fields are views over unsliced GPU buffers plus three private markers
attached via ``__dict__``:

``_combined_gpu``  (num_queries, 6) int32, unsliced
    Per-slot record laid out as ``[x1, y1, x2, y2, conf_bits_i32, class_id]``.
    Valid slots are ``[0, n_survivors)`` where ``n_survivors`` is read from
    ``_counter_gpu``. The ``confidence`` column is fp32 bits stored as int32;
    the host extracts it via ``numpy.view(np.float32)``.

``_counter_gpu``  (1,) int32
    Atomic counter written by the filter kernel. The consumer DtoHs it into
    a pinned host buffer to learn ``n_survivors`` without a CPU-blocking
    ``counter.item()`` on the postproc stream.

``_postproc_done_event``  ``torch.cuda.Event``
    Recorded on the postproc stream after both Triton kernels are queued.
    Cross-stream consumers must ``event.wait(consumer_stream)`` before
    reading the buffers.

Consumer: ``inference/core/models/inference_models_adapters.py`` (the
``InferenceModelsInstanceSegmentationAdapter.postprocess`` fast path).
"""
import os
from typing import List, Optional, Tuple, Union

import torch
from torchvision.transforms import functional

from inference_models import InstanceDetections

_RFDETR_TRITON_POSTPROC = os.getenv("RFDETR_TRITON_POSTPROC", "false").lower() in (
    "true",
    "1",
)
if _RFDETR_TRITON_POSTPROC:
    try:
        from inference_models.models.rfdetr.triton_postprocess import (
            TRITON_AVAILABLE as _TRITON_POSTPROC_AVAILABLE,
            triton_rfdetr_conf_filter,
        )
        _TRITON_POSTPROC_READY = _TRITON_POSTPROC_AVAILABLE and torch.cuda.is_available()
    except Exception:
        _TRITON_POSTPROC_READY = False
        triton_rfdetr_conf_filter = None
else:
    _TRITON_POSTPROC_READY = False
    triton_rfdetr_conf_filter = None

_RFDETR_TRITON_FULLPOSTPROC = os.getenv("RFDETR_TRITON_FULLPOSTPROC", "false").lower() in (
    "true",
    "1",
)
if _RFDETR_TRITON_FULLPOSTPROC:
    try:
        from inference_models.models.rfdetr.triton_fullpostproc import (
            TRITON_AVAILABLE as _TRITON_FULLPOST_AVAILABLE,
            triton_rfdetr_fullpost,
        )
        _TRITON_FULLPOST_READY = _TRITON_FULLPOST_AVAILABLE and torch.cuda.is_available()
    except Exception:
        _TRITON_FULLPOST_READY = False
        triton_rfdetr_fullpost = None
else:
    _TRITON_FULLPOST_READY = False
    triton_rfdetr_fullpost = None
from inference_models.entities import ImageDimensions
from inference_models.errors import CorruptedModelPackageError
from inference_models.models.common.roboflow.model_packages import (
    PreProcessingMetadata,
    StaticCropOffset,
)
from inference_models.models.common.roboflow.post_processing import (
    align_instance_segmentation_results,
)
from inference_models.models.rfdetr.class_remapping import ClassesReMapping
from inference_models.utils.file_system import read_json


def parse_model_type(config_path: str) -> str:
    try:
        parsed_config = read_json(path=config_path)
        if not isinstance(parsed_config, dict):
            raise ValueError(
                f"decoded value is {type(parsed_config)}, but dictionary expected"
            )
        if "model_type" not in parsed_config or not isinstance(
            parsed_config["model_type"], str
        ):
            raise ValueError(
                "could not find required entries in config - either "
                "'model_type' field is missing or not a string"
            )
        return parsed_config["model_type"]
    except (IOError, OSError, ValueError) as error:
        raise CorruptedModelPackageError(
            message=f"Model type config file is malformed: "
            f"{error}. In case that the package is "
            f"hosted on the Roboflow platform - contact support. If you created model package manually, please "
            f"verify its consistency in docs.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        ) from error


def _fullpost_eligible(
    bboxes: torch.Tensor,
    pre_processing_meta: List[PreProcessingMetadata],
    classes_re_mapping: Optional[ClassesReMapping],
) -> bool:
    """True iff the full-fusion Triton post-process can handle this call.

    Each condition exists because the fused kernel assumes it:
    * batch=1 — the filter kernel grid is per-query for a single image.
    * Single preprocess meta — same reason as above.
    * ``nonsquare_intermediate_size is None`` — the kernel denormalizes bbox
      percents by a single ``inference_size`` (W, H) pair; a separate
      intermediate size would need a second scale application.
    * Zero static-crop offsets — the bbox math does not account for an
      origin offset into the original frame.
    * Class remapping present — the kernel unconditionally writes the
      remapped class id; without a mapping it would emit raw DETR class ids.
    """
    if not _TRITON_FULLPOST_READY or not bboxes.is_cuda:
        return False
    if bboxes.shape[0] != 1 or len(pre_processing_meta) != 1:
        return False
    meta = pre_processing_meta[0]
    if meta.nonsquare_intermediate_size is not None:
        return False
    if meta.static_crop_offset.offset_x != 0 or meta.static_crop_offset.offset_y != 0:
        return False
    if classes_re_mapping is None:
        return False
    return True


def post_process_instance_segmentation_results(
    bboxes: torch.Tensor,
    logits: torch.Tensor,
    masks: torch.Tensor,
    pre_processing_meta: List[PreProcessingMetadata],
    threshold: Union[float, torch.Tensor],
    num_classes: int,
    classes_re_mapping: Optional[ClassesReMapping],
) -> List[InstanceDetections]:
    results = []
    device = bboxes.device
    if _fullpost_eligible(bboxes, pre_processing_meta, classes_re_mapping):
        meta = pre_processing_meta[0]
        thr_arg = threshold if isinstance(threshold, torch.Tensor) else float(threshold)
        combined, mask_bin, mask_any, counter, done_event = triton_rfdetr_fullpost(
            bboxes=bboxes,
            logits=logits,
            masks=masks,
            threshold=thr_arg,
            num_classes=num_classes,
            class_mapping=classes_re_mapping.class_mapping,
            inference_size_wh=(meta.inference_size.width, meta.inference_size.height),
            pad_ltrb=(meta.pad_left, meta.pad_top, meta.pad_right, meta.pad_bottom),
            scale_wh=(meta.scale_width, meta.scale_height),
            orig_size_wh=(meta.original_size.width, meta.original_size.height),
        )
        detections = InstanceDetections(
            xyxy=combined[:, :4],
            confidence=combined[:, 4],  # int32 bits; adapter bitcasts to fp32
            class_id=combined[:, 5],
            mask=mask_bin,
        )
        # Contract: see module docstring (``_combined_gpu`` / ``_counter_gpu``
        # / ``_postproc_done_event``).
        detections.__dict__["_combined_gpu"] = combined
        detections.__dict__["_counter_gpu"] = counter
        detections.__dict__["_postproc_done_event"] = done_event
        results.append(detections)
        return results

    use_triton = _TRITON_POSTPROC_READY and bboxes.is_cuda
    # The Triton conf-filter expects the threshold in the logits' dtype; the
    # torch path operates on ``sigmoid(logits)``, which is still in the logits'
    # dtype, so the two paths can share the same conversion.
    if isinstance(threshold, torch.Tensor):
        threshold = threshold.to(device=device, dtype=logits.dtype)
    if use_triton:
        iterator = zip(bboxes, logits, masks, pre_processing_meta)
    else:
        logits_sigmoid = torch.nn.functional.sigmoid(logits)
        iterator = zip(bboxes, logits_sigmoid, masks, pre_processing_meta)
    cmap = classes_re_mapping.class_mapping if classes_re_mapping is not None else None
    for image_bboxes, image_logits, image_masks, image_meta in iterator:
        if use_triton:
            confidence, top_classes, keep = triton_rfdetr_conf_filter(
                image_logits, threshold, num_classes, class_mapping=cmap
            )
            confidence = confidence[keep]
            top_classes = top_classes[keep].long()
            selected_boxes = image_bboxes[keep]
            selected_masks = image_masks[keep]
        else:
            confidence, top_classes = image_logits.max(dim=1)
            if classes_re_mapping is not None:
                remapping_mask = torch.isin(
                    top_classes, classes_re_mapping.remaining_class_ids
                )
                top_classes = classes_re_mapping.class_mapping[top_classes[remapping_mask]]
                confidence = confidence[remapping_mask]
                image_bboxes = image_bboxes[remapping_mask]
                image_masks = image_masks[remapping_mask]
            else:
                # drop DETR no-object rows
                named = top_classes < num_classes
                confidence = confidence[named]
                top_classes = top_classes[named]
                image_bboxes = image_bboxes[named]
                image_masks = image_masks[named]
            confidence_mask = confidence > (threshold[top_classes.long()] if isinstance(threshold, torch.Tensor) else threshold)
            confidence = confidence[confidence_mask]
            top_classes = top_classes[confidence_mask]
            selected_boxes = image_bboxes[confidence_mask]
            selected_masks = image_masks[confidence_mask]
        confidence, sorted_indices = torch.sort(confidence, descending=True)
        top_classes = top_classes[sorted_indices]
        selected_boxes = selected_boxes[sorted_indices]
        selected_masks = selected_masks[sorted_indices]
        cxcy = selected_boxes[:, :2]
        wh = selected_boxes[:, 2:]
        xy_min = cxcy - 0.5 * wh
        xy_max = cxcy + 0.5 * wh
        selected_boxes_xyxy_pct = torch.cat([xy_min, xy_max], dim=-1)
        denorm_size = (
            image_meta.nonsquare_intermediate_size or image_meta.inference_size
        )
        denorm_size_whwh = torch.tensor(
            [
                denorm_size.width,
                denorm_size.height,
                denorm_size.width,
                denorm_size.height,
            ],
            device=device,
        )
        padding = (
            image_meta.pad_left,
            image_meta.pad_top,
            image_meta.pad_right,
            image_meta.pad_bottom,
        )
        selected_boxes_xyxy = selected_boxes_xyxy_pct * denorm_size_whwh
        aligned_boxes, aligned_masks = align_instance_segmentation_results(
            image_bboxes=selected_boxes_xyxy,
            masks=selected_masks,
            padding=padding,
            scale_height=image_meta.scale_height,
            scale_width=image_meta.scale_width,
            original_size=image_meta.original_size,
            size_after_pre_processing=image_meta.size_after_pre_processing,
            inference_size=denorm_size,
            static_crop_offset=image_meta.static_crop_offset,
        )
        detections = InstanceDetections(
            xyxy=aligned_boxes.round().int(),
            confidence=confidence,
            class_id=top_classes.int(),
            mask=aligned_masks,
        )
        results.append(detections)
    return results
