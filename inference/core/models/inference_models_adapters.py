import base64
import io
import os
from io import BytesIO
from time import perf_counter
from typing import Any, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
import supervision as sv
import torch
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils
from supervision.config import CLASS_NAME_DATA_FIELD

from inference.core.entities.requests import (
    ClassificationInferenceRequest,
    InferenceRequest,
)
from inference.core.entities.responses.inference import (
    ClassificationInferenceResponse,
    InferenceResponse,
    InferenceResponseImage,
    InferenceResponseImageDC,
    INSTANCE_SEGMENTATION_WORKFLOW_V3_FAST_SOURCE,
    InstanceSegmentationInferenceResponse,
    InstanceSegmentationInferenceResponseDC,
    InstanceSegmentationPrediction,
    InstanceSegmentationRLEPrediction,
    Keypoint,
    KeypointsDetectionInferenceResponse,
    KeypointsPrediction,
    MultiLabelClassificationInferenceResponse,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
    Point,
    SemanticSegmentationInferenceResponse,
    SemanticSegmentationPrediction,
)
from inference.core.env import (
    ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
    ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
    API_KEY,
    DISABLED_INFERENCE_MODELS_BACKENDS,
    RFDETR_ONNX_MAX_RESOLUTION,
    VALID_INFERENCE_MODELS_BACKENDS,
)
from inference.core.exceptions import PostProcessingError
from inference.core.models.base import Model
from inference.core.roboflow_api import get_extra_weights_provider_headers
from inference.core.utils.image_utils import load_image_bgr, load_image_rgb
from inference.core.utils.postprocess import bitpacked_masks2poly, mask2poly, masks2poly
from inference.core.utils.visualisation import draw_detection_predictions
from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    PREDICTION_TYPE_KEY,
)
from inference.models.aliases import resolve_roboflow_model_alias
from inference_models import (
    AutoModel,
    ClassificationModel,
    ClassificationPrediction,
    Detections,
    InstanceDetections,
    InstanceSegmentationModel,
    KeyPoints,
    KeyPointsDetectionModel,
    MultiLabelClassificationModel,
    MultiLabelClassificationPrediction,
    ObjectDetectionModel,
    PreProcessingOverrides,
    SemanticSegmentationModel,
)
from inference_models.models.base.instance_segmentation import InferenceFuture
from inference_models.models.base.semantic_segmentation import (
    SemanticSegmentationResult,
)
from inference_models.models.base.types import InstancesRLEMasks, PreprocessingMetadata
from inference_models.models.common.rle_utils import (
    coco_rle_masks_to_numpy_mask,
    torch_mask_to_coco_rle,
)

DEFAULT_COLOR_PALETTE = [
    "#A351FB",
    "#FF4040",
    "#FFA1A0",
    "#FF7633",
    "#FFB633",
    "#D1D435",
    "#4CFB12",
    "#94CF1A",
    "#40DE8A",
    "#1B9640",
    "#00D6C1",
    "#2E9CAA",
    "#00C4FF",
    "#364797",
    "#6675FF",
    "#0019EF",
    "#863AFF",
    "#530087",
    "#CD3AFF",
    "#FF97CA",
    "#FF39C9",
]

# Pinned host buffers for async DtoH on the full-postproc Triton fast path.
# Keyed by (name, dtype); reused across frames provided the cached buffer is
# at least as large as the requested shape in every dimension.
PINNED_HOST_BUFFERS: dict = {}


def get_pinned_buffer(name: str, shape, dtype: torch.dtype) -> torch.Tensor:
    key = (name, dtype)
    buf = PINNED_HOST_BUFFERS.get(key)
    if buf is not None and all(buf.shape[i] >= shape[i] for i in range(len(shape))):
        return buf[tuple(slice(0, s) for s in shape)]
    buf = torch.empty(shape, dtype=dtype, pin_memory=True)
    PINNED_HOST_BUFFERS[key] = buf
    return buf


def _is_instance_segmentation_workflow_v3_fast_source(source: Optional[str]) -> bool:
    return source == INSTANCE_SEGMENTATION_WORKFLOW_V3_FAST_SOURCE


def _numpy_masks_to_bool_view(masks: np.ndarray) -> np.ndarray:
    masks_np = np.asarray(masks)
    if masks_np.dtype == np.bool_:
        return masks_np
    if masks_np.dtype == np.uint8:
        return masks_np.view(np.bool_)
    return masks_np.astype(bool, copy=False)


class _LazyWorkflowDetectionsData(dict):
    def __init__(self) -> None:
        super().__init__()
        self._owner = None

    def bind(self, owner: "LazyWorkflowSVDetections") -> None:
        self._owner = owner

    def __getitem__(self, key):
        if key not in self and self._owner is not None:
            self._owner._ensure_data_key_materialized(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key not in self and self._owner is not None:
            self._owner._ensure_data_key_materialized(key)
        return super().get(key, default)

    def __contains__(self, key):
        return super().__contains__(key)


class LazyWorkflowSVDetections(sv.Detections):
    """Delay CPU materialization of RF-DETR workflow detections until accessed."""

    def __init__(
        self,
        *,
        combined_gpu: torch.Tensor,
        counter_gpu: Optional[torch.Tensor],
        mask_gpu: Optional[torch.Tensor],
        mask_packed_gpu: Optional[torch.Tensor],
        selection_done_event: Optional["torch.cuda.Event"],
        postproc_done_event: Optional["torch.cuda.Event"],
        class_names: List[str],
        image_height: int,
        image_width: int,
    ) -> None:
        placeholder_count = int(combined_gpu.shape[0])
        data_proxy = _LazyWorkflowDetectionsData()
        object.__setattr__(self, "_lazy_ready", False)
        super().__init__(
            xyxy=np.empty((placeholder_count, 4), dtype=np.float32),
            mask=None,
            confidence=np.empty((placeholder_count,), dtype=np.float32),
            class_id=np.empty((placeholder_count,), dtype=np.int64),
            data=data_proxy,
        )
        data_proxy.bind(self)
        self._placeholder_count = placeholder_count
        self._materialized_count: Optional[int] = None
        self._boxes_materialized = False
        self._mask_materialized = False
        self._combined_gpu = combined_gpu
        self._counter_gpu = counter_gpu
        self._mask_gpu = mask_gpu
        self._mask_packed_gpu = mask_packed_gpu
        self._selection_done_event = (
            selection_done_event
            if selection_done_event is not None
            else postproc_done_event
        )
        self._postproc_done_event = (
            postproc_done_event
            if postproc_done_event is not None
            else selection_done_event
        )
        self._class_names = class_names
        self._image_height = image_height
        self._image_width = image_width
        object.__getattribute__(self, "data")[IMAGE_DIMENSIONS_KEY] = np.broadcast_to(
            np.array([image_height, image_width], dtype=np.int32),
            (placeholder_count, 2),
        )
        object.__setattr__(self, "_lazy_ready", True)

    def _fast_len_hint(self) -> int:
        return self._placeholder_count

    def _trim_existing_data(self, n_detections: int) -> None:
        data = object.__getattribute__(self, "data")
        placeholder_count = object.__getattribute__(self, "_placeholder_count")
        for key, value in list(data.items()):
            if isinstance(value, np.ndarray) and value.shape[0] == placeholder_count:
                data[key] = value[:n_detections]
            elif isinstance(value, list) and len(value) == placeholder_count:
                data[key] = value[:n_detections]

    def _ensure_count_materialized(self) -> int:
        cached = object.__getattribute__(self, "_materialized_count")
        if cached is not None:
            return cached
        counter_gpu = object.__getattribute__(self, "_counter_gpu")
        if counter_gpu is None:
            n_detections = object.__getattribute__(self, "_placeholder_count")
        else:
            device = counter_gpu.device
            stream = torch.cuda.current_stream(device)
            ready_event = object.__getattribute__(self, "_selection_done_event")
            if ready_event is not None:
                ready_event.wait(stream)
            counter_host = get_pinned_buffer("lazy_counter", (1,), counter_gpu.dtype)
            counter_host.copy_(counter_gpu, non_blocking=True)
            stream.synchronize()
            n_detections = int(counter_host.numpy()[0])
            n_detections = max(
                0,
                min(n_detections, object.__getattribute__(self, "_placeholder_count")),
            )
        object.__setattr__(self, "_materialized_count", n_detections)
        self._trim_existing_data(n_detections)
        return n_detections

    def _ensure_boxes_materialized(self) -> None:
        if object.__getattribute__(self, "_boxes_materialized"):
            return
        n_detections = self._ensure_count_materialized()
        if n_detections == 0:
            object.__setattr__(self, "xyxy", np.empty((0, 4), dtype=np.float32))
            object.__setattr__(self, "confidence", np.empty((0,), dtype=np.float32))
            object.__setattr__(self, "class_id", np.empty((0,), dtype=np.int64))
        else:
            combined_gpu = object.__getattribute__(self, "_combined_gpu")
            combined_slice = combined_gpu[:n_detections]
            device = combined_slice.device
            stream = torch.cuda.current_stream(device)
            ready_event = object.__getattribute__(self, "_selection_done_event")
            if ready_event is not None:
                ready_event.wait(stream)
            combined_host = get_pinned_buffer(
                "lazy_combined",
                tuple(combined_slice.shape),
                combined_slice.dtype,
            )
            combined_host.copy_(combined_slice, non_blocking=True)
            stream.synchronize()
            combined_np = combined_host.numpy()
            object.__setattr__(
                self,
                "xyxy",
                np.array(combined_np[:, :4], dtype=np.float32, copy=True),
            )
            object.__setattr__(
                self,
                "confidence",
                combined_np[:, 4].view(np.float32).copy(),
            )
            object.__setattr__(
                self,
                "class_id",
                np.array(combined_np[:, 5], dtype=np.int64, copy=True),
            )
        object.__setattr__(self, "_boxes_materialized", True)
        self._ensure_image_dimensions_data()
        self._ensure_detection_ids_data()

    def _ensure_mask_materialized(self) -> None:
        if object.__getattribute__(self, "_mask_materialized"):
            return
        n_detections = self._ensure_count_materialized()
        image_height = object.__getattribute__(self, "_image_height")
        image_width = object.__getattribute__(self, "_image_width")
        if n_detections == 0:
            mask = np.empty((0, image_height, image_width), dtype=bool)
        else:
            mask_packed_gpu = object.__getattribute__(self, "_mask_packed_gpu")
            if isinstance(mask_packed_gpu, torch.Tensor):
                packed_slice = mask_packed_gpu[:n_detections]
                device = packed_slice.device
                stream = torch.cuda.current_stream(device)
                ready_event = object.__getattribute__(self, "_postproc_done_event")
                if ready_event is not None:
                    ready_event.wait(stream)
                packed_host = get_pinned_buffer(
                    "lazy_mask_packed",
                    tuple(packed_slice.shape),
                    packed_slice.dtype,
                )
                packed_host.copy_(packed_slice, non_blocking=True)
                stream.synchronize()
                packed_np = packed_host.numpy()
                mask = _numpy_masks_to_bool_view(
                    np.unpackbits(
                        packed_np, axis=-1, bitorder="little"
                    )[..., :image_width]
                )
            else:
                mask_gpu = object.__getattribute__(self, "_mask_gpu")
                mask_slice = mask_gpu[:n_detections]
                device = mask_slice.device
                stream = torch.cuda.current_stream(device)
                ready_event = object.__getattribute__(self, "_postproc_done_event")
                if ready_event is not None:
                    ready_event.wait(stream)
                mask_host = get_pinned_buffer(
                    "lazy_mask",
                    tuple(mask_slice.shape),
                    mask_slice.dtype,
                )
                mask_host.copy_(mask_slice, non_blocking=True)
                stream.synchronize()
                mask = _numpy_masks_to_bool_view(mask_host.numpy())
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "_mask_materialized", True)

    def _ensure_image_dimensions_data(self) -> None:
        data = object.__getattribute__(self, "data")
        if IMAGE_DIMENSIONS_KEY in data:
            return
        n_detections = self._ensure_count_materialized()
        data[IMAGE_DIMENSIONS_KEY] = np.broadcast_to(
            np.array(
                [
                    object.__getattribute__(self, "_image_height"),
                    object.__getattribute__(self, "_image_width"),
                ],
                dtype=np.int32,
            ),
            (n_detections, 2),
        )

    def _ensure_detection_ids_data(self) -> None:
        data = object.__getattribute__(self, "data")
        if DETECTION_ID_KEY in data:
            return
        n_detections = self._ensure_count_materialized()
        data[DETECTION_ID_KEY] = np.asarray(
            [str(uuid4()) for _ in range(n_detections)],
            dtype=object,
        )

    def _ensure_class_names_data(self) -> None:
        data = object.__getattribute__(self, "data")
        if CLASS_NAME_DATA_FIELD in data:
            return
        self._ensure_boxes_materialized()
        class_names = object.__getattribute__(self, "_class_names")
        class_ids = object.__getattribute__(self, "class_id")
        data[CLASS_NAME_DATA_FIELD] = np.asarray(
            [class_names[int(class_id)] for class_id in class_ids],
            dtype=object,
        )

    def _ensure_data_key_materialized(self, key: str) -> None:
        if key == CLASS_NAME_DATA_FIELD:
            self._ensure_class_names_data()
        elif key == DETECTION_ID_KEY:
            self._ensure_detection_ids_data()
        elif key == IMAGE_DIMENSIONS_KEY:
            self._ensure_image_dimensions_data()

    def _mask_shape_matches_root_dimensions_fast(self) -> bool:
        return True

    def _shallow_clone_fast(self) -> "LazyWorkflowSVDetections":
        placeholder_count = object.__getattribute__(self, "_placeholder_count")
        data_proxy = _LazyWorkflowDetectionsData()
        clone = object.__new__(LazyWorkflowSVDetections)
        object.__setattr__(clone, "_lazy_ready", False)
        sv.Detections.__init__(
            clone,
            xyxy=np.empty((placeholder_count, 4), dtype=np.float32),
            mask=None,
            confidence=np.empty((placeholder_count,), dtype=np.float32),
            class_id=np.empty((placeholder_count,), dtype=np.int64),
            data=data_proxy,
            metadata=dict(object.__getattribute__(self, "metadata")),
        )
        data_proxy.bind(clone)
        object.__setattr__(
            clone,
            "xyxy",
            object.__getattribute__(self, "__dict__").get("xyxy"),
        )
        object.__setattr__(
            clone,
            "confidence",
            object.__getattribute__(self, "__dict__").get("confidence"),
        )
        object.__setattr__(
            clone,
            "class_id",
            object.__getattribute__(self, "__dict__").get("class_id"),
        )
        object.__setattr__(
            clone,
            "mask",
            object.__getattribute__(self, "__dict__").get("mask"),
        )
        for attr in (
            "_placeholder_count",
            "_materialized_count",
            "_boxes_materialized",
            "_mask_materialized",
            "_combined_gpu",
            "_counter_gpu",
            "_mask_gpu",
            "_mask_packed_gpu",
            "_selection_done_event",
            "_postproc_done_event",
            "_class_names",
            "_image_height",
            "_image_width",
        ):
            object.__setattr__(clone, attr, object.__getattribute__(self, attr))
        clone_data = object.__getattribute__(clone, "data")
        for key, value in object.__getattribute__(self, "data").items():
            clone_data[key] = value
        object.__setattr__(clone, "_lazy_ready", True)
        return clone

    def __getattribute__(self, name: str):
        if name == "_lazy_ready" or not object.__getattribute__(self, "_lazy_ready"):
            return object.__getattribute__(self, name)
        if name in {"xyxy", "confidence", "class_id"}:
            object.__getattribute__(self, "_ensure_boxes_materialized")()
        elif name == "mask":
            object.__getattribute__(self, "_ensure_mask_materialized")()
        return object.__getattribute__(self, name)

    def __len__(self) -> int:
        if not object.__getattribute__(self, "_lazy_ready"):
            return super().__len__()
        return self._ensure_count_materialized()

    def __getitem__(self, index):
        if not object.__getattribute__(self, "_lazy_ready"):
            return super().__getitem__(index)
        if isinstance(index, str):
            return object.__getattribute__(self, "data").get(index)
        self._ensure_boxes_materialized()
        self._ensure_mask_materialized()
        self._ensure_class_names_data()
        self._ensure_detection_ids_data()
        self._ensure_image_dimensions_data()
        return super().__getitem__(index)

    def __iter__(self):
        if not object.__getattribute__(self, "_lazy_ready"):
            return super().__iter__()
        self._ensure_boxes_materialized()
        self._ensure_mask_materialized()
        self._ensure_class_names_data()
        self._ensure_detection_ids_data()
        self._ensure_image_dimensions_data()
        return super().__iter__()


def _empty_workflow_sv_detections(height: int, width: int) -> sv.Detections:
    return sv.Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        mask=np.empty((0, height, width), dtype=bool),
        confidence=np.empty((0,), dtype=np.float32),
        class_id=np.empty((0,), dtype=np.int64),
        data={
            CLASS_NAME_DATA_FIELD: np.empty((0,), dtype=object),
            DETECTION_ID_KEY: np.empty((0,), dtype=object),
            IMAGE_DIMENSIONS_KEY: np.empty((0, 2), dtype=np.int32),
            PREDICTION_TYPE_KEY: np.empty((0,), dtype=object),
        },
    )


def _build_workflow_sv_detections(
    *,
    xyxy: np.ndarray,
    confs: np.ndarray,
    class_ids: np.ndarray,
    masks: np.ndarray,
    class_names: List[str],
    image_height: int,
    image_width: int,
    copy_masks: bool = True,
) -> sv.Detections:
    n_detections = int(class_ids.shape[0])
    if n_detections == 0:
        return _empty_workflow_sv_detections(height=image_height, width=image_width)
    image_dimensions = np.broadcast_to(
        np.array([image_height, image_width], dtype=np.int32),
        (n_detections, 2),
    )
    return sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32, copy=True),
        mask=np.array(masks, dtype=bool, copy=copy_masks),
        confidence=np.array(confs, dtype=np.float32, copy=True),
        class_id=np.array(class_ids, dtype=np.int64, copy=True),
        data={
            CLASS_NAME_DATA_FIELD: np.asarray(class_names, dtype=object),
            DETECTION_ID_KEY: np.asarray(
                [str(uuid4()) for _ in range(n_detections)], dtype=object
            ),
            IMAGE_DIMENSIONS_KEY: image_dimensions,
            PREDICTION_TYPE_KEY: np.full(
                (n_detections,), "instance-segmentation", dtype=object
            ),
        },
    )


class _PipelinePrimingSentinel:
    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return "<_PIPELINE_PRIMING>"


_PIPELINE_PRIMING = _PipelinePrimingSentinel()


class InferenceModelsObjectDetectionAdapter(Model):
    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY
        model_id = resolve_roboflow_model_alias(model_id=model_id)

        self.task_type = "object-detection"

        extra_weights_provider_headers = get_extra_weights_provider_headers(
            countinference=kwargs.get("countinference"),
            service_secret=kwargs.get("service_secret"),
        )
        backend = list(
            VALID_INFERENCE_MODELS_BACKENDS.difference(
                DISABLED_INFERENCE_MODELS_BACKENDS
            )
        )
        self._model: ObjectDetectionModel = AutoModel.from_pretrained(
            model_id_or_path=model_id,
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            weights_provider_extra_headers=extra_weights_provider_headers,
            backend=backend,
            rf_detr_max_input_resolution=RFDETR_ONNX_MAX_RESOLUTION,
            **kwargs,
        )
        self.class_names = list(self._model.class_names)

    def map_inference_kwargs(self, kwargs: dict) -> dict:
        kwargs["input_color_format"] = "bgr"
        pre_processing_overrides = PreProcessingOverrides(
            disable_contrast_enhancement=kwargs.get("disable_preproc_contrast", False),
            disable_grayscale=kwargs.get("disable_preproc_grayscale", False),
            disable_static_crop=kwargs.get("disable_preproc_static_crop", False),
        )
        kwargs["pre_processing_overrides"] = pre_processing_overrides
        return kwargs

    def preprocess(self, image: Any, **kwargs):
        is_batch = isinstance(image, list)
        images = image if is_batch else [image]
        np_images: List[np.ndarray] = [
            load_image_bgr(
                v,
                disable_preproc_auto_orient=kwargs.get(
                    "disable_preproc_auto_orient", False
                ),
            )
            for v in images
        ]
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.pre_process(np_images, **mapped_kwargs)

    def predict(self, img_in, **kwargs):
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.forward(img_in, **mapped_kwargs)

    def postprocess(
        self,
        predictions: List[Detections],
        preprocess_return_metadata: PreprocessingMetadata,
        **kwargs,
    ) -> List[ObjectDetectionInferenceResponse]:
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        detections_list = self._model.post_process(
            predictions, preprocess_return_metadata, **mapped_kwargs
        )

        responses: List[ObjectDetectionInferenceResponse] = []
        for preproc_metadata, det in zip(preprocess_return_metadata, detections_list):
            H = preproc_metadata.original_size.height
            W = preproc_metadata.original_size.width

            xyxy = det.xyxy.detach().cpu().numpy()
            confs = det.confidence.detach().cpu().numpy()
            class_ids = det.class_id.detach().cpu().numpy()

            predictions: List[ObjectDetectionPrediction] = []

            for (x1, y1, x2, y2), conf, class_id in zip(xyxy, confs, class_ids):
                cx = (float(x1) + float(x2)) / 2.0
                cy = (float(y1) + float(y2)) / 2.0
                w = float(x2) - float(x1)
                h = float(y2) - float(y1)
                class_id_int = int(class_id)
                class_name = (
                    self.class_names[class_id_int]
                    if 0 <= class_id_int < len(self.class_names)
                    else str(class_id_int)
                )
                if (
                    kwargs.get("class_filter")
                    and class_name not in kwargs["class_filter"]
                ):
                    continue
                predictions.append(
                    ObjectDetectionPrediction(
                        x=cx,
                        y=cy,
                        width=w,
                        height=h,
                        confidence=float(conf),
                        **{"class": class_name},
                        class_id=class_id_int,
                    )
                )

            responses.append(
                ObjectDetectionInferenceResponse(
                    predictions=predictions,
                    image=InferenceResponseImage(width=W, height=H),
                )
            )
        return responses

    def clear_cache(self, delete_from_disk: bool = True) -> None:
        """Clears any cache if necessary. TODO: Implement this to delete the cache from the experimental model.

        Args:
            delete_from_disk (bool, optional): Whether to delete cached files from disk. Defaults to True.
        """
        pass

    def draw_predictions(
        self,
        inference_request: InferenceRequest,
        inference_response: InferenceResponse,
    ) -> bytes:
        """Draw predictions from an inference response onto the original image provided by an inference request

        Args:
            inference_request (ObjectDetectionInferenceRequest): The inference request containing the image on which to draw predictions
            inference_response (ObjectDetectionInferenceResponse): The inference response containing predictions to be drawn

        Returns:
            str: A base64 encoded image string
        """
        class_id_2_color = {
            i: DEFAULT_COLOR_PALETTE[i % len(DEFAULT_COLOR_PALETTE)]
            for i, class_name in enumerate(self._model.class_names)
        }
        return draw_detection_predictions(
            inference_request=inference_request,
            inference_response=inference_response,
            colors=class_id_2_color,
        )


class InferenceModelsInstanceSegmentationAdapter(Model):
    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY
        model_id = resolve_roboflow_model_alias(model_id=model_id)

        self.task_type = "instance-segmentation"

        extra_weights_provider_headers = get_extra_weights_provider_headers(
            countinference=kwargs.get("countinference"),
            service_secret=kwargs.get("service_secret"),
        )
        backend = list(
            VALID_INFERENCE_MODELS_BACKENDS.difference(
                DISABLED_INFERENCE_MODELS_BACKENDS
            )
        )
        self._model: InstanceSegmentationModel = AutoModel.from_pretrained(
            model_id_or_path=model_id,
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            weights_provider_extra_headers=extra_weights_provider_headers,
            backend=backend,
            rf_detr_max_input_resolution=RFDETR_ONNX_MAX_RESOLUTION,
            **kwargs,
        )
        self.class_names = list(self._model.class_names)
        # Two-stage pipelining: depth=1 means original synchronous behavior
        # (preprocess→forward→postprocess on each frame, in order); depth=2
        # overlaps frame N+1's preprocess+forward with frame N's postprocess
        # decode by stashing a future and deferring CPU-side response build
        # by one frame. depth=2 requires that callers accept a one-frame
        # priming latency at stream start and call `flush()` at stream end.
        self._pipeline_depth = max(
            1, int(os.getenv("RFDETR_PIPELINE_DEPTH", "1"))
        )
        # Per-adapter in-flight future + metadata for the previous frame,
        # held across the (predict → postprocess) boundary of the current
        # frame. Not thread-safe; the InferencePipeline is single-producer
        # and the adapter is owned by a single worker.
        self._prev_future: Optional[InferenceFuture] = None
        self._prev_kwargs: Optional[dict] = None

    def map_inference_kwargs(self, kwargs: dict) -> dict:
        kwargs["input_color_format"] = "bgr"
        pre_processing_overrides = PreProcessingOverrides(
            disable_contrast_enhancement=kwargs.get("disable_preproc_contrast", False),
            disable_grayscale=kwargs.get("disable_preproc_grayscale", False),
            disable_static_crop=kwargs.get("disable_preproc_static_crop", False),
        )
        kwargs["pre_processing_overrides"] = pre_processing_overrides
        if "rle" in self._model.supported_mask_formats:
            kwargs["mask_format"] = "rle"
        kwargs["defer_count_to_adapter"] = (
            kwargs.get("response_mask_format") != "rle"
        )
        kwargs["defer_triton_preprocess_to_combined_graph"] = (
            self._pipeline_depth > 1 and kwargs["defer_count_to_adapter"]
        )
        return kwargs

    def preprocess(self, image: Any, **kwargs):
        is_batch = isinstance(image, list)
        images = image if is_batch else [image]
        np_images: List[np.ndarray] = [
            load_image_bgr(
                v,
                disable_preproc_auto_orient=kwargs.get(
                    "disable_preproc_auto_orient", False
                ),
            )
            for v in images
        ]
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.pre_process(np_images, **mapped_kwargs)

    def predict(self, img_in, **kwargs):
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        if self._pipeline_depth <= 1:
            # Original path: forward on current frame, postprocess on
            # current frame, all synchronous.
            return self._model.forward(img_in, **mapped_kwargs)

        # Depth-2 path: enqueue forward for the *current* frame onto the
        # inference stream (which releases the host immediately thanks to
        # the captured TRT CUDA graph), park the resulting future in
        # `_prev_future`, and return the *previous* future — which will be
        # decoded during this call's `postprocess`. The first frame of a
        # stream therefore returns a `_PrimingSentinel` that `postprocess`
        # recognises as "no output to emit yet". Callers are expected to
        # call `flush()` at stream end to drain the final pending future.
        #
        # BEFORE submitting frame N's forward, we eagerly enqueue frame
        # N-1's postprocess GPU kernels. This re-orders the device queue
        # so the ~200µs postproc runs BEFORE the next ~9.5ms forward —
        # without this, nsys traces show postproc kernels waiting ~2ms
        # (median, 10ms worst case) behind the forward on ~94% of frames,
        # and that wait reflects directly as host-side
        # stream.synchronize() latency inside postprocess. The change is
        # purely a GPU-scheduling reorder — all CPU-visible results still
        # happen in the same spots.
        prev = self._prev_future
        if prev is not None:
            # prev here is a `_DirectInferenceFuture`. We need the
            # metadata belonging to the frame that produced it, not the
            # current frame. The adapter keeps this in
            # `_pending_flush_meta_prev` (stashed by the previous
            # `postprocess` call). Pass it in so `post_process` can
            # compute box coordinates against the right image size.
            prev_meta = getattr(self, "_pending_flush_meta_prev", None)
            prev_adapter_kwargs = self._prev_kwargs
            if prev_meta is not None and prev_adapter_kwargs is not None:
                # Splice the correct meta + kwargs into the future so the
                # eager GPU submit happens with the right call state.
                prev._meta = prev_meta  # type: ignore[attr-defined]
                prev._kwargs = prev_adapter_kwargs.get("mapped_kwargs", {})  # type: ignore[attr-defined]
                submit = getattr(prev, "submit_gpu_work", None)
                if submit is not None:
                    submit(prev_meta)
        # NB: forward_async's meta arg is unused here because the adapter
        # carries preprocess metadata through `_pending_flush_meta_prev`
        # and splices it into the future inside `_finalize_future`. We only
        # need the future to hold the raw forward output + produce-event.
        fut = self._model.forward_async(img_in, None, **mapped_kwargs)
        prev_kwargs = self._prev_kwargs
        self._prev_future = fut
        self._prev_kwargs = {"mapped_kwargs": mapped_kwargs}
        if prev is None:
            return _PIPELINE_PRIMING
        # Stash previous call's mapped_kwargs on the future so postprocess
        # can reconstruct post_process args without depending on the
        # current frame's kwargs.
        prev._adapter_kwargs = prev_kwargs  # type: ignore[attr-defined]
        return prev

    def flush(self) -> List[InstanceSegmentationInferenceResponse]:
        """Drain the tail of the pipelined queue.

        Returns responses for any in-flight frames whose forward pass was
        submitted but whose postprocess has not yet been driven by a
        subsequent call to `postprocess`. Callers that use
        `RFDETR_PIPELINE_DEPTH>=2` MUST invoke this at stream end or the
        final frame will be dropped.
        """
        if self._pipeline_depth <= 1:
            return []
        fut = self._prev_future
        kw = self._prev_kwargs
        self._prev_future = None
        self._prev_kwargs = None
        if fut is None:
            return []
        # The future's preprocess metadata was passed in as `None` during
        # predict, so `post_process` has no size/offset info to work with.
        # That metadata must have been stashed by the caller before flush
        # — in the InferencePipeline path it lives on the adapter's
        # `_flush_meta` stack (populated during postprocess).
        meta = getattr(self, "_pending_flush_meta", None)
        self._pending_flush_meta = None
        if meta is None:
            return []
        return self._finalize_future(fut, meta, (kw or {}).get("mapped_kwargs", {}))

    def postprocess(
        self,
        predictions,
        preprocess_return_metadata: PreprocessingMetadata,
        **kwargs,
    ) -> List[InstanceSegmentationInferenceResponse]:
        if self._pipeline_depth <= 1:
            return self._postprocess_sync(
                predictions, preprocess_return_metadata, **kwargs
            )
        # Depth-2 path: `predictions` is either `_PIPELINE_PRIMING` (first
        # frame: emit empty list so the pipeline advances) or a prior
        # frame's InferenceFuture. Either way, stash the current frame's
        # preprocess metadata on the adapter so `flush()` can use it to
        # decode the in-flight future at stream end.
        self._pending_flush_meta = preprocess_return_metadata
        if predictions is _PIPELINE_PRIMING:
            # Stash the priming frame's metadata so the NEXT postprocess
            # (which will receive the priming frame's future) can decode
            # boxes into pixel coordinates that match this frame.
            self._pending_flush_meta_prev = preprocess_return_metadata
            # Return empty responses for the first frame so the stream
            # pipeline has something to dispatch. The real frame-0 output
            # arrives one frame later. Use the same response class the
            # downstream consumer is expecting (dataclass twin for the
            # workflow fast path, pydantic otherwise) so isinstance
            # dispatch in the v3 block picks the right decoder.
            if _is_instance_segmentation_workflow_v3_fast_source(
                kwargs.get("source")
            ):
                return [
                    InstanceSegmentationInferenceResponseDC(
                        predictions=[],
                        image=InferenceResponseImageDC(
                            width=m.original_size.width,
                            height=m.original_size.height,
                        ),
                        sv_detections=_empty_workflow_sv_detections(
                            height=m.original_size.height,
                            width=m.original_size.width,
                        ),
                    )
                    for m in preprocess_return_metadata
                ]
            return [
                InstanceSegmentationInferenceResponse(
                    predictions=[],
                    image=InferenceResponseImage(
                        width=m.original_size.width,
                        height=m.original_size.height,
                    ),
                )
                for m in preprocess_return_metadata
            ]
        fut: InferenceFuture = predictions
        # `preprocess_return_metadata` here corresponds to the *current*
        # frame but the future belongs to the *previous* frame. Use the
        # metadata that was stashed one call ago: we kept it as
        # `_pending_flush_meta_prev` from the previous postprocess.
        prev_meta = getattr(self, "_pending_flush_meta_prev", None)
        self._pending_flush_meta_prev = preprocess_return_metadata
        if prev_meta is None:
            # Should not happen under normal sequence (first postprocess
            # took the priming branch above), but be defensive.
            prev_meta = preprocess_return_metadata
        mapped_kwargs = getattr(fut, "_adapter_kwargs", {}).get(
            "mapped_kwargs", {}
        )
        return self._finalize_future(fut, prev_meta, mapped_kwargs)

    def _finalize_future(
        self,
        fut: InferenceFuture,
        preprocess_return_metadata: PreprocessingMetadata,
        mapped_kwargs: dict,
    ) -> List[InstanceSegmentationInferenceResponse]:
        # Override the future's stashed meta (which was `None` at submit
        # time) with the correct metadata for the frame whose forward pass
        # the future represents. This is an allowed private-surface tweak
        # because _DirectInferenceFuture's post_process is memoised.
        fut._meta = preprocess_return_metadata  # type: ignore[attr-defined]
        fut._kwargs = mapped_kwargs  # type: ignore[attr-defined]
        detections_list = fut.result()
        return self._build_responses_from_detections(
            detections_list, preprocess_return_metadata, **mapped_kwargs
        )

    def _postprocess_sync(
        self,
        predictions: List[InstanceDetections],
        preprocess_return_metadata: PreprocessingMetadata,
        **kwargs,
    ) -> List[InstanceSegmentationInferenceResponse]:
        return_in_rle = kwargs.get("response_mask_format") == "rle"
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        detections_list = self._model.post_process(
            predictions, preprocess_return_metadata, **mapped_kwargs
        )
        return self._build_responses_from_detections(
            detections_list, preprocess_return_metadata, **kwargs
        )

    def _build_responses_from_detections(
        self,
        detections_list: List[InstanceDetections],
        preprocess_return_metadata: PreprocessingMetadata,
        **kwargs,
    ) -> List[InstanceSegmentationInferenceResponse]:
        return_in_rle = kwargs.get("response_mask_format") == "rle"
        workflow_v3_fast_path = (
            _is_instance_segmentation_workflow_v3_fast_source(kwargs.get("source"))
            and not return_in_rle
        )
        class_filter = kwargs.get("class_filter")

        responses: List[InstanceSegmentationInferenceResponse] = []
        for preproc_metadata, det in zip(preprocess_return_metadata, detections_list):
            H = preproc_metadata.original_size.height
            W = preproc_metadata.original_size.width
            workflow_masks = None
            workflow_masks_copy = True

            combined_gpu = getattr(det, "_combined_gpu", None)
            mask_gpu = getattr(det, "_mask_gpu", None)
            mask_packed_gpu = getattr(det, "_mask_packed_gpu", None)
            mask_cpu = getattr(det, "_mask_cpu", None)
            defer_count_to_adapter = getattr(det, "_defer_count_to_adapter", False)
            done_event = getattr(det, "_postproc_done_event", None)
            dense_mask_cuda = isinstance(mask_gpu, torch.Tensor) and mask_gpu.is_cuda
            packed_mask_cuda = (
                isinstance(mask_packed_gpu, torch.Tensor) and mask_packed_gpu.is_cuda
            )
            if (
                not return_in_rle
                and done_event is not None
                and (dense_mask_cuda or packed_mask_cuda)
            ):
                if (
                    workflow_v3_fast_path
                    and defer_count_to_adapter
                    and isinstance(combined_gpu, torch.Tensor)
                    and combined_gpu.is_cuda
                ):
                    responses.append(
                        InstanceSegmentationInferenceResponseDC(
                            predictions=[],
                            image=InferenceResponseImageDC(width=W, height=H),
                            sv_detections=LazyWorkflowSVDetections(
                                combined_gpu=combined_gpu,
                                counter_gpu=getattr(det, "_counter_gpu", None),
                                mask_gpu=mask_gpu if dense_mask_cuda else None,
                                mask_packed_gpu=(
                                    mask_packed_gpu if packed_mask_cuda else None
                                ),
                                selection_done_event=getattr(
                                    det, "_selection_done_event", done_event
                                ),
                                postproc_done_event=done_event,
                                class_names=self.class_names,
                                image_height=H,
                                image_width=W,
                            ),
                        )
                    )
                    continue
                device = mask_gpu.device if dense_mask_cuda else mask_packed_gpu.device
                stream = torch.cuda.current_stream(device)
                done_event.wait(stream)

                if (
                    defer_count_to_adapter
                    and isinstance(combined_gpu, torch.Tensor)
                    and combined_gpu.is_cuda
                ):
                        combined_host = get_pinned_buffer(
                            "combined_full",
                            tuple(combined_gpu.shape),
                            combined_gpu.dtype,
                        )
                        combined_host.copy_(combined_gpu, non_blocking=True)
                        stream.synchronize()
                        combined_np = combined_host.numpy()
                        class_column = combined_np[:, 5]
                        inactive_indices = np.flatnonzero(class_column < 0)
                        n_survivors = (
                            int(inactive_indices[0])
                            if inactive_indices.size > 0
                            else int(class_column.shape[0])
                        )
                        if n_survivors == 0:
                            xyxy = np.empty((0, 4), dtype=np.int32)
                            confs = np.empty((0,), dtype=np.float32)
                            class_ids = np.empty((0,), dtype=np.int32)
                            if workflow_v3_fast_path:
                                workflow_masks = np.empty((0, H, W), dtype=bool)
                            else:
                                polys_or_rles = []
                        else:
                            combined_slice = combined_np[:n_survivors]
                            xyxy = combined_slice[:, :4]
                            confs = combined_slice[:, 4].view(np.float32)
                            class_ids = combined_slice[:, 5]
                            if packed_mask_cuda:
                                if workflow_v3_fast_path:
                                    packed_slice = mask_packed_gpu[:n_survivors]
                                    packed_host = get_pinned_buffer(
                                        "mask_packed",
                                        tuple(packed_slice.shape),
                                        packed_slice.dtype,
                                    )
                                    packed_host.copy_(packed_slice, non_blocking=True)
                                    stream.synchronize()
                                    packed_np = packed_host.numpy()
                                    workflow_masks = _numpy_masks_to_bool_view(
                                        np.unpackbits(
                                            packed_np, axis=-1, bitorder="little"
                                        )[..., :W]
                                    )
                                else:
                                    packed_slice = mask_packed_gpu[:n_survivors]
                                    packed_host = get_pinned_buffer(
                                        "mask_packed",
                                        tuple(packed_slice.shape),
                                        packed_slice.dtype,
                                    )
                                    packed_host.copy_(packed_slice, non_blocking=True)
                                    stream.synchronize()
                                    packed_np = packed_host.numpy()
                                    polys_or_rles = bitpacked_masks2poly(
                                        packed_np, width=W
                                    )
                            else:
                                mask_slice = mask_gpu[:n_survivors]
                                mask_host = get_pinned_buffer(
                                    "mask", tuple(mask_slice.shape), mask_slice.dtype
                                )
                                mask_host.copy_(mask_slice, non_blocking=True)
                                stream.synchronize()
                                if workflow_v3_fast_path:
                                    workflow_masks = _numpy_masks_to_bool_view(
                                        mask_host.numpy()
                                    )
                                else:
                                    polys_or_rles = masks2poly(mask_host.numpy())
                else:
                    n_survivors = int(det.xyxy.shape[0])
                    if n_survivors == 0:
                        xyxy = np.empty((0, 4), dtype=np.int32)
                        confs = np.empty((0,), dtype=np.float32)
                        class_ids = np.empty((0,), dtype=np.int32)
                        if workflow_v3_fast_path:
                            workflow_masks = np.empty((0, H, W), dtype=bool)
                        else:
                            polys_or_rles = []
                    else:
                        mask_slice = mask_gpu[:n_survivors]
                        mask_host = get_pinned_buffer(
                            "mask", tuple(mask_slice.shape), mask_slice.dtype
                        )
                        if (
                            isinstance(combined_gpu, torch.Tensor)
                            and combined_gpu.is_cuda
                            and tuple(combined_gpu.shape)
                            == (n_survivors, det.xyxy.shape[1] + 2)
                        ):
                            combined_slice = combined_gpu[:n_survivors]
                            combined_host = get_pinned_buffer(
                                "combined",
                                tuple(combined_slice.shape),
                                combined_slice.dtype,
                            )
                            combined_host.copy_(combined_slice, non_blocking=True)
                            mask_host.copy_(mask_slice, non_blocking=True)
                            stream.synchronize()
                            combined_np = combined_host.numpy()
                            xyxy = combined_np[:, :4]
                            confs = combined_np[:, 4].view(np.float32)
                            class_ids = combined_np[:, 5]
                            if workflow_v3_fast_path:
                                workflow_masks = _numpy_masks_to_bool_view(
                                    mask_host.numpy()
                                )
                            else:
                                polys_or_rles = masks2poly(mask_host.numpy())
                        else:
                            xyxy_host = get_pinned_buffer(
                                "xyxy", tuple(det.xyxy.shape), det.xyxy.dtype
                            )
                            conf_host = get_pinned_buffer(
                                "conf",
                                tuple(det.confidence.shape),
                                det.confidence.dtype,
                            )
                            class_host = get_pinned_buffer(
                                "class_id",
                                tuple(det.class_id.shape),
                                det.class_id.dtype,
                            )
                            xyxy_host.copy_(det.xyxy, non_blocking=True)
                            conf_host.copy_(det.confidence, non_blocking=True)
                            class_host.copy_(det.class_id, non_blocking=True)
                            mask_host.copy_(mask_slice, non_blocking=True)
                            stream.synchronize()
                            xyxy = xyxy_host.numpy()
                            confs = conf_host.numpy()
                            class_ids = class_host.numpy()
                            if workflow_v3_fast_path:
                                workflow_masks = _numpy_masks_to_bool_view(
                                    mask_host.numpy()
                                )
                            else:
                                polys_or_rles = masks2poly(mask_host.numpy())
            elif not return_in_rle and isinstance(mask_cpu, np.ndarray):
                xyxy = det.xyxy.detach().cpu().numpy()
                confs = det.confidence.detach().cpu().numpy()
                class_ids = det.class_id.detach().cpu().numpy()
                if workflow_v3_fast_path:
                    workflow_masks = _numpy_masks_to_bool_view(mask_cpu)
                    workflow_masks_copy = False
                else:
                    polys_or_rles = masks2poly(mask_cpu)
            else:
                xyxy = det.xyxy.detach().cpu().numpy()
                confs = det.confidence.detach().cpu().numpy()
                if isinstance(det.mask, torch.Tensor):
                    masks = det.mask.detach().cpu().numpy()
                    if return_in_rle:
                        polys_or_rles = [
                            torch_mask_to_coco_rle(mask=mask) for mask in masks
                        ]
                    elif workflow_v3_fast_path:
                        workflow_masks = _numpy_masks_to_bool_view(masks)
                        workflow_masks_copy = False
                    else:
                        polys_or_rles = masks2poly(masks)
                else:
                    if return_in_rle:
                        polys_or_rles = det.mask.to_coco_rle_masks()
                    elif workflow_v3_fast_path:
                        workflow_masks = coco_rle_masks_to_numpy_mask(det.mask)
                        workflow_masks_copy = False
                    else:
                        polys_or_rles = rle_masks2poly(det.mask)
                class_ids = det.class_id.detach().cpu().numpy()

            if workflow_v3_fast_path:
                if workflow_masks is None:
                    raise RuntimeError(
                        "Workflow v3 fast path expected dense masks to be available."
                    )
                class_names = [
                    (
                        self.class_names[int(class_id)]
                        if 0 <= int(class_id) < len(self.class_names)
                        else str(int(class_id))
                    )
                    for class_id in class_ids
                ]
                if class_filter:
                    keep = np.array(
                        [class_name in class_filter for class_name in class_names],
                        dtype=bool,
                    )
                    xyxy = xyxy[keep]
                    confs = confs[keep]
                    class_ids = class_ids[keep]
                    workflow_masks = workflow_masks[keep]
                    class_names = [
                        class_name
                        for class_name, keep_flag in zip(class_names, keep)
                        if keep_flag
                    ]
                responses.append(
                    InstanceSegmentationInferenceResponseDC(
                        predictions=[],
                        image=InferenceResponseImageDC(width=W, height=H),
                        sv_detections=_build_workflow_sv_detections(
                            xyxy=xyxy,
                            confs=confs,
                        class_ids=class_ids,
                        masks=workflow_masks,
                        class_names=class_names,
                        image_height=H,
                        image_width=W,
                        copy_masks=workflow_masks_copy,
                    ),
                )
                )
                continue

            predictions: List[
                Union[InstanceSegmentationPrediction, InstanceSegmentationRLEPrediction]
            ] = []

            for (x1, y1, x2, y2), mask_as_poly_or_rle, conf, class_id in zip(
                xyxy, polys_or_rles, confs, class_ids
            ):
                cx = (float(x1) + float(x2)) / 2.0
                cy = (float(y1) + float(y2)) / 2.0
                w = float(x2) - float(x1)
                h = float(y2) - float(y1)
                class_id_int = int(class_id)
                class_name = (
                    self.class_names[class_id_int]
                    if 0 <= class_id_int < len(self.class_names)
                    else str(class_id_int)
                )
                if class_filter and class_name not in class_filter:
                    continue
                if not return_in_rle:
                    predictions.append(
                        InstanceSegmentationPrediction(
                            x=cx,
                            y=cy,
                            width=w,
                            height=h,
                            confidence=float(conf),
                            points=[
                                Point(x=point[0], y=point[1])
                                for point in mask_as_poly_or_rle
                            ],
                            **{"class": class_name},
                            class_id=class_id_int,
                        )
                    )
                else:
                    if isinstance(mask_as_poly_or_rle["counts"], bytes):
                        mask_as_poly_or_rle["counts"] = mask_as_poly_or_rle[
                            "counts"
                        ].decode("ascii")
                    predictions.append(
                        InstanceSegmentationRLEPrediction(
                            x=cx,
                            y=cy,
                            width=w,
                            height=h,
                            confidence=float(conf),
                            rle=mask_as_poly_or_rle,
                            **{"class": class_name},
                            class_id=class_id_int,
                        )
                    )

            responses.append(
                InstanceSegmentationInferenceResponse(
                    predictions=predictions,
                    image=InferenceResponseImage(width=W, height=H),
                )
            )
        return responses

    def clear_cache(self, delete_from_disk: bool = True) -> None:
        """Clears any cache if necessary. TODO: Implement this to delete the cache from the experimental model.

        Args:
            delete_from_disk (bool, optional): Whether to delete cached files from disk. Defaults to True.
        """
        pass

    def draw_predictions(
        self,
        inference_request: InferenceRequest,
        inference_response: InferenceResponse,
    ) -> bytes:
        """Draw predictions from an inference response onto the original image provided by an inference request

        Args:
            inference_request (ObjectDetectionInferenceRequest): The inference request containing the image on which to draw predictions
            inference_response (ObjectDetectionInferenceResponse): The inference response containing predictions to be drawn

        Returns:
            str: A base64 encoded image string
        """
        class_id_2_color = {
            i: DEFAULT_COLOR_PALETTE[i % len(DEFAULT_COLOR_PALETTE)]
            for i, class_name in enumerate(self._model.class_names)
        }
        return draw_detection_predictions(
            inference_request=inference_request,
            inference_response=inference_response,
            colors=class_id_2_color,
        )


def rle_masks2poly(masks: InstancesRLEMasks) -> List[np.ndarray]:
    segments = []
    h, w = masks.image_size
    for counts in masks.masks:
        rle_dict = {"size": [h, w], "counts": counts}
        decoded_rle = np.ascontiguousarray(
            mask_utils.decode(rle_dict)
        )  # (H, W) uint8, already C-contiguous
        if not np.any(decoded_rle):
            segments.append(np.zeros((0, 2), dtype=np.float32))
            continue
        segments.append(mask2poly(decoded_rle))
    return segments


class InferenceModelsKeyPointsDetectionAdapter(Model):
    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY
        model_id = resolve_roboflow_model_alias(model_id=model_id)

        self.task_type = "keypoint-detection"

        extra_weights_provider_headers = get_extra_weights_provider_headers(
            countinference=kwargs.get("countinference"),
            service_secret=kwargs.get("service_secret"),
        )
        backend = list(
            VALID_INFERENCE_MODELS_BACKENDS.difference(
                DISABLED_INFERENCE_MODELS_BACKENDS
            )
        )
        self._model: KeyPointsDetectionModel = AutoModel.from_pretrained(
            model_id_or_path=model_id,
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            weights_provider_extra_headers=extra_weights_provider_headers,
            backend=backend,
            **kwargs,
        )
        self.class_names = list(self._model.class_names)

    def map_inference_kwargs(self, kwargs: dict) -> dict:
        kwargs["input_color_format"] = "bgr"
        if "request" in kwargs:
            keypoint_confidence_threshold = kwargs["request"].keypoint_confidence
            kwargs["key_points_threshold"] = keypoint_confidence_threshold
        pre_processing_overrides = PreProcessingOverrides(
            disable_contrast_enhancement=kwargs.get("disable_preproc_contrast", False),
            disable_grayscale=kwargs.get("disable_preproc_grayscale", False),
            disable_static_crop=kwargs.get("disable_preproc_static_crop", False),
        )
        kwargs["pre_processing_overrides"] = pre_processing_overrides
        return kwargs

    def preprocess(self, image: Any, **kwargs):
        is_batch = isinstance(image, list)
        images = image if is_batch else [image]
        np_images: List[np.ndarray] = [
            load_image_bgr(
                v,
                disable_preproc_auto_orient=kwargs.get(
                    "disable_preproc_auto_orient", False
                ),
            )
            for v in images
        ]
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.pre_process(np_images, **mapped_kwargs)

    def predict(self, img_in, **kwargs):
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.forward(img_in, **mapped_kwargs)

    def postprocess(
        self,
        predictions: Tuple[List[KeyPoints], Optional[List[Detections]]],
        preprocess_return_metadata: PreprocessingMetadata,
        **kwargs,
    ) -> List[KeypointsDetectionInferenceResponse]:
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        keypoints_list, detections_list = self._model.post_process(
            predictions, preprocess_return_metadata, **mapped_kwargs
        )
        if detections_list is None:
            raise RuntimeError(
                "Keypoints detection model does not provide instances detection - this is not supported for "
                "models from `inference-models` package which are adapted to work with `inference`."
            )
        key_points_classes = self._model.key_points_classes
        responses: List[KeypointsDetectionInferenceResponse] = []
        for preproc_metadata, keypoints, det in zip(
            preprocess_return_metadata, keypoints_list, detections_list
        ):

            H = preproc_metadata.original_size.height
            W = preproc_metadata.original_size.width

            xyxy = det.xyxy.detach().cpu().numpy()
            confs = det.confidence.detach().cpu().numpy()
            class_ids = det.class_id.detach().cpu().numpy()
            keypoints_xy = keypoints.xy.detach().cpu().tolist()
            keypoints_class_id = keypoints.class_id.detach().cpu().tolist()
            keypoints_confidence = keypoints.confidence.detach().cpu().tolist()
            predictions: List[KeypointsPrediction] = []

            for (
                (x1, y1, x2, y2),
                conf,
                class_id,
                instance_keypoints_xy,
                instance_keypoints_class_id,
                instance_keypoints_confidence,
            ) in zip(
                xyxy,
                confs,
                class_ids,
                keypoints_xy,
                keypoints_class_id,
                keypoints_confidence,
            ):
                cx = (float(x1) + float(x2)) / 2.0
                cy = (float(y1) + float(y2)) / 2.0
                w = float(x2) - float(x1)
                h = float(y2) - float(y1)
                class_id_int = int(class_id)
                class_name = (
                    self.class_names[class_id_int]
                    if 0 <= class_id_int < len(self.class_names)
                    else str(class_id_int)
                )
                if (
                    kwargs.get("class_filter")
                    and class_name not in kwargs["class_filter"]
                ):
                    continue
                predictions.append(
                    KeypointsPrediction(
                        x=cx,
                        y=cy,
                        width=w,
                        height=h,
                        confidence=float(conf),
                        **{"class": class_name},
                        class_id=class_id_int,
                        keypoints=model_keypoints_to_response(
                            instance_keypoints_xy=instance_keypoints_xy,
                            instance_keypoints_confidence=instance_keypoints_confidence,
                            instance_keypoints_class_id=instance_keypoints_class_id,
                            key_points_classes=key_points_classes,
                        ),
                    )
                )

            responses.append(
                KeypointsDetectionInferenceResponse(
                    predictions=predictions,
                    image=InferenceResponseImage(width=W, height=H),
                )
            )

        return responses

    def clear_cache(self, delete_from_disk: bool = True) -> None:
        """Clears any cache if necessary. TODO: Implement this to delete the cache from the experimental model.

        Args:
            delete_from_disk (bool, optional): Whether to delete cached files from disk. Defaults to True.
        """
        pass

    def draw_predictions(
        self,
        inference_request: InferenceRequest,
        inference_response: InferenceResponse,
    ) -> bytes:
        """Draw predictions from an inference response onto the original image provided by an inference request

        Args:
            inference_request (ObjectDetectionInferenceRequest): The inference request containing the image on which to draw predictions
            inference_response (ObjectDetectionInferenceResponse): The inference response containing predictions to be drawn

        Returns:
            str: A base64 encoded image string
        """
        class_id_2_color = {
            i: DEFAULT_COLOR_PALETTE[i % len(DEFAULT_COLOR_PALETTE)]
            for i, class_name in enumerate(self._model.class_names)
        }
        return draw_detection_predictions(
            inference_request=inference_request,
            inference_response=inference_response,
            colors=class_id_2_color,
        )


def model_keypoints_to_response(
    instance_keypoints_xy: List[
        List[Union[float, int]]
    ],  # (num_key_points_foc_class_of_object, 2)
    instance_keypoints_confidence: List[float],  # (instance_key_points, )
    instance_keypoints_class_id: int,
    key_points_classes: List[List[str]],
) -> List[Keypoint]:
    keypoint_classes = key_points_classes[instance_keypoints_class_id]
    results = []
    for keypoint_class_id, ((x, y), confidence, keypoint_class_name) in enumerate(
        zip(instance_keypoints_xy, instance_keypoints_confidence, keypoint_classes)
    ):
        if confidence <= 0.0:
            continue
        keypoint = Keypoint(
            x=x,
            y=y,
            confidence=confidence,
            class_id=keypoint_class_id,
            **{"class": keypoint_class_name},
        )
        results.append(keypoint)
    return results


class InferenceModelsClassificationAdapter(Model):
    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY
        model_id = resolve_roboflow_model_alias(model_id=model_id)

        self.task_type = "classification"
        extra_weights_provider_headers = get_extra_weights_provider_headers(
            countinference=kwargs.get("countinference"),
            service_secret=kwargs.get("service_secret"),
        )
        backend = list(
            VALID_INFERENCE_MODELS_BACKENDS.difference(
                DISABLED_INFERENCE_MODELS_BACKENDS
            )
        )
        self._model: Union[ClassificationModel, MultiLabelClassificationModel] = (
            AutoModel.from_pretrained(
                model_id_or_path=model_id,
                api_key=self.api_key,
                allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
                allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
                weights_provider_extra_headers=extra_weights_provider_headers,
                backend=backend,
                **kwargs,
            )
        )
        self.class_names = list(self._model.class_names)

    def map_inference_kwargs(self, kwargs: dict) -> dict:
        kwargs["input_color_format"] = "bgr"
        pre_processing_overrides = PreProcessingOverrides(
            disable_contrast_enhancement=kwargs.get("disable_preproc_contrast", False),
            disable_grayscale=kwargs.get("disable_preproc_grayscale", False),
            disable_static_crop=kwargs.get("disable_preproc_static_crop", False),
        )
        kwargs["pre_processing_overrides"] = pre_processing_overrides
        return kwargs

    def preprocess(self, image: Any, **kwargs):
        is_batch = isinstance(image, list)
        images = image if is_batch else [image]
        np_images: List[np.ndarray] = [
            load_image_bgr(
                v,
                disable_preproc_auto_orient=kwargs.get(
                    "disable_preproc_auto_orient", False
                ),
            )
            for v in images
        ]
        images_shapes = [i.shape[:2] for i in np_images]
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.pre_process(np_images, **mapped_kwargs), images_shapes

    def predict(self, img_in, **kwargs):
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.forward(img_in, **mapped_kwargs)

    def postprocess(
        self,
        predictions: Tuple[List[KeyPoints], Optional[List[Detections]]],
        returned_metadata: List[Tuple[int, int]],
        **kwargs,
    ) -> Union[
        List[MultiLabelClassificationInferenceResponse],
        List[ClassificationInferenceResponse],
    ]:
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        if isinstance(self._model, MultiLabelClassificationModel):
            post_processed_predictions = self._model.post_process(
                predictions, **mapped_kwargs
            )
            return prepare_multi_label_classification_response(
                post_processed_predictions,
                image_sizes=returned_metadata,
                class_names=self.class_names,
            )
        # Single-label classification: top-1 always wins regardless of
        # confidence, so per-class refinement isn't meaningful here. The base
        # class deliberately opts out of recommendedParameters entirely. The
        # response builder still uses the confidence as a cutoff that decides
        # which alternative classes show up — string-valued "best"/"default"
        # have no meaningful mapping here, so fall back to 0.5.
        post_processed_predictions = self._model.post_process(
            predictions, **mapped_kwargs
        )
        raw_confidence = kwargs.get("confidence")
        confidence_threshold = (
            raw_confidence if isinstance(raw_confidence, (int, float)) else 0.5
        )
        return prepare_classification_response(
            post_processed_predictions,
            image_sizes=returned_metadata,
            class_names=self.class_names,
            confidence_threshold=confidence_threshold,
        )

    def clear_cache(self, delete_from_disk: bool = True) -> None:
        """Clears any cache if necessary. TODO: Implement this to delete the cache from the experimental model.

        Args:
            delete_from_disk (bool, optional): Whether to delete cached files from disk. Defaults to True.
        """
        pass

    def infer_from_request(
        self,
        request: ClassificationInferenceRequest,
    ) -> Union[List[InferenceResponse], InferenceResponse]:
        """
        Handle an inference request to produce an appropriate response.

        Args:
            request (ClassificationInferenceRequest): The request object encapsulating the image(s) and relevant parameters.

        Returns:
            Union[List[InferenceResponse], InferenceResponse]: The response object(s) containing the predictions, visualization, and other pertinent details. If a list of images was provided, a list of responses is returned. Otherwise, a single response is returned.

        Notes:
            - Starts a timer at the beginning to calculate inference time.
            - Processes the image(s) through the `infer` method.
            - Generates the appropriate response object(s) using `make_response`.
            - Calculates and sets the time taken for inference.
            - If visualization is requested, the predictions are drawn on the image.
        """
        t1 = perf_counter()
        responses = self.infer(**request.dict(), return_image_dims=True)
        for response in responses:
            response.time = perf_counter() - t1
            response.inference_id = getattr(request, "id", None)

        if request.visualize_predictions:
            for response in responses:
                response.visualization = draw_predictions(
                    request, response, self.class_names
                )

        if not isinstance(request.image, list):
            responses = responses[0]

        return responses


def prepare_multi_label_classification_response(
    post_processed_predictions: List[MultiLabelClassificationPrediction],
    image_sizes: List[Tuple[int, int]],
    class_names: List[str],
) -> List[MultiLabelClassificationInferenceResponse]:
    """Build the API response from a model's post-processed predictions.

    `prediction.class_ids` is the authoritative list of "passed" classes —
    the model's `post_process` already applied the
    full priority chain (user → per-class → global → default), so the
    response builder doesn't re-threshold here. The full per-class score
    vector is still emitted in `image_predictions_dict` for UI display.
    """
    results = []
    for prediction, image_size in zip(post_processed_predictions, image_sizes):
        class_confidences = _reshape_classification_confidences(
            confidence=prediction.confidence.cpu(),
            expected_num_images=1,
            class_names=class_names,
        )[0].tolist()
        image_predictions_dict = {
            class_names[class_id]: {
                "confidence": confidence,
                "class_id": class_id,
            }
            for class_id, confidence in enumerate(class_confidences)
        }
        predicted_classes = [
            class_names[class_id] for class_id in prediction.class_ids.tolist()
        ]
        results.append(
            MultiLabelClassificationInferenceResponse(
                predictions=image_predictions_dict,
                predicted_classes=predicted_classes,
                image=InferenceResponseImage(width=image_size[1], height=image_size[0]),
                # essentially pushing a dummy values as I have no intention breaking the new API for the sake of delivering value that has no practical use
            )
        )
    return results


def prepare_classification_response(
    post_processed_predictions: ClassificationPrediction,
    image_sizes: List[Tuple[int, int]],
    class_names: List[str],
    confidence_threshold: float,
) -> List[ClassificationInferenceResponse]:
    responses = []
    batch_confidences = _reshape_classification_confidences(
        confidence=post_processed_predictions.confidence.cpu(),
        expected_num_images=len(image_sizes),
        class_names=class_names,
    )
    for classes_confidence, image_size in zip(batch_confidences.tolist(), image_sizes):
        individual_classes_predictions = []
        for i, cls_name in enumerate(class_names):
            class_score = float(classes_confidence[i])
            if class_score < confidence_threshold:
                continue
            class_prediction = {
                "class_id": i,
                "class": cls_name,
                "confidence": round(class_score, 4),
            }
            individual_classes_predictions.append(class_prediction)
        individual_classes_predictions = sorted(
            individual_classes_predictions, key=lambda x: x["confidence"], reverse=True
        )
        response = ClassificationInferenceResponse(
            image=InferenceResponseImage(width=image_size[1], height=image_size[0]),
            # essentially pushing a dummy values as I have no intention breaking the new API for the sake of delivering value that has no practical use
            predictions=individual_classes_predictions,
            top=(
                individual_classes_predictions[0]["class"]
                if individual_classes_predictions
                else ""
            ),
            confidence=(
                individual_classes_predictions[0]["confidence"]
                if individual_classes_predictions
                else 0.0
            ),
        )
        responses.append(response)
    return responses


def _reshape_classification_confidences(
    confidence: torch.Tensor,
    expected_num_images: int,
    class_names: List[str],
) -> torch.Tensor:
    expected_num_classes = len(class_names)
    expected_num_scores = expected_num_images * expected_num_classes
    actual_num_scores = confidence.numel()
    if actual_num_scores != expected_num_scores:
        raise PostProcessingError(
            "Classification model output has shape "
            f"{tuple(confidence.shape)} containing {actual_num_scores} confidence "
            f"score(s), but response metadata expects {expected_num_images} image(s) "
            f"x {expected_num_classes} class name(s) = {expected_num_scores} score(s). "
            "This usually means the model package class names metadata does not match "
            "the classifier head."
        )
    return confidence.reshape(expected_num_images, expected_num_classes)


def draw_predictions(inference_request, inference_response, class_names: List[str]):
    """Draw prediction visuals on an image.

    This method overlays the predictions on the input image, including drawing rectangles and text to visualize the predicted classes.

    Args:
        inference_request: The request object containing the image and parameters.
        inference_response: The response object containing the predictions and other details.
        class_names: List of class names corresponding to the model's classes.

    Returns:
        bytes: The bytes of the visualized image in JPEG format.
    """
    image = load_image_rgb(inference_request.image)
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    class_id_2_color = {
        i: DEFAULT_COLOR_PALETTE[i % len(DEFAULT_COLOR_PALETTE)]
        for i, class_name in enumerate(class_names)
    }
    if isinstance(inference_response.predictions, list):
        prediction = inference_response.predictions[0]
        color = class_id_2_color.get(prediction.class_id, "#4892EA")
        draw.rectangle(
            [0, 0, image.size[1], image.size[0]],
            outline=color,
            width=inference_request.visualization_stroke_width,
        )
        text = f"{prediction.class_id} - {prediction.class_name} {prediction.confidence:.2f}"
        text_size = font.getbbox(text)

        # set button size + 10px margins
        button_size = (text_size[2] + 20, text_size[3] + 20)
        button_img = Image.new("RGBA", button_size, color)
        # put text on button with 10px margins
        button_draw = ImageDraw.Draw(button_img)
        button_draw.text((10, 10), text, font=font, fill=(255, 255, 255, 255))

        # put button on source image in position (0, 0)
        image.paste(button_img, (0, 0))
    else:
        if len(inference_response.predictions) > 0:
            box_color = "#4892EA"
            draw.rectangle(
                [0, 0, image.size[1], image.size[0]],
                outline=box_color,
                width=inference_request.visualization_stroke_width,
            )
        row = 0
        predictions = [
            (cls_name, pred)
            for cls_name, pred in inference_response.predictions.items()
        ]
        predictions = sorted(predictions, key=lambda x: x[1].confidence, reverse=True)
        for i, (cls_name, pred) in enumerate(predictions):
            color = class_id_2_color.get(cls_name, "#4892EA")
            text = f"{cls_name} {pred.confidence:.2f}"
            text_size = font.getbbox(text)

            # set button size + 10px margins
            button_size = (text_size[2] + 20, text_size[3] + 20)
            button_img = Image.new("RGBA", button_size, color)
            # put text on button with 10px margins
            button_draw = ImageDraw.Draw(button_img)
            button_draw.text((10, 10), text, font=font, fill=(255, 255, 255, 255))

            # put button on source image in position (0, 0)
            image.paste(button_img, (0, row))
            row += button_size[1]

    buffered = BytesIO()
    image = image.convert("RGB")
    image.save(buffered, format="JPEG")
    return buffered.getvalue()


class InferenceModelsSemanticSegmentationAdapter(Model):
    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        super().__init__()

        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}

        self.api_key = api_key if api_key else API_KEY
        model_id = resolve_roboflow_model_alias(model_id=model_id)

        self.task_type = "semantic-segmentation"

        extra_weights_provider_headers = get_extra_weights_provider_headers(
            countinference=kwargs.get("countinference"),
            service_secret=kwargs.get("service_secret"),
        )
        backend = list(
            VALID_INFERENCE_MODELS_BACKENDS.difference(
                DISABLED_INFERENCE_MODELS_BACKENDS
            )
        )
        self._model: SemanticSegmentationModel = AutoModel.from_pretrained(
            model_id_or_path=model_id,
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            weights_provider_extra_headers=extra_weights_provider_headers,
            backend=backend,
            **kwargs,
        )
        self.class_names = list(self._model.class_names)

    @property
    def class_map(self):
        # match segment.roboflow.com
        return {str(k): v for k, v in enumerate(self.class_names)}

    def map_inference_kwargs(self, kwargs: dict) -> dict:
        kwargs["input_color_format"] = "bgr"
        pre_processing_overrides = PreProcessingOverrides(
            disable_contrast_enhancement=kwargs.get("disable_preproc_contrast", False),
            disable_grayscale=kwargs.get("disable_preproc_grayscale", False),
            disable_static_crop=kwargs.get("disable_preproc_static_crop", False),
        )
        kwargs["pre_processing_overrides"] = pre_processing_overrides
        return kwargs

    def preprocess(self, image: Any, **kwargs):
        is_batch = isinstance(image, list)
        images = image if is_batch else [image]
        np_images: List[np.ndarray] = [
            load_image_bgr(
                v,
                disable_preproc_auto_orient=kwargs.get(
                    "disable_preproc_auto_orient", False
                ),
            )
            for v in images
        ]
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.pre_process(np_images, **mapped_kwargs)

    def predict(self, img_in, **kwargs):
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        return self._model.forward(img_in, **mapped_kwargs)

    def postprocess(
        self,
        predictions: torch.Tensor,
        preprocess_return_metadata: PreprocessingMetadata,
        **kwargs,
    ) -> List[SemanticSegmentationInferenceResponse]:
        mapped_kwargs = self.map_inference_kwargs(kwargs)
        segmentation_results = self._model.post_process(
            predictions, preprocess_return_metadata, **mapped_kwargs
        )

        responses: List[SemanticSegmentationInferenceResponse] = []
        for preproc_metadata, segmentation in zip(
            preprocess_return_metadata, segmentation_results
        ):
            height = preproc_metadata.original_size.height
            width = preproc_metadata.original_size.width
            response_image = InferenceResponseImage(width=width, height=height)
            # WARNING! This way of conversion is hazardous - first of all, if background class is not in class names,
            # for certain pre-processing, we end up with -1 values which will be wrapped to 255 - second of all,
            # we can support only 256 classes - those constraints should be fine until inference 2.0
            response_predictions = SemanticSegmentationPrediction(
                segmentation_mask=self.img_to_b64_str(
                    segmentation.segmentation_map.to(torch.uint8)
                ),
                confidence_mask=self.img_to_b64_str(
                    (segmentation.confidence * 255).to(torch.uint8)
                ),
                class_map=self.class_map,
                image=dict(response_image),
            )
            response = SemanticSegmentationInferenceResponse(
                predictions=response_predictions,
                image=response_image,
            )
            responses.append(response)
        return responses

    def clear_cache(self, delete_from_disk: bool = True) -> None:
        """Clears any cache if necessary. TODO: Implement this to delete the cache from the experimental model.

        Args:
            delete_from_disk (bool, optional): Whether to delete cached files from disk. Defaults to True.
        """
        pass

    def img_to_b64_str(self, img: torch.Tensor) -> str:
        if img.dtype != torch.uint8:
            raise ValueError(
                f"img_to_b64_str requires uint8 tensor but got dtype {img.dtype}"
            )

        img = Image.fromarray(img.cpu().numpy())
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")

        img_str = base64.b64encode(buffered.getvalue())
        img_str = img_str.decode("ascii")

        return img_str

    def draw_predictions(
        self,
        inference_request: InferenceRequest,
        inference_response: InferenceResponse,
    ) -> bytes:
        raise NotImplementedError(
            "draw_predictions(...) is not implemented for semantic segmentation models - responses contain "
            "visualization already."
        )
