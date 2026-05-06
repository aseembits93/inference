import base64
import io
import os
from io import BytesIO
from time import perf_counter
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


# Cache of pinned host buffers for async DtoH, keyed by (name, dtype).
# Pinned memory lets torch's copy_(non_blocking=True) actually run async.
# We grow on first use and reuse thereafter; buffer is sliced to the
# current n_survivors for each copy.
_PINNED_HOST_BUFFERS: dict = {}


# Sentinel returned from `predict()` on the first frame of a pipelined
# run so `postprocess()` knows to emit empty responses (the real frame-0
# output follows on the next call). Using an instance-level sentinel
# rather than None so downstream code that checks `is None` keeps
# working unchanged.
class _PipelinePrimingSentinel:
    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return "<_PIPELINE_PRIMING>"


_PIPELINE_PRIMING = _PipelinePrimingSentinel()


def _get_pinned_buffer(name: str, shape, dtype: torch.dtype) -> torch.Tensor:
    shape = tuple(int(s) for s in shape)
    key = (name, dtype)
    buf = _PINNED_HOST_BUFFERS.get(key)
    if buf is not None:
        # Reuse if the cached buffer is at least as large in every dim.
        if all(buf.shape[i] >= shape[i] for i in range(len(shape))):
            return buf[tuple(slice(0, s) for s in shape)]
    buf = torch.empty(shape, dtype=dtype, pin_memory=True)
    _PINNED_HOST_BUFFERS[key] = buf
    return buf

from inference.core.entities.requests import (
    ClassificationInferenceRequest,
    InferenceRequest,
)
from inference.core.entities.responses.inference import (
    ClassificationInferenceResponse,
    InferenceResponse,
    InferenceResponseImage,
    InferenceResponseImageDC,
    InstanceSegmentationInferenceResponse,
    InstanceSegmentationInferenceResponseDC,
    InstanceSegmentationPrediction,
    InstanceSegmentationPredictionDC,
    Keypoint,
    KeypointsDetectionInferenceResponse,
    KeypointsPrediction,
    MultiLabelClassificationInferenceResponse,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
    Point,
    PointDC,
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
from inference.core.models.base import Model
from inference.core.roboflow_api import get_extra_weights_provider_headers
from inference.core.utils.image_utils import load_image_bgr, load_image_rgb
from inference.core.utils.postprocess import masks2poly
from inference.core.utils.visualisation import draw_detection_predictions
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
from inference_models.models.base.types import PreprocessingMetadata

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
            if kwargs.get("source") == "workflow-execution":
                return [
                    InstanceSegmentationInferenceResponseDC(
                        predictions=[],
                        image=InferenceResponseImageDC(
                            width=m.original_size.width,
                            height=m.original_size.height,
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
        gpu_fastpath = os.getenv("RFDETR_GPU_POSTPROCESS", "true").lower() in ("true", "1")
        # Workflow callers consume a plain dict via `_is_response_dc_to_dict`;
        # dataclasses avoid pydantic validation + `model_dump` overhead per
        # frame. Every other caller (HTTP, cache, visualization) keeps the
        # pydantic path because it depends on the pydantic class identity.
        use_dc = kwargs.get("source") == "workflow-execution"

        responses: List[InstanceSegmentationInferenceResponse] = []
        for preproc_metadata, det in zip(preprocess_return_metadata, detections_list):
            H = preproc_metadata.original_size.height
            W = preproc_metadata.original_size.width

            # Fast path: triton_rfdetr_fullpost returns UNSLICED buffers plus
            # a GPU counter tensor and a done-event. We:
            #   (1) wait the done-event on the current stream so post_process
            #       stream writes are visible,
            #   (2) async-DtoH the 4-byte counter into a pinned host buffer
            #       and sync once — this replaces the old in-kernel
            #       counter.item() that blocked the postproc stream,
            #   (3) slice combined/mask to n_survivors and async-DtoH both,
            #       then sync.
            combined_gpu = getattr(det, "_combined_gpu", None)
            counter_gpu = getattr(det, "_counter_gpu", None)
            done_event = getattr(det, "_postproc_done_event", None)
            if (
                combined_gpu is not None
                and counter_gpu is not None
                and done_event is not None
                and det.mask.is_cuda
            ):
                device = combined_gpu.device
                stream = torch.cuda.current_stream(device)
                done_event.wait(stream)

                counter_host = _get_pinned_buffer("counter", (1,), torch.int32)
                counter_host.copy_(counter_gpu, non_blocking=True)
                stream.synchronize()
                n_survivors = int(counter_host[0].item())

                if n_survivors == 0:
                    xyxy = np.empty((0, 4), dtype=np.int32)
                    confs = np.empty((0,), dtype=np.float32)
                    class_ids = np.empty((0,), dtype=np.int32)
                    masks = np.empty((0, 0, 0), dtype=np.uint8)
                else:
                    mask_gpu = det.mask
                    combined_slice = combined_gpu[:n_survivors]
                    mask_slice = mask_gpu[:n_survivors]
                    combined_host = _get_pinned_buffer(
                        "combined", combined_slice.shape, combined_slice.dtype
                    )
                    mask_host = _get_pinned_buffer(
                        "mask", mask_slice.shape, mask_slice.dtype
                    )
                    combined_host.copy_(combined_slice, non_blocking=True)
                    mask_host.copy_(mask_slice, non_blocking=True)
                    stream.synchronize()
                    combined_cpu = combined_host.numpy()
                    xyxy = combined_cpu[:, :4]
                    confs = combined_cpu[:, 4].view(np.float32)
                    class_ids = combined_cpu[:, 5]
                    masks = mask_host.numpy()
            elif combined_gpu is not None:
                combined_cpu = combined_gpu.detach().cpu().numpy()
                xyxy = combined_cpu[:, :4]
                confs = combined_cpu[:, 4].view(np.float32)
                class_ids = combined_cpu[:, 5]
                masks = det.mask.detach().cpu().numpy()
            else:
                xyxy = det.xyxy.detach().cpu().numpy()
                confs = det.confidence.detach().cpu().numpy()
                class_ids = det.class_id.detach().cpu().numpy()
                masks = det.mask.detach().cpu().numpy()
            polys = masks2poly(masks)

            predictions: List[InstanceSegmentationPrediction] = []

            for (x1, y1, x2, y2), mask_as_poly, conf, class_id in zip(
                xyxy, polys, confs, class_ids
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
                if use_dc:
                    predictions.append(
                        InstanceSegmentationPredictionDC(
                            x=cx,
                            y=cy,
                            width=w,
                            height=h,
                            confidence=float(conf),
                            class_name=class_name,
                            class_id=class_id_int,
                            points=[
                                PointDC(x=float(point[0]), y=float(point[1]))
                                for point in mask_as_poly
                            ],
                        )
                    )
                else:
                    predictions.append(
                        InstanceSegmentationPrediction(
                            x=cx,
                            y=cy,
                            width=w,
                            height=h,
                            confidence=float(conf),
                            points=[
                                Point(x=point[0], y=point[1])
                                for point in mask_as_poly
                            ],
                            **{"class": class_name},
                            class_id=class_id_int,
                        )
                    )

            if use_dc:
                responses.append(
                    InstanceSegmentationInferenceResponseDC(
                        predictions=predictions,
                        image=InferenceResponseImageDC(width=W, height=H),
                    )
                )
            else:
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
        image_predictions_dict = {
            class_names[class_id]: {
                "confidence": confidence,
                "class_id": class_id,
            }
            for class_id, confidence in enumerate(prediction.confidence.cpu().tolist())
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
    for classes_confidence, image_size in zip(
        post_processed_predictions.confidence.cpu().tolist(), image_sizes
    ):
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
