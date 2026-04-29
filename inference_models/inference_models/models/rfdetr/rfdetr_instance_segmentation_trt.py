import os
import threading
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from inference_models import (
    InstanceDetections,
    InstanceSegmentationModel,
    PreProcessingOverrides,
)
from inference_models.configuration import (
    DEFAULT_DEVICE,
    INFERENCE_MODELS_RFDETR_DEFAULT_CONFIDENCE,
)
from inference_models.entities import Confidence, ColorFormat
from inference_models.errors import (
    CorruptedModelPackageError,
    MissingDependencyError,
    ModelRuntimeError,
)
from inference_models.models.common.cuda import (
    use_cuda_context,
    use_primary_cuda_context,
)
from inference_models.models.common.model_packages import get_model_package_contents
from inference_models.models.common.roboflow.model_packages import (
    InferenceConfig,
    PreProcessingMetadata,
    ResizeMode,
    TRTConfig,
    parse_class_names_file,
    parse_inference_config,
    parse_trt_config,
)
from inference_models.models.common.trt import (
    TRTCudaGraphCache,
    establish_trt_cuda_graph_cache,
    get_trt_engine_inputs_and_outputs,
    infer_from_trt_engine,
    load_trt_model,
)
from inference_models.models.rfdetr.class_remapping import (
    ClassesReMapping,
    prepare_class_remapping,
)
from inference_models.models.rfdetr.common import (
    post_process_instance_segmentation_results,
)
from inference_models.models.rfdetr.pre_processing import pre_process_network_input
from inference_models.entities import ImageDimensions as _ImageDimensions
from inference_models.models.common.roboflow.model_packages import (
    StaticCropOffset as _StaticCropOffset,
)
from inference_models.models.common.roboflow.post_processing import ConfidenceFilter
from inference_models.weights_providers.entities import RecommendedParameters

_RFDETR_USE_TRITON_PREPROC = os.getenv("RFDETR_USE_TRITON_PREPROC", "false").lower() in (
    "true",
    "1",
)
if _RFDETR_USE_TRITON_PREPROC:
    try:
        from inference_models.models.rfdetr.triton_preprocess import (
            TRITON_AVAILABLE as _TRITON_AVAILABLE,
            triton_preprocess_rfdetr_stretch,
        )
        _TRITON_READY = _TRITON_AVAILABLE and torch.cuda.is_available()
    except Exception:  # pragma: no cover
        _TRITON_READY = False
        triton_preprocess_rfdetr_stretch = None
else:
    _TRITON_READY = False
    triton_preprocess_rfdetr_stretch = None

try:
    import tensorrt as trt
except ImportError as import_error:
    raise MissingDependencyError(
        message=f"Could not import RFDetr model with TRT backend - this error means that some additional dependencies "
        f"are not installed in the environment.  If you see this error running locally, "
        "please follow our installation guide: https://inference-models.roboflow.com/getting-started/installation/"
        " If you see this error using Roboflow infrastructure, make sure the service you use does support the "
        f"model, You can also contact Roboflow to get support. "
        "Additionally - if AutoModel.from_pretrained(...) "
        f"automatically selects model package which does not match your environment - that's a serious problem and "
        f"we will really appreciate letting us know - https://github.com/roboflow/inference/issues",
        help_url="https://inference-models.roboflow.com/errors/runtime-environment/#missingdependencyerror",
    ) from import_error

try:
    import pycuda.driver as cuda
except ImportError as import_error:
    raise MissingDependencyError(
        message="Running model RFDETR with TRT backend on GPU requires pycuda installation, which is brought with "
        "`trt-*` extras of `inference-models` library. If you see this error running locally, "
        "please follow our installation guide: https://inference-models.roboflow.com/getting-started/installation/"
        " If you see this error using Roboflow infrastructure, make sure the service you use does support the "
        f"model, You can also contact Roboflow to get support."
        "Additionally - if AutoModel.from_pretrained(...) "
        f"automatically selects model package which does not match your environment - that's a serious problem and "
        f"we will really appreciate letting us know - https://github.com/roboflow/inference/issues",
        help_url="https://inference-models.roboflow.com/errors/runtime-environment/#missingdependencyerror",
    ) from import_error


class RFDetrForInstanceSegmentationTRT(
    InstanceSegmentationModel[
        torch.Tensor,
        PreProcessingMetadata,
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]
):
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        engine_host_code_allowed: bool = False,
        trt_cuda_graph_cache: Optional[TRTCudaGraphCache] = None,
        default_trt_cuda_graph_cache_size: int = 8,
        rf_detr_max_input_resolution: Optional[Union[int, Tuple[int, int]]] = None,
        recommended_parameters: Optional[RecommendedParameters] = None,
        **kwargs,
    ) -> "RFDetrForInstanceSegmentationTRT":
        if device.type != "cuda":
            raise ModelRuntimeError(
                message=f"TRT engine only runs on CUDA device - {device} device detected.",
                help_url="https://inference-models.roboflow.com/errors/models-runtime/#modelruntimeerror",
            )
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=[
                "class_names.txt",
                "inference_config.json",
                "trt_config.json",
                "engine.plan",
            ],
        )
        class_names = parse_class_names_file(
            class_names_path=model_package_content["class_names.txt"]
        )
        inference_config = parse_inference_config(
            config_path=model_package_content["inference_config.json"],
            allowed_resize_modes={
                ResizeMode.STRETCH_TO,
                ResizeMode.LETTERBOX,
                ResizeMode.CENTER_CROP,
                ResizeMode.LETTERBOX_REFLECT_EDGES,
            },
            implicit_resize_mode_substitutions={
                ResizeMode.FIT_LONGER_EDGE: (
                    ResizeMode.STRETCH_TO,
                    None,
                    "RFDetr Instance Segmentation model running with TRT backend was trained with "
                    "`fit-longer-edge` input resize mode. This transform cannot be applied properly for "
                    "RFDetr models. To ensure interoperability, `stretch` "
                    "resize mode will be used instead. If model was trained on Roboflow platform, "
                    "we recommend using preprocessing method different that `fit-longer-edge`.",
                )
            },
            max_allowed_input_size=rf_detr_max_input_resolution,
        )
        classes_re_mapping = None
        if inference_config.class_names_operations:
            class_names, classes_re_mapping = prepare_class_remapping(
                class_names=class_names,
                class_names_operations=inference_config.class_names_operations,
                device=device,
            )
        trt_config = parse_trt_config(
            config_path=model_package_content["trt_config.json"]
        )
        cuda.init()
        cuda_device = cuda.Device(device.index or 0)
        with use_primary_cuda_context(cuda_device=cuda_device) as cuda_context:
            engine = load_trt_model(
                model_path=model_package_content["engine.plan"],
                engine_host_code_allowed=engine_host_code_allowed,
            )
            execution_context = engine.create_execution_context()
        inputs, outputs = get_trt_engine_inputs_and_outputs(engine=engine)
        if len(inputs) != 1:
            raise CorruptedModelPackageError(
                message=f"Implementation assume single model input, found: {len(inputs)}.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        if len(outputs) != 3:
            raise CorruptedModelPackageError(
                message=f"Implementation assume 3 model outputs, found: {len(outputs)}.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        trt_cuda_graph_cache = establish_trt_cuda_graph_cache(
            default_cuda_graph_cache_size=default_trt_cuda_graph_cache_size,
            cuda_graph_cache=trt_cuda_graph_cache,
        )
        return cls(
            engine=engine,
            input_name=inputs[0],
            output_names=outputs,
            class_names=class_names,
            classes_re_mapping=classes_re_mapping,
            inference_config=inference_config,
            trt_config=trt_config,
            device=device,
            cuda_context=cuda_context,
            execution_context=execution_context,
            trt_cuda_graph_cache=trt_cuda_graph_cache,
            recommended_parameters=recommended_parameters,
        )

    def __init__(
        self,
        engine: trt.ICudaEngine,
        input_name: str,
        output_names: List[str],
        class_names: List[str],
        classes_re_mapping: Optional[ClassesReMapping],
        inference_config: InferenceConfig,
        trt_config: TRTConfig,
        device: torch.device,
        cuda_context: cuda.Context,
        execution_context: trt.IExecutionContext,
        trt_cuda_graph_cache: Optional[TRTCudaGraphCache],
        recommended_parameters=None,
    ):
        self._engine = engine
        self._input_name = input_name
        self._output_names = output_names
        self._inference_config = inference_config
        self._class_names = class_names
        self._classes_re_mapping = classes_re_mapping
        self._device = device
        self._cuda_context = cuda_context
        self._execution_context = execution_context
        self._trt_config = trt_config
        self._trt_cuda_graph_cache = trt_cuda_graph_cache
        self._lock = threading.Lock()
        self._inference_stream = torch.cuda.Stream(device=self._device)
        self._thread_local_storage = threading.local()
        self.recommended_parameters = recommended_parameters

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        image_size: Optional[Tuple[int, int]] = None,
        pre_processing_overrides: Optional[PreProcessingOverrides] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
        fast = self._try_fast_preprocess(
            images=images,
            input_color_format=input_color_format,
            image_size=image_size,
            pre_processing_overrides=pre_processing_overrides,
        )
        if fast is not None:
            return fast
        with torch.cuda.stream(self._pre_process_stream):
            pre_processed_images, pre_processing_meta = pre_process_network_input(
                images=images,
                image_pre_processing=self._inference_config.image_pre_processing,
                network_input=self._inference_config.network_input,
                target_device=self._device,
                input_color_format=input_color_format,
                image_size_wh=image_size,
                pre_processing_overrides=pre_processing_overrides,
            )
        self._pre_process_stream.synchronize()
        return pre_processed_images, pre_processing_meta

    def _try_fast_preprocess(
        self,
        images,
        input_color_format,
        image_size,
        pre_processing_overrides,
    ):
        if not _TRITON_READY:
            return None
        if image_size is not None:
            return None
        ipp = self._inference_config.image_pre_processing
        if (
            ipp.static_crop is not None
            and ipp.static_crop.enabled
            or ipp.contrast is not None
            and ipp.contrast.enabled
            or ipp.grayscale is not None
            and ipp.grayscale.enabled
        ):
            return None
        ni = self._inference_config.network_input
        if ni.resize_mode != ResizeMode.STRETCH_TO:
            return None
        if ni.input_channels != 3:
            return None
        if ni.dataset_version_resize_dimensions is not None:
            return None
        if ni.scaling_factor not in (None, 255):
            return None
        if ni.normalization is None:
            return None
        means, stds = ni.normalization
        # Only handle numpy HWC BGR uint8 (the common hot path).
        if isinstance(images, list):
            if len(images) != 1:
                return None
            candidate = images[0]
        else:
            candidate = images
        if not isinstance(candidate, np.ndarray):
            return None
        if (
            candidate.dtype != np.uint8
            or candidate.ndim != 3
            or candidate.shape[2] != 3
        ):
            return None
        images = candidate
        # Color: if caller says RGB, skip; we do BGR->model_color_mode.
        from inference_models.models.common.roboflow.model_packages import ColorMode
        caller_mode = ColorMode(input_color_format) if input_color_format is not None else ColorMode.BGR
        if caller_mode != ColorMode.BGR or ni.color_mode != ColorMode.RGB:
            return None

        target_h = ni.training_input_size.height
        target_w = ni.training_input_size.width
        orig_h, orig_w = images.shape[0], images.shape[1]

        if not getattr(self, "_fast_buffer_initialized", False):
            self._fast_input_buffer = torch.empty(
                (1, 3, target_h, target_w),
                dtype=torch.float32,
                device=self._device,
            )
            # Marker: tells the TRT CUDA-graph capture path to use this
            # tensor as the graph's own input buffer, eliminating the
            # per-frame DtoD copy from our preproc output into the graph's
            # internal buffer. Our preproc always writes in-place here.
            self._fast_input_buffer._trt_reuse_as_input_buffer = True
            self._fast_means = tuple(means)
            self._fast_stds = tuple(stds)
            # Pinned host buffer for the raw BGR frame — lets us do a
            # truly async HtoD into src_gpu. Grown lazily below if the
            # frame size changes.
            self._fast_src_host_pinned = None
            self._fast_src_gpu = None
            self._fast_buffer_initialized = True

        # Reuse a pinned host staging buffer so torch.Tensor.copy_ with
        # non_blocking=True actually runs async. Without pinning,
        # non_blocking is silently promoted to a sync copy.
        src_shape = images.shape
        src_nbytes = images.nbytes
        pinned = self._fast_src_host_pinned
        if (
            pinned is None
            or pinned.numel() * pinned.element_size() < src_nbytes
            or tuple(pinned.shape) != src_shape
        ):
            pinned = torch.empty(src_shape, dtype=torch.uint8, pin_memory=True)
            self._fast_src_host_pinned = pinned
            self._fast_src_gpu = torch.empty(
                src_shape, dtype=torch.uint8, device=self._device
            )
        # Copy the numpy BGR frame into pinned host memory (fast CPU memcpy),
        # then async DtoH->GPU while the Triton launch happens on CPU side.
        pinned_np = pinned.numpy()
        np.copyto(pinned_np, images, casting="no")
        src_gpu = self._fast_src_gpu
        with torch.cuda.stream(self._pre_process_stream):
            src_gpu.copy_(pinned, non_blocking=True)
            triton_preprocess_rfdetr_stretch(
                src_gpu,
                target_h=target_h,
                target_w=target_w,
                means=self._fast_means,
                stds=self._fast_stds,
                out=self._fast_input_buffer,
            )
            # Record an event so the inference stream can wait on preproc
            # completion without blocking the CPU.
            self._fast_preproc_event = torch.cuda.Event()
            self._fast_preproc_event.record(self._pre_process_stream)
            self._fast_input_buffer.record_stream(self._pre_process_stream)

        size_after = _ImageDimensions(height=orig_h, width=orig_w)
        target = _ImageDimensions(height=target_h, width=target_w)
        metadata = PreProcessingMetadata(
            pad_left=0,
            pad_top=0,
            pad_right=0,
            pad_bottom=0,
            original_size=size_after,
            size_after_pre_processing=size_after,
            inference_size=target,
            scale_width=target_w / orig_w,
            scale_height=target_h / orig_h,
            static_crop_offset=_StaticCropOffset(
                offset_x=0, offset_y=0, crop_width=orig_w, crop_height=orig_h
            ),
        )
        return self._fast_input_buffer, [metadata]

    def forward(
        self,
        pre_processed_images: torch.Tensor,
        disable_cuda_graphs: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache = self._trt_cuda_graph_cache if not disable_cuda_graphs else None
        ev = getattr(self, "_fast_preproc_event", None)
        if ev is not None:
            ev.wait(self._inference_stream)
            self._fast_preproc_event = None
        with self._lock:
            with use_cuda_context(context=self._cuda_context):
                detections, labels, masks = infer_from_trt_engine(
                    pre_processed_images=pre_processed_images,
                    trt_config=self._trt_config,
                    engine=self._engine,
                    context=self._execution_context,
                    device=self._device,
                    input_name=self._input_name,
                    outputs=self._output_names,
                    stream=self._inference_stream,
                    trt_cuda_graph_cache=cache,
                )
                return detections, labels, masks

    def post_process(
        self,
        model_results: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        pre_processing_meta: List[PreProcessingMetadata],
        confidence: Confidence = "default",
        **kwargs,
    ) -> List[InstanceDetections]:
        confidence_filter = ConfidenceFilter(
            confidence=confidence,
            recommended_parameters=self.recommended_parameters,
            default_confidence=INFERENCE_MODELS_RFDETR_DEFAULT_CONFIDENCE,
        )
        # Wait on the TRT-stream "produce" event so our post_process stream
        # can start reading the (graph-owned) output buffers as soon as the
        # engine finishes, without a CPU-side synchronize().
        produce_event = getattr(model_results[0], "_trt_produce_event", None)
        graph_state = getattr(model_results[0], "_trt_graph_state", None)
        with torch.cuda.stream(self._post_process_stream):
            if produce_event is not None:
                produce_event.wait(self._post_process_stream)
            for result_element in model_results:
                result_element.record_stream(self._post_process_stream)
            bboxes, logits, masks = model_results
            results = post_process_instance_segmentation_results(
                bboxes=bboxes,
                logits=logits,
                masks=masks,
                pre_processing_meta=pre_processing_meta,
                threshold=confidence_filter.get_threshold(self.class_names),
                num_classes=len(self.class_names),
                classes_re_mapping=self._classes_re_mapping,
            )
            # Record "consumer done" so the next TRT replay can wait on it
            # before overwriting the graph-owned output buffers.
            if graph_state is not None:
                ev = graph_state.consumer_done_event
                if ev is None:
                    ev = torch.cuda.Event()
                    graph_state.consumer_done_event = ev
                ev.record(self._post_process_stream)
        return results

    @property
    def _pre_process_stream(self) -> torch.cuda.Stream:
        if not hasattr(self._thread_local_storage, "pre_process_stream"):
            self._thread_local_storage.pre_process_stream = torch.cuda.Stream(
                device=self._device
            )
        return self._thread_local_storage.pre_process_stream

    @property
    def _post_process_stream(self) -> torch.cuda.Stream:
        if not hasattr(self._thread_local_storage, "post_process_stream"):
            self._thread_local_storage.post_process_stream = torch.cuda.Stream(
                device=self._device
            )
        return self._thread_local_storage.post_process_stream
