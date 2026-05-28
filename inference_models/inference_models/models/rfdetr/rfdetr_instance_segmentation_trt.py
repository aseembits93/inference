from dataclasses import dataclass
import os
import threading
from typing import List, Optional, Set, Tuple, Union

import numpy as np
import torch

from inference_models import (
    InstanceDetections,
    InstanceSegmentationMaskFormat,
    InstanceSegmentationModel,
    PreProcessingOverrides,
)
# Hoisted to module scope to avoid per-call `from ... import` inside the hot
# forward_async path. Re-import inside the function added ~13µs/frame in the
# instrumented run on Jetson Orin. Import here is a no-op on every call.
from inference_models.models.base.instance_segmentation import _DirectInferenceFuture
from inference_models.configuration import (
    DEFAULT_DEVICE,
    INFERENCE_MODELS_RFDETR_DEFAULT_CONFIDENCE,
)
from inference_models.entities import ColorFormat, Confidence
from inference_models.errors import (
    CorruptedModelPackageError,
    MissingDependencyError,
    ModelInputError,
    ModelRuntimeError,
)
from inference_models.models.common.cuda import (
    use_cuda_context,
    use_primary_cuda_context,
)
from inference_models.models.common.model_packages import get_model_package_contents
from inference_models.entities import ImageDimensions
from inference_models.models.common.roboflow.model_packages import (
    ColorMode,
    InferenceConfig,
    PreProcessingMetadata,
    ResizeMode,
    StaticCropOffset,
    TRTConfig,
    parse_class_names_file,
    parse_inference_config,
    parse_trt_config,
)
from inference_models.models.common.roboflow.post_processing import ConfidenceFilter
from inference_models.models.common.trt import (
    TRTCudaGraphCache,
    bind_trt_aux_streams,
    create_trt_user_aux_streams,
    establish_trt_cuda_graph_cache,
    get_trt_engine_inputs_and_outputs,
    infer_from_trt_engine,
    load_trt_model,
    mark_trt_results_consumed,
    prepare_trt_results_for_consumer,
)
from inference_models.models.rfdetr.class_remapping import (
    ClassesReMapping,
    prepare_class_remapping,
)
from inference_models.models.rfdetr.common import (
    build_deferred_dense_postproc_detection,
    post_triton_eligible,
    post_process_instance_segmentation_results,
    post_process_instance_segmentation_results_to_rle_masks,
)
from inference_models.models.rfdetr.pre_processing import pre_process_network_input
from inference_models.utils.environment import get_boolean_from_env

try:
    from inference_models.models.rfdetr.triton_preprocess import (
        TRITON_AVAILABLE as _TRITON_AVAILABLE,
        build_resample_tables,
        triton_preprocess_rfdetr_stretch,
    )
except ImportError:
    _TRITON_AVAILABLE = False
    build_resample_tables = None
    triton_preprocess_rfdetr_stretch = None

try:
    from inference_models.models.rfdetr.triton_fullpostproc import (
        _POSTPROC_GRAPH_RING_DEPTH,
        _TOPK_QUERY_BLOCK,
        _allocate_scratch_buffers,
        _get_class_mapping_int32,
        _get_empty_float32_on_device,
        _get_empty_int32_on_device,
        _get_resize_tables,
        _launch_dense_split_postproc,
        _next_power_of_two,
        _prepare_threshold,
        _simple_mask_fastpath_supported,
        get_rfdetr_triton_postproc_geometry,
    )

    _TRITON_FULLPOSTPROC_AVAILABLE = True
except ImportError:
    _TRITON_FULLPOSTPROC_AVAILABLE = False
    _POSTPROC_GRAPH_RING_DEPTH = 1
    _TOPK_QUERY_BLOCK = 12
    _allocate_scratch_buffers = None
    _get_class_mapping_int32 = None
    _get_empty_float32_on_device = None
    _get_empty_int32_on_device = None
    _get_resize_tables = None
    _launch_dense_split_postproc = None
    _next_power_of_two = None
    _prepare_threshold = None
    _simple_mask_fastpath_supported = None
    get_rfdetr_triton_postproc_geometry = None

_COMBINED_DENSE_GRAPH_STREAM_COUNT = os.environ.get(
    "RFDETR_TRITON_POSTPROC_GRAPH_STREAM_COUNT"
)
if _COMBINED_DENSE_GRAPH_STREAM_COUNT is not None:
    _COMBINED_DENSE_GRAPH_STREAM_COUNT = max(
        1, int(_COMBINED_DENSE_GRAPH_STREAM_COUNT)
    )

# Kill switch: set INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED=false to force
# the PIL reference path for every call, regardless of other predicates.
_FAST_PATH_ENABLED = get_boolean_from_env(
    "INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED", default=True
)
from inference_models.weights_providers.entities import RecommendedParameters

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


class _FastPathState:
    """Per-(src_shape, target_shape) cache of GPU buffers + resample tables
    that the Triton fast path reuses across frames."""

    __slots__ = (
        "src_h",
        "src_w",
        "target_h",
        "target_w",
        "pinned_host",
        "src_gpu",
        "out_buffers",
        "out_events",
        "next_out_buffer_idx",
        "tables",
        "combined_dense_graph_rings",
    )

    def __init__(
        self,
        src_h: int,
        src_w: int,
        target_h: int,
        target_w: int,
        pinned_host: torch.Tensor,
        src_gpu: torch.Tensor,
        out_buffers,
        out_events,
        tables,
    ) -> None:
        self.src_h = src_h
        self.src_w = src_w
        self.target_h = target_h
        self.target_w = target_w
        self.pinned_host = pinned_host
        self.src_gpu = src_gpu
        self.out_buffers = out_buffers
        self.out_events = out_events
        self.next_out_buffer_idx = 0
        self.tables = tables
        self.combined_dense_graph_rings = {}

    @classmethod
    def build(
        cls,
        src_h: int,
        src_w: int,
        target_h: int,
        target_w: int,
        device: torch.device,
    ) -> "_FastPathState":
        pinned_host = torch.empty((src_h, src_w, 3), dtype=torch.uint8, pin_memory=True)
        src_gpu = torch.empty((src_h, src_w, 3), dtype=torch.uint8, device=device)
        out_buffers = tuple(
            torch.empty((1, 3, target_h, target_w), dtype=torch.float32, device=device)
            for _ in range(2)
        )
        out_events = tuple(torch.cuda.Event() for _ in range(2))
        tables = build_resample_tables(
            src_h=src_h,
            src_w=src_w,
            target_h=target_h,
            target_w=target_w,
            device=device,
        )
        return cls(
            src_h=src_h,
            src_w=src_w,
            target_h=target_h,
            target_w=target_w,
            pinned_host=pinned_host,
            src_gpu=src_gpu,
            out_buffers=out_buffers,
            out_events=out_events,
            tables=tables,
        )

    def is_stale(
        self, src_h: int, src_w: int, target_h: int, target_w: int
    ) -> bool:
        return (
            self.src_h != src_h
            or self.src_w != src_w
            or self.target_h != target_h
            or self.target_w != target_w
        )

    def acquire_out_buffer(self) -> Tuple[torch.Tensor, int, torch.cuda.Event]:
        slot_idx = self.next_out_buffer_idx
        self.next_out_buffer_idx = (slot_idx + 1) % len(self.out_buffers)
        return self.out_buffers[slot_idx], slot_idx, self.out_events[slot_idx]


@dataclass
class _CombinedDenseGraphState:
    cuda_graph: "torch.cuda.CUDAGraph"
    stream: torch.cuda.Stream
    execution_context: trt.IExecutionContext
    input_buffer: torch.Tensor
    output_buffers: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    combined: torch.Tensor
    mask_bin: torch.Tensor
    counter: torch.Tensor
    selected_queries: torch.Tensor
    partial_topk: torch.Tensor
    selection_done_event: "torch.cuda.Event"
    done_event: "torch.cuda.Event"
    aux_streams: Optional[Tuple["torch.cuda.Stream", ...]] = None
    preproc_src_gpu: Optional[torch.Tensor] = None
    fast_path_state: Optional["_FastPathState"] = None


@dataclass(frozen=True)
class _CombinedDenseRuntimeConfig:
    cache_key: Tuple[object, ...]
    thr_tensor: torch.Tensor
    per_class: bool
    cmap: torch.Tensor
    has_remap: bool
    geometry: object


@dataclass
class _CombinedDenseGraphRing:
    states: List[Optional[_CombinedDenseGraphState]]
    next_idx: int = 0


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
        self._fast_path_state: Optional[_FastPathState] = None
        self._combined_dense_graph_cache: dict = {}
        self._combined_dense_runtime_cache: dict = {}

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    @property
    def supported_mask_formats(self) -> Set[InstanceSegmentationMaskFormat]:
        return {"dense", "rle"}

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        image_size: Optional[Tuple[int, int]] = None,
        pre_processing_overrides: Optional[PreProcessingOverrides] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[PreProcessingMetadata]]:
        defer_fast_preprocess_to_combined_graph = bool(
            kwargs.get("defer_triton_preprocess_to_combined_graph", False)
        )
        fast = self._try_fast_preprocess(
            images=images,
            input_color_format=input_color_format,
            image_size=image_size,
            pre_processing_overrides=pre_processing_overrides,
            defer_fast_preprocess_to_combined_graph=defer_fast_preprocess_to_combined_graph,
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
            pre_processed_images.record_stream(self._pre_process_stream)
            done_event = torch.cuda.Event()
            done_event.record(self._pre_process_stream)
        pre_processed_images._trt_preproc_done_event = done_event  # type: ignore[attr-defined]
        pre_processed_images._rfdetr_preprocess_meta = pre_processing_meta  # type: ignore[attr-defined]
        return pre_processed_images, pre_processing_meta

    def _try_fast_preprocess(
        self,
        images,
        input_color_format,
        image_size,
        pre_processing_overrides,
        defer_fast_preprocess_to_combined_graph,
    ) -> Optional[Tuple[torch.Tensor, List[PreProcessingMetadata]]]:
        if not _FAST_PATH_ENABLED:
            return None
        if not _TRITON_AVAILABLE:
            return None
        if image_size is not None:
            return None
        # pre_processing_overrides can only *disable* transforms; it has no
        # "enable" knob. The fast path never applies static_crop / grayscale /
        # contrast regardless, so the override flags are irrelevant — we just
        # gate on whether the image_pre_processing config itself asks for them.
        ipp = self._inference_config.image_pre_processing
        if (
            (ipp.static_crop is not None and ipp.static_crop.enabled)
            or (ipp.contrast is not None and ipp.contrast.enabled)
            or (ipp.grayscale is not None and ipp.grayscale.enabled)
        ):
            return None

        ni = self._inference_config.network_input
        if ni.dataset_version_resize_dimensions is not None:
            return None
        if ni.input_channels != 3:
            return None
        if ni.scaling_factor not in (None, 255):
            return None
        if ni.normalization is None:
            return None
        # When dataset_version_resize_dimensions is None, the prod path collapses
        # non-stretch resize modes to a single PIL stretch as well
        # (pre_processing.py:_needs_two_step_resize), so we accept all modes here.
        if ni.resize_mode not in (
            ResizeMode.STRETCH_TO,
            ResizeMode.LETTERBOX,
            ResizeMode.CENTER_CROP,
            ResizeMode.LETTERBOX_REFLECT_EDGES,
        ):
            return None

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

        caller_mode = (
            ColorMode(input_color_format)
            if input_color_format is not None
            else ColorMode.BGR
        )
        swap_rb = caller_mode != ni.color_mode

        means, stds = ni.normalization
        means_t = (float(means[0]), float(means[1]), float(means[2]))
        stds_t = (float(stds[0]), float(stds[1]), float(stds[2]))
        target_h = ni.training_input_size.height
        target_w = ni.training_input_size.width
        orig_h, orig_w = int(candidate.shape[0]), int(candidate.shape[1])

        state = self._fast_path_state
        if state is None or state.is_stale(
            src_h=orig_h,
            src_w=orig_w,
            target_h=target_h,
            target_w=target_w,
        ):
            state = _FastPathState.build(
                src_h=orig_h,
                src_w=orig_w,
                target_h=target_h,
                target_w=target_w,
                device=self._device,
            )
            self._fast_path_state = state

        pinned_np = state.pinned_host.numpy()
        np.copyto(pinned_np, candidate, casting="no")

        out_buffer, slot_idx, done_event = state.acquire_out_buffer()

        meta = PreProcessingMetadata(
            pad_left=0,
            pad_top=0,
            pad_right=0,
            pad_bottom=0,
            original_size=ImageDimensions(width=orig_w, height=orig_h),
            size_after_pre_processing=ImageDimensions(width=orig_w, height=orig_h),
            inference_size=ImageDimensions(width=target_w, height=target_h),
            scale_width=target_w / orig_w,
            scale_height=target_h / orig_h,
            static_crop_offset=StaticCropOffset(
                offset_x=0, offset_y=0, crop_width=orig_w, crop_height=orig_h
            ),
        )
        if (
            defer_fast_preprocess_to_combined_graph
            and _TRITON_FULLPOSTPROC_AVAILABLE
            and self._trt_cuda_graph_cache is not None
        ):
            out_buffer._trt_reuse_as_input_buffer = True  # type: ignore[attr-defined]
            out_buffer._trt_reuse_key = slot_idx  # type: ignore[attr-defined]
            out_buffer._trt_preprocess_deferred = True  # type: ignore[attr-defined]
            out_buffer._trt_fast_path_state = state  # type: ignore[attr-defined]
            out_buffer._trt_preprocess_swap_rb = swap_rb  # type: ignore[attr-defined]
            out_buffer._trt_preproc_done_event = None  # type: ignore[attr-defined]
            out_buffer._trt_preproc_slot_event = done_event  # type: ignore[attr-defined]
            out_buffer._rfdetr_preprocess_meta = [meta]  # type: ignore[attr-defined]
            return out_buffer, [meta]

        with torch.cuda.stream(self._pre_process_stream):
            state.src_gpu.copy_(state.pinned_host, non_blocking=True)
            triton_preprocess_rfdetr_stretch(
                src=state.src_gpu,
                tables=state.tables,
                target_h=target_h,
                target_w=target_w,
                means=means_t,
                stds=stds_t,
                swap_rb=swap_rb,
                out=out_buffer,
            )
            out_buffer.record_stream(self._pre_process_stream)
            done_event.record(self._pre_process_stream)
        out_buffer._trt_reuse_as_input_buffer = True  # type: ignore[attr-defined]
        out_buffer._trt_reuse_key = slot_idx  # type: ignore[attr-defined]
        out_buffer._trt_preprocess_deferred = False  # type: ignore[attr-defined]
        out_buffer._trt_preproc_done_event = done_event  # type: ignore[attr-defined]
        out_buffer._rfdetr_preprocess_meta = [meta]  # type: ignore[attr-defined]
        return out_buffer, [meta]

    def _materialize_deferred_fast_preprocess(
        self, pre_processed_images: torch.Tensor
    ) -> torch.Tensor:
        if not bool(
            getattr(pre_processed_images, "_trt_preprocess_deferred", False)
        ):
            return pre_processed_images
        fast_path_state = getattr(pre_processed_images, "_trt_fast_path_state", None)
        if fast_path_state is None:
            return pre_processed_images
        done_event = getattr(pre_processed_images, "_trt_preproc_slot_event", None)
        if done_event is None:
            done_event = torch.cuda.Event()
            pre_processed_images._trt_preproc_slot_event = done_event  # type: ignore[attr-defined]
        means, stds = self._inference_config.network_input.normalization
        means_t = (float(means[0]), float(means[1]), float(means[2]))
        stds_t = (float(stds[0]), float(stds[1]), float(stds[2]))
        target_h = self._inference_config.network_input.training_input_size.height
        target_w = self._inference_config.network_input.training_input_size.width
        swap_rb = bool(getattr(pre_processed_images, "_trt_preprocess_swap_rb", False))
        with torch.cuda.stream(self._pre_process_stream):
            fast_path_state.src_gpu.copy_(
                fast_path_state.pinned_host, non_blocking=True
            )
            triton_preprocess_rfdetr_stretch(
                src=fast_path_state.src_gpu,
                tables=fast_path_state.tables,
                target_h=target_h,
                target_w=target_w,
                means=means_t,
                stds=stds_t,
                swap_rb=swap_rb,
                out=pre_processed_images,
            )
            pre_processed_images.record_stream(self._pre_process_stream)
            done_event.record(self._pre_process_stream)
        pre_processed_images._trt_preproc_done_event = done_event  # type: ignore[attr-defined]
        pre_processed_images._trt_preprocess_deferred = False  # type: ignore[attr-defined]
        return pre_processed_images

    def forward(
        self,
        pre_processed_images: torch.Tensor,
        disable_cuda_graphs: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pre_processed_images = self._materialize_deferred_fast_preprocess(
            pre_processed_images
        )
        cache = self._trt_cuda_graph_cache if not disable_cuda_graphs else None
        pre_processed_images._trt_zero_copy_consumer = True
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

    def _combined_dense_graph_eligible(
        self,
        pre_processed_images: torch.Tensor,
        pre_processing_meta: Optional[List[PreProcessingMetadata]],
        **kwargs,
    ) -> bool:
        return (
            _TRITON_FULLPOSTPROC_AVAILABLE
            and self._trt_cuda_graph_cache is not None
            and getattr(pre_processed_images, "_trt_reuse_key", None) is not None
            and kwargs.get("mask_format") == "rle"
            and kwargs.get("defer_count_to_adapter", False)
            and kwargs.get("response_mask_format") != "rle"
            and isinstance(pre_processing_meta, list)
            and len(pre_processing_meta) == 1
        )

    def _capture_combined_dense_graph_state(
        self,
        *,
        pre_processed_images: torch.Tensor,
        pre_processing_meta: List[PreProcessingMetadata],
        thr_tensor: torch.Tensor,
        per_class: bool,
        cmap: torch.Tensor,
        has_remap: bool,
        graph_stream: Optional[torch.cuda.Stream] = None,
    ) -> Optional[_CombinedDenseGraphState]:
        assert _allocate_scratch_buffers is not None
        assert _get_empty_float32_on_device is not None
        assert _get_empty_int32_on_device is not None
        assert _get_resize_tables is not None
        assert _launch_dense_split_postproc is not None
        assert _next_power_of_two is not None
        assert _simple_mask_fastpath_supported is not None
        assert get_rfdetr_triton_postproc_geometry is not None

        device = self._device
        meta = pre_processing_meta[0]
        deferred_fast_preprocess = bool(
            getattr(pre_processed_images, "_trt_preprocess_deferred", False)
        )
        fast_path_state = getattr(pre_processed_images, "_trt_fast_path_state", None)
        graph_context = self._engine.create_execution_context()
        status = graph_context.set_input_shape(
            self._input_name, tuple(pre_processed_images.shape)
        )
        if not status:
            raise ModelRuntimeError(
                message="Failed to set TRT input shape for combined dense graph capture.",
                help_url="https://inference-models.roboflow.com/errors/models-runtime/#modelruntimeerror",
            )
        status = graph_context.set_tensor_address(
            self._input_name, pre_processed_images.data_ptr()
        )
        if not status:
            raise ModelRuntimeError(
                message="Failed to bind TRT input buffer for combined dense graph capture.",
                help_url="https://inference-models.roboflow.com/errors/models-runtime/#modelruntimeerror",
            )

        output_buffers = []
        for output_name in self._output_names:
            output_shape = graph_context.get_tensor_shape(output_name)
            output_dtype = {
                trt.DataType.FLOAT: torch.float32,
                trt.DataType.HALF: torch.float16,
                trt.DataType.INT32: torch.int32,
                trt.DataType.INT8: torch.int8,
                trt.DataType.BOOL: torch.bool,
            }[self._engine.get_tensor_dtype(output_name)]
            output_buffer = torch.empty(
                tuple(output_shape), dtype=output_dtype, device=device
            )
            graph_context.set_tensor_address(output_name, output_buffer.data_ptr())
            output_buffers.append(output_buffer)
        bboxes_out, logits_out, masks_out = output_buffers
        if not post_triton_eligible(
            bboxes_out,
            logits_out,
            masks_out,
            pre_processing_meta,
            self._classes_re_mapping,
        ):
            return None

        num_queries = int(logits_out.shape[1])
        num_classes_total = int(logits_out.shape[2])
        mask_h = int(masks_out.shape[2])
        mask_w = int(masks_out.shape[3])
        selection_block = _next_power_of_two(num_queries)
        num_query_blocks = (num_queries + _TOPK_QUERY_BLOCK - 1) // _TOPK_QUERY_BLOCK
        combined, mask_bin, counter, selected_queries, partial_topk = (
            _allocate_scratch_buffers(
                num_queries=num_queries,
                orig_h=meta.original_size.height,
                orig_w=meta.original_size.width,
                device=device,
                pack_dense_masks=True,
                selection_block=selection_block,
                num_query_blocks=num_query_blocks,
            )
        )

        denorm_size = meta.nonsquare_intermediate_size or meta.inference_size
        geometry = get_rfdetr_triton_postproc_geometry(
            denorm_size_wh=(denorm_size.width, denorm_size.height),
            pad_ltrb=(
                meta.pad_left,
                meta.pad_top,
                meta.pad_right,
                meta.pad_bottom,
            ),
            scale_wh=(meta.scale_width, meta.scale_height),
            orig_size_wh=(meta.original_size.width, meta.original_size.height),
            size_after_pre_processing_wh=(
                meta.size_after_pre_processing.width,
                meta.size_after_pre_processing.height,
            ),
            static_crop_offset_xy=(
                meta.static_crop_offset.offset_x,
                meta.static_crop_offset.offset_y,
            ),
            mask_size_hw=(mask_h, mask_w),
        )
        simple_mask_fastpath = _simple_mask_fastpath_supported(
            geometry,
            mask_h=mask_h,
            mask_w=mask_w,
            emit_rle=False,
        )
        dummy_int32 = _get_empty_int32_on_device(device)
        dummy_float32 = _get_empty_float32_on_device(device)
        if simple_mask_fastpath:
            y_indices = dummy_int32
            y_weights = dummy_float32
            y_counts = dummy_int32
            x_indices = dummy_int32
            x_weights = dummy_float32
            x_counts = dummy_int32
            max_y_taps = 1
            max_x_taps = 1
        else:
            (
                y_indices,
                y_weights,
                y_counts,
                x_indices,
                x_weights,
                x_counts,
            ) = _get_resize_tables(
                input_h=geometry.mask_input_h,
                input_w=geometry.mask_input_w,
                output_h=geometry.output_h,
                output_w=geometry.output_w,
                device=device,
            )
            max_y_taps = int(y_indices.shape[1])
            max_x_taps = int(x_indices.shape[1])

        stream = (
            graph_stream
            if graph_stream is not None
            else torch.cuda.Stream(device=device)
        )
        aux_streams = create_trt_user_aux_streams(engine=self._engine, device=device)
        selection_done_event = torch.cuda.Event(external=True)
        done_event = torch.cuda.Event(external=True)
        preproc_done_event = None
        preproc_src_gpu = None
        if deferred_fast_preprocess:
            if fast_path_state is None:
                return None
            assert triton_preprocess_rfdetr_stretch is not None
            preproc_src_gpu = torch.empty_like(fast_path_state.src_gpu)
            means, stds = self._inference_config.network_input.normalization
            means_t = (float(means[0]), float(means[1]), float(means[2]))
            stds_t = (float(stds[0]), float(stds[1]), float(stds[2]))
            target_h = self._inference_config.network_input.training_input_size.height
            target_w = self._inference_config.network_input.training_input_size.width
            swap_rb = bool(
                getattr(pre_processed_images, "_trt_preprocess_swap_rb", False)
            )
        else:
            preproc_done_event = getattr(
                pre_processed_images, "_trt_preproc_done_event", None
            )
        with torch.cuda.stream(stream):
            if deferred_fast_preprocess:
                preproc_src_gpu.copy_(
                    fast_path_state.pinned_host, non_blocking=True
                )
                triton_preprocess_rfdetr_stretch(
                    src=preproc_src_gpu,
                    tables=fast_path_state.tables,
                    target_h=target_h,
                    target_w=target_w,
                    means=means_t,
                    stds=stds_t,
                    swap_rb=swap_rb,
                    out=pre_processed_images,
                )
            elif preproc_done_event is not None:
                stream.wait_event(preproc_done_event)
            bind_trt_aux_streams(context=graph_context, aux_streams=aux_streams)
            status = graph_context.execute_async_v3(stream_handle=stream.cuda_stream)
            if not status:
                raise ModelRuntimeError(
                    message="Failed to execute TRT warmup before combined dense graph capture.",
                    help_url="https://inference-models.roboflow.com/errors/models-runtime/#modelruntimeerror",
                )
            _launch_dense_split_postproc(
                logits_2d=logits_out[0],
                bboxes_2d=bboxes_out[0],
                masks_3d=masks_out[0],
                thr_tensor=thr_tensor,
                cmap=cmap,
                y_indices=y_indices,
                y_weights=y_weights,
                y_counts=y_counts,
                x_indices=x_indices,
                x_weights=x_weights,
                x_counts=x_counts,
                selected_queries=selected_queries,
                combined=combined,
                counter=counter,
                mask_bin=mask_bin,
                partial_topk=partial_topk,
                num_classes=len(self.class_names),
                geometry=geometry,
                orig_h=meta.original_size.height,
                orig_w=meta.original_size.width,
                per_class=per_class,
                has_remap=has_remap,
                pack_dense_masks=True,
                simple_mask_fastpath=simple_mask_fastpath,
                num_queries=num_queries,
                num_classes_total=num_classes_total,
                mask_h=mask_h,
                mask_w=mask_w,
                selection_block=selection_block,
                num_query_blocks=num_query_blocks,
                max_y_taps=max_y_taps,
                max_x_taps=max_x_taps,
                selection_done_event=selection_done_event,
                done_event=done_event,
            )
        stream.synchronize()

        cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cuda_graph, stream=stream):
            if deferred_fast_preprocess:
                preproc_src_gpu.copy_(
                    fast_path_state.pinned_host, non_blocking=True
                )
                triton_preprocess_rfdetr_stretch(
                    src=preproc_src_gpu,
                    tables=fast_path_state.tables,
                    target_h=target_h,
                    target_w=target_w,
                    means=means_t,
                    stds=stds_t,
                    swap_rb=swap_rb,
                    out=pre_processed_images,
                )
            bind_trt_aux_streams(context=graph_context, aux_streams=aux_streams)
            status = graph_context.execute_async_v3(stream_handle=stream.cuda_stream)
            if not status:
                raise ModelRuntimeError(
                    message="Failed to capture combined dense CUDA graph.",
                    help_url="https://inference-models.roboflow.com/errors/models-runtime/#modelruntimeerror",
                )
            _launch_dense_split_postproc(
                logits_2d=logits_out[0],
                bboxes_2d=bboxes_out[0],
                masks_3d=masks_out[0],
                thr_tensor=thr_tensor,
                cmap=cmap,
                y_indices=y_indices,
                y_weights=y_weights,
                y_counts=y_counts,
                x_indices=x_indices,
                x_weights=x_weights,
                x_counts=x_counts,
                selected_queries=selected_queries,
                combined=combined,
                counter=counter,
                mask_bin=mask_bin,
                partial_topk=partial_topk,
                num_classes=len(self.class_names),
                geometry=geometry,
                orig_h=meta.original_size.height,
                orig_w=meta.original_size.width,
                per_class=per_class,
                has_remap=has_remap,
                pack_dense_masks=True,
                simple_mask_fastpath=simple_mask_fastpath,
                num_queries=num_queries,
                num_classes_total=num_classes_total,
                mask_h=mask_h,
                mask_w=mask_w,
                selection_block=selection_block,
                num_query_blocks=num_query_blocks,
                max_y_taps=max_y_taps,
                max_x_taps=max_x_taps,
                selection_done_event=selection_done_event,
                done_event=done_event,
            )
        return _CombinedDenseGraphState(
            cuda_graph=cuda_graph,
            stream=stream,
            execution_context=graph_context,
            input_buffer=pre_processed_images,
            output_buffers=(bboxes_out, logits_out, masks_out),
            combined=combined,
            mask_bin=mask_bin,
            counter=counter,
            selected_queries=selected_queries,
            partial_topk=partial_topk,
            selection_done_event=selection_done_event,
            done_event=done_event,
            aux_streams=aux_streams,
            preproc_src_gpu=preproc_src_gpu,
            fast_path_state=fast_path_state,
        )

    def _get_or_prepare_combined_dense_runtime(
        self,
        pre_processing_meta: List[PreProcessingMetadata],
        confidence: Confidence,
    ) -> _CombinedDenseRuntimeConfig:
        assert _get_class_mapping_int32 is not None
        assert _get_empty_int32_on_device is not None
        assert _prepare_threshold is not None
        assert get_rfdetr_triton_postproc_geometry is not None

        meta = pre_processing_meta[0]
        denorm_size = meta.nonsquare_intermediate_size or meta.inference_size
        remap_key = None
        if self._classes_re_mapping is not None:
            remap_key = id(self._classes_re_mapping.class_mapping)
        runtime_key = None
        if not isinstance(confidence, torch.Tensor):
            runtime_key = (
                confidence,
                remap_key,
                denorm_size.width,
                denorm_size.height,
                meta.pad_left,
                meta.pad_top,
                meta.pad_right,
                meta.pad_bottom,
                meta.scale_width,
                meta.scale_height,
                meta.original_size.width,
                meta.original_size.height,
                meta.size_after_pre_processing.width,
                meta.size_after_pre_processing.height,
                meta.static_crop_offset.offset_x,
                meta.static_crop_offset.offset_y,
            )
            cached = self._combined_dense_runtime_cache.get(runtime_key)
            if cached is not None:
                return cached

        confidence_filter = ConfidenceFilter(
            confidence=confidence,
            recommended_parameters=self.recommended_parameters,
            default_confidence=INFERENCE_MODELS_RFDETR_DEFAULT_CONFIDENCE,
        )
        threshold = confidence_filter.get_threshold(self.class_names)
        thr_tensor, per_class = _prepare_threshold(
            threshold, self._device, len(self.class_names)
        )
        if self._classes_re_mapping is not None:
            has_remap = True
            cmap = _get_class_mapping_int32(
                self._classes_re_mapping.class_mapping, self._device
            )
        else:
            has_remap = False
            cmap = _get_empty_int32_on_device(self._device)
        geometry = get_rfdetr_triton_postproc_geometry(
            denorm_size_wh=(denorm_size.width, denorm_size.height),
            pad_ltrb=(
                meta.pad_left,
                meta.pad_top,
                meta.pad_right,
                meta.pad_bottom,
            ),
            scale_wh=(meta.scale_width, meta.scale_height),
            orig_size_wh=(meta.original_size.width, meta.original_size.height),
            size_after_pre_processing_wh=(
                meta.size_after_pre_processing.width,
                meta.size_after_pre_processing.height,
            ),
            static_crop_offset_xy=(
                meta.static_crop_offset.offset_x,
                meta.static_crop_offset.offset_y,
            ),
        )
        runtime = _CombinedDenseRuntimeConfig(
            cache_key=(
                int(thr_tensor.data_ptr()),
                per_class,
                int(cmap.data_ptr()),
                has_remap,
                geometry,
            ),
            thr_tensor=thr_tensor,
            per_class=per_class,
            cmap=cmap,
            has_remap=has_remap,
            geometry=geometry,
        )
        if runtime_key is not None:
            self._combined_dense_runtime_cache[runtime_key] = runtime
        return runtime

    def _get_or_capture_combined_dense_graph_state(
        self,
        pre_processed_images: torch.Tensor,
        pre_processing_meta: List[PreProcessingMetadata],
        **kwargs,
    ) -> Optional[_CombinedDenseGraphState]:
        confidence = kwargs.get("confidence", "default")
        runtime = self._get_or_prepare_combined_dense_runtime(
            pre_processing_meta, confidence
        )
        thr_tensor = runtime.thr_tensor
        per_class = runtime.per_class
        cmap = runtime.cmap
        has_remap = runtime.has_remap
        config_key = runtime.cache_key

        def cache_key_for(image_tensor: torch.Tensor):
            fast_path_state = getattr(image_tensor, "_trt_fast_path_state", None)
            pinned_host_ptr = (
                int(fast_path_state.pinned_host.data_ptr())
                if fast_path_state is not None
                else 0
            )
            return (
                int(self._device.index if self._device.index is not None else -1),
                int(image_tensor.data_ptr()),
                tuple(image_tensor.shape),
                image_tensor.dtype,
                getattr(image_tensor, "_trt_reuse_key", None),
                bool(getattr(image_tensor, "_trt_preprocess_deferred", False)),
                bool(getattr(image_tensor, "_trt_preprocess_swap_rb", False)),
                pinned_host_ptr,
                config_key,
            )

        fast_path_state = getattr(pre_processed_images, "_trt_fast_path_state", None)
        if (
            bool(getattr(pre_processed_images, "_trt_preprocess_deferred", False))
            and fast_path_state is not None
        ):
            current_slot = getattr(pre_processed_images, "_trt_reuse_key", None)
            slot_rings = fast_path_state.combined_dense_graph_rings.get(config_key)
            if slot_rings is None:
                stream_pool = None
                if _COMBINED_DENSE_GRAPH_STREAM_COUNT is not None:
                    total_states = (
                        len(fast_path_state.out_buffers) * _POSTPROC_GRAPH_RING_DEPTH
                    )
                    stream_pool = tuple(
                        torch.cuda.Stream(device=self._device)
                        for _ in range(
                            min(_COMBINED_DENSE_GRAPH_STREAM_COUNT, total_states)
                        )
                    )
                stream_idx = 0
                captured_rings = []
                for slot_idx, slot_tensor in enumerate(fast_path_state.out_buffers):
                    slot_tensor._trt_reuse_as_input_buffer = True  # type: ignore[attr-defined]
                    slot_tensor._trt_reuse_key = slot_idx  # type: ignore[attr-defined]
                    slot_tensor._trt_preprocess_deferred = True  # type: ignore[attr-defined]
                    slot_tensor._trt_fast_path_state = fast_path_state  # type: ignore[attr-defined]
                    slot_tensor._trt_preprocess_swap_rb = bool(  # type: ignore[attr-defined]
                        getattr(pre_processed_images, "_trt_preprocess_swap_rb", False)
                    )
                    slot_tensor._trt_preproc_done_event = None  # type: ignore[attr-defined]
                    slot_tensor._rfdetr_preprocess_meta = pre_processing_meta  # type: ignore[attr-defined]
                    states: List[Optional[_CombinedDenseGraphState]] = []
                    for _ in range(_POSTPROC_GRAPH_RING_DEPTH):
                        graph_stream = None
                        if stream_pool is not None:
                            graph_stream = stream_pool[stream_idx % len(stream_pool)]
                            stream_idx += 1
                        states.append(
                            self._capture_combined_dense_graph_state(
                                pre_processed_images=slot_tensor,
                                pre_processing_meta=pre_processing_meta,
                                thr_tensor=thr_tensor,
                                per_class=per_class,
                                cmap=cmap,
                                has_remap=has_remap,
                                graph_stream=graph_stream,
                            )
                        )
                    captured_rings.append(_CombinedDenseGraphRing(states=states))
                slot_rings = tuple(captured_rings)
                fast_path_state.combined_dense_graph_rings[config_key] = slot_rings
            slot_idx = current_slot if isinstance(current_slot, int) else 0
            if slot_idx < 0 or slot_idx >= len(slot_rings):
                slot_idx = 0
            ring = slot_rings[slot_idx]
            state_index = ring.next_idx
            ring.next_idx = (state_index + 1) % len(ring.states)
            return ring.states[state_index]
        cache_key = cache_key_for(pre_processed_images)
        ring = self._combined_dense_graph_cache.get(cache_key)
        if ring is None:
            stream_pool = None
            if _COMBINED_DENSE_GRAPH_STREAM_COUNT is not None:
                stream_pool = tuple(
                    torch.cuda.Stream(device=self._device)
                    for _ in range(
                        min(
                            _COMBINED_DENSE_GRAPH_STREAM_COUNT,
                            _POSTPROC_GRAPH_RING_DEPTH,
                        )
                    )
                )
            states: List[Optional[_CombinedDenseGraphState]] = []
            for state_idx in range(_POSTPROC_GRAPH_RING_DEPTH):
                graph_stream = None
                if stream_pool is not None:
                    graph_stream = stream_pool[state_idx % len(stream_pool)]
                states.append(
                    self._capture_combined_dense_graph_state(
                        pre_processed_images=pre_processed_images,
                        pre_processing_meta=pre_processing_meta,
                        thr_tensor=thr_tensor,
                        per_class=per_class,
                        cmap=cmap,
                        has_remap=has_remap,
                        graph_stream=graph_stream,
                    )
                )
            ring = _CombinedDenseGraphRing(states=states)
            self._combined_dense_graph_cache[cache_key] = ring
        state_index = ring.next_idx
        ring.next_idx = (state_index + 1) % len(ring.states)
        return ring.states[state_index]

    def _maybe_forward_async_combined_dense_graph(
        self,
        pre_processed_images: torch.Tensor,
        **kwargs,
    ):
        pre_processing_meta = getattr(
            pre_processed_images, "_rfdetr_preprocess_meta", None
        )
        if not self._combined_dense_graph_eligible(
            pre_processed_images, pre_processing_meta, **kwargs
        ):
            return None
        state = self._get_or_capture_combined_dense_graph_state(
            pre_processed_images, pre_processing_meta, **kwargs
        )
        if state is None:
            return None
        preproc_done_event = getattr(
            pre_processed_images, "_trt_preproc_done_event", None
        )
        with torch.cuda.stream(state.stream):
            if preproc_done_event is not None:
                state.stream.wait_event(preproc_done_event)
            state.cuda_graph.replay()
        meta = pre_processing_meta[0]
        detections = [
            build_deferred_dense_postproc_detection(
                combined=state.combined,
                mask_packed_gpu=state.mask_bin,
                counter=state.counter,
                selection_done_event=state.selection_done_event,
                done_event=state.done_event,
                orig_h=meta.original_size.height,
                orig_w=meta.original_size.width,
            )
        ]
        return _DirectInferenceFuture(
            self, detections, pre_processing_meta, state.done_event, kwargs
        )

    def forward_async(
        self,
        pre_processed_images: torch.Tensor,
        pre_processing_meta,
        **kwargs,
    ):
        """Async launch variant that isolates graph outputs when needed.

        When the fast preprocess path is active, inputs rotate across two
        stable slot buffers and the CUDA-graph cache keeps one captured
        graph per slot. In that case graph-owned outputs can be handed
        directly to postprocess, and same-slot replay simply waits for
        postprocess completion before reusing the slot.

        For all other graph paths we still need the clone fallback: a
        single captured graph would otherwise overwrite its own output
        buffers before the previous frame's postprocess has read them.

        Non-graph path returns newly-allocated tensors already, so we
        reuse the base `forward_async` in that case.

        Hot-path CPU optimisations (vs the naive `tuple(t.clone() for t in
        raw)` form):

          * Keep three per-output reusable destination buffers (one small,
            one medium, one large mask) around the future's lifetime and
            `copy_` into them with ``non_blocking=True`` instead of
            allocating new tensors every frame — saves ~40µs/frame of
            torch.empty + internal allocator work.
          * Enter the inference stream exactly once (replacing
            ``torch.cuda.stream(stream)`` context manager, which does
            save-current + set + restore and costs ~20µs by itself,
            with a pair of ``torch.cuda.set_stream`` calls at ~2µs
            each).
          * Reuse a single pre-allocated ``torch.cuda.Event()`` for
            ``consumer_done`` across frames — saves the Event() ctor.
        """
        combined_future = self._maybe_forward_async_combined_dense_graph(
            pre_processed_images, **kwargs
        )
        if combined_future is not None:
            return combined_future
        raw = self.forward(pre_processed_images, **kwargs)
        graph_state = getattr(raw[0], "_trt_graph_state", None)
        if graph_state is None:
            # Non-graph (execute_async_v3) path: outputs are freshly
            # allocated per call, no aliasing hazard.
            return super().forward_async(
                pre_processed_images, pre_processing_meta, **kwargs
            )
        if getattr(pre_processed_images, "_trt_reuse_key", None) is not None:
            produce_event = getattr(raw[0], "_trt_produce_event", None)
            return _DirectInferenceFuture(
                self, raw, pre_processing_meta, produce_event, kwargs
            )
        stream = graph_state.cuda_stream

        # Reusable per-call clone buffers. We keep a small ring of three
        # sets in thread-local storage so that at pipeline depth=2 we
        # never alias "buffers that the previous in-flight future is
        # still decoding" with "buffers the current call is writing".
        tls = self._thread_local_storage
        clone_sets = getattr(tls, "clone_sets", None)
        if clone_sets is None:
            raw0, raw1, raw2 = raw
            clone_sets = [
                (
                    torch.empty_like(raw0),
                    torch.empty_like(raw1),
                    torch.empty_like(raw2),
                )
                for _ in range(3)  # pipeline depth + flush headroom
            ]
            tls.clone_sets = clone_sets
            tls.clone_idx = 0
        idx = tls.clone_idx
        clones = clone_sets[idx]
        tls.clone_idx = (idx + 1) % len(clone_sets)

        # Enter the inference stream without the ``torch.cuda.stream(...)``
        # context manager — its save-and-restore costs ~20µs per call on
        # Orin. We restore the current stream explicitly at the end.
        prev_stream = torch.cuda.current_stream(self._device)
        torch.cuda.set_stream(stream)
        try:
            raw0, raw1, raw2 = raw
            clones[0].copy_(raw0, non_blocking=True)
            clones[1].copy_(raw1, non_blocking=True)
            clones[2].copy_(raw2, non_blocking=True)
            # Record "consumer done" right after the clone so the next
            # graph replay can wait on this event and overwrite the
            # graph's own output buffers without colliding with the
            # in-flight future. We reuse a single event object.
            ev = graph_state.consumer_done_event
            if ev is None:
                ev = torch.cuda.Event()
                graph_state.consumer_done_event = ev
            ev.record(stream)
        finally:
            torch.cuda.set_stream(prev_stream)

        # The reusable event is recorded *after* the clone copies finish.
        # That gives postprocess the correct readiness point for the
        # clone buffers, and it also marks when the graph-owned output
        # buffers are free to be overwritten by the next replay.
        clones[0]._trt_produce_event = ev  # type: ignore[attr-defined]
        return _DirectInferenceFuture(
            self, clones, pre_processing_meta, ev, kwargs
        )

    def post_process(
        self,
        model_results: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        pre_processing_meta: List[PreProcessingMetadata],
        confidence: Confidence = "default",
        mask_format: InstanceSegmentationMaskFormat = "dense",
        **kwargs,
    ) -> List[InstanceDetections]:
        if isinstance(model_results, list) and (
            len(model_results) == 0
            or all(isinstance(result, InstanceDetections) for result in model_results)
        ):
            return model_results
        if mask_format not in self.supported_mask_formats:
            raise ModelInputError(
                message=f"RFDetr Instance Segmentation models support the following mask "
                f"formats: {self.supported_mask_formats}. Requested format: {mask_format} "
                f"is not supported. If you see this error while running on Roboflow platform, "
                f"contact support or raise an issue at https://github.com/roboflow/inference/issues. "
                f"When running locally - please verify your integration to make sure that appropriate "
                f"value of `mask_format` parameter is set.",
                help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
            )
        confidence_filter = ConfidenceFilter(
            confidence=confidence,
            recommended_parameters=self.recommended_parameters,
            default_confidence=INFERENCE_MODELS_RFDETR_DEFAULT_CONFIDENCE,
        )
        with torch.cuda.stream(self._post_process_stream):
            prepare_trt_results_for_consumer(
                model_results=model_results,
                consumer_stream=self._post_process_stream,
            )
            bboxes, logits, masks = model_results
            if mask_format == "dense":
                results = post_process_instance_segmentation_results(
                    bboxes=bboxes,
                    logits=logits,
                    masks=masks,
                    pre_processing_meta=pre_processing_meta,
                    threshold=confidence_filter.get_threshold(self.class_names),
                    num_classes=len(self.class_names),
                    classes_re_mapping=self._classes_re_mapping,
                )
            else:
                results = post_process_instance_segmentation_results_to_rle_masks(
                    bboxes=bboxes,
                    logits=logits,
                    masks=masks,
                    pre_processing_meta=pre_processing_meta,
                    threshold=confidence_filter.get_threshold(self.class_names),
                    num_classes=len(self.class_names),
                    classes_re_mapping=self._classes_re_mapping,
                    emit_in_kernel_rle=kwargs.get("response_mask_format") == "rle",
                    defer_count_to_adapter=kwargs.get("defer_count_to_adapter", False),
                )
            mark_trt_results_consumed(
                model_results=model_results,
                consumer_stream=self._post_process_stream,
            )
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
