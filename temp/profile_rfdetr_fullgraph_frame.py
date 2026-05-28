import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from inference_models import AutoModel
from inference_models.configuration import INFERENCE_MODELS_RFDETR_DEFAULT_CONFIDENCE
from inference_models.models.common.cuda import use_cuda_context
from inference_models.models.common.roboflow.post_processing import ConfidenceFilter
from inference_models.models.common.trt import _trt_dtype_to_torch
from inference_models.models.rfdetr.common import build_deferred_dense_postproc_detection
from inference_models.models.rfdetr.triton_fullpostproc import (
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
    rfdetr_topk_merge_finalize_triton_kernel,
    rfdetr_topk_partial_triton_kernel,
    get_rfdetr_triton_postproc_geometry,
)


VIDEO_PATH = os.environ.get(
    "VIDEO_PATH", "/home/ubuntu/inference/vehicles_312px.mp4"
)
DEVICE = os.environ.get("DEVICE", "cuda:0")
WARMUP = int(os.environ.get("WARMUP", "10"))
CYCLES = int(os.environ.get("CYCLES", "40"))
MODEL_ID = os.environ.get("MODEL_ID", "rfdetr-seg-nano")
CONFIDENCE = os.environ.get("CONFIDENCE", "default")
GRAPH_POSTPROC_MODE = os.environ.get("GRAPH_POSTPROC_MODE", "full_events")


def load_frames(video_path: str, count: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(count):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames.append(frame)
    cap.release()
    if len(frames) < count:
        raise RuntimeError(
            f"Could not read {count} frames from {video_path}; got {len(frames)}."
        )
    return frames


def load_frame(video_path: str):
    return load_frames(video_path=video_path, count=1)[0]


def materialize_deferred_detection(det) -> Dict[str, np.ndarray]:
    done_event = getattr(det, "_postproc_done_event")
    done_event.synchronize()
    counter_gpu = getattr(det, "_counter_gpu")
    combined_gpu = getattr(det, "_combined_gpu")
    mask_packed_gpu = getattr(det, "_mask_packed_gpu")
    n_survivors = int(counter_gpu.cpu().item())
    combined = combined_gpu[:n_survivors].cpu().numpy().copy()
    mask_packed = mask_packed_gpu[:n_survivors].cpu().numpy().copy()
    return {
        "count": n_survivors,
        "combined": combined,
        "mask_packed": mask_packed,
    }


def assert_detection_buffers_equal(
    baseline: Dict[str, np.ndarray],
    candidate: Dict[str, np.ndarray],
    label: str,
) -> None:
    if baseline["count"] != candidate["count"]:
        raise RuntimeError(
            f"{label}: count mismatch baseline={baseline['count']} "
            f"candidate={candidate['count']}"
        )
    if not np.array_equal(baseline["combined"], candidate["combined"]):
        raise RuntimeError(f"{label}: combined detection buffer mismatch.")
    if not np.array_equal(baseline["mask_packed"], candidate["mask_packed"]):
        raise RuntimeError(f"{label}: packed mask buffer mismatch.")


def launch_selection_only(
    *,
    logits_2d: torch.Tensor,
    bboxes_2d: torch.Tensor,
    thr_tensor: torch.Tensor,
    cmap: torch.Tensor,
    selected_queries: torch.Tensor,
    combined: torch.Tensor,
    counter: torch.Tensor,
    partial_topk: torch.Tensor,
    num_classes: int,
    geometry,
    per_class: bool,
    has_remap: bool,
    num_queries: int,
    num_classes_total: int,
    selection_block: int,
    num_query_blocks: int,
) -> None:
    rfdetr_topk_partial_triton_kernel[(num_query_blocks,)](
        logits_2d,
        partial_topk,
        logits_2d.stride(0),
        TOPK_PAD=selection_block,
        CLASS_BLOCK=selection_block,
        QUERY_BLOCK=_TOPK_QUERY_BLOCK,
        NUM_QUERIES=num_queries,
        NUM_CLASSES_TOTAL=num_classes_total,
        num_warps=4,
        num_stages=1,
    )
    rfdetr_topk_merge_finalize_triton_kernel[(1,)](
        partial_topk,
        bboxes_2d,
        thr_tensor,
        cmap,
        selected_queries,
        combined,
        counter,
        int(num_classes),
        int(geometry.denorm_w),
        int(geometry.denorm_h),
        int(geometry.pad_left),
        int(geometry.pad_top),
        float(geometry.inv_scale_w),
        float(geometry.inv_scale_h),
        int(geometry.output_offset_x),
        int(geometry.output_offset_y),
        bboxes_2d.stride(0),
        PER_CLASS=1 if per_class else 0,
        HAS_REMAPPING=1 if has_remap else 0,
        NUM_QUERY_BLOCKS=num_query_blocks,
        NUM_QUERIES=num_queries,
        NUM_CLASSES_TOTAL=num_classes_total,
        TOPK_PAD=selection_block,
        num_warps=4,
        num_stages=1,
    )


def launch_partial_only(
    *,
    logits_2d: torch.Tensor,
    partial_topk: torch.Tensor,
    num_queries: int,
    num_classes_total: int,
    selection_block: int,
    num_query_blocks: int,
) -> None:
    rfdetr_topk_partial_triton_kernel[(num_query_blocks,)](
        logits_2d,
        partial_topk,
        logits_2d.stride(0),
        TOPK_PAD=selection_block,
        CLASS_BLOCK=selection_block,
        QUERY_BLOCK=_TOPK_QUERY_BLOCK,
        NUM_QUERIES=num_queries,
        NUM_CLASSES_TOTAL=num_classes_total,
        num_warps=4,
        num_stages=1,
    )


def launch_graph_postproc_mode(
    *,
    logits_out: torch.Tensor,
    bboxes_out: torch.Tensor,
    masks_out: torch.Tensor,
    thr_tensor: torch.Tensor,
    cmap: torch.Tensor,
    y_indices: torch.Tensor,
    y_weights: torch.Tensor,
    y_counts: torch.Tensor,
    x_indices: torch.Tensor,
    x_weights: torch.Tensor,
    x_counts: torch.Tensor,
    selected_queries: torch.Tensor,
    combined: torch.Tensor,
    counter: torch.Tensor,
    mask_bin: torch.Tensor,
    partial_topk: torch.Tensor,
    output_copy_buffers: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    num_classes: int,
    geometry,
    orig_h: int,
    orig_w: int,
    per_class: bool,
    has_remap: bool,
    pack_dense_masks: bool,
    simple_mask_fastpath: bool,
    num_queries: int,
    num_classes_total: int,
    mask_h: int,
    mask_w: int,
    selection_block: int,
    num_query_blocks: int,
    max_y_taps: int,
    max_x_taps: int,
    selection_done_event: Optional[torch.cuda.Event],
    done_event: Optional[torch.cuda.Event],
) -> None:
    if GRAPH_POSTPROC_MODE == "trt_only":
        return
    if GRAPH_POSTPROC_MODE == "copy_outputs":
        if output_copy_buffers is None:
            raise RuntimeError("Output copy buffers were not allocated.")
        copy_bboxes, copy_logits, copy_masks = output_copy_buffers
        copy_bboxes.copy_(bboxes_out, non_blocking=True)
        copy_logits.copy_(logits_out, non_blocking=True)
        copy_masks.copy_(masks_out, non_blocking=True)
        return
    if GRAPH_POSTPROC_MODE == "partial_only":
        launch_partial_only(
            logits_2d=logits_out[0],
            partial_topk=partial_topk,
            num_queries=num_queries,
            num_classes_total=num_classes_total,
            selection_block=selection_block,
            num_query_blocks=num_query_blocks,
        )
        return
    if GRAPH_POSTPROC_MODE == "selection_only":
        launch_selection_only(
            logits_2d=logits_out[0],
            bboxes_2d=bboxes_out[0],
            thr_tensor=thr_tensor,
            cmap=cmap,
            selected_queries=selected_queries,
            combined=combined,
            counter=counter,
            partial_topk=partial_topk,
            num_classes=num_classes,
            geometry=geometry,
            per_class=per_class,
            has_remap=has_remap,
            num_queries=num_queries,
            num_classes_total=num_classes_total,
            selection_block=selection_block,
            num_query_blocks=num_query_blocks,
        )
        return
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
        num_classes=num_classes,
        geometry=geometry,
        orig_h=orig_h,
        orig_w=orig_w,
        per_class=per_class,
        has_remap=has_remap,
        pack_dense_masks=pack_dense_masks,
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


@dataclass
class FullGraphRunner:
    model: object
    frame: np.ndarray
    preproc_meta: list
    stream: torch.cuda.Stream
    cuda_graph: torch.cuda.CUDAGraph
    graph_context: object
    done_event: torch.cuda.Event
    selection_done_event: torch.cuda.Event
    combined: torch.Tensor
    mask_bin: torch.Tensor
    counter: torch.Tensor
    input_buffer: torch.Tensor
    src_gpu: torch.Tensor
    preproc_out: torch.Tensor
    output_buffers: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    output_copy_buffers: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]

    @classmethod
    def build(cls, model, frame: np.ndarray) -> "FullGraphRunner":
        print("  build: init fast path", flush=True)
        # Initialize fast-path state and metadata on the exact frame shape.
        pre_processed_images, preproc_meta = model.pre_process(frame)
        state = model._fast_path_state
        if state is None:
            raise RuntimeError("Fast preprocess state was not initialized.")

        preproc_out = state.out_buffers[0]
        input_name = model._input_name
        output_names = model._output_names
        device = model._device
        engine = model._engine
        graph_context = engine.create_execution_context()
        status = graph_context.set_input_shape(input_name, tuple(preproc_out.shape))
        if not status:
            raise RuntimeError("Failed to set TRT input shape for unified graph.")
        status = graph_context.set_tensor_address(input_name, preproc_out.data_ptr())
        if not status:
            raise RuntimeError("Failed to bind TRT input buffer for unified graph.")

        output_buffers = []
        for output_name in output_names:
            output_shape = graph_context.get_tensor_shape(output_name)
            output_dtype = _trt_dtype_to_torch(engine.get_tensor_dtype(output_name))
            output_buffer = torch.empty(
                tuple(output_shape), dtype=output_dtype, device=device
            )
            graph_context.set_tensor_address(output_name, output_buffer.data_ptr())
            output_buffers.append(output_buffer)
        bboxes_out, logits_out, masks_out = output_buffers
        output_copy_buffers = (
            tuple(torch.empty_like(buffer) for buffer in output_buffers)
            if GRAPH_POSTPROC_MODE == "copy_outputs"
            else None
        )

        meta = preproc_meta[0]
        denorm_size = meta.nonsquare_intermediate_size or meta.inference_size
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

        print("  build: threshold + mapping", flush=True)
        confidence_filter = ConfidenceFilter(
            confidence=CONFIDENCE,
            recommended_parameters=model.recommended_parameters,
            default_confidence=INFERENCE_MODELS_RFDETR_DEFAULT_CONFIDENCE,
        )
        threshold = confidence_filter.get_threshold(model.class_names)
        thr_tensor, per_class = _prepare_threshold(
            threshold, device, len(model.class_names)
        )
        class_mapping = (
            model._classes_re_mapping.class_mapping
            if model._classes_re_mapping is not None
            else None
        )
        if class_mapping is not None:
            has_remap = True
            cmap = _get_class_mapping_int32(class_mapping, device)
        else:
            has_remap = False
            cmap = _get_empty_int32_on_device(device)

        stream = torch.cuda.Stream(device=device)
        done_event = torch.cuda.Event(external=True)
        selection_done_event = torch.cuda.Event(external=True)
        cuda_graph = torch.cuda.CUDAGraph()

        print("  build: warmup launch", flush=True)
        with use_cuda_context(context=model._cuda_context):
            with torch.cuda.stream(stream):
                state.src_gpu.copy_(state.pinned_host, non_blocking=True)
                model_module = __import__(
                    "inference_models.models.rfdetr.rfdetr_instance_segmentation_trt",
                    fromlist=["triton_preprocess_rfdetr_stretch"],
                )
                model_module.triton_preprocess_rfdetr_stretch(
                    src=state.src_gpu,
                    tables=state.tables,
                    target_h=model._inference_config.network_input.training_input_size.height,
                    target_w=model._inference_config.network_input.training_input_size.width,
                    means=tuple(float(v) for v in model._inference_config.network_input.normalization[0]),
                    stds=tuple(float(v) for v in model._inference_config.network_input.normalization[1]),
                    swap_rb=True,
                    out=preproc_out,
                )
                status = graph_context.execute_async_v3(stream_handle=stream.cuda_stream)
                if not status:
                    raise RuntimeError("TRT warmup failed for unified graph.")
                launch_graph_postproc_mode(
                    logits_out=logits_out,
                    bboxes_out=bboxes_out,
                    masks_out=masks_out,
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
                    output_copy_buffers=output_copy_buffers,
                    num_classes=len(model.class_names),
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
                    selection_done_event=(
                        selection_done_event
                        if GRAPH_POSTPROC_MODE == "full_events"
                        else None
                    ),
                    done_event=(
                        done_event if GRAPH_POSTPROC_MODE == "full_events" else None
                    ),
                )
            stream.synchronize()
            print("  build: capture graph", flush=True)
            with torch.cuda.graph(cuda_graph, stream=stream):
                state.src_gpu.copy_(state.pinned_host, non_blocking=True)
                model_module.triton_preprocess_rfdetr_stretch(
                    src=state.src_gpu,
                    tables=state.tables,
                    target_h=model._inference_config.network_input.training_input_size.height,
                    target_w=model._inference_config.network_input.training_input_size.width,
                    means=tuple(float(v) for v in model._inference_config.network_input.normalization[0]),
                    stds=tuple(float(v) for v in model._inference_config.network_input.normalization[1]),
                    swap_rb=True,
                    out=preproc_out,
                )
                status = graph_context.execute_async_v3(stream_handle=stream.cuda_stream)
                if not status:
                    raise RuntimeError("TRT capture failed for unified graph.")
                launch_graph_postproc_mode(
                    logits_out=logits_out,
                    bboxes_out=bboxes_out,
                    masks_out=masks_out,
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
                    output_copy_buffers=output_copy_buffers,
                    num_classes=len(model.class_names),
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
                    selection_done_event=(
                        selection_done_event
                        if GRAPH_POSTPROC_MODE == "full_events"
                        else None
                    ),
                    done_event=(
                        done_event if GRAPH_POSTPROC_MODE == "full_events" else None
                    ),
                )

        print("  build: done", flush=True)
        return cls(
            model=model,
            frame=frame,
            preproc_meta=preproc_meta,
            stream=stream,
            cuda_graph=cuda_graph,
            graph_context=graph_context,
            done_event=done_event,
            selection_done_event=selection_done_event,
            combined=combined,
            mask_bin=mask_bin,
            counter=counter,
            input_buffer=state.pinned_host,
            src_gpu=state.src_gpu,
            preproc_out=preproc_out,
            output_buffers=(bboxes_out, logits_out, masks_out),
            output_copy_buffers=output_copy_buffers,
        )

    def replay_once(self) -> Optional[Dict[str, np.ndarray]]:
        print("  replay: host copy", flush=True)
        np.copyto(self.input_buffer.numpy(), self.frame, casting="no")
        with use_cuda_context(context=self.model._cuda_context):
            print("  replay: graph replay", flush=True)
            self.cuda_graph.replay()
            if GRAPH_POSTPROC_MODE not in {"full_noevents", "full_events"}:
                print("  replay: stream sync", flush=True)
                self.stream.synchronize()
                return None
            print("  replay: done_event sync", flush=True)
            self.done_event.synchronize()
        print("  replay: materialize", flush=True)
        det = build_deferred_dense_postproc_detection(
            combined=self.combined,
            mask_packed_gpu=self.mask_bin,
            counter=self.counter,
            selection_done_event=self.selection_done_event,
            done_event=self.done_event,
            orig_h=self.preproc_meta[0].original_size.height,
            orig_w=self.preproc_meta[0].original_size.width,
        )
        return materialize_deferred_detection(det)

    def benchmark(self, cycles: int) -> float:
        start = time.perf_counter()
        for _ in range(cycles):
            np.copyto(self.input_buffer.numpy(), self.frame, casting="no")
            with use_cuda_context(context=self.model._cuda_context):
                self.cuda_graph.replay()
        with use_cuda_context(context=self.model._cuda_context):
            self.done_event.synchronize()
        elapsed = time.perf_counter() - start
        return cycles / elapsed if elapsed > 0 else 0.0


@dataclass
class TwoStepGraphStep:
    frame: np.ndarray
    preproc_meta: list
    pinned_host: torch.Tensor
    src_gpu: torch.Tensor
    preproc_out: torch.Tensor
    graph_context: object
    output_buffers: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    output_copy_buffers: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    combined: torch.Tensor
    mask_bin: torch.Tensor
    counter: torch.Tensor
    selected_queries: torch.Tensor
    partial_topk: torch.Tensor
    selection_done_event: torch.cuda.Event
    done_event: torch.cuda.Event


@dataclass
class TwoStepFullGraphRunner:
    model: object
    steps: Tuple[TwoStepGraphStep, TwoStepGraphStep]
    stream: torch.cuda.Stream
    cuda_graph: torch.cuda.CUDAGraph

    @classmethod
    def build(
        cls, model, frames: Sequence[np.ndarray]
    ) -> "TwoStepFullGraphRunner":
        if len(frames) != 2:
            raise ValueError(
                f"TwoStepFullGraphRunner requires exactly 2 frames, got {len(frames)}."
            )
        print("  build pair: init fast path", flush=True)
        per_step_preproc = [model.pre_process(frame) for frame in frames]
        state = model._fast_path_state
        if state is None:
            raise RuntimeError("Fast preprocess state was not initialized.")

        model_module = __import__(
            "inference_models.models.rfdetr.rfdetr_instance_segmentation_trt",
            fromlist=["triton_preprocess_rfdetr_stretch"],
        )
        triton_preprocess = model_module.triton_preprocess_rfdetr_stretch
        target_h = model._inference_config.network_input.training_input_size.height
        target_w = model._inference_config.network_input.training_input_size.width
        means = tuple(
            float(v)
            for v in model._inference_config.network_input.normalization[0]
        )
        stds = tuple(
            float(v)
            for v in model._inference_config.network_input.normalization[1]
        )
        device = model._device
        steps = []
        for idx, (frame, (preproc_out, preproc_meta)) in enumerate(
            zip(frames, per_step_preproc)
        ):
            preproc_out = preproc_out
            pinned_host = torch.empty_like(
                state.pinned_host, device="cpu", pin_memory=True
            )
            np.copyto(pinned_host.numpy(), frame, casting="no")
            src_gpu = torch.empty_like(state.src_gpu)
            graph_context = model._engine.create_execution_context()
            status = graph_context.set_input_shape(
                model._input_name, tuple(preproc_out.shape)
            )
            if not status:
                raise RuntimeError(
                    f"Failed to set TRT input shape for pair step {idx}."
                )
            status = graph_context.set_tensor_address(
                model._input_name, preproc_out.data_ptr()
            )
            if not status:
                raise RuntimeError(
                    f"Failed to bind TRT input buffer for pair step {idx}."
                )

            output_buffers = []
            for output_name in model._output_names:
                output_shape = graph_context.get_tensor_shape(output_name)
                output_dtype = _trt_dtype_to_torch(
                    model._engine.get_tensor_dtype(output_name)
                )
                output_buffer = torch.empty(
                    tuple(output_shape), dtype=output_dtype, device=device
                )
                graph_context.set_tensor_address(
                    output_name, output_buffer.data_ptr()
                )
                output_buffers.append(output_buffer)
            bboxes_out, logits_out, masks_out = output_buffers
            output_copy_buffers = (
                tuple(torch.empty_like(buffer) for buffer in output_buffers)
                if GRAPH_POSTPROC_MODE == "copy_outputs"
                else None
            )

            meta = preproc_meta[0]
            denorm_size = meta.nonsquare_intermediate_size or meta.inference_size
            num_queries = int(logits_out.shape[1])
            num_classes_total = int(logits_out.shape[2])
            mask_h = int(masks_out.shape[2])
            mask_w = int(masks_out.shape[3])
            selection_block = _next_power_of_two(num_queries)
            num_query_blocks = (
                num_queries + _TOPK_QUERY_BLOCK - 1
            ) // _TOPK_QUERY_BLOCK

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

            confidence_filter = ConfidenceFilter(
                confidence=CONFIDENCE,
                recommended_parameters=model.recommended_parameters,
                default_confidence=INFERENCE_MODELS_RFDETR_DEFAULT_CONFIDENCE,
            )
            threshold = confidence_filter.get_threshold(model.class_names)
            thr_tensor, per_class = _prepare_threshold(
                threshold, device, len(model.class_names)
            )
            class_mapping = (
                model._classes_re_mapping.class_mapping
                if model._classes_re_mapping is not None
                else None
            )
            if class_mapping is not None:
                has_remap = True
                cmap = _get_class_mapping_int32(class_mapping, device)
            else:
                has_remap = False
                cmap = _get_empty_int32_on_device(device)

            selection_done_event = torch.cuda.Event(external=True)
            done_event = torch.cuda.Event(external=True)
            step = TwoStepGraphStep(
                frame=frame,
                preproc_meta=preproc_meta,
                pinned_host=pinned_host,
                src_gpu=src_gpu,
                preproc_out=preproc_out,
                graph_context=graph_context,
                output_buffers=(bboxes_out, logits_out, masks_out),
                output_copy_buffers=output_copy_buffers,
                combined=combined,
                mask_bin=mask_bin,
                counter=counter,
                selected_queries=selected_queries,
                partial_topk=partial_topk,
                selection_done_event=selection_done_event,
                done_event=done_event,
            )
            step._thr_tensor = thr_tensor  # type: ignore[attr-defined]
            step._cmap = cmap  # type: ignore[attr-defined]
            step._per_class = per_class  # type: ignore[attr-defined]
            step._has_remap = has_remap  # type: ignore[attr-defined]
            step._geometry = geometry  # type: ignore[attr-defined]
            step._simple_mask_fastpath = simple_mask_fastpath  # type: ignore[attr-defined]
            step._num_queries = num_queries  # type: ignore[attr-defined]
            step._num_classes_total = num_classes_total  # type: ignore[attr-defined]
            step._mask_h = mask_h  # type: ignore[attr-defined]
            step._mask_w = mask_w  # type: ignore[attr-defined]
            step._selection_block = selection_block  # type: ignore[attr-defined]
            step._num_query_blocks = num_query_blocks  # type: ignore[attr-defined]
            step._max_y_taps = max_y_taps  # type: ignore[attr-defined]
            step._max_x_taps = max_x_taps  # type: ignore[attr-defined]
            step._y_indices = y_indices  # type: ignore[attr-defined]
            step._y_weights = y_weights  # type: ignore[attr-defined]
            step._y_counts = y_counts  # type: ignore[attr-defined]
            step._x_indices = x_indices  # type: ignore[attr-defined]
            step._x_weights = x_weights  # type: ignore[attr-defined]
            step._x_counts = x_counts  # type: ignore[attr-defined]
            steps.append(step)

        stream = torch.cuda.Stream(device=device)
        cuda_graph = torch.cuda.CUDAGraph()

        print("  build pair: warmup launch", flush=True)
        with use_cuda_context(context=model._cuda_context):
            with torch.cuda.stream(stream):
                for step in steps:
                    step.src_gpu.copy_(step.pinned_host, non_blocking=True)
                    triton_preprocess(
                        src=step.src_gpu,
                        tables=state.tables,
                        target_h=target_h,
                        target_w=target_w,
                        means=means,
                        stds=stds,
                        swap_rb=True,
                        out=step.preproc_out,
                    )
                    status = step.graph_context.execute_async_v3(
                        stream_handle=stream.cuda_stream
                    )
                    if not status:
                        raise RuntimeError("TRT warmup failed for pair graph.")
                    bboxes_out, logits_out, masks_out = step.output_buffers
                    launch_graph_postproc_mode(
                        logits_out=logits_out,
                        bboxes_out=bboxes_out,
                        masks_out=masks_out,
                        thr_tensor=step._thr_tensor,  # type: ignore[attr-defined]
                        cmap=step._cmap,  # type: ignore[attr-defined]
                        y_indices=step._y_indices,  # type: ignore[attr-defined]
                        y_weights=step._y_weights,  # type: ignore[attr-defined]
                        y_counts=step._y_counts,  # type: ignore[attr-defined]
                        x_indices=step._x_indices,  # type: ignore[attr-defined]
                        x_weights=step._x_weights,  # type: ignore[attr-defined]
                        x_counts=step._x_counts,  # type: ignore[attr-defined]
                        selected_queries=step.selected_queries,
                        combined=step.combined,
                        counter=step.counter,
                        mask_bin=step.mask_bin,
                        partial_topk=step.partial_topk,
                        output_copy_buffers=step.output_copy_buffers,
                        num_classes=len(model.class_names),
                        geometry=step._geometry,  # type: ignore[attr-defined]
                        orig_h=step.preproc_meta[0].original_size.height,
                        orig_w=step.preproc_meta[0].original_size.width,
                        per_class=step._per_class,  # type: ignore[attr-defined]
                        has_remap=step._has_remap,  # type: ignore[attr-defined]
                        pack_dense_masks=True,
                        simple_mask_fastpath=step._simple_mask_fastpath,  # type: ignore[attr-defined]
                        num_queries=step._num_queries,  # type: ignore[attr-defined]
                        num_classes_total=step._num_classes_total,  # type: ignore[attr-defined]
                        mask_h=step._mask_h,  # type: ignore[attr-defined]
                        mask_w=step._mask_w,  # type: ignore[attr-defined]
                        selection_block=step._selection_block,  # type: ignore[attr-defined]
                        num_query_blocks=step._num_query_blocks,  # type: ignore[attr-defined]
                        max_y_taps=step._max_y_taps,  # type: ignore[attr-defined]
                        max_x_taps=step._max_x_taps,  # type: ignore[attr-defined]
                        selection_done_event=(
                            step.selection_done_event
                            if GRAPH_POSTPROC_MODE == "full_events"
                            else None
                        ),
                        done_event=(
                            step.done_event
                            if GRAPH_POSTPROC_MODE == "full_events"
                            else None
                        ),
                    )
            stream.synchronize()

            print("  build pair: capture graph", flush=True)
            with torch.cuda.graph(cuda_graph, stream=stream):
                for step in steps:
                    step.src_gpu.copy_(step.pinned_host, non_blocking=True)
                    triton_preprocess(
                        src=step.src_gpu,
                        tables=state.tables,
                        target_h=target_h,
                        target_w=target_w,
                        means=means,
                        stds=stds,
                        swap_rb=True,
                        out=step.preproc_out,
                    )
                    status = step.graph_context.execute_async_v3(
                        stream_handle=stream.cuda_stream
                    )
                    if not status:
                        raise RuntimeError("TRT capture failed for pair graph.")
                    bboxes_out, logits_out, masks_out = step.output_buffers
                    launch_graph_postproc_mode(
                        logits_out=logits_out,
                        bboxes_out=bboxes_out,
                        masks_out=masks_out,
                        thr_tensor=step._thr_tensor,  # type: ignore[attr-defined]
                        cmap=step._cmap,  # type: ignore[attr-defined]
                        y_indices=step._y_indices,  # type: ignore[attr-defined]
                        y_weights=step._y_weights,  # type: ignore[attr-defined]
                        y_counts=step._y_counts,  # type: ignore[attr-defined]
                        x_indices=step._x_indices,  # type: ignore[attr-defined]
                        x_weights=step._x_weights,  # type: ignore[attr-defined]
                        x_counts=step._x_counts,  # type: ignore[attr-defined]
                        selected_queries=step.selected_queries,
                        combined=step.combined,
                        counter=step.counter,
                        mask_bin=step.mask_bin,
                        partial_topk=step.partial_topk,
                        output_copy_buffers=step.output_copy_buffers,
                        num_classes=len(model.class_names),
                        geometry=step._geometry,  # type: ignore[attr-defined]
                        orig_h=step.preproc_meta[0].original_size.height,
                        orig_w=step.preproc_meta[0].original_size.width,
                        per_class=step._per_class,  # type: ignore[attr-defined]
                        has_remap=step._has_remap,  # type: ignore[attr-defined]
                        pack_dense_masks=True,
                        simple_mask_fastpath=step._simple_mask_fastpath,  # type: ignore[attr-defined]
                        num_queries=step._num_queries,  # type: ignore[attr-defined]
                        num_classes_total=step._num_classes_total,  # type: ignore[attr-defined]
                        mask_h=step._mask_h,  # type: ignore[attr-defined]
                        mask_w=step._mask_w,  # type: ignore[attr-defined]
                        selection_block=step._selection_block,  # type: ignore[attr-defined]
                        num_query_blocks=step._num_query_blocks,  # type: ignore[attr-defined]
                        max_y_taps=step._max_y_taps,  # type: ignore[attr-defined]
                        max_x_taps=step._max_x_taps,  # type: ignore[attr-defined]
                        selection_done_event=(
                            step.selection_done_event
                            if GRAPH_POSTPROC_MODE == "full_events"
                            else None
                        ),
                        done_event=(
                            step.done_event
                            if GRAPH_POSTPROC_MODE == "full_events"
                            else None
                        ),
                    )

        print("  build pair: done", flush=True)
        return cls(
            model=model,
            steps=(steps[0], steps[1]),
            stream=stream,
            cuda_graph=cuda_graph,
        )

    def replay_once(self) -> Optional[List[Dict[str, np.ndarray]]]:
        print("  replay pair: host copy", flush=True)
        for step in self.steps:
            np.copyto(step.pinned_host.numpy(), step.frame, casting="no")
        with use_cuda_context(context=self.model._cuda_context):
            print("  replay pair: graph replay", flush=True)
            self.cuda_graph.replay()
            if GRAPH_POSTPROC_MODE not in {"full_noevents", "full_events"}:
                print("  replay pair: stream sync", flush=True)
                self.stream.synchronize()
                return None
            print("  replay pair: done_event sync", flush=True)
            self.steps[-1].done_event.synchronize()
        print("  replay pair: materialize", flush=True)
        detections = []
        for step in self.steps:
            det = build_deferred_dense_postproc_detection(
                combined=step.combined,
                mask_packed_gpu=step.mask_bin,
                counter=step.counter,
                selection_done_event=step.selection_done_event,
                done_event=step.done_event,
                orig_h=step.preproc_meta[0].original_size.height,
                orig_w=step.preproc_meta[0].original_size.width,
            )
            detections.append(materialize_deferred_detection(det))
        return detections

    def benchmark(self, cycles: int) -> float:
        start = time.perf_counter()
        with use_cuda_context(context=self.model._cuda_context):
            for _ in range(cycles):
                for step in self.steps:
                    np.copyto(step.pinned_host.numpy(), step.frame, casting="no")
                self.cuda_graph.replay()
            for step in self.steps:
                step.done_event.synchronize()
        elapsed = time.perf_counter() - start
        return (len(self.steps) * cycles) / elapsed if elapsed > 0 else 0.0


@dataclass
class ParallelTwoStepFullGraphRunner:
    model: object
    steps: Tuple[TwoStepGraphStep, TwoStepGraphStep]
    capture_stream: torch.cuda.Stream
    lane_streams: Tuple[torch.cuda.Stream, torch.cuda.Stream]
    cuda_graph: torch.cuda.CUDAGraph

    @classmethod
    def build(
        cls, model, frames: Sequence[np.ndarray]
    ) -> "ParallelTwoStepFullGraphRunner":
        if len(frames) != 2:
            raise ValueError(
                "ParallelTwoStepFullGraphRunner requires exactly 2 frames."
            )
        print("  build pair parallel: init fast path", flush=True)
        per_step_preproc = [model.pre_process(frame) for frame in frames]
        state = model._fast_path_state
        if state is None:
            raise RuntimeError("Fast preprocess state was not initialized.")

        model_module = __import__(
            "inference_models.models.rfdetr.rfdetr_instance_segmentation_trt",
            fromlist=["triton_preprocess_rfdetr_stretch"],
        )
        triton_preprocess = model_module.triton_preprocess_rfdetr_stretch
        target_h = model._inference_config.network_input.training_input_size.height
        target_w = model._inference_config.network_input.training_input_size.width
        means = tuple(
            float(v) for v in model._inference_config.network_input.normalization[0]
        )
        stds = tuple(
            float(v) for v in model._inference_config.network_input.normalization[1]
        )
        device = model._device

        confidence_filter = ConfidenceFilter(
            confidence=CONFIDENCE,
            recommended_parameters=model.recommended_parameters,
            default_confidence=INFERENCE_MODELS_RFDETR_DEFAULT_CONFIDENCE,
        )
        threshold = confidence_filter.get_threshold(model.class_names)
        thr_tensor, per_class = _prepare_threshold(
            threshold, device, len(model.class_names)
        )
        class_mapping = (
            model._classes_re_mapping.class_mapping
            if model._classes_re_mapping is not None
            else None
        )
        if class_mapping is not None:
            has_remap = True
            cmap = _get_class_mapping_int32(class_mapping, device)
        else:
            has_remap = False
            cmap = _get_empty_int32_on_device(device)

        steps = []
        for idx, (frame, (preproc_out, preproc_meta)) in enumerate(
            zip(frames, per_step_preproc)
        ):
            pinned_host = torch.empty_like(
                state.pinned_host, device="cpu", pin_memory=True
            )
            np.copyto(pinned_host.numpy(), frame, casting="no")
            src_gpu = torch.empty_like(state.src_gpu)
            graph_context = model._engine.create_execution_context()
            status = graph_context.set_input_shape(
                model._input_name, tuple(preproc_out.shape)
            )
            if not status:
                raise RuntimeError(
                    f"Failed to set TRT input shape for parallel pair step {idx}."
                )
            status = graph_context.set_tensor_address(
                model._input_name, preproc_out.data_ptr()
            )
            if not status:
                raise RuntimeError(
                    f"Failed to bind TRT input buffer for parallel pair step {idx}."
                )

            output_buffers = []
            for output_name in model._output_names:
                output_shape = graph_context.get_tensor_shape(output_name)
                output_dtype = _trt_dtype_to_torch(
                    model._engine.get_tensor_dtype(output_name)
                )
                output_buffer = torch.empty(
                    tuple(output_shape), dtype=output_dtype, device=device
                )
                graph_context.set_tensor_address(
                    output_name, output_buffer.data_ptr()
                )
                output_buffers.append(output_buffer)
            bboxes_out, logits_out, masks_out = output_buffers
            output_copy_buffers = (
                tuple(torch.empty_like(buffer) for buffer in output_buffers)
                if GRAPH_POSTPROC_MODE == "copy_outputs"
                else None
            )

            meta = preproc_meta[0]
            denorm_size = meta.nonsquare_intermediate_size or meta.inference_size
            num_queries = int(logits_out.shape[1])
            num_classes_total = int(logits_out.shape[2])
            mask_h = int(masks_out.shape[2])
            mask_w = int(masks_out.shape[3])
            selection_block = _next_power_of_two(num_queries)
            num_query_blocks = (
                num_queries + _TOPK_QUERY_BLOCK - 1
            ) // _TOPK_QUERY_BLOCK

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

            selection_done_event = torch.cuda.Event(external=True)
            done_event = torch.cuda.Event(external=True)
            step = TwoStepGraphStep(
                frame=frame,
                preproc_meta=preproc_meta,
                pinned_host=pinned_host,
                src_gpu=src_gpu,
                preproc_out=preproc_out,
                graph_context=graph_context,
                output_buffers=(bboxes_out, logits_out, masks_out),
                output_copy_buffers=output_copy_buffers,
                combined=combined,
                mask_bin=mask_bin,
                counter=counter,
                selected_queries=selected_queries,
                partial_topk=partial_topk,
                selection_done_event=selection_done_event,
                done_event=done_event,
            )
            step._thr_tensor = thr_tensor  # type: ignore[attr-defined]
            step._cmap = cmap  # type: ignore[attr-defined]
            step._per_class = per_class  # type: ignore[attr-defined]
            step._has_remap = has_remap  # type: ignore[attr-defined]
            step._geometry = geometry  # type: ignore[attr-defined]
            step._simple_mask_fastpath = simple_mask_fastpath  # type: ignore[attr-defined]
            step._num_queries = num_queries  # type: ignore[attr-defined]
            step._num_classes_total = num_classes_total  # type: ignore[attr-defined]
            step._mask_h = mask_h  # type: ignore[attr-defined]
            step._mask_w = mask_w  # type: ignore[attr-defined]
            step._selection_block = selection_block  # type: ignore[attr-defined]
            step._num_query_blocks = num_query_blocks  # type: ignore[attr-defined]
            step._max_y_taps = max_y_taps  # type: ignore[attr-defined]
            step._max_x_taps = max_x_taps  # type: ignore[attr-defined]
            step._y_indices = y_indices  # type: ignore[attr-defined]
            step._y_weights = y_weights  # type: ignore[attr-defined]
            step._y_counts = y_counts  # type: ignore[attr-defined]
            step._x_indices = x_indices  # type: ignore[attr-defined]
            step._x_weights = x_weights  # type: ignore[attr-defined]
            step._x_counts = x_counts  # type: ignore[attr-defined]
            steps.append(step)

        capture_stream = torch.cuda.Stream(device=device)
        lane_streams = (
            torch.cuda.Stream(device=device),
            torch.cuda.Stream(device=device),
        )
        cuda_graph = torch.cuda.CUDAGraph()

        def _launch_step(step: TwoStepGraphStep, lane_stream: torch.cuda.Stream) -> None:
            with torch.cuda.stream(lane_stream):
                step.src_gpu.copy_(step.pinned_host, non_blocking=True)
                triton_preprocess(
                    src=step.src_gpu,
                    tables=state.tables,
                    target_h=target_h,
                    target_w=target_w,
                    means=means,
                    stds=stds,
                    swap_rb=True,
                    out=step.preproc_out,
                )
                status = step.graph_context.execute_async_v3(
                    stream_handle=lane_stream.cuda_stream
                )
                if not status:
                    raise RuntimeError(
                        "TRT launch failed for parallel pair graph."
                    )
                bboxes_out, logits_out, masks_out = step.output_buffers
                launch_graph_postproc_mode(
                    logits_out=logits_out,
                    bboxes_out=bboxes_out,
                    masks_out=masks_out,
                    thr_tensor=step._thr_tensor,  # type: ignore[attr-defined]
                    cmap=step._cmap,  # type: ignore[attr-defined]
                    y_indices=step._y_indices,  # type: ignore[attr-defined]
                    y_weights=step._y_weights,  # type: ignore[attr-defined]
                    y_counts=step._y_counts,  # type: ignore[attr-defined]
                    x_indices=step._x_indices,  # type: ignore[attr-defined]
                    x_weights=step._x_weights,  # type: ignore[attr-defined]
                    x_counts=step._x_counts,  # type: ignore[attr-defined]
                    selected_queries=step.selected_queries,
                    combined=step.combined,
                    counter=step.counter,
                    mask_bin=step.mask_bin,
                    partial_topk=step.partial_topk,
                    output_copy_buffers=step.output_copy_buffers,
                    num_classes=len(model.class_names),
                    geometry=step._geometry,  # type: ignore[attr-defined]
                    orig_h=step.preproc_meta[0].original_size.height,
                    orig_w=step.preproc_meta[0].original_size.width,
                    per_class=step._per_class,  # type: ignore[attr-defined]
                    has_remap=step._has_remap,  # type: ignore[attr-defined]
                    pack_dense_masks=True,
                    simple_mask_fastpath=step._simple_mask_fastpath,  # type: ignore[attr-defined]
                    num_queries=step._num_queries,  # type: ignore[attr-defined]
                    num_classes_total=step._num_classes_total,  # type: ignore[attr-defined]
                    mask_h=step._mask_h,  # type: ignore[attr-defined]
                    mask_w=step._mask_w,  # type: ignore[attr-defined]
                    selection_block=step._selection_block,  # type: ignore[attr-defined]
                    num_query_blocks=step._num_query_blocks,  # type: ignore[attr-defined]
                    max_y_taps=step._max_y_taps,  # type: ignore[attr-defined]
                    max_x_taps=step._max_x_taps,  # type: ignore[attr-defined]
                    selection_done_event=(
                        step.selection_done_event
                        if GRAPH_POSTPROC_MODE == "full_events"
                        else None
                    ),
                    done_event=(
                        step.done_event
                        if GRAPH_POSTPROC_MODE == "full_events"
                        else None
                    ),
                )

        print("  build pair parallel: warmup launch", flush=True)
        with use_cuda_context(context=model._cuda_context):
            with torch.cuda.stream(capture_stream):
                for lane_stream in lane_streams:
                    lane_stream.wait_stream(capture_stream)
                for step, lane_stream in zip(steps, lane_streams):
                    _launch_step(step, lane_stream)
                for lane_stream in lane_streams:
                    capture_stream.wait_stream(lane_stream)
            capture_stream.synchronize()

            print("  build pair parallel: capture graph", flush=True)
            with torch.cuda.graph(cuda_graph, stream=capture_stream):
                for lane_stream in lane_streams:
                    lane_stream.wait_stream(capture_stream)
                for step, lane_stream in zip(steps, lane_streams):
                    _launch_step(step, lane_stream)
                for lane_stream in lane_streams:
                    capture_stream.wait_stream(lane_stream)

        print("  build pair parallel: done", flush=True)
        return cls(
            model=model,
            steps=(steps[0], steps[1]),
            capture_stream=capture_stream,
            lane_streams=lane_streams,
            cuda_graph=cuda_graph,
        )

    def replay_once(self) -> Optional[List[Dict[str, np.ndarray]]]:
        print("  replay pair parallel: host copy", flush=True)
        for step in self.steps:
            np.copyto(step.pinned_host.numpy(), step.frame, casting="no")
        with use_cuda_context(context=self.model._cuda_context):
            print("  replay pair parallel: graph replay", flush=True)
            self.cuda_graph.replay()
            if GRAPH_POSTPROC_MODE not in {"full_noevents", "full_events"}:
                print("  replay pair parallel: capture stream sync", flush=True)
                self.capture_stream.synchronize()
                return None
            print("  replay pair parallel: done_event sync", flush=True)
            for step in self.steps:
                step.done_event.synchronize()
        print("  replay pair parallel: materialize", flush=True)
        detections = []
        for step in self.steps:
            det = build_deferred_dense_postproc_detection(
                combined=step.combined,
                mask_packed_gpu=step.mask_bin,
                counter=step.counter,
                selection_done_event=step.selection_done_event,
                done_event=step.done_event,
                orig_h=step.preproc_meta[0].original_size.height,
                orig_w=step.preproc_meta[0].original_size.width,
            )
            detections.append(materialize_deferred_detection(det))
        return detections

    def benchmark(self, cycles: int) -> float:
        start = time.perf_counter()
        with use_cuda_context(context=self.model._cuda_context):
            for _ in range(cycles):
                for step in self.steps:
                    np.copyto(step.pinned_host.numpy(), step.frame, casting="no")
                self.cuda_graph.replay()
            for step in self.steps:
                step.done_event.synchronize()
        elapsed = time.perf_counter() - start
        return (len(self.steps) * cycles) / elapsed if elapsed > 0 else 0.0


def run_current_gpu_path(model, frame: np.ndarray) -> Dict[str, np.ndarray]:
    pre_processed_images, preproc_meta = model.pre_process(frame)
    raw = model.forward(pre_processed_images)
    detections = model.post_process(
        raw,
        preproc_meta,
        confidence=CONFIDENCE,
        mask_format="rle",
        defer_count_to_adapter=True,
    )
    return materialize_deferred_detection(detections[0])


def benchmark_current_gpu_path(model, frame: np.ndarray, cycles: int) -> float:
    last_event = None
    start = time.perf_counter()
    for _ in range(cycles):
        pre_processed_images, preproc_meta = model.pre_process(frame)
        raw = model.forward(pre_processed_images)
        detections = model.post_process(
            raw,
            preproc_meta,
            confidence=CONFIDENCE,
            mask_format="rle",
            defer_count_to_adapter=True,
        )
        last_event = getattr(detections[0], "_postproc_done_event")
    if last_event is not None:
        last_event.synchronize()
    elapsed = time.perf_counter() - start
    return cycles / elapsed if elapsed > 0 else 0.0


def benchmark_current_gpu_path_pair(
    model, frames: Sequence[np.ndarray], cycles: int
) -> float:
    last_event = None
    start = time.perf_counter()
    for _ in range(cycles):
        for frame in frames:
            pre_processed_images, preproc_meta = model.pre_process(frame)
            raw = model.forward(pre_processed_images)
            detections = model.post_process(
                raw,
                preproc_meta,
                confidence=CONFIDENCE,
                mask_format="rle",
                defer_count_to_adapter=True,
            )
            last_event = getattr(detections[0], "_postproc_done_event")
    if last_event is not None:
        last_event.synchronize()
    elapsed = time.perf_counter() - start
    return (len(frames) * cycles) / elapsed if elapsed > 0 else 0.0


def benchmark_single_fullgraph_pair(
    runners: Sequence[FullGraphRunner], cycles: int
) -> float:
    if len(runners) == 0:
        return 0.0
    start = time.perf_counter()
    with use_cuda_context(context=runners[0].model._cuda_context):
        for _ in range(cycles):
            for runner in runners:
                np.copyto(runner.input_buffer.numpy(), runner.frame, casting="no")
                runner.cuda_graph.replay()
        for runner in runners:
            runner.done_event.synchronize()
    elapsed = time.perf_counter() - start
    return (len(runners) * cycles) / elapsed if elapsed > 0 else 0.0


def benchmark_single_fullgraph_pair_serial(
    runners: Sequence[FullGraphRunner], cycles: int
) -> float:
    if len(runners) == 0:
        return 0.0
    start = time.perf_counter()
    with use_cuda_context(context=runners[0].model._cuda_context):
        for _ in range(cycles):
            for runner in runners:
                np.copyto(runner.input_buffer.numpy(), runner.frame, casting="no")
                runner.cuda_graph.replay()
                runner.done_event.synchronize()
    elapsed = time.perf_counter() - start
    return (len(runners) * cycles) / elapsed if elapsed > 0 else 0.0


def main() -> None:
    print("stage: load frames", flush=True)
    frames = load_frames(VIDEO_PATH, count=2)
    print("stage: load model", flush=True)
    model = AutoModel.from_pretrained(
        model_id_or_path=MODEL_ID,
        device=torch.device(DEVICE),
        backend="trt",
    )

    print("stage: baseline once", flush=True)
    baselines = [run_current_gpu_path(model, frame) for frame in frames]

    print("stage: build single fullgraph runners", flush=True)
    single_runners = [FullGraphRunner.build(model, frame) for frame in frames]
    for idx, (runner, baseline) in enumerate(zip(single_runners, baselines)):
        replay = runner.replay_once()
        if replay is None:
            print(f"single replay: mode {GRAPH_POSTPROC_MODE} survived")
            return
        assert_detection_buffers_equal(
            baseline=baseline,
            candidate=replay,
            label=f"single_fullgraph[{idx}]",
        )
    print("parity: both single fullgraph replays matched exactly", flush=True)

    print("stage: build pair fullgraph runner", flush=True)
    pair_runner = TwoStepFullGraphRunner.build(model, frames)
    pair_replay = pair_runner.replay_once()
    if pair_replay is None:
        print(f"replay: mode {GRAPH_POSTPROC_MODE} survived")
        return
    for idx, (baseline, replay) in enumerate(zip(baselines, pair_replay)):
        assert_detection_buffers_equal(
            baseline=baseline,
            candidate=replay,
            label=f"pair_fullgraph[{idx}]",
        )
    print("parity: both pair fullgraph replays matched exactly", flush=True)

    print("stage: build pair parallel fullgraph runner", flush=True)
    pair_parallel_runner = ParallelTwoStepFullGraphRunner.build(model, frames)
    pair_parallel_replay = pair_parallel_runner.replay_once()
    if pair_parallel_replay is None:
        print(f"parallel replay: mode {GRAPH_POSTPROC_MODE} survived")
        return
    for idx, (baseline, replay) in enumerate(zip(baselines, pair_parallel_replay)):
        assert_detection_buffers_equal(
            baseline=baseline,
            candidate=replay,
            label=f"pair_parallel_fullgraph[{idx}]",
        )
    print(
        "parity: both pair parallel fullgraph replays matched exactly",
        flush=True,
    )

    for _ in range(WARMUP):
        for frame in frames:
            _ = run_current_gpu_path(model, frame)
        for runner in single_runners:
            _ = runner.replay_once()
        _ = pair_runner.replay_once()
        _ = pair_parallel_runner.replay_once()

    current_fps = benchmark_current_gpu_path_pair(model, frames, CYCLES)
    single_fullgraph_overlap_fps = benchmark_single_fullgraph_pair(
        single_runners, CYCLES
    )
    single_fullgraph_serial_fps = benchmark_single_fullgraph_pair_serial(
        single_runners, CYCLES
    )
    pair_fullgraph_fps = pair_runner.benchmark(CYCLES)
    pair_parallel_fullgraph_fps = pair_parallel_runner.benchmark(CYCLES)
    print(f"current_gpu_path_pair_fps={current_fps:.2f}")
    print(
        f"single_fullgraph_pair_overlap_fps={single_fullgraph_overlap_fps:.2f}"
    )
    print(f"single_fullgraph_pair_serial_fps={single_fullgraph_serial_fps:.2f}")
    print(f"pair_fullgraph_fps={pair_fullgraph_fps:.2f}")
    print(f"pair_parallel_fullgraph_fps={pair_parallel_fullgraph_fps:.2f}")


if __name__ == "__main__":
    main()
