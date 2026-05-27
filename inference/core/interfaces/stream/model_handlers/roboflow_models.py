from typing import List, Optional

from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.entities import (
    InferenceHandlerResult,
    ModelConfig,
)
from inference.core.interfaces.stream.utils import wrap_in_list
from inference.core.models.roboflow import OnnxRoboflowInferenceModel


def default_process_frame(
    video_frame: List[VideoFrame],
    model: OnnxRoboflowInferenceModel,
    inference_config: ModelConfig,
) -> List[dict]:
    postprocessing_args = inference_config.to_postprocessing_params()
    # TODO: handle batch input in usage
    fps = _resolve_usage_fps(video_frames=video_frame)
    predictions = wrap_in_list(
        model.infer(
            [f.image for f in video_frame],
            usage_fps=fps,
            usage_api_key=model.api_key,
            **postprocessing_args,
        )
    )
    return _serialise_predictions(predictions=predictions)


class RoboflowModelHandler:
    """Stateful stream handler for Roboflow models.

    Some models, such as RF-DETR instance segmentation with
    ``RFDETR_PIPELINE_DEPTH=2``, intentionally emit frame ``N``'s output
    while processing frame ``N+1``. This handler buffers frame metadata so
    the pipeline can dispatch predictions against the correct frame and
    drain the final pending result at stream end.
    """

    def __init__(
        self,
        model: OnnxRoboflowInferenceModel,
        inference_config: ModelConfig,
    ):
        self._model = model
        self._inference_config = inference_config
        self._pending_video_frames: Optional[List[VideoFrame]] = None

    def __call__(
        self, video_frames: List[VideoFrame]
    ) -> Optional[InferenceHandlerResult]:
        predictions = default_process_frame(
            video_frame=video_frames,
            model=self._model,
            inference_config=self._inference_config,
        )
        if not self._uses_stream_buffering():
            self._pending_video_frames = None
            return InferenceHandlerResult(
                predictions=predictions,
                video_frames=video_frames,
            )
        previous_video_frames = self._pending_video_frames
        self._pending_video_frames = video_frames
        if previous_video_frames is None:
            return None
        return InferenceHandlerResult(
            predictions=predictions,
            video_frames=previous_video_frames,
        )

    def flush(self) -> Optional[InferenceHandlerResult]:
        if not self._uses_stream_buffering():
            self._pending_video_frames = None
            return None
        if self._pending_video_frames is None:
            return None
        flush_fn = getattr(self._model, "flush", None)
        if not callable(flush_fn):
            self._pending_video_frames = None
            return None
        emit_video_frames = self._pending_video_frames
        self._pending_video_frames = None
        predictions = wrap_in_list(flush_fn())
        return InferenceHandlerResult(
            predictions=_serialise_predictions(predictions=predictions),
            video_frames=emit_video_frames,
        )

    def _uses_stream_buffering(self) -> bool:
        return (
            getattr(self._model, "_pipeline_depth", 1) > 1
            and callable(getattr(self._model, "flush", None))
        )


def _serialise_predictions(predictions: list) -> List[dict]:
    return [
        p.dict(
            by_alias=True,
            exclude_none=True,
        )
        for p in predictions
    ]


def _resolve_usage_fps(video_frames: List[VideoFrame]) -> float:
    fps = video_frames[0].fps
    if video_frames[0].measured_fps:
        fps = video_frames[0].measured_fps
    if not fps:
        fps = 0
    return fps
