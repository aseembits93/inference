"""Parity between the pydantic instance-segmentation response models and their
slotted dataclass twins used on the workflow-execution fast path.

The fast path's correctness rests on the two serializers emitting byte-equal
dicts; if a future maintainer adds a field to the pydantic model without
mirroring it into the dataclass twin (or `_is_*_dc_to_dict`), this test
catches the drift.
"""
from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    InferenceResponseImageDC,
    InstanceSegmentationInferenceResponse,
    InstanceSegmentationInferenceResponseDC,
    InstanceSegmentationPrediction,
    InstanceSegmentationPredictionDC,
    Point,
    PointDC,
    _is_pred_dc_to_dict,
    _is_response_dc_to_dict,
)


def _build_pair(*, class_confidence, parent_id, points, detection_id):
    pyd = InstanceSegmentationPrediction(
        x=1.0,
        y=2.0,
        width=3.0,
        height=4.0,
        confidence=0.9,
        points=[Point(x=p[0], y=p[1]) for p in points],
        class_id=7,
        detection_id=detection_id,
        parent_id=parent_id,
        class_confidence=class_confidence,
        **{"class": "cat"},
    )
    dc = InstanceSegmentationPredictionDC(
        x=1.0,
        y=2.0,
        width=3.0,
        height=4.0,
        confidence=0.9,
        class_name="cat",
        class_id=7,
        points=[PointDC(x=p[0], y=p[1]) for p in points],
        detection_id=detection_id,
        parent_id=parent_id,
        class_confidence=class_confidence,
    )
    return pyd, dc


def test_prediction_parity_with_class_confidence_none():
    pyd, dc = _build_pair(
        class_confidence=None,
        parent_id=None,
        points=[(10.0, 20.0), (30.0, 40.0)],
        detection_id="abc-123",
    )
    assert _is_pred_dc_to_dict(dc) == pyd.model_dump(by_alias=True, exclude_none=True)


def test_prediction_parity_with_class_confidence_set():
    pyd, dc = _build_pair(
        class_confidence=0.3,
        parent_id="parent-xyz",
        points=[(0.0, 0.0)],
        detection_id="abc-123",
    )
    assert _is_pred_dc_to_dict(dc) == pyd.model_dump(by_alias=True, exclude_none=True)


def test_response_parity_empty_predictions_minimal_envelope():
    pyd = InstanceSegmentationInferenceResponse(
        image=InferenceResponseImage(width=640, height=480),
        predictions=[],
    )
    dc = InstanceSegmentationInferenceResponseDC(
        image=InferenceResponseImageDC(width=640, height=480),
        predictions=[],
    )
    assert _is_response_dc_to_dict(dc) == pyd.model_dump(by_alias=True, exclude_none=True)


def test_response_parity_with_envelope_fields_and_predictions():
    pyd_pred, dc_pred = _build_pair(
        class_confidence=0.25,
        parent_id=None,
        points=[(1.0, 2.0), (3.0, 4.0)],
        detection_id="det-1",
    )
    pyd = InstanceSegmentationInferenceResponse(
        image=InferenceResponseImage(width=100, height=200),
        predictions=[pyd_pred],
        inference_id="infer-1",
        frame_id=42,
        time=0.123,
    )
    dc = InstanceSegmentationInferenceResponseDC(
        image=InferenceResponseImageDC(width=100, height=200),
        predictions=[dc_pred],
        inference_id="infer-1",
        frame_id=42,
        time=0.123,
    )
    assert _is_response_dc_to_dict(dc) == pyd.model_dump(by_alias=True, exclude_none=True)


def test_response_parity_envelope_fields_unset_are_omitted():
    pyd = InstanceSegmentationInferenceResponse(
        image=InferenceResponseImage(width=8, height=8),
        predictions=[],
    )
    dc = InstanceSegmentationInferenceResponseDC(
        image=InferenceResponseImageDC(width=8, height=8),
        predictions=[],
    )
    result = _is_response_dc_to_dict(dc)
    assert "inference_id" not in result
    assert "frame_id" not in result
    assert "time" not in result
    assert "visualization" not in result
    assert result == pyd.model_dump(by_alias=True, exclude_none=True)
