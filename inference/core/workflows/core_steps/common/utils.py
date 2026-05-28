import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, TypeVar, Union

import numpy as np
import supervision as sv
from supervision.config import CLASS_NAME_DATA_FIELD

from inference.core.entities.requests.clip import ClipCompareRequest
from inference.core.entities.requests.doctr import DoctrOCRInferenceRequest
from inference.core.entities.requests.easy_ocr import EasyOCRInferenceRequest
from inference.core.entities.requests.sam2 import Sam2InferenceRequest
from inference.core.entities.requests.yolo_world import YOLOWorldInferenceRequest
from inference.core.managers.base import ModelManager
from inference.core.roboflow_api import ModelEndpointType
from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    HEIGHT_KEY,
    IMAGE_DIMENSIONS_KEY,
    INFERENCE_ID_KEY,
    KEYPOINTS_CLASS_ID_KEY_IN_INFERENCE_RESPONSE,
    KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_CLASS_NAME_KEY_IN_INFERENCE_RESPONSE,
    KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_CONFIDENCE_KEY_IN_INFERENCE_RESPONSE,
    KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_KEY_IN_INFERENCE_RESPONSE,
    KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
    PARENT_COORDINATES_KEY,
    PARENT_DIMENSIONS_KEY,
    PARENT_ID_KEY,
    POLYGON_KEY_IN_SV_DETECTIONS,
    PREDICTION_TYPE_KEY,
    RLE_MASK_KEY_IN_INFERENCE_RESPONSE,
    RLE_MASK_KEY_IN_SV_DETECTIONS,
    ROOT_PARENT_COORDINATES_KEY,
    ROOT_PARENT_DIMENSIONS_KEY,
    ROOT_PARENT_ID_KEY,
    SCALING_RELATIVE_TO_PARENT_KEY,
    SCALING_RELATIVE_TO_ROOT_PARENT_KEY,
    WIDTH_KEY,
    X_KEY,
    Y_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    OriginCoordinatesSystem,
    WorkflowImageData,
)
from inference.core.workflows.prototypes.block import BlockResult

T = TypeVar("T")


def load_core_model(
    model_manager: ModelManager,
    inference_request: Union[
        DoctrOCRInferenceRequest,
        EasyOCRInferenceRequest,
        ClipCompareRequest,
        YOLOWorldInferenceRequest,
        Sam2InferenceRequest,
    ],
    core_model: str,
) -> str:
    version_id_field = f"{core_model}_version_id"
    core_model_id = (
        f"{core_model}/{inference_request.__getattribute__(version_id_field)}"
    )
    model_manager.add_model(
        core_model_id,
        inference_request.api_key,
        endpoint_type=ModelEndpointType.CORE_MODEL,
    )
    return core_model_id


def attach_prediction_type_info(
    predictions: List[Dict[str, Any]],
    prediction_type: str,
    key: str = PREDICTION_TYPE_KEY,
) -> List[Dict[str, Any]]:
    for result in predictions:
        result[key] = prediction_type
    return predictions


def filter_out_invalid_polygons(predictions: List[dict]) -> List[dict]:
    return [
        d for d in predictions if "points" not in d or len(d.get("points", [])) >= 3
    ]


def attach_prediction_type_info_to_sv_detections_batch(
    predictions: List[sv.Detections],
    prediction_type: str,
    key: str = PREDICTION_TYPE_KEY,
) -> List[sv.Detections]:
    for prediction in predictions:
        existing = prediction.data.get(key)
        if (
            existing is not None
            and len(existing) == get_sv_detections_fast_size(prediction)
            and (len(existing) == 0 or np.all(existing == prediction_type))
        ):
            continue
        prediction[key] = np.full(
            (get_sv_detections_fast_size(prediction),),
            prediction_type,
            dtype=object,
        )
    return predictions


def convert_inference_detections_batch_to_sv_detections(
    predictions: List[Dict[str, Union[List[Dict[str, Any]], Any]]],
    predictions_key: str = "predictions",
    image_key: str = "image",
) -> List[sv.Detections]:
    batch_of_detections: List[sv.Detections] = []
    for p in predictions:
        width, height = p[image_key][WIDTH_KEY], p[image_key][HEIGHT_KEY]
        detections = sv.Detections.from_inference(p)
        raw_predictions = p[predictions_key]
        if len(detections) != len(raw_predictions):
            raw_predictions = filter_out_invalid_polygons(predictions=raw_predictions)
        parent_ids = [d.get(PARENT_ID_KEY, "") for d in raw_predictions]
        detection_ids = [
            d.get(DETECTION_ID_KEY, str(uuid.uuid4())) for d in raw_predictions
        ]
        detections[DETECTION_ID_KEY] = np.array(detection_ids)
        detections[PARENT_ID_KEY] = np.array(parent_ids)
        detections[IMAGE_DIMENSIONS_KEY] = np.array([[height, width]] * len(detections))
        if INFERENCE_ID_KEY in p:
            detections[INFERENCE_ID_KEY] = np.array(
                [p[INFERENCE_ID_KEY]] * len(detections)
            )
        rle_masks = [
            d.get(RLE_MASK_KEY_IN_INFERENCE_RESPONSE) or d.get("rle")
            for d in raw_predictions
        ]
        if any(m is not None for m in rle_masks):
            detections.data[RLE_MASK_KEY_IN_SV_DETECTIONS] = np.array(
                rle_masks, dtype=object
            )
        batch_of_detections.append(detections)
    return batch_of_detections


def add_inference_keypoints_to_sv_detections(
    inference_prediction: List[dict],
    detections: sv.Detections,
) -> sv.Detections:
    if len(inference_prediction) != len(detections):
        raise ValueError(
            f"Detected missmatch in number of detections in sv.Detections instance ({len(detections)}) "
            f"and `inference` predictions ({len(inference_prediction)}) while attempting to add keypoints metadata."
        )
    keypoints_class_names = []
    keypoints_class_ids = []
    keypoints_confidences = []
    keypoints_xy = []
    for inference_detection in inference_prediction:
        keypoints = inference_detection.get(KEYPOINTS_KEY_IN_INFERENCE_RESPONSE, [])
        keypoints_class_names.append(
            [k[KEYPOINTS_CLASS_NAME_KEY_IN_INFERENCE_RESPONSE] for k in keypoints]
        )
        keypoints_class_ids.append(
            [k[KEYPOINTS_CLASS_ID_KEY_IN_INFERENCE_RESPONSE] for k in keypoints]
        )
        keypoints_confidences.append(
            [k[KEYPOINTS_CONFIDENCE_KEY_IN_INFERENCE_RESPONSE] for k in keypoints]
        )
        keypoints_xy.append([[k[X_KEY], k[Y_KEY]] for k in keypoints])
    # Pad to uniform length so arrays are proper N-d numpy arrays instead of
    # object-dtype ragged arrays. Object-dtype arrays break supervision's
    # is_data_equal (used in Detections indexing/comparison).
    max_kps = max((len(kp) for kp in keypoints_xy), default=0)
    n = len(inference_prediction)
    padded_xy = np.zeros((n, max_kps, 2), dtype=np.float32)
    padded_conf = np.zeros((n, max_kps), dtype=np.float32)
    padded_class_id = np.zeros((n, max_kps), dtype=int)
    padded_class_name = np.full((n, max_kps), "", dtype=object)
    for i in range(n):
        k = len(keypoints_xy[i])
        if k > 0:
            padded_xy[i, :k] = keypoints_xy[i]
            padded_conf[i, :k] = keypoints_confidences[i]
            padded_class_id[i, :k] = keypoints_class_ids[i]
            padded_class_name[i, :k] = keypoints_class_names[i]
    detections[KEYPOINTS_XY_KEY_IN_SV_DETECTIONS] = padded_xy
    detections[KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS] = padded_conf
    detections[KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS] = padded_class_id
    detections[KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS] = padded_class_name
    return detections


def attach_parents_coordinates_to_batch_of_sv_detections(
    predictions: List[sv.Detections],
    images: Iterable[WorkflowImageData],
) -> List[sv.Detections]:
    result = []
    for prediction, image in zip(predictions, images):
        result.append(
            attach_parents_coordinates_to_sv_detections(
                detections=prediction,
                image=image,
            )
        )
    return result


def attach_parents_coordinates_to_sv_detections(
    detections: sv.Detections,
    image: WorkflowImageData,
) -> sv.Detections:
    detections = attach_parent_coordinates_to_detections(
        detections=detections,
        parent_metadata=image.workflow_root_ancestor_metadata,
        parent_id_key=ROOT_PARENT_ID_KEY,
        coordinates_key=ROOT_PARENT_COORDINATES_KEY,
        dimensions_key=ROOT_PARENT_DIMENSIONS_KEY,
    )
    return attach_parent_coordinates_to_detections(
        detections=detections,
        parent_metadata=image.parent_metadata,
        parent_id_key=PARENT_ID_KEY,
        coordinates_key=PARENT_COORDINATES_KEY,
        dimensions_key=PARENT_DIMENSIONS_KEY,
    )


def attach_parent_coordinates_to_detections(
    detections: sv.Detections,
    parent_metadata: ImageParentMetadata,
    parent_id_key: str,
    coordinates_key: str,
    dimensions_key: str,
) -> sv.Detections:
    n_detections = get_sv_detections_fast_size(detections)
    parent_coordinates_system = parent_metadata.origin_coordinates
    detections[parent_id_key] = np.full(
        (n_detections,), parent_metadata.parent_id, dtype=object
    )
    coordinates = np.broadcast_to(
        np.array(
            [
                parent_coordinates_system.left_top_x,
                parent_coordinates_system.left_top_y,
            ],
            dtype=np.int32,
        ),
        (n_detections, 2),
    )
    detections[coordinates_key] = coordinates
    dimensions = np.broadcast_to(
        np.array(
            [
                parent_coordinates_system.origin_height,
                parent_coordinates_system.origin_width,
            ],
            dtype=np.int32,
        ),
        (n_detections, 2),
    )
    detections[dimensions_key] = dimensions
    return detections


def get_sv_detections_fast_size(detections: sv.Detections) -> int:
    fast_len_hint = getattr(detections, "_fast_len_hint", None)
    if callable(fast_len_hint):
        return int(fast_len_hint())
    return len(detections)


KEYS_REQUIRED_TO_EMBED_IN_ROOT_COORDINATES = {
    ROOT_PARENT_COORDINATES_KEY,
    ROOT_PARENT_DIMENSIONS_KEY,
    ROOT_PARENT_ID_KEY,
}


def _shallow_clone_sv_detections(detections: sv.Detections) -> sv.Detections:
    clone_fast = getattr(detections, "_shallow_clone_fast", None)
    if callable(clone_fast):
        return clone_fast()
    return sv.Detections(
        xyxy=detections.xyxy,
        mask=detections.mask,
        confidence=detections.confidence,
        class_id=detections.class_id,
        tracker_id=detections.tracker_id,
        data=dict(detections.data),
        metadata=dict(detections.metadata),
    )


def _sv_detections_root_conversion_is_noop(
    detections: sv.Detections,
) -> bool:
    if get_sv_detections_fast_size(detections) == 0:
        return False
    if any(key not in detections.data for key in KEYS_REQUIRED_TO_EMBED_IN_ROOT_COORDINATES):
        return False

    root_parent_coordinates = detections.data.get(ROOT_PARENT_COORDINATES_KEY)
    root_parent_dimensions = detections.data.get(ROOT_PARENT_DIMENSIONS_KEY)
    if root_parent_coordinates is None or root_parent_dimensions is None:
        return False
    if len(root_parent_coordinates) == 0 or len(root_parent_dimensions) == 0:
        return False
    if not np.all(root_parent_coordinates == 0):
        return False

    parent_coordinates = detections.data.get(PARENT_COORDINATES_KEY)
    parent_dimensions = detections.data.get(PARENT_DIMENSIONS_KEY)
    if parent_coordinates is None or parent_dimensions is None:
        return False
    if len(parent_coordinates) == 0 or len(parent_dimensions) == 0:
        return False
    if not np.all(parent_coordinates == 0):
        return False
    if not np.array_equal(parent_dimensions, root_parent_dimensions):
        return False

    parent_ids = detections.data.get(PARENT_ID_KEY)
    root_parent_ids = detections.data.get(ROOT_PARENT_ID_KEY)
    if parent_ids is None or root_parent_ids is None:
        return False
    if len(parent_ids) == 0 or len(root_parent_ids) == 0:
        return False
    if not np.array_equal(parent_ids, root_parent_ids):
        return False

    image_dimensions = detections.data.get(IMAGE_DIMENSIONS_KEY)
    if image_dimensions is None:
        return False
    if len(image_dimensions) == 0:
        return False
    if not np.array_equal(image_dimensions, root_parent_dimensions):
        return False

    scaling_relative_to_parent = detections.data.get(SCALING_RELATIVE_TO_PARENT_KEY)
    if scaling_relative_to_parent is not None and not np.allclose(
        scaling_relative_to_parent, 1.0
    ):
        return False
    scaling_relative_to_root_parent = detections.data.get(
        SCALING_RELATIVE_TO_ROOT_PARENT_KEY
    )
    if scaling_relative_to_root_parent is not None and not np.allclose(
        scaling_relative_to_root_parent, 1.0
    ):
        return False

    raw_mask = object.__getattribute__(detections, "mask")
    if raw_mask is not None:
        origin_height = int(root_parent_dimensions[0][0])
        origin_width = int(root_parent_dimensions[0][1])
        if tuple(raw_mask.shape[1:]) != (origin_height, origin_width):
            return False
    else:
        mask_shape_matches_root = getattr(
            detections, "_mask_shape_matches_root_dimensions_fast", None
        )
        if callable(mask_shape_matches_root) and not mask_shape_matches_root():
            return False
    return True


def sv_detections_to_root_coordinates(
    detections: sv.Detections, keypoints_key: str = KEYPOINTS_XY_KEY_IN_SV_DETECTIONS
) -> sv.Detections:
    n_detections = get_sv_detections_fast_size(detections)
    if n_detections == 0:
        return _shallow_clone_sv_detections(detections=detections)
    if _sv_detections_root_conversion_is_noop(detections=detections):
        detections_copy = _shallow_clone_sv_detections(detections=detections)
        fast_copy_size = get_sv_detections_fast_size(detections_copy)
        scaling = np.ones((fast_copy_size,), dtype=np.float32)
        detections_copy[SCALING_RELATIVE_TO_PARENT_KEY] = scaling
        detections_copy[SCALING_RELATIVE_TO_ROOT_PARENT_KEY] = scaling.copy()
        if fast_copy_size:
            origin_height = int(detections_copy[ROOT_PARENT_DIMENSIONS_KEY][0][0])
            origin_width = int(detections_copy[ROOT_PARENT_DIMENSIONS_KEY][0][1])
            detections_copy[IMAGE_DIMENSIONS_KEY] = np.repeat(
                np.array([[origin_height, origin_width]], dtype=np.int32),
                fast_copy_size,
                axis=0,
            )
        return detections_copy

    detections_copy = deepcopy(detections)
    if len(detections_copy) == 0:
        return detections_copy

    if any(
        key not in detections_copy.data
        for key in KEYS_REQUIRED_TO_EMBED_IN_ROOT_COORDINATES
    ):
        logging.warning(
            "Could not execute detections_to_root_coordinates(...) on detections with "
            f"the following metadata registered: {list(detections_copy.data.keys())}"
        )
        return detections_copy
    if SCALING_RELATIVE_TO_ROOT_PARENT_KEY in detections_copy.data:
        scale = detections_copy[SCALING_RELATIVE_TO_ROOT_PARENT_KEY][0]
        detections_copy = scale_sv_detections(
            detections=detections,
            scale=1 / scale,
        )
    detections_copy[SCALING_RELATIVE_TO_PARENT_KEY] = np.array(
        [1.0] * len(detections_copy)
    )
    detections_copy[SCALING_RELATIVE_TO_ROOT_PARENT_KEY] = np.array(
        [1.0] * len(detections_copy)
    )
    origin_height = detections_copy[ROOT_PARENT_DIMENSIONS_KEY][0][0]
    origin_width = detections_copy[ROOT_PARENT_DIMENSIONS_KEY][0][1]
    detections_copy[IMAGE_DIMENSIONS_KEY] = np.array(
        [[origin_height, origin_width]] * len(detections_copy)
    )
    root_parent_id = detections_copy[ROOT_PARENT_ID_KEY][0]
    shift_x, shift_y = detections_copy[ROOT_PARENT_COORDINATES_KEY][0]
    detections_copy.xyxy += [shift_x, shift_y, shift_x, shift_y]
    if keypoints_key in detections_copy.data:
        for keypoints in detections_copy[keypoints_key]:
            if len(keypoints):
                keypoints += [shift_x, shift_y]
    if detections_copy.mask is not None:
        origin_mask_base = np.full((origin_height, origin_width), False)
        new_anchored_masks = np.array(
            [origin_mask_base.copy() for _ in detections_copy]
        )
        for anchored_mask, original_mask in zip(
            new_anchored_masks, detections_copy.mask
        ):
            mask_h, mask_w = original_mask.shape
            # TODO: instead of shifting mask we could store contours in data instead of storing mask (even if calculated)
            #       it would be faster to shift contours but at expense of having to remember to generate mask from contour when it's needed
            anchored_mask[shift_y : shift_y + mask_h, shift_x : shift_x + mask_w] = (
                original_mask
            )
        detections_copy.mask = new_anchored_masks
    new_root_metadata = ImageParentMetadata(
        parent_id=root_parent_id,
        origin_coordinates=OriginCoordinatesSystem(
            left_top_y=0,
            left_top_x=0,
            origin_width=origin_width,
            origin_height=origin_height,
        ),
    )
    detections_copy = attach_parent_coordinates_to_detections(
        detections=detections_copy,
        parent_metadata=new_root_metadata,
        parent_id_key=ROOT_PARENT_ID_KEY,
        coordinates_key=ROOT_PARENT_COORDINATES_KEY,
        dimensions_key=ROOT_PARENT_DIMENSIONS_KEY,
    )
    return attach_parent_coordinates_to_detections(
        detections=detections_copy,
        parent_metadata=new_root_metadata,
        parent_id_key=PARENT_ID_KEY,
        coordinates_key=PARENT_COORDINATES_KEY,
        dimensions_key=PARENT_DIMENSIONS_KEY,
    )


def filter_out_unwanted_classes_from_sv_detections_batch(
    predictions: List[sv.Detections],
    classes_to_accept: Optional[List[str]],
) -> List[sv.Detections]:
    if not classes_to_accept:
        return predictions
    filtered_predictions = []
    for prediction in predictions:
        filtered_prediction = prediction[
            np.isin(prediction[CLASS_NAME_DATA_FIELD], classes_to_accept)
        ]
        filtered_predictions.append(filtered_prediction)
    return filtered_predictions


def grab_batch_parameters(
    operations_parameters: Dict[str, Any],
    main_batch_size: int,
) -> Dict[str, Any]:
    return {
        key: value.broadcast(n=main_batch_size)
        for key, value in operations_parameters.items()
        if isinstance(value, Batch)
    }


def grab_non_batch_parameters(operations_parameters: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in operations_parameters.items()
        if not isinstance(value, Batch)
    }


def scale_sv_detections(
    detections: sv.Detections,
    scale: float,
    keypoints_key: str = KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
) -> sv.Detections:
    detections_copy = deepcopy(detections)
    if len(detections_copy) == 0:
        return detections_copy
    detections_copy.xyxy = (detections_copy.xyxy * scale).round()
    if keypoints_key in detections_copy.data:
        for i in range(len(detections_copy[keypoints_key])):
            detections_copy[keypoints_key][i] = (
                detections_copy[keypoints_key][i].astype(np.float32) * scale
            ).round()
    detections_copy[IMAGE_DIMENSIONS_KEY] = (
        detections_copy[IMAGE_DIMENSIONS_KEY] * scale
    ).round()
    if detections_copy.mask is not None:
        scaled_masks = []
        original_mask_size_wh = (
            detections_copy.mask.shape[2],
            detections_copy.mask.shape[1],
        )
        scaled_mask_size_wh = round(original_mask_size_wh[0] * scale), round(
            original_mask_size_wh[1] * scale
        )
        for detection_mask in detections_copy.mask:
            polygons = sv.mask_to_polygons(mask=detection_mask)
            polygon_masks = []
            for polygon in polygons:
                scaled_polygon = (polygon * scale).round().astype(np.int32)
                polygon_masks.append(
                    sv.polygon_to_mask(
                        polygon=scaled_polygon, resolution_wh=scaled_mask_size_wh
                    )
                )
            scaled_detection_mask = np.sum(polygon_masks, axis=0) > 0
            scaled_masks.append(scaled_detection_mask)
        detections_copy.mask = np.array(scaled_masks)
    if POLYGON_KEY_IN_SV_DETECTIONS in detections_copy.data:
        detections_copy.data[POLYGON_KEY_IN_SV_DETECTIONS] = (
            (detections_copy.data[POLYGON_KEY_IN_SV_DETECTIONS] * scale)
            .round()
            .astype(np.int32)
        )
    if SCALING_RELATIVE_TO_PARENT_KEY in detections_copy.data:
        detections_copy[SCALING_RELATIVE_TO_PARENT_KEY] = (
            detections_copy[SCALING_RELATIVE_TO_PARENT_KEY] * scale
        )
    else:
        detections_copy[SCALING_RELATIVE_TO_PARENT_KEY] = np.array(
            [scale] * len(detections_copy)
        )
    if SCALING_RELATIVE_TO_ROOT_PARENT_KEY in detections_copy.data:
        detections_copy[SCALING_RELATIVE_TO_ROOT_PARENT_KEY] = (
            detections_copy[SCALING_RELATIVE_TO_ROOT_PARENT_KEY] * scale
        )
    else:
        detections_copy[SCALING_RELATIVE_TO_ROOT_PARENT_KEY] = np.array(
            [scale] * len(detections_copy)
        )
    return detections_copy


def remove_unexpected_keys_from_dictionary(
    dictionary: dict,
    expected_keys: set,
) -> dict:
    """This function mutates input `dictionary`"""
    unexpected_keys = set(dictionary.keys()).difference(expected_keys)
    for unexpected_key in unexpected_keys:
        del dictionary[unexpected_key]
    return dictionary


def post_process_ocr_result(
    images: Batch[WorkflowImageData],
    predictions: List[dict],
    expected_output_keys: Set[str],
) -> BlockResult:
    for prediction, image in zip(predictions, images):
        raw_predictions = prediction.get("predictions", [])
        prediction["predictions"] = sv.Detections.from_inference(prediction)
        if len(prediction["predictions"]) != len(raw_predictions):
            raw_predictions = filter_out_invalid_polygons(predictions=raw_predictions)
        detection_ids = [
            p.get("detection_id", str(uuid.uuid4())) for p in raw_predictions
        ]
        prediction["predictions"]["detection_id"] = detection_ids
        prediction[PREDICTION_TYPE_KEY] = "ocr"
        prediction[PARENT_ID_KEY] = image.parent_metadata.parent_id
        prediction[ROOT_PARENT_ID_KEY] = image.workflow_root_ancestor_metadata.parent_id
        _ = remove_unexpected_keys_from_dictionary(
            dictionary=prediction,
            expected_keys=expected_output_keys,
        )
    return predictions


def run_in_parallel(tasks: List[Callable[[], T]], max_workers: int = 1) -> List[T]:
    tasks = _propagate_inference_context(tasks)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(lambda f: f(), tasks))


def _propagate_inference_context(
    tasks: List[Callable[[], T]],
) -> List[Callable[[], T]]:
    """Wrap each task so that inference_sdk context vars and server-side
    context vars are propagated into worker threads.  Returns the tasks
    unchanged when no context is active.
    """
    from inference.core.managers.model_load_collector import (
        model_load_info,
        request_model_ids,
    )

    load_collector = model_load_info.get(None)
    ids_collector = request_model_ids.get(None)

    try:
        from asgi_correlation_id import correlation_id as _cid_ctx

        corr_id = _cid_ctx.get()
    except Exception:
        corr_id = None

    try:
        from inference_sdk.config import (
            apply_duration_minimum,
            execution_id,
            remote_processing_times,
        )

        exec_id = execution_id.get() if execution_id is not None else None
        rpt_collector = (
            remote_processing_times.get()
            if remote_processing_times is not None
            else None
        )
        duration_min = (
            apply_duration_minimum.get() if apply_duration_minimum is not None else None
        )
    except ImportError:
        exec_id = None
        rpt_collector = None
        duration_min = None

    if (
        exec_id is None
        and rpt_collector is None
        and duration_min is None
        and load_collector is None
        and ids_collector is None
        and corr_id is None
    ):
        return tasks

    def _wrap(fun: Callable[[], T]) -> Callable[[], T]:
        def _with_context() -> T:
            if corr_id is not None:
                from asgi_correlation_id import correlation_id as _cid_ctx

                _cid_ctx.set(corr_id)
            if exec_id is not None:
                from inference_sdk.config import execution_id

                execution_id.set(exec_id)
            if rpt_collector is not None:
                from inference_sdk.config import remote_processing_times

                remote_processing_times.set(rpt_collector)
            if duration_min is not None:
                from inference_sdk.config import apply_duration_minimum

                apply_duration_minimum.set(duration_min)
            if load_collector is not None:
                model_load_info.set(load_collector)
            if ids_collector is not None:
                request_model_ids.set(ids_collector)
            return fun()

        return _with_context

    return [_wrap(t) for t in tasks]
