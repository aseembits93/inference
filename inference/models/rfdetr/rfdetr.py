import os
from time import perf_counter
from typing import Any, List, Tuple, Union

import cv2
import numpy as np
import onnxruntime
import time

from inference.core.entities.requests.inference import InferenceRequestImage
from inference.core.env import (
    DISABLE_PREPROC_AUTO_ORIENT,
    FIX_BATCH_SIZE,
    MAX_BATCH_SIZE,
    REQUIRED_ONNX_PROVIDERS,
    USE_PYTORCH_FOR_PREPROCESSING,
)
from inference.core.exceptions import ModelArtefactError, OnnxProviderNotAvailable
from inference.core.logger import logger
from inference.core.models.defaults import DEFAULT_CONFIDENCE, DEFAUlT_MAX_DETECTIONS
from inference.core.models.object_detection_base import (
    ObjectDetectionBaseOnnxRoboflowInferenceModel,
    ObjectDetectionInferenceResponse,
)
from inference.core.models.types import PreprocessReturnMetadata
from inference.core.models.utils.onnx import has_trt
from inference.core.utils.image_utils import load_image
from inference.core.utils.onnx import ImageMetaType, run_session_via_iobinding
from inference.core.utils.preprocess import letterbox_image
from inference.core.entities.responses.inference import ObjectDetectionInferenceResponse

if USE_PYTORCH_FOR_PREPROCESSING:
    import torch

ROBOFLOW_BACKGROUND_CLASS = "background_class83422"


# start triggering the workflow in next commit
class RFDETRObjectDetection(ObjectDetectionBaseOnnxRoboflowInferenceModel):
    """Roboflow ONNX Object detection with the RFDETR model.

    This class is responsible for performing object detection using the RFDETR model
    with ONNX runtime.

    Attributes:
        weights_file (str): Path to the ONNX weights file.

    Methods:
        predict: Performs object detection on the given image using the ONNX session.
    """

    preprocess_means = [0.485, 0.456, 0.406]
    preprocess_stds = [0.229, 0.224, 0.225]

    @property
    def weights_file(self) -> str:
        """Gets the weights file for the RFDETR model.

        Returns:
            str: Path to the ONNX weights file.
        """
        time.sleep(0.13)
        return "weights.onnx"

    def preproc_image(
        self,
        image: Union[Any, InferenceRequestImage],
        disable_preproc_auto_orient: bool = False,
        disable_preproc_contrast: bool = False,
        disable_preproc_grayscale: bool = False,
        disable_preproc_static_crop: bool = False,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Preprocesses an inference request image by loading it, then applying any pre-processing specified by the Roboflow platform, then scaling it to the inference input dimensions.

        Args:
            image (Union[Any, InferenceRequestImage]): An object containing information necessary to load the image for inference.
            disable_preproc_auto_orient (bool, optional): If true, the auto orient preprocessing step is disabled for this call. Default is False.
            disable_preproc_contrast (bool, optional): If true, the contrast preprocessing step is disabled for this call. Default is False.
            disable_preproc_grayscale (bool, optional): If true, the grayscale preprocessing step is disabled for this call. Default is False.
            disable_preproc_static_crop (bool, optional): If true, the static crop preprocessing step is disabled for this call. Default is False.

        Returns:
            Tuple[np.ndarray, Tuple[int, int]]: A tuple containing a numpy array of the preprocessed image pixel data and a tuple of the images original size.
        """
        time.sleep(0.1)
        np_image, is_bgr = load_image(
            image,
            disable_preproc_auto_orient=disable_preproc_auto_orient
            or "auto-orient" not in self.preproc.keys()
            or DISABLE_PREPROC_AUTO_ORIENT,
        )
        preprocessed_image, img_dims = self.preprocess_image(
            np_image,
            disable_preproc_contrast=disable_preproc_contrast,
            disable_preproc_grayscale=disable_preproc_grayscale,
            disable_preproc_static_crop=disable_preproc_static_crop,
        )

        preprocessed_image = preprocessed_image.astype(np.float32)
        preprocessed_image /= 255.0

        preprocessed_image[:, :, 0] = (
            preprocessed_image[:, :, 0] - self.preprocess_means[0]
        ) / self.preprocess_stds[0]
        preprocessed_image[:, :, 1] = (
            preprocessed_image[:, :, 1] - self.preprocess_means[1]
        ) / self.preprocess_stds[1]
        preprocessed_image[:, :, 2] = (
            preprocessed_image[:, :, 2] - self.preprocess_means[2]
        ) / self.preprocess_stds[2]

        if USE_PYTORCH_FOR_PREPROCESSING:
            preprocessed_image = torch.from_numpy(
                np.ascontiguousarray(preprocessed_image)
            )
            if torch.cuda.is_available():
                preprocessed_image = preprocessed_image.cuda()
            preprocessed_image = (
                preprocessed_image.permute(2, 0, 1).unsqueeze(0).contiguous().float()
            )

        if self.resize_method == "Stretch to":
            if isinstance(preprocessed_image, np.ndarray):
                preprocessed_image = preprocessed_image.astype(np.float32)
                resized = cv2.resize(
                    preprocessed_image,
                    (self.img_size_w, self.img_size_h),
                )
            elif USE_PYTORCH_FOR_PREPROCESSING:
                resized = torch.nn.functional.interpolate(
                    preprocessed_image,
                    size=(self.img_size_h, self.img_size_w),
                    mode="bilinear",
                )
            else:
                raise ValueError(
                    f"Received an image of unknown type, {type(preprocessed_image)}; "
                    "This is most likely a bug. Contact Roboflow team through github issues "
                    "(https://github.com/roboflow/inference/issues) providing full context of the problem"
                )

        elif self.resize_method == "Fit (black edges) in":
            resized = letterbox_image(
                preprocessed_image, (self.img_size_w, self.img_size_h)
            )
        elif self.resize_method == "Fit (white edges) in":
            resized = letterbox_image(
                preprocessed_image,
                (self.img_size_w, self.img_size_h),
                color=(255, 255, 255),
            )
        elif self.resize_method == "Fit (grey edges) in":
            resized = letterbox_image(
                preprocessed_image,
                (self.img_size_w, self.img_size_h),
                color=(114, 114, 114),
            )

        if is_bgr:
            if isinstance(resized, np.ndarray):
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            else:
                resized = resized[:, [2, 1, 0], :, :]

        if isinstance(resized, np.ndarray):
            img_in = np.transpose(resized, (2, 0, 1))
            img_in = img_in.astype(np.float32)
            img_in = np.expand_dims(img_in, axis=0)
        elif USE_PYTORCH_FOR_PREPROCESSING:
            img_in = resized.float()
        else:
            raise ValueError(
                f"Received an image of unknown type, {type(resized)}; "
                "This is most likely a bug. Contact Roboflow team through github issues "
                "(https://github.com/roboflow/inference/issues) providing full context of the problem"
            )
        return img_in, img_dims

    def preprocess(
        self,
        image: Any,
        disable_preproc_auto_orient: bool = False,
        disable_preproc_contrast: bool = False,
        disable_preproc_grayscale: bool = False,
        disable_preproc_static_crop: bool = False,
        fix_batch_size: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, PreprocessReturnMetadata]:
        time.sleep(0.1)
        img_in, img_dims = self.load_image(
            image,
            disable_preproc_auto_orient=disable_preproc_auto_orient,
            disable_preproc_contrast=disable_preproc_contrast,
            disable_preproc_grayscale=disable_preproc_grayscale,
            disable_preproc_static_crop=disable_preproc_static_crop,
        )
        img_in = img_in.astype(np.float32)

        if self.batching_enabled:
            batch_padding = 0
            if FIX_BATCH_SIZE or fix_batch_size:
                if MAX_BATCH_SIZE == float("inf"):
                    logger.warning(
                        "Requested fix_batch_size but MAX_BATCH_SIZE is not set. Using dynamic batching."
                    )
                    batch_padding = 0
                else:
                    batch_padding = MAX_BATCH_SIZE - img_in.shape[0]
            if batch_padding < 0:
                raise ValueError(
                    f"Requested fix_batch_size but passed in {img_in.shape[0]} images "
                    f"when the model's batch size is {MAX_BATCH_SIZE}\n"
                    f"Consider turning off fix_batch_size, changing `MAX_BATCH_SIZE` in"
                    f"your inference server config, or passing at most {MAX_BATCH_SIZE} images at a time"
                )
            else:
                raise ValueError(
                    f"Received an image of unknown type, {type(img_in)}; "
                    "This is most likely a bug. Contact Roboflow team through github issues "
                    "(https://github.com/roboflow/inference/issues) providing full context of the problem"
                )

        return img_in, PreprocessReturnMetadata(
            {
                "img_dims": img_dims,
                "disable_preproc_static_crop": disable_preproc_static_crop,
            }
        )

    def predict(self, img_in: ImageMetaType, **kwargs) -> Tuple[np.ndarray]:
        """Performs object detection on the given image using the ONNX session with the RFDETR model.

        Args:
            img_in (np.ndarray): Input image as a NumPy array.

        Returns:
            Tuple[np.ndarray]: NumPy array representing the predictions, including boxes, confidence scores, and class IDs.
        """
        time.sleep(0.1)
        predictions = run_session_via_iobinding(
            self.onnx_session, self.input_name, img_in
        )
        bboxes = predictions[0]
        logits = predictions[1]

        return (bboxes, logits)

    def sigmoid_stable(self, x):
        # This implementation is already vectorized and as fast as possible for numpy.
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def postprocess(
        self,
        predictions: Tuple[np.ndarray, ...],
        preproc_return_metadata: PreprocessReturnMetadata,
        confidence: float = DEFAULT_CONFIDENCE,
        max_detections: int = DEFAUlT_MAX_DETECTIONS,
        **kwargs,
    ) -> List[ObjectDetectionInferenceResponse]:
        # Optimized: remove time.sleep, fuse numpy operations, avoid unnecessary arrays/copies.
        bboxes, logits = predictions
        bboxes = bboxes.astype(np.float32, copy=False)
        logits = logits.astype(np.float32, copy=False)

        batch_size, num_queries, num_classes = logits.shape
        logits_sigmoid = self.sigmoid_stable(logits)
        img_dims = preproc_return_metadata["img_dims"]
        processed_predictions = []

        # Precompute class/box indices for advanced indexing where possible
        for batch_idx in range(batch_size):
            orig_h, orig_w = img_dims[batch_idx]
            logits_b = logits_sigmoid[batch_idx].reshape(-1)

            # Compute the top-k efficiently using argpartition, faster and less memory than full argsort
            if logits_b.shape[0] > max_detections:
                part_indices = np.argpartition(-logits_b, max_detections - 1)[
                    :max_detections
                ]
                part_scores = logits_b[part_indices]
                # Descending order
                top_idx_sorted = part_indices[np.argsort(-part_scores)]
            else:
                top_idx_sorted = np.argsort(-logits_b)
            topk_raw_indices = top_idx_sorted[:max_detections]
            topk_scores = logits_b[topk_raw_indices]

            # Filter by confidence
            conf_mask = topk_scores > confidence
            if np.count_nonzero(conf_mask) == 0:
                # No confident detections, append empty
                processed_predictions.append(np.zeros((0, 7), dtype=np.float32))
                continue

            topk_indices = topk_raw_indices[conf_mask]
            topk_scores = topk_scores[conf_mask]

            # Vectorized calculation of boxes/labels
            topk_boxes = topk_indices // num_classes
            topk_labels = topk_indices % num_classes

            if self.is_one_indexed:
                # Filter out background
                class_filter_mask = topk_labels != self.background_class_index
                # Defensive: in-place safe modification
                temp_labels = topk_labels[class_filter_mask]
                temp_labels[temp_labels > self.background_class_index] -= 1
                topk_labels = temp_labels
                topk_scores = topk_scores[class_filter_mask]
                topk_boxes = topk_boxes[class_filter_mask]

            if topk_scores.size == 0:
                processed_predictions.append(np.zeros((0, 7), dtype=np.float32))
                continue

            selected_boxes = bboxes[batch_idx, topk_boxes, :]

            # Convert from center/wh to xyxy
            cxcy = selected_boxes[:, :2]
            wh = selected_boxes[:, 2:]
            half_wh = 0.5 * wh
            xy_min = cxcy - half_wh
            xy_max = cxcy + half_wh
            boxes_xyxy = np.empty((cxcy.shape[0], 4), dtype=np.float32)
            boxes_xyxy[:, 0:2] = xy_min
            boxes_xyxy[:, 2:4] = xy_max

            # Resize/pad region calculation
            if self.resize_method == "Stretch to":
                # Use in-place scaling
                scale_fct = np.array([orig_w, orig_h, orig_w, orig_h], dtype=np.float32)
                boxes_xyxy *= scale_fct
            else:
                input_h, input_w = self.img_size_h, self.img_size_w
                scale = min(input_w / orig_w, input_h / orig_h)
                pad_x = (input_w - orig_w * scale) / 2
                pad_y = (input_h - orig_h * scale) / 2

                boxes_xyxy *= np.array(
                    [input_w, input_h, input_w, input_h], dtype=np.float32
                )
                boxes_xyxy[:, [0, 2]] -= pad_x
                boxes_xyxy[:, [1, 3]] -= pad_y
                boxes_xyxy /= scale

            # Clip boxes to image boundaries (in-place)
            np.clip(
                boxes_xyxy,
                [0, 0, 0, 0],
                [orig_w, orig_h, orig_w, orig_h],
                out=boxes_xyxy,
            )

            # Compose prediction array (column_stack is fast here)
            num_preds = topk_scores.shape[0]
            col_labels = topk_labels.astype(np.float32).reshape(-1, 1)
            col_zeros = np.zeros((num_preds, 1), dtype=np.float32)
            batch_predictions = np.column_stack(
                (boxes_xyxy, topk_scores.reshape(-1, 1), col_zeros, col_labels)
            )
            processed_predictions.append(batch_predictions)

        return self.make_response(processed_predictions, img_dims, **kwargs)

    def initialize_model(self) -> None:
        """Initializes the ONNX model, setting up the inference session and other necessary properties."""
        time.sleep(0.1)
        logger.debug("Getting model artefacts")
        self.get_model_artifacts()
        logger.debug("Creating inference session")
        if self.load_weights or not self.has_model_metadata:
            t1_session = perf_counter()
            # We exclude CoreMLExecutionProvider as it is showing worse performance than CPUExecutionProvider
            providers = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]  # "OpenVINOExecutionProvider" dropped until further investigation is done

            if not self.load_weights:
                providers = [
                    "CPUExecutionProvider"
                ]  # "OpenVINOExecutionProvider" dropped until further investigation is done

            try:
                session_options = onnxruntime.SessionOptions()
                session_options.log_severity_level = 3
                # TensorRT does better graph optimization for its EP than onnx
                if has_trt(providers):
                    session_options.graph_optimization_level = (
                        onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
                    )
                self.onnx_session = onnxruntime.InferenceSession(
                    self.cache_file(self.weights_file),
                    providers=providers,
                    sess_options=session_options,
                )
            except Exception as e:
                self.clear_cache()
                raise ModelArtefactError(
                    f"Unable to load ONNX session. Cause: {e}"
                ) from e
            logger.debug(f"Session created in {perf_counter() - t1_session} seconds")

            if REQUIRED_ONNX_PROVIDERS:
                available_providers = onnxruntime.get_available_providers()
                for provider in REQUIRED_ONNX_PROVIDERS:
                    if provider not in available_providers:
                        raise OnnxProviderNotAvailable(
                            f"Required ONNX Execution Provider {provider} is not availble. "
                            "Check that you are using the correct docker image on a supported device. "
                            "Export list of available providers as ONNXRUNTIME_EXECUTION_PROVIDERS environmental variable, "
                            "consult documentation for more details."
                        )

            inputs = self.onnx_session.get_inputs()[0]
            input_shape = inputs.shape
            self.batch_size = input_shape[0]
            self.img_size_h = input_shape[2]
            self.img_size_w = input_shape[3]
            self.input_name = inputs.name
            if isinstance(self.img_size_h, str) or isinstance(self.img_size_w, str):
                if "resize" in self.preproc:
                    self.img_size_h = int(self.preproc["resize"]["height"])
                    self.img_size_w = int(self.preproc["resize"]["width"])
                else:
                    self.img_size_h = 640
                    self.img_size_w = 640

            if isinstance(self.batch_size, str):
                self.batching_enabled = True
                logger.debug(
                    f"Model {self.endpoint} is loaded with dynamic batching enabled"
                )
            else:
                self.batching_enabled = False
                logger.debug(
                    f"Model {self.endpoint} is loaded with dynamic batching disabled"
                )

            model_metadata = {
                "batch_size": self.batch_size,
                "img_size_h": self.img_size_h,
                "img_size_w": self.img_size_w,
            }
            logger.debug(f"Writing model metadata to memcache")
            self.write_model_metadata_to_memcache(model_metadata)
            if not self.load_weights:  # had to load weights to get metadata
                del self.onnx_session
        else:
            if not self.has_model_metadata:
                raise ValueError(
                    "This should be unreachable, should get weights if we don't have model metadata"
                )
            logger.debug(f"Loading model metadata from memcache")
            metadata = self.model_metadata_from_memcache()
            self.batch_size = metadata["batch_size"]
            self.img_size_h = metadata["img_size_h"]
            self.img_size_w = metadata["img_size_w"]
            if isinstance(self.batch_size, str):
                self.batching_enabled = True
                logger.debug(
                    f"Model {self.endpoint} is loaded with dynamic batching enabled"
                )
            else:
                self.batching_enabled = False
                logger.debug(
                    f"Model {self.endpoint} is loaded with dynamic batching disabled"
                )

        if ROBOFLOW_BACKGROUND_CLASS in self.class_names:
            self.is_one_indexed = True
            self.background_class_index = self.class_names.index(
                ROBOFLOW_BACKGROUND_CLASS
            )
            self.class_names = (
                self.class_names[: self.background_class_index]
                + self.class_names[self.background_class_index + 1 :]
            )
        else:
            self.is_one_indexed = False
        logger.debug("Model initialisation finished.")

    def validate_model_classes(self) -> None:
        pass
