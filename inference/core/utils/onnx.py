from typing import TYPE_CHECKING, List, Union

import numpy as np
import onnxruntime as ort

if TYPE_CHECKING:
    import torch

ImageMetaType = Union[np.ndarray, "torch.Tensor"]


def get_onnxruntime_execution_providers(value: str) -> List[str]:
    """Extracts the ONNX runtime execution providers from the given string.

    The input string is expected to be a comma-separated list, possibly enclosed
    within square brackets and containing single quotes.

    Args:
        value (str): The string containing the list of ONNX runtime execution providers.

    Returns:
        List[str]: A list of strings representing each execution provider.
    """
    if len(value) == 0:
        return []
    value = value.replace("[", "").replace("]", "").replace("'", "").replace(" ", "")
    return value.split(",")


def run_session_via_iobinding(
    session: ort.InferenceSession, input_name: str, input_data: ImageMetaType
) -> List[np.ndarray]:
    if isinstance(input_data, (np.ndarray, list)):
        # skip the iobinding and just run the session
        # we likely won't get any gains by pointing to the input data directly
        predictions = session.run(None, {input_name: input_data})
    elif "CUDAExecutionProvider" not in session.get_providers():
        # no point in doing iobinding as the input must live on CPU anyway
        input_data = (
            input_data.cpu().numpy()
        )  # since we must be a tensor but ONNX needs a numpy array
        predictions = session.run(None, {input_name: input_data})
    else:
        # we live on GPU and we can use CUDA ONNX, so point to the input data directly
        import torch
        binding = session.io_binding()

        gpu_predictions = []
        dtype = None
        for output in session.get_outputs():
            # assemble GPU-based output buffers for the ONNX runtime to write to directly
            if dtype is None:
                dtype = np.float16 if "16" in output.type else np.float32
            torch_dtype = torch.float16 if dtype == np.float16 else torch.float32
            # Allocate output tensor on GPU
            prediction_tensor = torch.empty(output.shape, dtype=torch_dtype, device=input_data.device)
            binding.bind_output(
                name=output.name,
                device_type=input_data.device.type,
                device_id=(input_data.device.index if input_data.device.index is not None else 0),
                element_type=dtype,
                shape=output.shape,
                buffer_ptr=prediction_tensor.data_ptr(),
            )
            gpu_predictions.append(prediction_tensor)

        input_data = input_data.contiguous()
        binding.bind_input(
            name=input_name,
            device_type=input_data.device.type,
            device_id=(
                input_data.device.index if input_data.device.index is not None else 0
            ),
            element_type=dtype,
            shape=input_data.shape,
            buffer_ptr=input_data.data_ptr(),
        )

        session.run_with_iobinding(binding)

        # Convert GPU tensors to numpy on CPU only when needed
        # Keep as float16 if that's what the model outputs to avoid extra conversion
        predictions = [pred.cpu().numpy().astype(np.float32) if pred.dtype == torch.float16
                       else pred.cpu().numpy() for pred in gpu_predictions]

    return predictions
