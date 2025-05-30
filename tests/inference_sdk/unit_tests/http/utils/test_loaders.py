import base64
import os.path
from typing import Tuple
from unittest import mock
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from aioresponses import aioresponses
from PIL import Image
from requests import HTTPError, Response

from inference_sdk.http.errors import EncodingError, InvalidInputFormatError
from inference_sdk.http.utils import loaders
from inference_sdk.http.utils.loaders import (
    load_directory_inference_input,
    load_image_from_string,
    load_image_from_string_async,
    load_image_from_url,
    load_image_from_url_async,
    load_nested_batches_of_inference_input,
    load_static_inference_input,
    load_static_inference_input_async,
    load_stream_inference_input,
    uri_is_http_link,
)


def test_uri_is_http_link_when_local_absolute_path_provided() -> None:
    # when
    result = uri_is_http_link("/some/directory/file.txt")

    # then
    assert result is False


def test_uri_is_http_link_when_local_relative_path_provided() -> None:
    # when
    result = uri_is_http_link("directory/file.txt")

    # then
    assert result is False


def test_uri_is_http_link_when_http_url_provided() -> None:
    # when
    result = uri_is_http_link("http://site/path/file.txt")

    # then
    assert result is True


def test_uri_is_http_link_when_https_url_provided() -> None:
    # when
    result = uri_is_http_link("https://site/path/file.txt")

    # then
    assert result is True


@mock.patch.object(loaders.requests, "get")
def test_load_file_from_url_on_unsuccessful_image_download(
    requests_get_mock: MagicMock,
) -> None:
    # given
    response = Response()
    response.status_code = 404
    requests_get_mock.return_value = response

    # when
    with pytest.raises(HTTPError):
        _ = load_image_from_url(url="http://some/file.jpg")

    # then
    requests_get_mock.assert_called_once_with("http://some/file.jpg")


@mock.patch.object(loaders.requests, "get")
def test_load_file_from_url_on_successful_image_download_without_resize(
    requests_get_mock: MagicMock,
) -> None:
    # given
    response = Response()
    response.status_code = 200
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    _, encoded_image = cv2.imencode(".jpg", image)
    response._content = encoded_image
    requests_get_mock.return_value = response

    # when
    serialised_image, scaling_factor = load_image_from_url(url="http://some/file.jpg")
    recovered_image = base64.b64decode(serialised_image)
    bytes_array = np.frombuffer(recovered_image, dtype=np.uint8)
    decoding_result = cv2.imdecode(bytes_array, cv2.IMREAD_UNCHANGED)

    # then
    assert scaling_factor is None
    assert decoding_result.shape == image.shape
    assert np.allclose(decoding_result, image)
    requests_get_mock.assert_called_once_with("http://some/file.jpg")


@pytest.mark.asyncio
async def test_load_file_from_url_async_on_successful_image_download_without_resize() -> (
    None
):
    with aioresponses() as m:
        # given
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        _, encoded_image = cv2.imencode(".jpg", image)
        m.get("https://some.com/file.jpg", status=200, body=encoded_image.tobytes())

        # when
        serialised_image, scaling_factor = await load_image_from_url_async(
            url="https://some.com/file.jpg"
        )
        recovered_image = base64.b64decode(serialised_image)
        bytes_array = np.frombuffer(recovered_image, dtype=np.uint8)
        decoding_result = cv2.imdecode(bytes_array, cv2.IMREAD_UNCHANGED)

        # then
        assert scaling_factor is None
        assert decoding_result.shape == image.shape
        assert np.allclose(decoding_result, image)


@mock.patch.object(loaders.requests, "get")
def test_load_file_from_url_on_successful_image_download_with_resize(
    requests_get_mock: MagicMock,
) -> None:
    # given
    response = Response()
    response.status_code = 200
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    _, encoded_image = cv2.imencode(".jpg", image)
    response._content = encoded_image
    requests_get_mock.return_value = response

    # when
    serialised_image, scaling_factor = load_image_from_url(
        url="http://some/file.jpg",
        max_width=64,
        max_height=128,
    )
    recovered_image = base64.b64decode(serialised_image)
    bytes_array = np.frombuffer(recovered_image, dtype=np.uint8)
    decoding_result = cv2.imdecode(bytes_array, cv2.IMREAD_UNCHANGED)

    # then
    assert abs(scaling_factor - 0.5) < 1e-5
    assert decoding_result.shape == (64, 64, 3)
    assert (decoding_result == 0).all()
    requests_get_mock.assert_called_once_with("http://some/file.jpg")


@pytest.mark.asyncio
async def test_load_file_from_url_async_on_successful_image_download_with_resize() -> (
    None
):
    with aioresponses() as m:
        # given
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        _, encoded_image = cv2.imencode(".jpg", image)
        m.get("https://some.com/file.jpg", status=200, body=encoded_image.tobytes())

        # when
        serialised_image, scaling_factor = await load_image_from_url_async(
            url="https://some.com/file.jpg",
            max_width=64,
            max_height=128,
        )
        recovered_image = base64.b64decode(serialised_image)
        bytes_array = np.frombuffer(recovered_image, dtype=np.uint8)
        decoding_result = cv2.imdecode(bytes_array, cv2.IMREAD_UNCHANGED)

        # then
        assert abs(scaling_factor - 0.5) < 1e-5
        assert decoding_result.shape == (64, 64, 3)
        assert (decoding_result == 0).all()


@mock.patch.object(loaders.requests, "get")
def test_load_image_from_string_when_file_to_be_downloaded(
    requests_get_mock: MagicMock,
) -> None:
    # given
    response = Response()
    response.status_code = 200
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    _, encoded_image = cv2.imencode(".jpg", image)
    response._content = encoded_image
    requests_get_mock.return_value = response

    # when
    serialised_image, scaling_factor = load_image_from_string(
        reference="https://some/file.jpg"
    )

    # then
    recovered_image = base64.b64decode(serialised_image)
    bytes_array = np.frombuffer(recovered_image, dtype=np.uint8)
    decoding_result = cv2.imdecode(bytes_array, cv2.IMREAD_UNCHANGED)

    # then
    assert scaling_factor is None
    assert decoding_result.shape == image.shape
    assert np.allclose(decoding_result, image)
    requests_get_mock.assert_called_once_with("https://some/file.jpg")


@pytest.mark.asyncio
async def test_load_image_from_string_async_when_file_to_be_downloaded() -> None:
    # given
    with aioresponses() as m:
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        _, encoded_image = cv2.imencode(".jpg", image)
        m.get("https://some.com/file.jpg", status=200, body=encoded_image.tobytes())

        # when
        serialised_image, scaling_factor = await load_image_from_string_async(
            reference="https://some.com/file.jpg"
        )

        # then
        recovered_image = base64.b64decode(serialised_image)
        bytes_array = np.frombuffer(recovered_image, dtype=np.uint8)
        decoding_result = cv2.imdecode(bytes_array, cv2.IMREAD_UNCHANGED)

        # then
        assert scaling_factor is None
        assert decoding_result.shape == image.shape
        assert np.allclose(decoding_result, image)


def test_load_image_from_string_when_local_image_to_be_loaded(
    example_local_image: Tuple[str, np.ndarray]
) -> None:
    # given
    file_path, _ = example_local_image

    # when
    serialised_image, scaling_factor = load_image_from_string(
        reference=file_path,
        max_width=64,
        max_height=128,
    )
    recovered_image = base64.b64decode(serialised_image)
    bytes_array = np.frombuffer(recovered_image, dtype=np.uint8)
    decoding_result = cv2.imdecode(bytes_array, cv2.IMREAD_UNCHANGED)

    # then
    assert abs(scaling_factor - 0.5) < 1e-5
    assert decoding_result.shape == (64, 64, 3)
    assert (decoding_result == 0).all()


@pytest.mark.asyncio
async def test_load_image_from_string_async_when_local_image_to_be_loaded(
    example_local_image: Tuple[str, np.ndarray]
) -> None:
    # given
    file_path, _ = example_local_image

    # when
    serialised_image, scaling_factor = await load_image_from_string_async(
        reference=file_path,
        max_width=64,
        max_height=128,
    )
    recovered_image = base64.b64decode(serialised_image)
    bytes_array = np.frombuffer(recovered_image, dtype=np.uint8)
    decoding_result = cv2.imdecode(bytes_array, cv2.IMREAD_UNCHANGED)

    # then
    assert abs(scaling_factor - 0.5) < 1e-5
    assert decoding_result.shape == (64, 64, 3)
    assert (decoding_result == 0).all()


def test_load_image_from_string_when_base64_image_given_and_no_resize_needed(
    example_local_image: Tuple[str, np.ndarray]
) -> None:
    # given
    _, image = example_local_image
    _, buffer = cv2.imencode(".jpg", image)
    base64_image = base64.b64encode(buffer).decode("ascii")

    # when
    serialised_image, scaling_factor = load_image_from_string(
        reference=base64_image,
        max_width=None,
        max_height=None,
    )

    # then
    assert (
        scaling_factor is None
    ), "No resize parameters given, so scaling factor should not be established"
    assert (
        serialised_image == base64_image
    ), "Serialised image should be identical with input"


@pytest.mark.asyncio
async def test_load_image_from_string_async_when_base64_image_given_and_no_resize_needed(
    example_local_image: Tuple[str, np.ndarray]
) -> None:
    # given
    _, image = example_local_image
    _, buffer = cv2.imencode(".jpg", image)
    base64_image = base64.b64encode(buffer).decode("ascii")

    # when
    serialised_image, scaling_factor = await load_image_from_string_async(
        reference=base64_image,
        max_width=None,
        max_height=None,
    )

    # then
    assert (
        scaling_factor is None
    ), "No resize parameters given, so scaling factor should not be established"
    assert (
        serialised_image == base64_image
    ), "Serialised image should be identical with input"


def test_load_image_from_string_when_base64_image_given_and_resize_needed(
    example_local_image: Tuple[str, np.ndarray]
) -> None:
    # given
    _, image = example_local_image
    _, buffer = cv2.imencode(".jpg", image)
    base64_image = base64.b64encode(buffer).decode("ascii")

    # when
    serialised_image, scaling_factor = load_image_from_string(
        reference=base64_image,
        max_width=64,
        max_height=128,
    )
    recovered_image = base64.b64decode(serialised_image)
    bytes_array = np.frombuffer(recovered_image, dtype=np.uint8)
    decoding_result = cv2.imdecode(bytes_array, cv2.IMREAD_UNCHANGED)

    # then
    assert abs(scaling_factor - 0.5) < 1e-5
    assert decoding_result.shape == (64, 64, 3)
    assert (decoding_result == 0).all()


@pytest.mark.asyncio
async def test_load_image_from_string_async_when_base64_image_given_and_resize_needed(
    example_local_image: Tuple[str, np.ndarray]
) -> None:
    # given
    _, image = example_local_image
    _, buffer = cv2.imencode(".jpg", image)
    base64_image = base64.b64encode(buffer).decode("ascii")

    # when
    serialised_image, scaling_factor = await load_image_from_string_async(
        reference=base64_image,
        max_width=64,
        max_height=128,
    )
    recovered_image = base64.b64decode(serialised_image)
    bytes_array = np.frombuffer(recovered_image, dtype=np.uint8)
    decoding_result = cv2.imdecode(bytes_array, cv2.IMREAD_UNCHANGED)

    # then
    assert abs(scaling_factor - 0.5) < 1e-5
    assert decoding_result.shape == (64, 64, 3)
    assert (decoding_result == 0).all()


def test_load_image_from_uri_when_local_non_image_file_to_be_loaded(
    example_local_text_file: str,
) -> None:
    # when
    with pytest.raises(EncodingError):
        _ = load_image_from_string(
            reference=example_local_text_file,
            max_width=64,
            max_height=128,
        )


def test_load_static_inference_input_when_single_path_is_passed(
    example_local_image: Tuple[str, np.ndarray]
) -> None:
    # given
    file_path, image = example_local_image

    # when
    result = load_static_inference_input(inference_input=[file_path])

    # then
    assert len(result) == 1
    assert result[0][1] is None
    recovered_image = base64.b64decode(result[0][0])
    bytes_array = np.frombuffer(recovered_image, dtype=np.uint8)
    decoding_result = cv2.imdecode(bytes_array, cv2.IMREAD_UNCHANGED)
    assert decoding_result.shape == image.shape
    assert np.allclose(decoding_result, image)


@pytest.mark.asyncio
async def test_load_static_inference_input_async_when_single_path_is_passed(
    example_local_image: Tuple[str, np.ndarray]
) -> None:
    # given
    file_path, image = example_local_image

    # when
    result = await load_static_inference_input_async(inference_input=[file_path])

    # then
    assert len(result) == 1
    assert result[0][1] is None
    recovered_image = base64.b64decode(result[0][0])
    bytes_array = np.frombuffer(recovered_image, dtype=np.uint8)
    decoding_result = cv2.imdecode(bytes_array, cv2.IMREAD_UNCHANGED)
    assert decoding_result.shape == image.shape
    assert np.allclose(decoding_result, image)


def test_load_static_inference_input_when_single_np_array_passed(
    example_local_image: Tuple[str, np.ndarray]
) -> None:
    # given
    file_path, image = example_local_image

    # when
    result = load_static_inference_input(inference_input=[image])

    # then
    assert len(result) == 1
    assert result[0][1] is None
    recovered_image = base64.b64decode(result[0][0])
    bytes_array = np.frombuffer(recovered_image, dtype=np.uint8)
    decoding_result = cv2.imdecode(bytes_array, cv2.IMREAD_UNCHANGED)
    assert decoding_result.shape == image.shape
    assert np.allclose(decoding_result, image)


@pytest.mark.asyncio
async def test_load_static_inference_input_async_when_single_np_array_passed(
    example_local_image: Tuple[str, np.ndarray]
) -> None:
    # given
    file_path, image = example_local_image

    # when
    result = await load_static_inference_input_async(inference_input=[image])

    # then
    assert len(result) == 1
    assert result[0][1] is None
    recovered_image = base64.b64decode(result[0][0])
    bytes_array = np.frombuffer(recovered_image, dtype=np.uint8)
    decoding_result = cv2.imdecode(bytes_array, cv2.IMREAD_UNCHANGED)
    assert decoding_result.shape == image.shape
    assert np.allclose(decoding_result, image)


def test_load_static_inference_input_when_single_pillow_image_passed(
    example_local_image: Tuple[str, np.ndarray]
) -> None:
    # given
    file_path, image = example_local_image

    # when
    result = load_static_inference_input(inference_input=Image.fromarray(image))

    # then
    assert result[0][1] is None
    recovered_image = base64.b64decode(result[0][0])
    bytes_array = np.frombuffer(recovered_image, dtype=np.uint8)
    decoding_result = cv2.imdecode(bytes_array, cv2.IMREAD_UNCHANGED)
    assert decoding_result.shape == image.shape
    assert np.allclose(decoding_result, image)


@pytest.mark.asyncio
async def test_load_static_inference_input_async_when_single_pillow_image_passed(
    example_local_image: Tuple[str, np.ndarray]
) -> None:
    # given
    file_path, image = example_local_image

    # when
    result = await load_static_inference_input_async(
        inference_input=Image.fromarray(image)
    )

    # then
    assert result[0][1] is None
    recovered_image = base64.b64decode(result[0][0])
    bytes_array = np.frombuffer(recovered_image, dtype=np.uint8)
    decoding_result = cv2.imdecode(bytes_array, cv2.IMREAD_UNCHANGED)
    assert decoding_result.shape == image.shape
    assert np.allclose(decoding_result, image)


def test_load_static_inference_input_when_invalid_input_passed() -> None:
    # given
    invalid_input = b"Some invalid input"

    # when
    with pytest.raises(InvalidInputFormatError):
        _ = load_static_inference_input(inference_input=invalid_input)  # type: ignore


@pytest.mark.asyncio
async def test_load_static_inference_input_async_when_invalid_input_passed() -> None:
    # given
    invalid_input = b"Some invalid input"

    # when
    with pytest.raises(InvalidInputFormatError):
        _ = await load_static_inference_input_async(inference_input=invalid_input)  # type: ignore


def test_load_static_inference_input_when_multiple_inputs_passed(
    example_local_image: Tuple[str, np.ndarray]
) -> None:
    # given
    file_path, image = example_local_image

    # when
    result = load_static_inference_input(
        inference_input=[image, file_path],
        max_width=64,
        max_height=64,
    )

    # then
    assert len(result) == 2
    for idx in range(2):
        assert abs(result[idx][1] - 0.5) < 1e-5
        recovered_image = base64.b64decode(result[idx][0])
        bytes_array = np.frombuffer(recovered_image, dtype=np.uint8)
        decoding_result = cv2.imdecode(bytes_array, cv2.IMREAD_UNCHANGED)
        assert decoding_result.shape == (64, 64, 3)
        assert (decoding_result == 0).all()


@pytest.mark.asyncio
async def test_load_static_inference_input_async_when_multiple_inputs_passed(
    example_local_image: Tuple[str, np.ndarray]
) -> None:
    # given
    file_path, image = example_local_image

    # when
    result = await load_static_inference_input_async(
        inference_input=[image, file_path],
        max_width=64,
        max_height=64,
    )

    # then
    assert len(result) == 2
    for idx in range(2):
        assert abs(result[idx][1] - 0.5) < 1e-5
        recovered_image = base64.b64decode(result[idx][0])
        bytes_array = np.frombuffer(recovered_image, dtype=np.uint8)
        decoding_result = cv2.imdecode(bytes_array, cv2.IMREAD_UNCHANGED)
        assert decoding_result.shape == (64, 64, 3)
        assert (decoding_result == 0).all()


def test_load_static_inference_input_when_invalid_input_passed_among_multiple_inputs(
    example_local_image: Tuple[str, np.ndarray]
) -> None:
    # given
    file_path, image = example_local_image
    invalid_input = b"Some invalid input"

    # when
    with pytest.raises(InvalidInputFormatError):
        _ = load_static_inference_input(inference_input=[invalid_input, file_path, image])  # type: ignore


@pytest.mark.asyncio
async def test_load_static_inference_input_async_when_invalid_input_passed_among_multiple_inputs(
    example_local_image: Tuple[str, np.ndarray]
) -> None:
    # given
    file_path, image = example_local_image
    invalid_input = b"Some invalid input"

    # when
    with pytest.raises(InvalidInputFormatError):
        _ = await load_static_inference_input_async(inference_input=[invalid_input, file_path, image])  # type: ignore


def test_load_directory_inference_input(example_directory_with_images: str) -> None:
    # when
    results = list(
        load_directory_inference_input(
            directory_path=example_directory_with_images,
            image_extensions=None,
        )
    )
    file_name2results = {os.path.basename(e[0]): e[1] for e in results}

    # then
    assert len(results) == 2
    for expected_file_name in ["file_1.jpg", "file_2.png"]:
        assert file_name2results[expected_file_name].shape == (128, 128, 3)


def test_load_directory_inference_input_when_filtering_by_extension_is_enabled(
    example_directory_with_images: str,
) -> None:
    # when
    results = list(
        load_directory_inference_input(
            directory_path=example_directory_with_images,
            image_extensions=["jpg"],
        )
    )
    file_name2results = {os.path.basename(e[0]): e[1] for e in results}

    # then
    assert len(results) == 1
    assert file_name2results["file_1.jpg"].shape == (128, 128, 3)


@mock.patch.object(loaders.sv, "get_video_frames_generator")
def test_load_stream_inference_input(
    get_video_frames_generator_mock: MagicMock,
) -> None:
    # given
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    get_video_frames_generator_mock.return_value = (image for _ in range(10))

    # when
    result = list(
        load_stream_inference_input(input_uri="/some/video.mp4", image_extensions=None)
    )

    # then
    assert len(result) == 10
    for idx in range(10):
        assert result[idx] == (idx, image)
    get_video_frames_generator_mock.assert_called_once_with(
        source_path="/some/video.mp4"
    )


@mock.patch.object(loaders, "load_static_inference_input")
def test_load_nested_batches_of_inference_input_when_single_element_is_given(
    load_static_inference_input_mock: MagicMock,
) -> None:
    # given
    load_static_inference_input_mock.side_effect = [
        ["image_1"]
    ]

    # when
    result = load_nested_batches_of_inference_input(
        inference_input="my_image",
    )

    # then
    assert result == "image_1", "Expected direct result from load_static_inference_input()"


@mock.patch.object(loaders, "load_static_inference_input")
def test_load_nested_batches_of_inference_input_when_1d_batch_is_given(
    load_static_inference_input_mock: MagicMock,
) -> None:
    # given
    load_static_inference_input_mock.side_effect = [
        ["image_1"],
        ["image_2"],
        ["image_3"]
    ]

    # when
    result = load_nested_batches_of_inference_input(
        inference_input=["1", "2", "3"],
    )

    # then
    assert result == ["image_1", "image_2", "image_3"], "Expected direct result from load_static_inference_input()"


@mock.patch.object(loaders, "load_static_inference_input")
def test_load_nested_batches_of_inference_input_when_nested_batch_is_given(
    load_static_inference_input_mock: MagicMock,
) -> None:
    # given
    load_static_inference_input_mock.side_effect = [
        ["image_1"],
        ["image_2"],
        ["image_3"],
        ["image_4"],
        ["image_5"],
    ]

    # when
    result = load_nested_batches_of_inference_input(
        inference_input=[["1", "2"], ["3"], [["4", "5"]]],
    )

    # then
    assert result == [["image_1", "image_2"], ["image_3"], [["image_4", "image_5"]]]
