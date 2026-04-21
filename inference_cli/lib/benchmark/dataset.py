import hashlib
import os.path
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from itertools import chain
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import requests
from tqdm import tqdm

from inference_cli.lib.exceptions import DatasetLoadingError
from inference_cli.lib.logger import CLI_LOGGER
from inference_sdk.http.utils.encoding import bytes_to_opencv_image

MAX_IMAGES_TO_LOAD = 8
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]
CACHE_DIR = Path.home() / ".cache" / "inference_cli" / "benchmark_images"
PREDEFINED_DATASETS = {
    "coco": [
        "https://source.roboflow.com/BTRTpB7nxxjUchrOQ9vT/aFq7tthQAK6d4pvtupX7/original.jpg",
        "https://source.roboflow.com/BTRTpB7nxxjUchrOQ9vT/KmFskd2RQMfcnDNjzeeA/original.jpg",
        "https://source.roboflow.com/BTRTpB7nxxjUchrOQ9vT/3FBCYL5SX7VPrg0OVkdN/original.jpg",
        "https://source.roboflow.com/BTRTpB7nxxjUchrOQ9vT/K2KrTzjxYu0kJCScGcoH/original.jpg",
        "https://source.roboflow.com/BTRTpB7nxxjUchrOQ9vT/XzDB9zVrIxJm17iVKleP/original.jpg",
        "https://source.roboflow.com/BTRTpB7nxxjUchrOQ9vT/0fsReHjmHk3hBadXdNk4/original.jpg",
        "https://source.roboflow.com/BTRTpB7nxxjUchrOQ9vT/t23lZ0inksJwRRLd3J1b/original.jpg",
        "https://source.roboflow.com/BTRTpB7nxxjUchrOQ9vT/3iCH40NuJxcf8l2tXgQn/original.jpg",
    ]
}


def load_dataset_images(dataset_reference: str) -> List[np.ndarray]:
    if os.path.isdir(dataset_reference):
        return load_images(directory=dataset_reference)
    if dataset_reference not in PREDEFINED_DATASETS:
        raise DatasetLoadingError(f"Could not find dataset: {dataset_reference}")
    return download_images(urls=PREDEFINED_DATASETS[dataset_reference])


def load_images(
    directory: str, max_images_to_load: int = MAX_IMAGES_TO_LOAD
) -> List[np.ndarray]:
    file_paths = sorted(
        list(
            chain.from_iterable(
                glob(os.path.join(directory, f"*{e}")) for e in IMAGE_EXTENSIONS
            )
        )
    )
    results = []
    progress_bar = tqdm(desc="Loading images...", total=max_images_to_load)
    for file_path in file_paths:
        image = load_image(path=file_path)
        if image is None:
            continue
        results.append(image)
        progress_bar.update()
    progress_bar.close()
    if len(results) < 1:
        raise DatasetLoadingError(f"Could not load images from {directory}")
    return results


def load_image(path: str) -> Optional[np.ndarray]:
    try:
        return cv2.imread(path)
    except Exception as error:
        CLI_LOGGER.warning(f"Could not load image: {path}. Cause: {error}")
        return None


def download_images(urls: List[str]) -> List[np.ndarray]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_url = {executor.submit(download_image, url): url for url in urls}

        with tqdm(total=len(urls), desc="Loading images...") as pbar:
            for future in as_completed(future_to_url):
                result = future.result()
                if result is not None:
                    results.append(result)
                pbar.update(1)

    if len(results) < 1:
        raise DatasetLoadingError(f"Could not load images")
    return results


def download_image(url: str) -> Optional[np.ndarray]:
    try:
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cache_path = CACHE_DIR / f"{url_hash}.jpg"

        if cache_path.exists():
            return cv2.imread(str(cache_path))

        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = bytes_to_opencv_image(payload=response.content)

        if image is not None:
            cv2.imwrite(str(cache_path), image)

        return image
    except Exception as error:
        CLI_LOGGER.warning(f"Could not load image: {url}. Cause: {error}")
        return None
