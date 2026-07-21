"""Reads image files from disk and resizes them to the network's input size."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class ImageLoader:
    """Single job: load an image file as a numpy array of a fixed size."""

    def __init__(self, target_size: tuple[int, int]) -> None:
        self._target_size = target_size

    def load(self, image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"could not read image: {image_path}")
        return cv2.resize(image, self._target_size)
