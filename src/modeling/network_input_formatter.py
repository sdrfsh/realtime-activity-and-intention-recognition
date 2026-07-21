"""Converts a raw (grayscale, arbitrarily-sized) motion image into the exact
tensor shape AlexNet expects: 3-channel, resized to ``input_shape``."""
from __future__ import annotations

import cv2
import numpy as np

from config import NetworkSettings


class NetworkInputFormatter:
    """Single job: adapt a motion image to the network's expected input tensor."""

    def __init__(self, settings: NetworkSettings) -> None:
        self._settings = settings

    def format(self, motion_image: np.ndarray) -> np.ndarray:
        height, width, channels = self._settings.input_shape
        resized = cv2.resize(motion_image, (width, height))
        if resized.ndim == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        return resized.astype(np.float32)
