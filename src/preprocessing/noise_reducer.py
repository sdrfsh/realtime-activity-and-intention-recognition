"""Docx 3.1: "dilation and erosion operations and a proper Gaussian filter are
used to eliminate noises" before background subtraction runs."""
from __future__ import annotations

import cv2
import numpy as np

from config import TrackingSettings


class NoiseReducer:
    """Single job: smooth a frame and clean up small speckle noise."""

    def __init__(self, settings: TrackingSettings) -> None:
        self._settings = settings

    def apply(self, frame: np.ndarray) -> np.ndarray:
        smoothed = cv2.GaussianBlur(frame, self._settings.gaussian_kernel, 0)
        dilated = cv2.dilate(smoothed, None, iterations=self._settings.morphology_iterations)
        eroded = cv2.erode(dilated, None, iterations=self._settings.morphology_iterations)
        return eroded
