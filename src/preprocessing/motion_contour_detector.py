"""Docx 3.1: "the condition for the presence of a moving contour in the image is
considered the starting point for tracking and confirms the presence of the
person(s) in the scene." Clips with no contours anywhere are discarded."""
from __future__ import annotations

import cv2
import numpy as np

from config import TrackingSettings


class MotionContourDetector:
    """Single job: decide whether a (background-subtracted) frame contains a moving subject."""

    def __init__(self, settings: TrackingSettings) -> None:
        self._settings = settings

    def contour_count(self, frame: np.ndarray) -> int:
        dilated = cv2.dilate(frame, None, iterations=self._settings.morphology_iterations)
        eroded = cv2.erode(dilated, None, iterations=self._settings.morphology_iterations)
        low, high = self._settings.canny_thresholds
        edges = cv2.Canny(eroded, low, high)
        contours = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        return len(contours)

    def has_motion(self, frames: list[np.ndarray]) -> bool:
        return sum(self.contour_count(frame) for frame in frames) > 0
