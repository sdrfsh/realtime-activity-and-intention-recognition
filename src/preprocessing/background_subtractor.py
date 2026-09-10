"""Docx 3.1: KNN background subtraction was chosen over the alternatives because
it "works better under a variety of lighting conditions than other methods"
(shadows, slow/sudden brightness changes, dynamic backgrounds)."""
from __future__ import annotations

import cv2
import numpy as np

from config import TrackingSettings


class BackgroundSubtractor:
    """Single job: turn a raw frame into a binary foreground (subject) mask.

    Stateful across the frames of one clip (the KNN model accumulates a
    background estimate), so call :meth:`reset` before starting a new clip.
    """

    def __init__(self, settings: TrackingSettings) -> None:
        self._settings = settings
        self._subtractor = self._new_subtractor()

    def _new_subtractor(self) -> cv2.BackgroundSubtractorKNN:
        return cv2.createBackgroundSubtractorKNN(
            history=self._settings.knn_history,
            dist2Threshold=self._settings.knn_dist2_threshold,
            detectShadows=self._settings.detect_shadows,
        )

    def reset(self) -> None:
        self._subtractor = self._new_subtractor()

    def apply(self, frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(
            frame,
            self._settings.frame_size,
            fx=0,
            fy=0,
            interpolation=cv2.INTER_CUBIC,
        )
        return self._subtractor.apply(resized)
