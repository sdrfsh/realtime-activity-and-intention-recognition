"""Docx 3.2: "the most recent sampled frames are more significant in this
representation. Therefore as we add images to each other the frames are
weighted and the addition of the new frame is done with 70% of the previous
frames ... each time the previous frames are decreased by 30%."

This is the "dynamic image" step: an ordered list of sampled frames collapses
into one still image where older motion fades and recent motion dominates.
Implemented with plain numpy (saturating like ``cv2.addWeighted`` would) so it
has no OpenCV dependency and is trivially unit-testable.
"""
from __future__ import annotations

import numpy as np

from config import SamplingSettings


class MotionImageEncoder:
    """Single job: collapse an ordered frame sequence into one recency-weighted image."""

    def __init__(self, settings: SamplingSettings) -> None:
        self._settings = settings

    def encode(self, frames: list[np.ndarray]) -> np.ndarray:
        if not frames:
            raise ValueError("cannot encode an empty frame sequence")

        accumulated = np.zeros_like(frames[0], dtype=np.float64)
        decay = self._settings.decay_weight
        for frame in frames:
            accumulated = decay * accumulated + frame.astype(np.float64)

        return np.clip(accumulated, 0, 255).astype(np.uint8)
