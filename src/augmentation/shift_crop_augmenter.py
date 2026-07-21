"""Ports ``legacy/DataRandomCrops.ipynb``: shifts an image a random number of
pixels toward each of the four edges (zero-filling the vacated border) to add
translation invariance, as the original notebook did per-direction."""
from __future__ import annotations

import random

import numpy as np

_MIN_SHIFT = 5
_MAX_SHIFT = 25


class ShiftCropAugmenter:
    """Single job: produce random-shifted crop variants of an image."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    def augment_all(self, image: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "down": self._shift(image, axis=0, towards_end=True),
            "up": self._shift(image, axis=0, towards_end=False),
            "right": self._shift(image, axis=1, towards_end=True),
            "left": self._shift(image, axis=1, towards_end=False),
        }

    def _shift(self, image: np.ndarray, axis: int, towards_end: bool) -> np.ndarray:
        shift = self._rng.randrange(_MIN_SHIFT, _MAX_SHIFT)
        shifted = np.roll(image, shift if towards_end else -shift, axis=axis)

        if axis == 0:
            if towards_end:
                shifted[:shift, ...] = 0
            else:
                shifted[-shift:, ...] = 0
        else:
            if towards_end:
                shifted[:, :shift, ...] = 0
            else:
                shifted[:, -shift:, ...] = 0

        return shifted
