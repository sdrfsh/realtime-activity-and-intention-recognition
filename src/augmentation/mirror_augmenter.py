"""Ports ``legacy/DataMirroring.ipynb``: a horizontal-flip augmentation that
teaches the classifier the activity looks the same facing either direction."""
from __future__ import annotations

import numpy as np


class MirrorAugmenter:
    """Single job: produce the horizontally-mirrored version of an image."""

    def augment(self, image: np.ndarray) -> np.ndarray:
        return np.fliplr(image)
