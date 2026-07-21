"""Writes a numpy image array to disk. The only class allowed to touch cv2.imwrite here."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class ImageWriter:
    """Single job: persist an image array to a file path."""

    def save(self, image: np.ndarray, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image)
