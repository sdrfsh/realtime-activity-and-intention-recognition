from __future__ import annotations

import time
from typing import Iterator

import cv2
import numpy as np


class CameraFrameSource:
    def __init__(self, device_index_or_url: int | str = 0) -> None:
        self._capture = cv2.VideoCapture(device_index_or_url)
        if not self._capture.isOpened():
            raise RuntimeError(f"could not open camera source: {device_index_or_url}")

    def frames(self) -> Iterator[np.ndarray]:
        while self._capture.isOpened():
            success, frame = self._capture.read()
            if not success:
                break
            yield frame

    def measure_fps(self, frame_count: int = 60) -> float:
        start = time.perf_counter()
        captured = 0
        while captured < frame_count:
            success, _ = self._capture.read()
            if not success:
                break
            captured += 1
        elapsed = time.perf_counter() - start
        if captured < 2 or elapsed <= 0:
            raise RuntimeError("could not measure camera frame rate")
        return (captured - 1) / elapsed

    def fps(self) -> float | None:
        value = self._capture.get(cv2.CAP_PROP_FPS)
        return float(value) if value and value > 0 else None

    def close(self) -> None:
        self._capture.release()