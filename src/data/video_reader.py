"""Reads frames out of a video file. The only class allowed to touch cv2.VideoCapture."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class VideoReader:
    """Single job: stream the frames of a video file."""

    def frames(self, video_path: Path) -> Iterator[np.ndarray]:
        capture = cv2.VideoCapture(str(video_path))
        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                yield frame
        finally:
            capture.release()

    def frame_count(self, video_path: Path) -> int:
        capture = cv2.VideoCapture(str(video_path))
        try:
            return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
