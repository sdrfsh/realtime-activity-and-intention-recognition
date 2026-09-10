from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from data.video_reader import VideoReader


class FileFrameSource:
    def __init__(self, path: Path, reader: VideoReader | None = None) -> None:
        self._path = path
        self._reader = reader or VideoReader()

    def frames(self) -> Iterator[np.ndarray]:
        yield from self._reader.frames(self._path)

    def fps(self) -> float | None:
        return None

    def close(self) -> None:
        return None