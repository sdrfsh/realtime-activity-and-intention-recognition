from __future__ import annotations

from collections import deque

import numpy as np


class FrameWindow:
    def __init__(self, max_frames: int) -> None:
        if max_frames < 2:
            raise ValueError("max_frames must be at least 2")
        self._frames: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=max_frames)

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def capacity(self) -> int:
        return self._frames.maxlen or 0

    @property
    def is_full(self) -> bool:
        return len(self._frames) == self._frames.maxlen

    def append(self, raw: np.ndarray, mask: np.ndarray) -> None:
        self._frames.append((raw, mask))

    def clear(self) -> None:
        self._frames.clear()

    def raw(self) -> list[np.ndarray]:
        return [raw for raw, _ in self._frames]

    def masks(self) -> list[np.ndarray]:
        return [mask for _, mask in self._frames]