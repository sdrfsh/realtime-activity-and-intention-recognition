from __future__ import annotations

from typing import Iterator, Protocol

import numpy as np


class FrameSource(Protocol):
    def frames(self) -> Iterator[np.ndarray]:
        ...

    def fps(self) -> float | None:
        ...

    def close(self) -> None:
        ...