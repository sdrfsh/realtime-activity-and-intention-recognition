from __future__ import annotations

import time

import numpy as np

from config import LiveSettings


class MotionTrigger:
    def __init__(self, settings: LiveSettings) -> None:
        self._settings = settings
        self._collecting = False
        self._cooldown_until = 0.0
        self._started_at: float | None = None

    def update(self, mask: np.ndarray, now: float | None = None) -> bool:
        current_time = time.monotonic() if now is None else now
        if current_time < self._cooldown_until:
            return False
        if not self._collecting and int(np.count_nonzero(mask)) >= self._settings.motion_pixel_threshold:
            self._collecting = True
            self._started_at = current_time
            return True
        return False

    @property
    def collecting(self) -> bool:
        return self._collecting

    @property
    def started_at(self) -> float | None:
        return self._started_at

    def finish(self, now: float | None = None) -> None:
        current_time = time.monotonic() if now is None else now
        self._collecting = False
        self._cooldown_until = current_time + self._settings.cooldown_seconds
        self._started_at = None