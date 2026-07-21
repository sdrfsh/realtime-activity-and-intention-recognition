"""Docx 3.2 "Sampling": "if the sampling rate is a fixed number, the data in
which the individual moves faster than the sampling rate will be missed and
the data in which the individual moves slower ... will be redundant. So,
based on individual's velocity an adaptive sampling rate is defined."

Pure arithmetic on frame counts / velocity — no OpenCV dependency, so it is
trivially unit-testable.
"""
from __future__ import annotations

from config import SamplingSettings


class AdaptiveFrameSampler:
    """Single job: decide *which* frame indices to keep from a clip."""

    def __init__(self, settings: SamplingSettings) -> None:
        self._settings = settings

    def compute_sampling_rate(self, frame_count: int, velocity: float) -> int:
        """Larger velocity -> smaller interval (sample more often), so fast
        motion isn't missed; smaller velocity -> larger interval, so slow
        motion doesn't produce redundant near-duplicate samples."""
        base_rate = max(frame_count // self._settings.max_samples_per_clip, 1)
        if velocity <= 0:
            return base_rate
        return max(int(round(base_rate / velocity)), 1)

    def select_frame_indices(self, frame_count: int, sampling_rate: int) -> list[int]:
        indices = list(range(0, frame_count, max(sampling_rate, 1)))
        indices = indices[: self._settings.max_samples_per_clip]
        return self._trim(indices)

    def _trim(self, indices: list[int]) -> list[int]:
        start, end = self._settings.trim_start, self._settings.trim_end
        if len(indices) <= start + end:
            return indices
        return indices[start : len(indices) - end]
