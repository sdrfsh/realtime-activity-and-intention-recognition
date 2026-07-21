"""Plain data objects shared across layers. No behavior, no I/O, no third-party imports."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LabeledClip:
    """One raw video clip and its activity label."""

    video_path: Path
    label: str


@dataclass(frozen=True)
class SampledFrames:
    """The frames selected from a clip by the adaptive sampler, plus the rate used."""

    frames: list[np.ndarray]
    sampling_rate: int


@dataclass(frozen=True)
class MotionImage:
    """The single recency-weighted image produced for one clip (the "dynamic image")."""

    image: np.ndarray
    source_clip: Path
    label: str | None = None


@dataclass(frozen=True)
class DatasetSplit:
    """Train/validation/test partition of image paths + labels."""

    train: list[tuple[Path, str]]
    validation: list[tuple[Path, str]]
    test: list[tuple[Path, str]]


@dataclass(frozen=True)
class PredictionResult:
    label_index: int
    label: str
    class_probabilities: np.ndarray
