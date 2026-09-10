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


@dataclass(frozen=True)
class DoorApproach:
    """Heading-based verdict for one window: where the subject's centroid
    started and ended (0..1 frame fractions), how far it moved toward and
    along the door edge, and the resulting label."""

    label: str
    entering_score: float
    start: tuple[float, float]
    end: tuple[float, float]
    toward_door: float
    across_door: float
    trajectory: list[tuple[float, float]]


@dataclass(frozen=True)
class LiveDecision:
    label: str
    confidence: float
    class_probabilities: np.ndarray
    started_at: float
    decided_at: float
    latency_s: float
