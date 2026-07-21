"""Docx 3.3: "the dataset was divided into three sections: 60% training,
30% validation, and 10% for testing." Pure logic, no ML framework dependency."""
from __future__ import annotations

import random
from pathlib import Path

from config import NetworkSettings
from domain.entities import DatasetSplit


class DatasetSplitter:
    """Single job: partition labeled samples into train/validation/test sets."""

    def __init__(self, settings: NetworkSettings, rng: random.Random | None = None) -> None:
        self._settings = settings
        self._rng = rng or random.Random()

    def split(self, samples: list[tuple[Path, str]]) -> DatasetSplit:
        shuffled = list(samples)
        self._rng.shuffle(shuffled)

        total = len(shuffled)
        train_end = round(total * self._settings.train_ratio)
        validation_end = train_end + round(total * self._settings.validation_ratio)

        return DatasetSplit(
            train=shuffled[:train_end],
            validation=shuffled[train_end:validation_end],
            test=shuffled[validation_end:],
        )
