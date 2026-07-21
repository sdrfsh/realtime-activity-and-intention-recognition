"""Reads and writes the "File,Label" CSV format used throughout the pipeline.

This replaces the ad-hoc ``pd.read_csv`` / ``csv.writer`` calls scattered
across the original notebooks (``Data_labeling*.ipynb``, ``FindContours*.ipynb``)
with one class that owns the on-disk label format.
"""
from __future__ import annotations

import csv
from pathlib import Path

from domain.entities import LabeledClip


class LabelRepository:
    """Single job: persist and retrieve (path, label) pairs as CSV."""

    _FIELDNAMES = ("File", "Label")

    def __init__(self, csv_path: Path) -> None:
        self._csv_path = csv_path

    def read(self) -> list[LabeledClip]:
        if not self._csv_path.exists():
            return []
        with self._csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return [
                LabeledClip(video_path=Path(row["File"]), label=row["Label"])
                for row in reader
            ]

    def write(self, clips: list[LabeledClip]) -> None:
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self._csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(self._FIELDNAMES)
            for clip in clips:
                writer.writerow([str(clip.video_path), clip.label])
