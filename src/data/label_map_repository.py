"""Persists the ordered list of class names a model was trained with, so
inference can turn a predicted index back into a human-readable label (e.g.
"entering", "passing_by") instead of a bare integer."""
from __future__ import annotations

import json
from pathlib import Path


class LabelMapRepository:
    """Single job: save/load the index-ordered list of class names."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def save(self, classes: list[str]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(list(classes)), encoding="utf-8")

    def load(self) -> list[str]:
        if not self._path.exists():
            return []
        return json.loads(self._path.read_text(encoding="utf-8"))
