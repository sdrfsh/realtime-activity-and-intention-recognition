"""Maps string labels (e.g. "entering" / "passing_by") to the integer class
indices Keras expects, and back again for human-readable predictions."""
from __future__ import annotations


class LabelEncoder:
    """Single job: convert between string labels and integer class indices."""

    def __init__(self, labels: list[str]) -> None:
        self._classes = sorted(set(labels))
        self._label_to_index = {label: index for index, label in enumerate(self._classes)}

    def encode(self, label: str) -> int:
        return self._label_to_index[label]

    def encode_all(self, labels: list[str]) -> list[int]:
        return [self.encode(label) for label in labels]

    def decode(self, index: int) -> str:
        return self._classes[index]

    @property
    def classes(self) -> list[str]:
        """Class names ordered by their integer index (index 0 first)."""
        return list(self._classes)

    @property
    def num_classes(self) -> int:
        return len(self._classes)
