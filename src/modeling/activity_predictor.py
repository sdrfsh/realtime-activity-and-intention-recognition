"""The inference-time counterpart of ``ModelTrainer``: loads a trained model
and classifies a single motion image (the real-time / embedded use case
described in docx 3.4), returning the human-readable class name (e.g.
"entering", "passing_by") the model was trained with."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model

from config import NetworkSettings
from data.label_map_repository import LabelMapRepository
from domain.entities import PredictionResult


class ActivityPredictor:
    """Single job: turn one motion image into a predicted class label."""

    def __init__(
        self,
        settings: NetworkSettings,
        model_path: Path,
        label_map_repository: LabelMapRepository,
    ) -> None:
        self._settings = settings
        self._model = load_model(str(model_path))
        self._classes = label_map_repository.load()

    def predict(self, motion_image: np.ndarray) -> PredictionResult:
        height, width, _ = self._settings.input_shape
        batch = motion_image.reshape(1, height, width, -1).astype(np.float32)
        probabilities = self._model.predict(batch)[0]
        label_index = int(np.argmax(probabilities))
        label = self._classes[label_index] if label_index < len(self._classes) else str(label_index)
        return PredictionResult(
            label_index=label_index,
            label=label,
            class_probabilities=probabilities,
        )
