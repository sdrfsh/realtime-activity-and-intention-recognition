"""The inference-time counterpart of ``ModelTrainer``: loads a trained model
and classifies a single motion image (the real-time / embedded use case
described in docx 3.4), returning the human-readable class name (e.g.
"entering", "passing_by") the model was trained with."""
from __future__ import annotations

from pathlib import Path

import keras
import numpy as np

from config import NetworkSettings
from domain.entities import PredictionResult
from modeling.model_source import resolve_model_source


class ActivityPredictor:
    """Single job: turn one motion image into a predicted class label.

    ``model_source`` is a local ``.keras`` path or an ``hf://namespace/repo``
    reference; ``classes`` lists the label names in output-index order.
    """

    def __init__(
        self,
        settings: NetworkSettings,
        model_source: str | Path,
        classes: list[str] | tuple[str, ...],
    ) -> None:
        self._settings = settings
        model_path = resolve_model_source(model_source, settings.model_filename)
        self._model = keras.saving.load_model(model_path, compile=False)
        self._classes = list(classes)

    @property
    def classes(self) -> list[str]:
        """Label names in output-index order."""
        return list(self._classes)

    def predict(self, motion_image: np.ndarray) -> PredictionResult:
        height, width, _ = self._settings.input_shape
        batch = motion_image.reshape(1, height, width, -1).astype(np.float32)
        probabilities = np.asarray(self._model.predict(batch, verbose=0))[0]
        label_index = int(np.argmax(probabilities))
        label = self._classes[label_index] if label_index < len(self._classes) else str(label_index)
        return PredictionResult(
            label_index=label_index,
            label=label,
            class_probabilities=probabilities,
        )
