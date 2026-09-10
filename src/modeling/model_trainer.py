"""Docx 3.3: SGD was chosen over faster optimizers (e.g. Adam) because, given
the wide range of data the system will receive, "higher training capacity
would be more important than the convergence speed" — SGD's slower descent
generalizes better, and is tuned via learning rate and momentum."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from keras import Model
from keras.optimizers import SGD

from config import NetworkSettings


class ModelTrainer:
    """Single job: compile, fit, and persist an AlexNet model."""

    def __init__(self, settings: NetworkSettings) -> None:
        self._settings = settings

    def compile(self, model: Model) -> Model:
        optimizer = SGD(
            learning_rate=self._settings.learning_rate,
            momentum=self._settings.momentum,
        )
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=optimizer,
            metrics=["accuracy"],
        )
        return model

    def train(
        self,
        model: Model,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_validation: np.ndarray,
        y_validation: np.ndarray,
    ):
        return model.fit(
            x_train,
            y_train,
            batch_size=self._settings.batch_size,
            epochs=self._settings.epochs,
            verbose=1,
            validation_data=(x_validation, y_validation),
        )

    def save(self, model: Model, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(path))
