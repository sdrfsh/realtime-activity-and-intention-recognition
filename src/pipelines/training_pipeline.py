"""Orchestrates docx 3.3: load the prepared training images, 60/30/10 split,
build + compile + fit AlexNet, save the trained model. Mirrors
``legacy/AlexNet(v1.9).ipynb`` but with each concern in its own class.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from config import NetworkSettings
from data.image_loader import ImageLoader
from data.label_map_repository import LabelMapRepository
from data.label_repository import LabelRepository
from modeling.alexnet_builder import AlexNetBuilder
from modeling.dataset_splitter import DatasetSplitter
from modeling.label_encoder import LabelEncoder
from modeling.model_trainer import ModelTrainer


class TrainingPipeline:
    """Single job: turn the prepared image dataset into a saved, trained model."""

    def __init__(
        self,
        settings: NetworkSettings,
        training_labels: LabelRepository,
        image_loader: ImageLoader,
        dataset_splitter: DatasetSplitter,
        model_builder: AlexNetBuilder,
        model_trainer: ModelTrainer,
        label_map_repository: LabelMapRepository,
        model_output_path: Path,
    ) -> None:
        self._settings = settings
        self._training_labels = training_labels
        self._image_loader = image_loader
        self._dataset_splitter = dataset_splitter
        self._model_builder = model_builder
        self._model_trainer = model_trainer
        self._label_map_repository = label_map_repository
        self._model_output_path = model_output_path

    def run(self):
        clips = self._training_labels.read()
        samples = [(clip.video_path, clip.label) for clip in clips]
        split = self._dataset_splitter.split(samples)

        encoder = LabelEncoder([label for _, label in samples])
        self._label_map_repository.save(encoder.classes)

        x_train, y_train = self._load_batch(split.train, encoder)
        x_validation, y_validation = self._load_batch(split.validation, encoder)

        model = self._model_builder.build()
        self._model_trainer.compile(model)
        history = self._model_trainer.train(model, x_train, y_train, x_validation, y_validation)
        self._model_trainer.save(model, self._model_output_path)
        return history

    def _load_batch(self, samples: list[tuple[Path, str]], encoder: LabelEncoder):
        images = np.asarray([self._image_loader.load(path) for path, _ in samples], dtype=np.float32)
        labels = np.asarray(encoder.encode_all([label for _, label in samples]), dtype=np.int64)
        return images, labels
