"""Docx 3.4 "Designing Real-Time embedded System": the end-to-end path from a
freshly captured clip to a predicted activity. Replaces ``legacy/FPGAside.py``
with the same steps, each owned by its own class.
"""
from __future__ import annotations

from pathlib import Path

from domain.entities import PredictionResult
from modeling.activity_predictor import ActivityPredictor
from modeling.network_input_formatter import NetworkInputFormatter
from pipelines.clip_preprocessing_pipeline import ClipPreprocessingPipeline


class RealtimeInferencePipeline:
    """Single job: classify one incoming video clip, end to end."""

    def __init__(
        self,
        clip_pipeline: ClipPreprocessingPipeline,
        input_formatter: NetworkInputFormatter,
        predictor: ActivityPredictor,
    ) -> None:
        self._clip_pipeline = clip_pipeline
        self._input_formatter = input_formatter
        self._predictor = predictor

    def predict(self, video_path: Path) -> PredictionResult | None:
        motion_image = self._clip_pipeline.process(video_path)
        if motion_image is None:
            return None

        network_input = self._input_formatter.format(motion_image.image)
        return self._predictor.predict(network_input)
