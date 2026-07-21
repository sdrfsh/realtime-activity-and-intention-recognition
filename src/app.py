"""The composition root: the one place that knows how to wire every
single-responsibility class together. Nothing here contains business logic —
it only builds dependencies and hands them to the three pipelines.
"""
from __future__ import annotations

from pathlib import Path

from augmentation.mirror_augmenter import MirrorAugmenter
from augmentation.shift_crop_augmenter import ShiftCropAugmenter
from config import Settings, load_settings
from data.image_loader import ImageLoader
from data.image_writer import ImageWriter
from data.label_map_repository import LabelMapRepository
from data.label_repository import LabelRepository
from data.video_reader import VideoReader
from data.video_repository import VideoRepository
from domain.entities import PredictionResult
from modeling.activity_predictor import ActivityPredictor
from modeling.alexnet_builder import AlexNetBuilder
from modeling.dataset_splitter import DatasetSplitter
from modeling.model_trainer import ModelTrainer
from modeling.network_input_formatter import NetworkInputFormatter
from pipelines.clip_preprocessing_pipeline import ClipPreprocessingPipeline
from pipelines.dataset_preparation_pipeline import DatasetPreparationPipeline
from pipelines.realtime_inference_pipeline import RealtimeInferencePipeline
from pipelines.training_pipeline import TrainingPipeline
from preprocessing.adaptive_frame_sampler import AdaptiveFrameSampler
from preprocessing.background_subtractor import BackgroundSubtractor
from preprocessing.motion_contour_detector import MotionContourDetector
from preprocessing.motion_image_encoder import MotionImageEncoder
from preprocessing.noise_reducer import NoiseReducer
from preprocessing.optical_flow_estimator import OpticalFlowEstimator


class Application:
    """The main class: exposes the three user-facing operations
    (prepare data, train, predict) and builds every collaborator needed to
    perform them from a single :class:`Settings` object.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or load_settings()
        self._settings.paths.ensure_exists()

    def _build_clip_pipeline(self) -> ClipPreprocessingPipeline:
        tracking = self._settings.tracking
        sampling = self._settings.sampling
        return ClipPreprocessingPipeline(
            video_reader=VideoReader(),
            noise_reducer=NoiseReducer(tracking),
            background_subtractor=BackgroundSubtractor(tracking),
            contour_detector=MotionContourDetector(tracking),
            flow_estimator=OpticalFlowEstimator(sampling),
            frame_sampler=AdaptiveFrameSampler(sampling),
            motion_encoder=MotionImageEncoder(sampling),
        )

    def prepare_dataset(self) -> None:
        """Docx 3.1 + 3.2 + augmentation: raw clips -> augmented training images."""
        paths = self._settings.paths
        pipeline = DatasetPreparationPipeline(
            video_repository=VideoRepository(paths.raw_video_dir),
            source_labels=LabelRepository(paths.labels_csv),
            filtered_labels=LabelRepository(paths.filtered_labels_csv),
            training_labels=LabelRepository(paths.training_csv),
            clip_pipeline=self._build_clip_pipeline(),
            image_writer=ImageWriter(),
            mirror_augmenter=MirrorAugmenter(),
            shift_crop_augmenter=ShiftCropAugmenter(),
            motion_image_dir=paths.motion_image_dir,
            augmented_image_dir=paths.augmented_image_dir,
        )
        pipeline.run()

    def train(self):
        """Docx 3.3: prepared images -> trained, saved AlexNet model."""
        network = self._settings.network
        paths = self._settings.paths
        pipeline = TrainingPipeline(
            settings=network,
            training_labels=LabelRepository(paths.training_csv),
            image_loader=ImageLoader(target_size=network.input_shape[:2]),
            dataset_splitter=DatasetSplitter(network),
            model_builder=AlexNetBuilder(network),
            model_trainer=ModelTrainer(network),
            label_map_repository=LabelMapRepository(paths.label_map_path),
            model_output_path=paths.trained_model_path,
        )
        return pipeline.run()

    def predict(self, video_path: Path) -> PredictionResult | None:
        """Docx 3.4: a single new clip -> predicted intention/activity, in real time."""
        network = self._settings.network
        paths = self._settings.paths
        pipeline = RealtimeInferencePipeline(
            clip_pipeline=self._build_clip_pipeline(),
            input_formatter=NetworkInputFormatter(network),
            predictor=ActivityPredictor(
                network, paths.trained_model_path, LabelMapRepository(paths.label_map_path)
            ),
        )
        return pipeline.predict(video_path)
