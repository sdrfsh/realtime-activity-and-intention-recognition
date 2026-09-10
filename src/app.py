"""The composition root: the one place that knows how to wire every
single-responsibility class together. Nothing here contains business logic —
it only builds dependencies and hands them to the three pipelines.
"""
from __future__ import annotations

import time
from pathlib import Path
from dataclasses import replace

import numpy as np

from augmentation.mirror_augmenter import MirrorAugmenter
from augmentation.shift_crop_augmenter import ShiftCropAugmenter
from config import LiveSettings, Settings, load_settings
from data.image_loader import ImageLoader
from data.image_writer import ImageWriter
from data.camera_frame_source import CameraFrameSource
from data.file_frame_source import FileFrameSource
from data.label_map_repository import LabelMapRepository
from data.label_repository import LabelRepository
from data.video_reader import VideoReader
from data.video_repository import VideoRepository
from domain.entities import LiveDecision, PredictionResult
from modeling.activity_predictor import ActivityPredictor
from modeling.alexnet_builder import AlexNetBuilder
from modeling.dataset_splitter import DatasetSplitter
from modeling.door_approach_decider import DoorApproachDecider
from modeling.model_source import is_hub_source
from modeling.model_trainer import ModelTrainer
from pipelines.clip_preprocessing_pipeline import ClipPreprocessingPipeline
from pipelines.dataset_preparation_pipeline import DatasetPreparationPipeline
from pipelines.live_session import LiveSession
from pipelines.training_pipeline import TrainingPipeline
from preprocessing.adaptive_frame_sampler import AdaptiveFrameSampler
from preprocessing.background_subtractor import BackgroundSubtractor
from preprocessing.mask_cleaner import MaskCleaner
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

    def _build_door_decider(self, door_side: str | None = None) -> DoorApproachDecider:
        """Heading-based entering/passing decider. ``door_side`` overrides
        ``SceneSettings.door_side`` for this call only."""
        scene = self._settings.scene
        if door_side is not None:
            scene = replace(scene, door_side=door_side)
        return DoorApproachDecider(scene)

    def _live_settings(self, decision_mode: str | None = None, **overrides) -> LiveSettings:
        base = self._settings.live
        values = {key: value for key, value in overrides.items() if value is not None}
        if decision_mode is not None:
            values["decision_mode"] = decision_mode
        return replace(base, **values)

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
            mask_cleaner=MaskCleaner(tracking),
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

    def _build_predictor(self, model_source: str | Path | None = None) -> ActivityPredictor:
        """Load the inference model from a local ``.keras`` path or an
        ``hf://namespace/repo`` reference (default: the published pretrained
        classifier). Local models use the ``label_map.json`` written next to
        them by ``train``; the Hub model uses the class order from settings."""
        network = self._settings.network
        source = network.model_source if model_source is None else model_source
        if is_hub_source(source):
            classes: list[str] | tuple[str, ...] = network.pretrained_classes
        else:
            classes = LabelMapRepository(Path(source).with_name("label_map.json")).load()
        return ActivityPredictor(network, source, classes)

    def predict(
        self,
        video_path: Path,
        model_source: str | Path | None = None,
        door_side: str | None = None,
        decision_mode: str | None = None,
    ) -> PredictionResult | None:
        """Development/regression prediction through the live session path."""
        network = self._settings.network
        live_settings = self._live_settings(decision_mode)
        heading = live_settings.decision_mode == "heading"
        session = LiveSession(
            source=FileFrameSource(video_path),
            tracking=self._settings.tracking,
            sampling=self._settings.sampling,
            network=network,
            live=live_settings,
            predictor=None if heading else self._build_predictor(model_source),
            door_decider=self._build_door_decider(door_side),
        )
        decision = session.run_until_decision()
        if decision is None:
            return None
        label_index = int(np.argmax(decision.class_probabilities))
        return PredictionResult(label_index, decision.label, decision.class_probabilities)

    def live(
        self,
        camera_source: int | str | None = None,
        window_seconds: float | None = None,
        motion_threshold: int | None = None,
        preview: bool | None = None,
        decide: bool = True,
        model_source: str | Path | None = None,
        door_side: str | None = None,
        decision_mode: str | None = None,
    ) -> None:
        network = self._settings.network
        live_settings = self._live_settings(
            decision_mode,
            camera_source=camera_source,
            window_seconds=window_seconds,
            motion_pixel_threshold=motion_threshold,
            preview=preview,
        )
        heading = live_settings.decision_mode == "heading"
        predictor = None if heading or not decide else self._build_predictor(model_source)
        door_decider = self._build_door_decider(door_side)
        source = CameraFrameSource(live_settings.camera_source)

        def print_decision(decision: LiveDecision) -> None:
            timestamp = time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"
            print(
                f"[{timestamp}] {decision.label} ({decision.confidence:.1%}) "
                f"latency={decision.latency_s:.2f}s"
            )

        def print_motion() -> None:
            timestamp = time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"
            print(f"[{timestamp}] motion")

        session = LiveSession(
            source=source,
            tracking=self._settings.tracking,
            sampling=self._settings.sampling,
            network=network,
            live=live_settings,
            predictor=predictor,
            decision_sink=print_decision,
            motion_sink=print_motion,
            decide=decide,
            door_decider=door_decider,
        )
        if heading:
            print(f"door side: {door_decider.door_side} (entering = subject heads toward that edge)")
        else:
            print("decision: network model (door side ignored)")
        print(f"calibrating... ({live_settings.warmup_seconds:g} s)")
        session.warm_up()
        try:
            session.run()
        except KeyboardInterrupt:
            session.stop()
