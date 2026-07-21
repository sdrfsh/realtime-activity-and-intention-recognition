"""Runs the clip-preprocessing pipeline across the whole raw dataset, then
augments the resulting motion images (mirroring + random shifts), producing
the final training CSV. Mirrors what ``Data_labeling*``, ``FindContours*``,
``MotionRepresentation*``, ``DataMirroring`` and ``DataRandomCrops`` did
together in the original notebooks, as one coherent, resumable step.
"""
from __future__ import annotations

from pathlib import Path

from augmentation.mirror_augmenter import MirrorAugmenter
from augmentation.shift_crop_augmenter import ShiftCropAugmenter
from data.image_writer import ImageWriter
from data.label_repository import LabelRepository
from data.video_repository import VideoRepository
from domain.entities import LabeledClip
from pipelines.clip_preprocessing_pipeline import ClipPreprocessingPipeline


class DatasetPreparationPipeline:
    """Single job: turn a folder of raw, labeled clips into an augmented,
    ready-to-train image dataset."""

    def __init__(
        self,
        video_repository: VideoRepository,
        source_labels: LabelRepository,
        filtered_labels: LabelRepository,
        training_labels: LabelRepository,
        clip_pipeline: ClipPreprocessingPipeline,
        image_writer: ImageWriter,
        mirror_augmenter: MirrorAugmenter,
        shift_crop_augmenter: ShiftCropAugmenter,
        motion_image_dir: Path,
        augmented_image_dir: Path,
    ) -> None:
        self._video_repository = video_repository
        self._source_labels = source_labels
        self._filtered_labels = filtered_labels
        self._training_labels = training_labels
        self._clip_pipeline = clip_pipeline
        self._image_writer = image_writer
        self._mirror_augmenter = mirror_augmenter
        self._shift_crop_augmenter = shift_crop_augmenter
        self._motion_image_dir = motion_image_dir
        self._augmented_image_dir = augmented_image_dir

    def run(self) -> None:
        clips = self._source_labels.read() or self._discover_unlabeled_clips()

        kept_clips: list[LabeledClip] = []
        training_clips: list[LabeledClip] = []

        for clip in clips:
            motion_image = self._clip_pipeline.process(clip.video_path, clip.label)
            if motion_image is None:
                continue

            kept_clips.append(clip)
            base_name = clip.video_path.stem
            motion_path = self._motion_image_dir / f"{base_name}.jpg"
            self._image_writer.save(motion_image.image, motion_path)
            training_clips.append(LabeledClip(motion_path, clip.label))
            training_clips.extend(
                self._augment(motion_image.image, base_name, clip.label)
            )

        self._filtered_labels.write(kept_clips)
        self._training_labels.write(training_clips)

    def _discover_unlabeled_clips(self) -> list[LabeledClip]:
        return [
            LabeledClip(video_path=video_path, label="unknown")
            for video_path in self._video_repository.list_videos()
        ]

    def _augment(self, image, base_name: str, label: str) -> list[LabeledClip]:
        augmented: list[LabeledClip] = []

        mirrored_path = self._augmented_image_dir / f"mirror_{base_name}.jpg"
        self._image_writer.save(self._mirror_augmenter.augment(image), mirrored_path)
        augmented.append(LabeledClip(mirrored_path, label))

        for direction, shifted_image in self._shift_crop_augmenter.augment_all(image).items():
            shifted_path = self._augmented_image_dir / f"crop_{direction}_{base_name}.jpg"
            self._image_writer.save(shifted_image, shifted_path)
            augmented.append(LabeledClip(shifted_path, label))

        return augmented
