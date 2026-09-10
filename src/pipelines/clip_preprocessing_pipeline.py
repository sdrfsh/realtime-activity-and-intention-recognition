"""Orchestrates docx 3.1 + 3.2 for a single clip: tracking (noise removal,
background subtraction, contour check) followed by adaptive, recency-weighted
sampling into one motion image. Each step is delegated to its own class —
this pipeline only sequences them.
"""
from __future__ import annotations

from pathlib import Path

from data.video_reader import VideoReader
from domain.entities import MotionImage
from preprocessing.adaptive_frame_sampler import AdaptiveFrameSampler
from preprocessing.background_subtractor import BackgroundSubtractor
from preprocessing.mask_cleaner import MaskCleaner
from preprocessing.motion_contour_detector import MotionContourDetector
from preprocessing.motion_image_encoder import MotionImageEncoder
from preprocessing.noise_reducer import NoiseReducer
from preprocessing.optical_flow_estimator import OpticalFlowEstimator
from pipelines.window_clip_pipeline import WindowClipPipeline


class ClipPreprocessingPipeline:
    """Single job: turn one raw video clip into one motion image, or ``None``
    if the clip has no detectable motion (discarded, as in the original
    contour-filtering step)."""

    def __init__(
        self,
        video_reader: VideoReader,
        noise_reducer: NoiseReducer,
        background_subtractor: BackgroundSubtractor,
        contour_detector: MotionContourDetector,
        flow_estimator: OpticalFlowEstimator,
        frame_sampler: AdaptiveFrameSampler,
        motion_encoder: MotionImageEncoder,
        mask_cleaner: MaskCleaner | None = None,
    ) -> None:
        self._video_reader = video_reader
        self._noise_reducer = noise_reducer
        self._background_subtractor = background_subtractor
        self._mask_cleaner = mask_cleaner
        self._contour_detector = contour_detector
        self._flow_estimator = flow_estimator
        self._frame_sampler = frame_sampler
        self._motion_encoder = motion_encoder
        self._window_pipeline = WindowClipPipeline(
            contour_detector,
            flow_estimator,
            frame_sampler,
            motion_encoder,
        )

    def process(self, video_path: Path, label: str | None = None) -> MotionImage | None:
        raw_frames = list(self._video_reader.frames(video_path))
        if len(raw_frames) < 2:
            return None

        self._background_subtractor.reset()
        clean_frames = [
            self._background_subtractor.apply(self._noise_reducer.apply(frame))
            for frame in raw_frames
        ]
        if self._mask_cleaner is not None:
            clean_frames = [self._mask_cleaner.apply(mask) for mask in clean_frames]

        return self._window_pipeline.process(raw_frames, clean_frames, video_path, label)
