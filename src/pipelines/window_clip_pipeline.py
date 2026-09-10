from __future__ import annotations

from pathlib import Path

import numpy as np

from domain.entities import MotionImage
from preprocessing.adaptive_frame_sampler import AdaptiveFrameSampler
from preprocessing.motion_contour_detector import MotionContourDetector
from preprocessing.motion_image_encoder import MotionImageEncoder
from preprocessing.optical_flow_estimator import OpticalFlowEstimator


class WindowClipPipeline:
    def __init__(
        self,
        contour_detector: MotionContourDetector,
        flow_estimator: OpticalFlowEstimator,
        frame_sampler: AdaptiveFrameSampler,
        motion_encoder: MotionImageEncoder,
    ) -> None:
        self._contour_detector = contour_detector
        self._flow_estimator = flow_estimator
        self._frame_sampler = frame_sampler
        self._motion_encoder = motion_encoder

    def process(
        self,
        raw_frames: list[np.ndarray],
        mask_frames: list[np.ndarray],
        source_clip: Path | None = None,
        label: str | None = None,
    ) -> MotionImage | None:
        if len(raw_frames) < 2 or not self._contour_detector.has_motion(mask_frames):
            return None
        velocity = self._flow_estimator.estimate_velocity(raw_frames[0], raw_frames[len(raw_frames) // 2])
        sampling_rate = self._frame_sampler.compute_sampling_rate(len(raw_frames), velocity)
        indices = self._frame_sampler.select_frame_indices(len(raw_frames), sampling_rate)
        if not indices:
            return None
        return MotionImage(
            image=self._motion_encoder.encode([mask_frames[i] for i in indices]),
            source_clip=source_clip or Path("<live>"),
            label=label,
        )