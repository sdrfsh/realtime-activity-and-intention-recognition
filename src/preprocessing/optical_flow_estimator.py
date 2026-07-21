"""Docx 3.2: "u_i is the outputs of the optical flow function ... and the
approximate movement distance is calculated by considering their average" —
this is the velocity signal the adaptive sampler uses to set its rate."""
from __future__ import annotations

import cv2
import numpy as np

from config import SamplingSettings


class OpticalFlowEstimator:
    """Single job: estimate how fast the subject is moving between two frames."""

    def __init__(self, settings: SamplingSettings) -> None:
        self._settings = settings

    def estimate_velocity(self, frame_a: np.ndarray, frame_b: np.ndarray) -> float:
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            gray_a,
            gray_b,
            None,
            pyr_scale=self._settings.optical_flow_pyr_scale,
            levels=self._settings.optical_flow_levels,
            winsize=self._settings.optical_flow_winsize,
            iterations=self._settings.optical_flow_iterations,
            poly_n=self._settings.optical_flow_poly_n,
            poly_sigma=self._settings.optical_flow_poly_sigma,
            flags=0,
        )
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return float(np.mean(magnitude))
