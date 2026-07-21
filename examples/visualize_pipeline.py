"""Runs the real preprocessing classes on one clip and saves a contact sheet
showing: a raw frame, the background-subtracted mask, and the final motion
image the network would actually classify. A debugging/demo aid only — it
does not duplicate any pipeline logic, it just calls the production classes
and captures their intermediate output.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np

from config import load_settings
from data.video_reader import VideoReader
from preprocessing.adaptive_frame_sampler import AdaptiveFrameSampler
from preprocessing.background_subtractor import BackgroundSubtractor
from preprocessing.motion_contour_detector import MotionContourDetector
from preprocessing.motion_image_encoder import MotionImageEncoder
from preprocessing.noise_reducer import NoiseReducer
from preprocessing.optical_flow_estimator import OpticalFlowEstimator


def _label(image: np.ndarray, text: str) -> np.ndarray:
    canvas = cv2.copyMakeBorder(image, 24, 4, 4, 4, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    cv2.putText(canvas, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _to_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image


def visualize(video_path: Path, output_path: Path) -> None:
    settings = load_settings()
    tracking, sampling = settings.tracking, settings.sampling

    video_reader = VideoReader()
    noise_reducer = NoiseReducer(tracking)
    background_subtractor = BackgroundSubtractor(tracking)
    contour_detector = MotionContourDetector(tracking)
    flow_estimator = OpticalFlowEstimator(sampling)
    frame_sampler = AdaptiveFrameSampler(sampling)
    motion_encoder = MotionImageEncoder(sampling)

    raw_frames = list(video_reader.frames(video_path))
    frame_count = len(raw_frames)
    print(f"read {frame_count} raw frames from {video_path}")

    background_subtractor.reset()
    clean_frames = [
        background_subtractor.apply(noise_reducer.apply(frame)) for frame in raw_frames
    ]

    has_motion = contour_detector.has_motion(clean_frames)
    print(f"motion detected: {has_motion}")

    velocity = flow_estimator.estimate_velocity(raw_frames[0], raw_frames[frame_count // 2])
    sampling_rate = frame_sampler.compute_sampling_rate(frame_count, velocity)
    indices = frame_sampler.select_frame_indices(frame_count, sampling_rate)
    print(f"estimated velocity: {velocity:.4f}")
    print(f"sampling rate (interval): {sampling_rate}")
    print(f"sampled frame indices ({len(indices)}): {indices}")

    sampled = [clean_frames[i] for i in indices]
    motion_image = motion_encoder.encode(sampled)

    mid_index = frame_count // 2
    raw_panel = _label(cv2.resize(raw_frames[mid_index], tracking.frame_size), "raw frame")
    bg_panel = _label(cv2.resize(_to_bgr(clean_frames[mid_index]), tracking.frame_size), "background subtracted")
    motion_panel = _label(cv2.resize(_to_bgr(motion_image), tracking.frame_size), "motion image (final)")

    contact_sheet = np.hstack([raw_panel, bg_panel, motion_panel])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), contact_sheet)
    print(f"wrote contact sheet to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    visualize(args.video_path, args.output_path)
