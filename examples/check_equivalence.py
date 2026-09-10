"""Gate A: compare the extracted window pipeline with the legacy steps."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from app import Application
from data.video_reader import VideoReader
from preprocessing.background_subtractor import BackgroundSubtractor
from preprocessing.mask_cleaner import MaskCleaner
from preprocessing.noise_reducer import NoiseReducer


def legacy_motion_image(app: Application, clip: Path):
    pipeline = app._build_clip_pipeline()
    raw_frames = list(VideoReader().frames(clip))
    subtractor = BackgroundSubtractor(app._settings.tracking)
    noise_reducer = NoiseReducer(app._settings.tracking)
    cleaner = MaskCleaner(app._settings.tracking)
    masks = [cleaner.apply(subtractor.apply(noise_reducer.apply(frame))) for frame in raw_frames]
    return pipeline._window_pipeline.process(raw_frames, masks, clip)


def main() -> None:
    app = Application()
    for clip in sorted(app._settings.paths.raw_video_dir.glob("*.mp4")):
        expected = legacy_motion_image(app, clip)
        actual = app._build_clip_pipeline().process(clip)
        matches = expected is None and actual is None
        if expected is not None and actual is not None:
            matches = np.array_equal(expected.image, actual.image)
        print(f"{clip.name}: {'MATCH' if matches else 'MISMATCH'}")


if __name__ == "__main__":
    main()
