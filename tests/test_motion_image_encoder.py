import numpy as np
import pytest

from config import SamplingSettings
from preprocessing.motion_image_encoder import MotionImageEncoder


def _encoder(decay_weight: float = 0.7) -> MotionImageEncoder:
    return MotionImageEncoder(SamplingSettings(decay_weight=decay_weight))


def test_encode_rejects_empty_sequence():
    with pytest.raises(ValueError):
        _encoder().encode([])


def test_single_frame_passes_through_unchanged():
    frame = np.full((4, 4), 100, dtype=np.uint8)
    result = _encoder().encode([frame])
    np.testing.assert_array_equal(result, frame)


def test_more_recent_frame_dominates_the_result():
    old_frame = np.full((4, 4), 255, dtype=np.uint8)
    new_frame = np.full((4, 4), 10, dtype=np.uint8)
    result = _encoder(decay_weight=0.7).encode([old_frame, new_frame])

    expected = np.clip(0.7 * old_frame.astype(np.float64) + new_frame, 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(result, expected)
    assert result.mean() < old_frame.mean()


def test_output_never_exceeds_uint8_range():
    bright_frames = [np.full((2, 2), 200, dtype=np.uint8) for _ in range(5)]
    result = _encoder(decay_weight=0.9).encode(bright_frames)
    assert result.max() <= 255
    assert result.dtype == np.uint8
