import numpy as np

from config import LiveSettings
from preprocessing.frame_window import FrameWindow
from preprocessing.motion_trigger import MotionTrigger


def test_frame_window_is_bounded_and_preserves_latest_frames() -> None:
    window = FrameWindow(2)
    window.append(np.array([1]), np.array([1]))
    window.append(np.array([2]), np.array([2]))
    window.append(np.array([3]), np.array([3]))

    assert window.is_full
    assert [int(frame[0]) for frame in window.raw()] == [2, 3]


def test_motion_trigger_starts_once_and_rearms_after_cooldown() -> None:
    settings = LiveSettings(motion_pixel_threshold=2, cooldown_seconds=2.0)
    trigger = MotionTrigger(settings)
    still = np.zeros((2, 2), dtype=np.uint8)
    motion = np.array([[255, 255], [0, 0]], dtype=np.uint8)

    assert not trigger.update(still, now=0.0)
    assert trigger.update(motion, now=1.0)
    assert not trigger.update(motion, now=1.1)
    trigger.finish(now=1.2)
    assert not trigger.update(motion, now=2.0)
    assert trigger.update(motion, now=3.3)