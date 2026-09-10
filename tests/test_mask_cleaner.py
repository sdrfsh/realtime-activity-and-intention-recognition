import cv2
import numpy as np

from config import TrackingSettings
from preprocessing.mask_cleaner import MaskCleaner


def _mask() -> np.ndarray:
    mask = np.zeros((90, 160), dtype=np.uint8)
    mask[20:60, 40:70] = 255  # a person-sized blob
    mask[35:40, 50:55] = 0  # with a hole inside it
    mask[5, 120] = 255  # an isolated speck
    mask[70:72, 130:132] = 255  # a tiny 2x2 blob
    return mask


def test_cleaner_removes_specks_and_small_blobs_but_keeps_the_subject() -> None:
    cleaned = MaskCleaner(TrackingSettings()).apply(_mask())

    assert cleaned[5, 120] == 0
    assert cleaned[70:72, 130:132].max() == 0
    # interior of the blob stays solid (corners may round off slightly)
    assert cleaned[23:57, 43:67].min() == 255
    assert cleaned[20:60, 40:70].mean() > 240


def test_cleaner_fills_holes_inside_the_subject() -> None:
    cleaned = MaskCleaner(TrackingSettings()).apply(_mask())

    assert cleaned[35:40, 50:55].min() == 255


def test_cleaner_does_not_grow_the_subject_by_default() -> None:
    original = _mask()
    cleaned = MaskCleaner(TrackingSettings()).apply(original)

    # nothing outside the original blob's bounding box becomes foreground
    outside = cleaned.copy()
    outside[20:60, 40:70] = 0
    assert outside.max() == 0


def test_optional_hull_fill_turns_a_crescent_into_a_solid_blob() -> None:
    # what KNN yields for a mostly-still head: only the leading edge moves
    settings = TrackingSettings(
        mask_median_kernel=3, mask_open_iterations=0, mask_close_kernel=9,
        mask_dilate_iterations=3, mask_fill_mode="hull",
    )
    mask = np.zeros((90, 160), dtype=np.uint8)
    cv2.ellipse(mask, (80, 45), (20, 20), 0, 200, 340, 255, thickness=3)
    cleaned = MaskCleaner(settings).apply(mask)

    assert cleaned[45, 80] == 255  # centre of the head is now foreground
    assert np.count_nonzero(cleaned) > 4 * np.count_nonzero(mask)


def test_cleaner_output_is_strictly_binary() -> None:
    mask = _mask()
    mask[80:85, 10:20] = 127  # KNN shadow value
    cleaned = MaskCleaner(TrackingSettings()).apply(mask)

    assert set(np.unique(cleaned)).issubset({0, 255})


def test_cleaner_can_be_disabled_through_settings() -> None:
    settings = TrackingSettings(
        mask_median_kernel=0,
        mask_open_iterations=0,
        mask_close_iterations=0,
        mask_min_component_area=0,
        mask_dilate_iterations=0,
        mask_fill_mode="none",
    )
    mask = _mask()

    cleaned = MaskCleaner(settings).apply(mask)

    assert np.array_equal(cleaned, mask)
