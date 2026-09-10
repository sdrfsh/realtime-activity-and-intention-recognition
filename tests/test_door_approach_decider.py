import numpy as np
import pytest

from config import SceneSettings
from preprocessing.door_approach_decider import DoorApproachDecider

H, W = 90, 160


def _masks(path: list[tuple[int, int]], size: int = 10) -> list[np.ndarray]:
    """One mask per (x, y) blob centre along the path."""
    masks = []
    for x, y in path:
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[max(0, y - size) : y + size, max(0, x - size) : x + size] = 255
        masks.append(mask)
    return masks


def _line(start: tuple[int, int], end: tuple[int, int], steps: int = 12) -> list[tuple[int, int]]:
    return [
        (round(start[0] + (end[0] - start[0]) * i / (steps - 1)), round(start[1] + (end[1] - start[1]) * i / (steps - 1)))
        for i in range(steps)
    ]


@pytest.mark.parametrize(
    "door_side, path",
    [
        ("top", _line((80, 75), (80, 15))),  # walks up the frame
        ("bottom", _line((80, 15), (80, 75))),
        ("left", _line((140, 45), (20, 45))),
        ("right", _line((20, 45), (140, 45))),
        ("top", _line((20, 75), (100, 20))),  # diagonal but mostly upward
    ],
)
def test_heading_at_the_door_edge_is_entering(door_side: str, path) -> None:
    decision = DoorApproachDecider(SceneSettings(door_side=door_side)).decide(_masks(path))
    assert decision is not None
    assert decision.label == "entering"
    assert decision.entering_score > 0.5


@pytest.mark.parametrize(
    "door_side, path",
    [
        ("top", _line((20, 45), (140, 45))),  # walks across, parallel to the door edge
        ("top", _line((80, 15), (80, 75))),  # walks away from the door
        ("left", _line((20, 45), (140, 45))),  # walks away from a left door
        ("right", _line((80, 15), (80, 75))),  # vertical walk, door on the right
    ],
)
def test_other_headings_are_passing_by(door_side: str, path) -> None:
    decision = DoorApproachDecider(SceneSettings(door_side=door_side)).decide(_masks(path))
    assert decision is not None
    assert decision.label == "passing_by"


def test_standing_still_is_passing_by() -> None:
    decision = DoorApproachDecider(SceneSettings(door_side="top")).decide(_masks([(80, 45)] * 12))
    assert decision is not None
    assert decision.label == "passing_by"
    assert decision.toward_door == pytest.approx(0.0)


def test_too_few_visible_frames_gives_no_decision() -> None:
    decider = DoorApproachDecider(SceneSettings(door_side="top"))
    empty = [np.zeros((H, W), dtype=np.uint8)] * 12
    assert decider.decide(empty) is None
    assert decider.decide(_masks([(80, 60), (80, 40)])) is None


def test_empty_masks_are_skipped_not_counted() -> None:
    path = _line((80, 75), (80, 15))
    masks = _masks(path)
    masks.insert(3, np.zeros((H, W), dtype=np.uint8))
    decision = DoorApproachDecider(SceneSettings(door_side="top")).decide(masks)
    assert decision is not None
    assert decision.label == "entering"
    assert len(decision.trajectory) == len(path)


def test_unsettled_all_foreground_masks_are_ignored() -> None:
    path = _line((20, 45), (140, 45))  # walks across, parallel to a top door
    masks = [np.full((H, W), 255, dtype=np.uint8)] * 4 + _masks(path)
    decision = DoorApproachDecider(SceneSettings(door_side="top")).decide(masks)
    assert decision is not None
    assert decision.label == "passing_by"
    assert len(decision.trajectory) == len(path)


def test_trajectory_is_normalised_to_unit_square() -> None:
    points = DoorApproachDecider(SceneSettings()).trajectory(_masks([(0, 0), (W - 1, H - 1)], size=1))
    assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in points)


def test_invalid_side_is_rejected_by_settings() -> None:
    with pytest.raises(ValueError):
        SceneSettings(door_side="front")
