"""Decides "entering" vs "passing_by" from where the moving subject is
heading, relative to the configured door edge.

Purely 2-D: the door is one of the four frame edges
(``SceneSettings.door_side``) and depth is ignored. For every mask in a
window we take the centroid of the foreground pixels; the displacement from
the first few centroids to the last few is the subject's heading. If that
heading points at the door edge (within ``approach_max_angle_deg`` of the
edge normal) and covers at least ``approach_min_travel`` of the frame, the
subject is entering; any other movement is passing by.

Plain numpy so it is unit-testable without OpenCV.
"""
from __future__ import annotations

import math

import numpy as np

from config import SceneSettings
from domain.entities import DoorApproach


class DoorApproachDecider:
    """Single job: turn a window of foreground masks into a door-approach decision."""

    CLASSES: tuple[str, str] = ("passing_by", "entering")

    # unit vector (dx, dy) pointing at each edge, in image coordinates (y grows downward)
    _TOWARD_DOOR: dict[str, tuple[float, float]] = {
        "right": (1.0, 0.0),
        "left": (-1.0, 0.0),
        "top": (0.0, -1.0),
        "bottom": (0.0, 1.0),
    }

    def __init__(self, settings: SceneSettings) -> None:
        self._settings = settings
        self._direction = self._TOWARD_DOOR[settings.door_side]
        self._min_cosine = math.cos(math.radians(settings.approach_max_angle_deg))

    @property
    def door_side(self) -> str:
        return self._settings.door_side

    def trajectory(self, masks: list[np.ndarray]) -> list[tuple[float, float]]:
        """Foreground centroids per mask, normalised to 0..1 of the frame
        width/height. Masks with too little foreground (nobody there) or too
        much (unsettled background model, exposure jump) are skipped."""
        points: list[tuple[float, float]] = []
        for mask in masks:
            ys, xs = np.nonzero(mask)
            height, width = mask.shape[:2]
            if len(xs) < self._settings.min_foreground_pixels:
                continue
            if len(xs) > self._settings.max_foreground_fraction * height * width:
                continue
            points.append((float(xs.mean()) / max(width - 1, 1), float(ys.mean()) / max(height - 1, 1)))
        return points

    def decide(self, masks: list[np.ndarray]) -> DoorApproach | None:
        """``None`` when the subject was visible in too few frames to tell."""
        points = self.trajectory(masks)
        smoothing = max(1, self._settings.trajectory_smoothing_frames)
        if len(points) < 2 * smoothing:
            return None
        start = tuple(np.mean(points[:smoothing], axis=0))
        end = tuple(np.mean(points[-smoothing:], axis=0))
        dx, dy = end[0] - start[0], end[1] - start[1]
        travel = math.hypot(dx, dy)
        toward = dx * self._direction[0] + dy * self._direction[1]
        across = math.hypot(dx - toward * self._direction[0], dy - toward * self._direction[1])
        cosine = toward / travel if travel > 0 else 0.0
        moved_enough = toward >= self._settings.approach_min_travel
        entering = moved_enough and cosine >= self._min_cosine
        # 0.5 = heading exactly along the door edge, 1.0 = straight at it
        score = 0.5 + 0.5 * cosine if moved_enough else 0.5 * max(cosine, 0.0)
        score = float(np.clip(score, 0.0, 1.0))
        return DoorApproach(
            label=self.CLASSES[1] if entering else self.CLASSES[0],
            entering_score=score,
            start=(float(start[0]), float(start[1])),
            end=(float(end[0]), float(end[1])),
            toward_door=float(toward),
            across_door=float(across),
            trajectory=points,
        )
