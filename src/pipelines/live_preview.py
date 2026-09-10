"""Draws the ``live`` preview window: the camera frame annotated with the
trigger state and the latest decision on the left, and a column with the
live cleaned mask and the last network input on the right. Pure rendering:
it holds no session state and makes no decisions."""
from __future__ import annotations

import cv2
import numpy as np

from domain.entities import LiveDecision


class LivePreview:
    """Single job: compose one preview image from the session's current state."""

    _ENTERING_COLOR = (0, 220, 0)
    _PASSING_COLOR = (0, 165, 255)
    _DOOR_COLOR = (255, 200, 0)
    _TEXT_COLOR = (255, 255, 255)
    _MUTED_COLOR = (200, 200, 200)

    def __init__(
        self,
        class_names: list[str],
        motion_pixel_threshold: int,
        decision_source: str,
        door_side: str | None = None,
    ) -> None:
        self._class_names = class_names
        self._motion_pixel_threshold = motion_pixel_threshold
        self._decision_source = decision_source
        self._door_side = door_side

    def render(
        self,
        frame: np.ndarray,
        mask: np.ndarray | None,
        network_input: np.ndarray | None,
        decision: LiveDecision | None,
        collecting: tuple[int, int] | None,
    ) -> np.ndarray:
        """``collecting`` is ``(frames so far, window capacity)`` while a
        window is being filled, else ``None``."""
        camera = frame.copy()
        height = camera.shape[0]
        if self._door_side is not None:
            self._draw_door_edge(camera)
        self._draw_status(camera, mask, collecting)
        self._draw_decision(camera, decision)

        panel_height = height // 2
        mask_panel = self._mask_panel(mask, panel_height)
        input_panel = self._network_input_panel(network_input, panel_height)
        column_width = max(mask_panel.shape[1], input_panel.shape[1])
        column = np.vstack((self._pad_width(mask_panel, column_width), self._pad_width(input_panel, column_width)))
        if column.shape[0] != height:  # odd frame heights
            column = self._pad_height(column, height)
        return np.hstack((camera, column))

    # ---------------------------------------------------------------- camera side
    def _draw_door_edge(self, camera: np.ndarray) -> None:
        """Highlight the frame edge the door lies along."""
        height, width = camera.shape[:2]
        thickness = max(4, min(height, width) // 60)
        if self._door_side == "top":
            box, origin = ((0, 0), (width - 1, thickness)), (width // 2 - 30, thickness + 22)
        elif self._door_side == "bottom":
            box, origin = ((0, height - 1 - thickness), (width - 1, height - 1)), (width // 2 - 30, height - thickness - 8)
        elif self._door_side == "left":
            box, origin = ((0, 0), (thickness, height - 1)), (thickness + 8, height // 2)
        else:
            box, origin = ((width - 1 - thickness, 0), (width - 1, height - 1)), (width - thickness - 70, height // 2)
        cv2.rectangle(camera, box[0], box[1], self._DOOR_COLOR, -1)
        self._put_text(camera, "door", origin, self._DOOR_COLOR, 0.6, 2)

    def _draw_status(self, camera: np.ndarray, mask: np.ndarray | None, collecting: tuple[int, int] | None) -> None:
        pixels = 0 if mask is None else int(np.count_nonzero(mask))
        if collecting is not None:
            status, color = f"collecting {collecting[0]}/{collecting[1]}", (0, 200, 255)
        else:
            status, color = "idle", self._MUTED_COLOR
        self._put_text(camera, f"{status}   fg px {pixels} / thr {self._motion_pixel_threshold}", (10, 28), color, 0.6)

    def _draw_decision(self, camera: np.ndarray, decision: LiveDecision | None) -> None:
        height, width = camera.shape[:2]
        if decision is None:
            self._put_text(camera, f"{self._decision_source}: no decision yet", (10, height - 16), self._MUTED_COLOR, 0.6)
            return
        self._put_text(camera, f"{decision.label}  {decision.confidence:.0%}", (10, height - 52), self._label_color(decision.label), 1.0, 2)
        # per-class probability bars along the bottom edge
        probabilities = np.asarray(decision.class_probabilities, dtype=np.float32).ravel()
        bar_left, bar_top, bar_height, gap = 10, height - 36, 10, 4
        bar_width = max(60, min(220, width // 3))
        for index, probability in enumerate(probabilities):
            name = self._class_names[index] if index < len(self._class_names) else f"class {index}"
            y = bar_top + index * (bar_height + gap)
            cv2.rectangle(camera, (bar_left, y), (bar_left + bar_width, y + bar_height), (40, 40, 40), -1)
            fill = int(bar_width * float(np.clip(probability, 0.0, 1.0)))
            cv2.rectangle(camera, (bar_left, y), (bar_left + fill, y + bar_height), self._label_color(name), -1)
            self._put_text(camera, f"{name} {probability:.0%}", (bar_left + bar_width + 8, y + bar_height), self._TEXT_COLOR, 0.45, 1)

    # ---------------------------------------------------------------- right column
    def _mask_panel(self, mask: np.ndarray | None, panel_height: int) -> np.ndarray:
        if mask is None:
            panel = np.zeros((panel_height, panel_height * 16 // 9, 3), dtype=np.uint8)
        else:
            mask_h, mask_w = mask.shape[:2]
            panel_width = max(1, round(panel_height * mask_w / mask_h))
            resized = cv2.resize(mask, (panel_width, panel_height), interpolation=cv2.INTER_NEAREST)
            panel = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        self._put_text(panel, "cleaned mask (live)", (10, 24), self._TEXT_COLOR, 0.55)
        return panel

    def _network_input_panel(self, network_input: np.ndarray | None, panel_height: int) -> np.ndarray:
        if network_input is None:
            panel = np.zeros((panel_height, panel_height, 3), dtype=np.uint8)
            title = "network input (none yet)"
        else:
            image = np.clip(network_input, 0, 255).astype(np.uint8)
            panel = cv2.resize(image, (panel_height, panel_height), interpolation=cv2.INTER_NEAREST)
            title = "network input"
        self._put_text(panel, title, (10, 24), self._TEXT_COLOR, 0.55)
        return panel

    # ---------------------------------------------------------------- helpers
    @classmethod
    def _label_color(cls, label: str) -> tuple[int, int, int]:
        return cls._ENTERING_COLOR if label == "entering" else cls._PASSING_COLOR

    @staticmethod
    def _pad_width(image: np.ndarray, width: int) -> np.ndarray:
        extra = width - image.shape[1]
        return image if extra <= 0 else cv2.copyMakeBorder(image, 0, 0, 0, extra, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    @staticmethod
    def _pad_height(image: np.ndarray, height: int) -> np.ndarray:
        extra = height - image.shape[0]
        if extra < 0:
            return image[:height]
        return image if extra == 0 else cv2.copyMakeBorder(image, 0, extra, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    @staticmethod
    def _put_text(
        image: np.ndarray,
        text: str,
        origin: tuple[int, int],
        color: tuple[int, int, int],
        scale: float = 0.65,
        thickness: int = 1,
    ) -> None:
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
