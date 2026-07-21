"""Generates short synthetic .mp4 clips of the two intentions this system
distinguishes at an automatic door — "entering" (approaching and reaching the
door) vs "passing_by" (walking along the corridor without turning toward the
door) — so the pipeline can be exercised end-to-end without a real camera
feed. Not part of the production package — a test-data helper only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 320, 240
DOOR_ZONE = (260, 80, 300, 160)  # (x1, y1, x2, y2): the automatic door's footprint
SUBJECT_SIZE = 26


def _background(rng: np.random.Generator) -> np.ndarray:
    frame = rng.integers(60, 70, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
    cv2.rectangle(frame, DOOR_ZONE[:2], DOOR_ZONE[2:], (110, 110, 110), 2)
    return frame


def _trajectory(intent: str, num_frames: int) -> list[tuple[int, int]]:
    door_center_x = (DOOR_ZONE[0] + DOOR_ZONE[2]) // 2
    door_center_y = (DOOR_ZONE[1] + DOOR_ZONE[3]) // 2
    shared_start = (20, HEIGHT - 40)

    if intent == "entering":
        # walks diagonally from the same starting point, turning toward and
        # ending inside the door zone
        start = shared_start
        end = (door_center_x, door_center_y)
    elif intent == "passing_by":
        # walks straight across from the same starting point, ignoring the door
        start = shared_start
        end = (WIDTH - 20, shared_start[1])
    else:
        raise ValueError(f"unknown intent: {intent!r} (expected 'entering' or 'passing_by')")

    return [
        (
            int(start[0] + (end[0] - start[0]) * i / (num_frames - 1)),
            int(start[1] + (end[1] - start[1]) * i / (num_frames - 1)),
        )
        for i in range(num_frames)
    ]


def generate_clip(output_path: Path, intent: str, num_frames: int = 90, fps: int = 30) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (WIDTH, HEIGHT))

    rng = np.random.default_rng(seed=42)
    background = _background(rng)
    positions = _trajectory(intent, num_frames)

    for x, y in positions:
        frame = background.copy()
        noise = rng.integers(-4, 4, size=frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        top_left = (x - SUBJECT_SIZE // 2, y - SUBJECT_SIZE // 2)
        bottom_right = (x + SUBJECT_SIZE // 2, y + SUBJECT_SIZE // 2)
        cv2.rectangle(frame, top_left, bottom_right, (230, 230, 230), -1)
        writer.write(frame)

    writer.release()
    print(f"wrote {num_frames} '{intent}' frames to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--intent", choices=("entering", "passing_by"), default="entering")
    args = parser.parse_args()
    generate_clip(args.output_path, args.intent)
