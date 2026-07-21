"""Discovers raw video files on disk. No labeling, no CSV concerns."""
from __future__ import annotations

from pathlib import Path


class VideoRepository:
    """Single job: list the video files that live under a directory."""

    def __init__(self, directory: Path, extension: str = ".mp4") -> None:
        self._directory = directory
        self._extension = extension

    def list_videos(self) -> list[Path]:
        if not self._directory.exists():
            return []
        return sorted(self._directory.glob(f"*{self._extension}"))
