"""Cleans a raw background-subtraction mask into one solid, person-shaped blob.

The legacy pipeline got much of this for free: masks were written to an MP4V
video and read back (lossy compression smears single-pixel specks away), then
eroded and dilated per sample. The live path has no such accidental
denoising, so this stage does it explicitly:

1. median blur — removes isolated specks without shifting edges;
2. binarise — KNN emits 0/127/255 when shadows are on, we want 0/255;
3. morphological opening — strips thin noise strands;
4. morphological closing (large kernel) — merges fragments of one body;
5. connected-component filter — drops blobs smaller than a person could be;
6. dilation — grows the survivors so the blob covers the whole subject,
   not just the edges the background model saw move;
7. fill — "hull" paints each blob's convex hull (a crescent of moving edge
   becomes a solid head), "contour" only fills interior holes.
"""
from __future__ import annotations

import cv2
import numpy as np

from config import TrackingSettings


class MaskCleaner:
    """Single job: turn a noisy foreground mask into a clean, solid binary one."""

    def __init__(self, settings: TrackingSettings) -> None:
        self._settings = settings
        # small kernel for opening so thin limbs survive; larger ones for
        # closing and dilation so fragments merge and the blob covers the body
        self._open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        close = max(3, settings.mask_close_kernel | 1)
        self._close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))
        self._dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def apply(self, mask: np.ndarray) -> np.ndarray:
        s = self._settings
        cleaned = mask
        if s.mask_median_kernel >= 3:
            kernel = s.mask_median_kernel | 1  # medianBlur needs an odd size
            cleaned = cv2.medianBlur(cleaned, kernel)
        _, cleaned = cv2.threshold(cleaned, 0, 255, cv2.THRESH_BINARY)
        if s.mask_open_iterations > 0:
            cleaned = cv2.morphologyEx(
                cleaned, cv2.MORPH_OPEN, self._open_kernel, iterations=s.mask_open_iterations
            )
        if s.mask_close_iterations > 0:
            cleaned = cv2.morphologyEx(
                cleaned, cv2.MORPH_CLOSE, self._close_kernel, iterations=s.mask_close_iterations
            )
        if s.mask_min_component_area > 0:
            cleaned = self._drop_small_components(cleaned, s.mask_min_component_area)
        if s.mask_dilate_iterations > 0:
            cleaned = cv2.dilate(cleaned, self._dilate_kernel, iterations=s.mask_dilate_iterations)
        if s.mask_fill_mode == "hull":
            cleaned = self._fill_contours(cleaned, hull=True)
        elif s.mask_fill_mode == "contour":
            cleaned = self._fill_contours(cleaned, hull=False)
        return cleaned

    @staticmethod
    def _drop_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count <= 1:
            return mask
        keep = stats[:, cv2.CC_STAT_AREA] >= min_area
        keep[0] = False  # background label
        return np.where(keep[labels], 255, 0).astype(np.uint8)

    @staticmethod
    def _fill_contours(mask: np.ndarray, hull: bool) -> np.ndarray:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask
        shapes = [cv2.convexHull(contour) for contour in contours] if hull else list(contours)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, shapes, -1, 255, thickness=cv2.FILLED)
        return filled
