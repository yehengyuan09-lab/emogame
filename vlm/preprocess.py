"""Image preprocessing for the VLM pipeline.

Implements the 5-step preprocessing from ``docs/02-vlm-pipeline.md``:
1. Format standardization (JPEG/PNG → RGB array, resize to target)
2. Multi-scale crops (full / center / face-region)
3. Color histogram extraction (HSV quantisation → top-5 dominant colors)
4. Canny edge density (proxy for effect complexity)
5. Laplacian variance (sharpness / blur detection)
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image

from vlm.config import get_settings
from vlm.schemas import PreprocessResult


class MultiScaleCrops(NamedTuple):
    """Three crops of the input image at different scales."""

    full: np.ndarray  # (H, W, 3) — full image
    center: np.ndarray  # (H//2, W//2, 3) — central 50 %
    face_region: np.ndarray | None  # upper-third crop (presumed face area)


class Preprocessor:
    """Image preprocessing for the VLM pipeline.

    Provides both the standardised image (for VLM inference) and
    traditional-CV features (for degradation fallback).
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    # ── loading ──

    def load_image(self, path: Path) -> np.ndarray:
        """Load a JPEG/PNG and return an RGB ``uint8`` array ``(H, W, 3)``."""
        img = Image.open(path).convert("RGB")
        return np.array(img)

    # ── step 1: standardisation ──

    def standardize(self, image: np.ndarray) -> np.ndarray:
        """Resize to the target resolution, preserving aspect ratio.

        Black letterbox bars are added when the aspect ratio differs.
        """
        target_w, target_h = self.settings.target_resolution
        h, w = image.shape[:2]

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        return canvas

    # ── step 2: multi-scale crops ──

    def multi_scale_crops(self, image: np.ndarray) -> MultiScaleCrops:
        """Generate full, center, and face-region crops."""
        h, w = image.shape[:2]

        # Center crop — central 50 % of the image
        ch, cw = int(h * 0.5), int(w * 0.5)
        cy, cx = (h - ch) // 2, (w - cw) // 2
        center = image[cy : cy + ch, cx : cx + cw].copy()

        # Face region — upper third, center 40 % width
        fh = int(h * 0.33)
        fw = int(w * 0.4)
        fy = int(h * 0.05)
        fx = (w - fw) // 2
        face = image[fy : fy + fh, fx : fx + fw].copy() if fh > 0 and fw > 0 else None

        return MultiScaleCrops(full=image, center=center, face_region=face)

    # ── step 3: color histogram ──

    @staticmethod
    def extract_color_histogram(image: np.ndarray, top_k: int = 5) -> list[str]:
        """HSV quantisation → top-K dominant colours as ``#RRGGBB`` hex strings."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        h_bins, s_bins, v_bins = 12, 5, 5
        hist = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            [h_bins, s_bins, v_bins],
            [0, 180, 0, 256, 0, 256],
        )

        flat = hist.flatten()
        top_indices = np.argsort(flat)[-top_k:][::-1]

        colors: list[str] = []
        for idx in top_indices:
            h_idx = idx // (s_bins * v_bins)
            s_idx = (idx // v_bins) % s_bins
            v_idx = idx % v_bins

            h_val = int(h_idx * 180 / h_bins)
            s_val = int(s_idx * 255 / s_bins)
            v_val = int(v_idx * 255 / v_bins)

            rgb = cv2.cvtColor(
                np.uint8([[[h_val, s_val, v_val]]]), cv2.COLOR_HSV2RGB
            )[0][0]
            colors.append(f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}")

        return colors

    # ── step 4: edge density ──

    @staticmethod
    def canny_edge_density(image: np.ndarray) -> float:
        """Canny edge detection → proportion of edge pixels.

        Proxy for visual-effect complexity.  Higher values mean more edges /
        particles / detail.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return float(np.count_nonzero(edges) / edges.size)

    # ── step 5: sharpness ──

    @staticmethod
    def laplacian_variance(image: np.ndarray) -> float:
        """Laplacian variance — sharpness / blur metric.  Higher = sharper."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # ── full pipeline ──

    def process(self, image_path: Path) -> tuple[np.ndarray, PreprocessResult]:
        """Run the full preprocessing pipeline.

        Returns
        -------
        standardized : np.ndarray
            The resized RGB image, ready for VLM inference.
        result : PreprocessResult
            Traditional-CV features (edge density, sharpness, colors).
        """
        raw = self.load_image(image_path)
        std = self.standardize(raw)

        edge_density = self.canny_edge_density(std)
        sharpness = self.laplacian_variance(std)
        colors = self.extract_color_histogram(std)

        return std, PreprocessResult(
            edge_density=edge_density,
            sharpness=sharpness,
            dominant_colors_cv=colors,
        )
