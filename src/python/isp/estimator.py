"""ISP 3A parameter estimation from the background plate.

Derives per-camera correction parameters:
  Auto exposure    — gamma LUT for exposure normalization
  Auto white balance — per-channel gains via gray-world
  Auto focus       — adaptive unsharp mask guided by per-region blur map

Results can be cached in camera_state/ for fast restart.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class ISPParams:
    """ISP correction parameters derived from the background plate.

    Not frozen because ndarray fields are mutable, but treated as
    immutable after creation. Use save/load for persistence.
    """

    auto_exposure_lut: np.ndarray
    auto_white_balance_gains: np.ndarray
    blur_map: np.ndarray

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ISPParams):
            return NotImplemented
        return (
            np.array_equal(self.auto_exposure_lut, other.auto_exposure_lut)
            and np.allclose(self.auto_white_balance_gains, other.auto_white_balance_gains)
            and np.allclose(self.blur_map, other.blur_map)
        )


# ---------------------------------------------------------------------------
# Tuning constants — empirical
#
# AUTO_FOCUS_ALPHA_MAX  1.5    Max unsharp mask strength. Higher = more sharpening
#                              in blurry regions. Empirical; >2.0 introduces halos.
# AUTO_FOCUS_KSIZE      5      Gaussian kernel for detail extraction. Must be odd.
#                              5 balances noise suppression vs detail preservation.
# target_mean         128.0    AE target: mid-gray in 8-bit. Assumes scene should
#                              average to 50% brightness. Wrong for night/high-key.
# grid_size             8      Spatial grid for per-region blur estimation (8x8 tiles).
# gamma clamp     [0.2, 5.0]   Prevents extreme AE correction. 0.2 = max darken,
#                              5.0 = max brighten. Arbitrary safety bounds.
# AWB gain clamp  [0.5, 2.0]   Prevents color shifts > 2x per channel. Gray-world
#                              can produce wild gains on non-neutral scenes.
# mean tolerance        5.0    Skip AE if mean luminance within ±5 of target.
#                              Avoids unnecessary correction on well-exposed scenes.
# ---------------------------------------------------------------------------
AUTO_FOCUS_ALPHA_MAX = 1.5
AUTO_FOCUS_KSIZE = 5


class ISPEstimator:
    """Derives ISP 3A correction parameters from the background plate."""

    def __init__(self, target_mean: float = 128.0, grid_size: int = 8) -> None:
        self.target_mean = target_mean
        self.grid_size = grid_size

    @staticmethod
    def compute_auto_exposure_lut(plate: np.ndarray, target_mean: float = 128.0) -> np.ndarray:
        """Compute auto exposure gamma correction LUT.

        Tone mapping via power-law (gamma) curve. Solves for gamma that maps
        the current mean luminance to the target:
            output = (input / 255) ^ gamma * 255
            gamma  = log(target/255) / log(mean/255)

        Gamma < 1 brightens dark images, gamma > 1 darkens bright images.
        The 256-entry LUT is applied to the Y channel in YCrCb space so
        chrominance is preserved (applying gamma to BGR directly shifts hue).
        """
        gray = cv2.cvtColor(plate.clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        current_mean = float(gray.mean())

        # Already near target — return identity LUT
        if abs(current_mean - target_mean) < 5.0:
            return np.arange(256, dtype=np.uint8)

        # Solve for gamma, clamped to [0.2, 5.0] to prevent extreme curves
        gamma = np.log(target_mean / 255.0) / np.log(max(current_mean, 1.0) / 255.0)
        gamma = float(np.clip(gamma, 0.2, 5.0))

        # Build monotonic LUT with fixed endpoints (0 → 0, 255 → 255)
        indices = np.arange(256, dtype=np.float64)
        lut = ((indices / 255.0) ** gamma * 255.0).clip(0, 255).astype(np.uint8)
        lut[0] = 0
        lut[255] = 255
        return lut

    @staticmethod
    def compute_auto_white_balance_gains(plate: np.ndarray) -> np.ndarray:
        """Compute color correction gains to neutralize lighting color cast.

        Uses the gray-world assumption: the average color of a typical outdoor
        scene should be neutral gray. If it isn't, the light source has a color
        cast (e.g. sodium vapor → orange, LED → blue, sunset → warm).

        Per-channel gains normalize each channel relative to green:
            gain_blue  = mean_green / mean_blue   (boost if blue is weak)
            gain_green = 1.0                       (reference channel)
            gain_red   = mean_green / mean_red    (reduce if red is strong)

        Gains are clamped to [0.5, 2.0] to prevent extreme corrections from
        near-zero channel means (e.g. monochromatic night scenes).
        """
        mean_b = float(plate[:, :, 0].mean())
        mean_g = float(plate[:, :, 1].mean())
        mean_r = float(plate[:, :, 2].mean())

        # Normalize to green channel; clamp to avoid runaway gains
        gain_b = np.clip(mean_g / max(mean_b, 1e-6), 0.5, 2.0)
        gain_r = np.clip(mean_g / max(mean_r, 1e-6), 0.5, 2.0)

        return np.array([gain_b, 1.0, gain_r], dtype=np.float32)

    @staticmethod
    def compute_blur_map(plate: np.ndarray, grid_size: int = 8) -> np.ndarray:
        """Compute per-region blur severity map (Laplacian variance per cell)."""
        gray = cv2.cvtColor(plate.clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        row_edges = np.linspace(0, h, grid_size + 1).astype(int)
        col_edges = np.linspace(0, w, grid_size + 1).astype(int)

        blur_map = np.zeros((grid_size, grid_size), dtype=np.float32)
        for r in range(grid_size):
            for c in range(grid_size):
                cell = gray[row_edges[r] : row_edges[r + 1], col_edges[c] : col_edges[c + 1]]
                if cell.size == 0:
                    continue
                lap = cv2.Laplacian(cell, cv2.CV_64F, ksize=3)
                blur_map[r, c] = float(lap.var())

        return blur_map

    def estimate(self, plate: np.ndarray) -> ISPParams:
        """Run all three ISP 3A estimations on the background plate."""
        return ISPParams(
            auto_exposure_lut=self.compute_auto_exposure_lut(plate, self.target_mean),
            auto_white_balance_gains=self.compute_auto_white_balance_gains(plate),
            blur_map=self.compute_blur_map(plate, self.grid_size),
        )

    @staticmethod
    def apply_auto_exposure(frame: np.ndarray, lut: np.ndarray) -> np.ndarray:
        """Apply auto exposure LUT to Y channel only (preserves chrominance)."""
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.LUT(ycrcb[:, :, 0], lut)
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    @staticmethod
    def apply_auto_white_balance(frame: np.ndarray, gains: np.ndarray) -> np.ndarray:
        """Apply color correction by scaling each BGR channel independently.

        Each pixel's channels are multiplied by the corresponding gain,
        then clipped to [0, 255]. This shifts the image toward neutral
        color, removing the tint from the ambient light source.
        """
        corrected = frame.astype(np.float32) * gains[np.newaxis, np.newaxis, :]
        return corrected.clip(0, 255).astype(np.uint8)

    @staticmethod
    def apply_auto_focus(frame: np.ndarray, blur_map: np.ndarray) -> np.ndarray:
        """Apply adaptive sharpening guided by the blur map.

        Uses a spatially-varying unsharp mask. The standard unsharp mask is:
            sharpened = original + alpha * (original - GaussianBlur(original))

        Here alpha varies per-region based on the blur map:
            - Low Laplacian variance (blurry region)  → high alpha (more sharpening)
            - High Laplacian variance (sharp region)   → low alpha (near zero)

        The blur map is normalized relative to its max (scene-adaptive), inverted,
        and scaled to [0, AUTO_FOCUS_ALPHA_MAX]. The 8x8 grid is bilinearly
        interpolated to full frame resolution for smooth transitions between regions.
        """
        bmax = float(blur_map.max())
        if bmax < 1e-6:
            return frame

        # Normalize to [0, 1] relative to sharpest region, then invert:
        # sharpest cell → alpha=0 (no sharpening), blurriest → alpha=ALPHA_MAX
        normalized = blur_map / bmax
        alpha_grid = ((1.0 - normalized) * AUTO_FOCUS_ALPHA_MAX).astype(np.float32)

        # Bilinear upscale to frame resolution (smooth transitions, no block edges)
        h, w = frame.shape[:2]
        alpha_map = cv2.resize(alpha_grid, (w, h), interpolation=cv2.INTER_LINEAR)

        # Unsharp mask: detail = high-frequency content the blur removed
        blurred = cv2.GaussianBlur(frame, (AUTO_FOCUS_KSIZE, AUTO_FOCUS_KSIZE), 0)
        detail = frame.astype(np.float32) - blurred.astype(np.float32)

        # Add back detail scaled by per-pixel alpha
        sharpened = frame.astype(np.float32) + alpha_map[:, :, np.newaxis] * detail
        return sharpened.clip(0, 255).astype(np.uint8)

    @staticmethod
    def apply(frame: np.ndarray, params: ISPParams) -> np.ndarray:
        """Apply full ISP correction (auto exposure → auto white balance → auto focus)."""
        frame = ISPEstimator.apply_auto_exposure(frame, params.auto_exposure_lut)
        frame = ISPEstimator.apply_auto_white_balance(frame, params.auto_white_balance_gains)
        frame = ISPEstimator.apply_auto_focus(frame, params.blur_map)
        return frame

    @staticmethod
    def apply_simd(frame: np.ndarray, params: ISPParams, simd_module: object) -> np.ndarray:
        """SIMD-accelerated ISP display path (fused AE+AWB + AF blend).

        AE applies gamma LUT directly to BGR (not YCrCb), which is a
        slight simplification vs the Python path but negligible for
        traffic camera footage.
        """
        gains = params.auto_white_balance_gains
        corrected = simd_module.apply_ae_awb(
            frame,
            params.auto_exposure_lut,
            float(gains[0]),
            float(gains[1]),
            float(gains[2]),
        )

        # AF: compute alpha map from blur map, GaussianBlur via OpenCV, blend via SIMD
        bmax = float(params.blur_map.max())
        if bmax < 1e-6:
            return corrected

        normalized = params.blur_map / bmax
        alpha_grid = ((1.0 - normalized) * AUTO_FOCUS_ALPHA_MAX).astype(np.float32)
        h, w = frame.shape[:2]
        alpha_map = cv2.resize(alpha_grid, (w, h), interpolation=cv2.INTER_LINEAR)

        blurred = cv2.GaussianBlur(corrected, (AUTO_FOCUS_KSIZE, AUTO_FOCUS_KSIZE), 0)
        return simd_module.apply_af_blend(corrected, blurred, alpha_map)


def url_hash(url: str) -> str:
    """Compute a stable hash for a camera URL."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def save_isp_params(params: ISPParams, directory: Path) -> None:
    """Save ISP params to a directory."""
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "auto_exposure_lut.npy", params.auto_exposure_lut)
    np.save(directory / "auto_white_balance_gains.npy", params.auto_white_balance_gains)
    np.save(directory / "blur_map.npy", params.blur_map)


def load_isp_params(directory: Path) -> ISPParams | None:
    """Load ISP params from a directory. Returns None if files missing."""
    exposure_path = directory / "auto_exposure_lut.npy"
    balance_path = directory / "auto_white_balance_gains.npy"
    blur_path = directory / "blur_map.npy"
    if not (exposure_path.exists() and balance_path.exists() and blur_path.exists()):
        return None
    return ISPParams(
        auto_exposure_lut=np.load(exposure_path),
        auto_white_balance_gains=np.load(balance_path),
        blur_map=np.load(blur_path),
    )
