"""Background plate quality assessment.

Measures noise level, compression blocking artifacts, blur severity,
and dynamic range on the background plate. These metrics configure
the ISP 3A estimator and enhancement kernel parameters.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class QualityScore:
    """Image quality metrics for a background plate."""

    noise_level: float
    blocking_severity: float
    blur_level: float
    dynamic_range: float


class QualityAssessor:
    """Measures image quality on the background plate.

    Each metric is a standalone static method for independent testability.
    The assess() method runs all four and returns a frozen QualityScore.
    """

    @staticmethod
    def estimate_noise(plate: np.ndarray) -> float:
        """Robust noise estimation via MAD of the Laplacian (Donoho estimator)."""
        gray = cv2.cvtColor(plate.clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        sigma = float(np.median(np.abs(lap)) * 1.4826)
        return sigma

    @staticmethod
    def estimate_blocking(plate: np.ndarray, block_size: int = 8) -> float:
        """Detect compression block artifacts at 8x8 boundaries."""
        gray = cv2.cvtColor(plate.clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(
            np.float64
        )
        gx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
        gy = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
        grad = gx + gy

        h, w = gray.shape
        boundary = np.zeros((h, w), dtype=bool)
        boundary[:, block_size::block_size] = True
        boundary[block_size::block_size, :] = True

        boundary_energy = float(grad[boundary].mean()) if boundary.any() else 0.0
        interior = ~boundary
        interior_energy = float(grad[interior].mean()) if interior.any() else 1e-6
        return boundary_energy / max(interior_energy, 1e-6)

    @staticmethod
    def estimate_blur(plate: np.ndarray) -> float:
        """Global sharpness via Laplacian variance."""
        gray = cv2.cvtColor(plate.clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        return float(lap.var())

    @staticmethod
    def estimate_dynamic_range(plate: np.ndarray) -> float:
        """Usable luminance range as fraction of [0, 255]."""
        gray = cv2.cvtColor(plate.clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        lo, hi = np.percentile(gray, [1, 99])
        return float((hi - lo) / 255.0)

    def assess(self, plate: np.ndarray) -> QualityScore:
        """Run all four quality metrics on the background plate."""
        return QualityScore(
            noise_level=self.estimate_noise(plate),
            blocking_severity=self.estimate_blocking(plate),
            blur_level=self.estimate_blur(plate),
            dynamic_range=self.estimate_dynamic_range(plate),
        )
