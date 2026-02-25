"""Tests for ISP quality assessment (noise, blocking, blur, dynamic range)."""

import cv2
import numpy as np
import pytest

from src.python.isp.quality import QualityAssessor, QualityScore

# --- Helpers ---


def make_clean_plate(h=240, w=320, value=128):
    """Uniform plate — low noise, no blocking, no edges."""
    return np.full((h, w, 3), value, dtype=np.float32)


def make_noisy_plate(h=240, w=320, sigma=20.0, seed=42):
    """Plate with additive Gaussian noise."""
    rng = np.random.RandomState(seed)
    base = np.full((h, w, 3), 128.0, dtype=np.float32)
    noise = rng.randn(h, w, 3).astype(np.float32) * sigma
    return (base + noise).clip(0, 255)


def make_blocky_plate(h=240, w=320, block_size=8):
    """Plate with artificial 8x8 block artifacts."""
    rng = np.random.RandomState(42)
    plate = np.zeros((h, w, 3), dtype=np.float32)
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            val = rng.uniform(60, 200)
            plate[y : y + block_size, x : x + block_size] = val
    return plate


def make_sharp_plate(h=240, w=320, seed=42):
    """Checkerboard plate with strong edges (high Laplacian variance)."""
    plate = np.zeros((h, w, 3), dtype=np.float32)
    for y in range(0, h, 16):
        for x in range(0, w, 16):
            val = 200.0 if ((y // 16 + x // 16) % 2 == 0) else 50.0
            plate[y : y + 16, x : x + 16] = val
    return plate


def make_blurry_plate(h=240, w=320):
    """Heavily blurred gradient (low Laplacian variance)."""
    grad = np.linspace(80, 180, w, dtype=np.float32)
    plate = np.tile(grad, (h, 1))[:, :, np.newaxis].repeat(3, axis=2)
    plate = cv2.GaussianBlur(plate.clip(0, 255).astype(np.uint8), (31, 31), 10.0).astype(np.float32)
    return plate


# --- Tests ---


class TestQualityScoreDataclass:
    def test_frozen(self):
        score = QualityScore(
            noise_level=5.0,
            blocking_severity=1.1,
            blur_level=500.0,
            dynamic_range=0.8,
        )
        with pytest.raises(AttributeError):
            score.noise_level = 10.0

    def test_fields(self):
        score = QualityScore(
            noise_level=5.0,
            blocking_severity=1.1,
            blur_level=500.0,
            dynamic_range=0.8,
        )
        assert score.noise_level == pytest.approx(5.0)
        assert score.blocking_severity == pytest.approx(1.1)
        assert score.blur_level == pytest.approx(500.0)
        assert score.dynamic_range == pytest.approx(0.8)


class TestNoiseEstimation:
    def test_clean_image_low_noise(self):
        plate = make_clean_plate()
        sigma = QualityAssessor.estimate_noise(plate)
        assert sigma < 1.0

    def test_noisy_image_higher_noise(self):
        plate = make_noisy_plate(sigma=20.0)
        sigma = QualityAssessor.estimate_noise(plate)
        assert sigma > 5.0

    def test_noise_monotonic_with_sigma(self):
        s_low = QualityAssessor.estimate_noise(make_noisy_plate(sigma=5.0))
        s_high = QualityAssessor.estimate_noise(make_noisy_plate(sigma=30.0))
        assert s_high > s_low

    def test_returns_float(self):
        plate = make_clean_plate()
        assert isinstance(QualityAssessor.estimate_noise(plate), float)


class TestBlockingEstimation:
    def test_smooth_image_no_blocking(self):
        plate = make_clean_plate()
        ratio = QualityAssessor.estimate_blocking(plate)
        assert ratio < 1.2

    def test_blocky_image_high_ratio(self):
        plate = make_blocky_plate()
        ratio = QualityAssessor.estimate_blocking(plate)
        assert ratio > 1.0

    def test_returns_positive_float(self):
        plate = make_clean_plate()
        ratio = QualityAssessor.estimate_blocking(plate)
        assert isinstance(ratio, float)
        assert ratio >= 0.0


class TestBlurEstimation:
    def test_sharp_image_high_variance(self):
        plate = make_sharp_plate()
        blur = QualityAssessor.estimate_blur(plate)
        assert blur > 100.0

    def test_blurry_image_low_variance(self):
        plate = make_blurry_plate()
        blur = QualityAssessor.estimate_blur(plate)
        assert blur < 50.0

    def test_blur_monotonic(self):
        sharp = make_sharp_plate()
        blurred = cv2.GaussianBlur(sharp.clip(0, 255).astype(np.uint8), (15, 15), 5.0).astype(
            np.float32
        )
        assert QualityAssessor.estimate_blur(blurred) < QualityAssessor.estimate_blur(sharp)

    def test_returns_float(self):
        plate = make_clean_plate()
        assert isinstance(QualityAssessor.estimate_blur(plate), float)


class TestDynamicRange:
    def test_full_range(self):
        """Grayscale from random BGR is compressed by channel weighting (~0.78)."""
        rng = np.random.RandomState(42)
        plate = rng.randint(0, 256, (240, 320, 3)).astype(np.float32)
        dr = QualityAssessor.estimate_dynamic_range(plate)
        assert dr > 0.7

    def test_narrow_range(self):
        plate = np.random.RandomState(42).randint(100, 121, (240, 320, 3)).astype(np.float32)
        dr = QualityAssessor.estimate_dynamic_range(plate)
        assert dr < 0.15

    def test_uniform_image(self):
        plate = make_clean_plate(value=128)
        dr = QualityAssessor.estimate_dynamic_range(plate)
        assert dr < 0.02

    def test_range_bounded(self):
        plate = make_noisy_plate()
        dr = QualityAssessor.estimate_dynamic_range(plate)
        assert 0.0 <= dr <= 1.0


class TestAssessIntegration:
    def test_returns_quality_score(self):
        plate = make_noisy_plate()
        assessor = QualityAssessor()
        score = assessor.assess(plate)
        assert isinstance(score, QualityScore)

    def test_all_fields_populated(self):
        plate = make_noisy_plate()
        assessor = QualityAssessor()
        score = assessor.assess(plate)
        assert score.noise_level > 0
        assert score.blocking_severity > 0
        assert score.blur_level >= 0
        assert 0.0 <= score.dynamic_range <= 1.0
