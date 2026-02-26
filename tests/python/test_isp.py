"""Tests for ISP 3A estimation (AE, AWB, AF blur map) and persistence."""

import cv2
import numpy as np
import pytest

from src.python.isp.converter import isp_to_reinhard
from src.python.isp.estimator import (
    ISPEstimator,
    ISPParams,
    load_isp_params,
    save_isp_params,
    url_hash,
)

# --- Helpers ---


def make_dark_plate(h=240, w=320):
    return np.full((h, w, 3), 50.0, dtype=np.float32)


def make_bright_plate(h=240, w=320):
    return np.full((h, w, 3), 220.0, dtype=np.float32)


def make_neutral_plate(h=240, w=320):
    return np.full((h, w, 3), 128.0, dtype=np.float32)


def make_warm_plate(h=240, w=320):
    """Strong red/green, weak blue (sodium vapor lighting)."""
    plate = np.zeros((h, w, 3), dtype=np.float32)
    plate[:, :, 0] = 60.0
    plate[:, :, 1] = 140.0
    plate[:, :, 2] = 180.0
    return plate


def make_textured_plate(h=240, w=320):
    """Left half: checkerboard (sharp). Right half: smooth gradient."""
    plate = np.zeros((h, w, 3), dtype=np.float32)
    for y in range(0, h, 8):
        for x in range(0, w // 2, 8):
            val = 200.0 if ((y // 8 + x // 8) % 2 == 0) else 50.0
            plate[y : y + 8, x : x + 8] = val
    grad = np.linspace(80, 180, w - w // 2, dtype=np.float32)
    plate[:, w // 2 :] = grad[np.newaxis, :, np.newaxis]
    return plate


def make_half_blur_frame(h=240, w=320):
    """Left half: blurred checkerboard. Right half: sharp checkerboard."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # Sharp checkerboard everywhere first
    for y in range(0, h, 16):
        for x in range(0, w, 16):
            val = 200 if ((y // 16 + x // 16) % 2 == 0) else 50
            frame[y : y + 16, x : x + 16] = val
    # Blur the left half
    left = frame[:, : w // 2]
    left_blurred = cv2.GaussianBlur(left, (15, 15), 5.0)
    frame[:, : w // 2] = left_blurred
    return frame


def make_half_blur_map(grid_size=8):
    """Left columns low variance (blurry), right columns high variance (sharp)."""
    bmap = np.zeros((grid_size, grid_size), dtype=np.float32)
    bmap[:, : grid_size // 2] = 10.0  # Low variance = blurry
    bmap[:, grid_size // 2 :] = 1000.0  # High variance = sharp
    return bmap


# --- Tests ---


class TestISPParamsDataclass:
    def test_equality(self):
        lut = np.arange(256, dtype=np.uint8)
        gains = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        blur = np.ones((8, 8), dtype=np.float32)
        a = ISPParams(auto_exposure_lut=lut, auto_white_balance_gains=gains, blur_map=blur)
        b = ISPParams(
            auto_exposure_lut=lut.copy(),
            auto_white_balance_gains=gains.copy(),
            blur_map=blur.copy(),
        )
        assert a == b

    def test_inequality(self):
        lut = np.arange(256, dtype=np.uint8)
        gains_a = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        gains_b = np.array([1.5, 1.0, 0.8], dtype=np.float32)
        blur = np.ones((8, 8), dtype=np.float32)
        a = ISPParams(auto_exposure_lut=lut, auto_white_balance_gains=gains_a, blur_map=blur)
        b = ISPParams(auto_exposure_lut=lut, auto_white_balance_gains=gains_b, blur_map=blur)
        assert a != b


class TestAutoExposureLut:
    def test_dark_image_brightens(self):
        plate = make_dark_plate()
        lut = ISPEstimator.compute_auto_exposure_lut(plate, target_mean=128.0)
        assert lut.shape == (256,)
        assert lut.dtype == np.uint8
        assert int(lut[128]) > 128

    def test_bright_image_darkens(self):
        plate = make_bright_plate()
        lut = ISPEstimator.compute_auto_exposure_lut(plate, target_mean=128.0)
        assert int(lut[128]) < 128

    def test_neutral_image_identity_lut(self):
        plate = make_neutral_plate()
        lut = ISPEstimator.compute_auto_exposure_lut(plate, target_mean=128.0)
        identity = np.arange(256, dtype=np.uint8)
        assert np.max(np.abs(lut.astype(int) - identity.astype(int))) < 5

    def test_lut_monotonic(self):
        plate = make_dark_plate()
        lut = ISPEstimator.compute_auto_exposure_lut(plate)
        diffs = np.diff(lut.astype(np.int16))
        assert np.all(diffs >= 0)

    def test_lut_endpoints(self):
        plate = make_dark_plate()
        lut = ISPEstimator.compute_auto_exposure_lut(plate)
        assert lut[0] == 0
        assert lut[255] == 255


class TestAutoWhiteBalanceGains:
    def test_neutral_plate_unity_gains(self):
        plate = make_neutral_plate()
        gains = ISPEstimator.compute_auto_white_balance_gains(plate)
        assert gains.shape == (3,)
        assert gains.dtype == np.float32
        np.testing.assert_allclose(gains, [1.0, 1.0, 1.0], atol=0.05)

    def test_warm_plate_boosts_blue(self):
        plate = make_warm_plate()
        gains = ISPEstimator.compute_auto_white_balance_gains(plate)
        assert gains[0] > 1.0  # Blue gain boosted
        assert gains[1] == pytest.approx(1.0)  # Green always 1.0
        assert gains[2] < 1.0  # Red gain reduced

    def test_gains_clamped(self):
        plate = np.zeros((100, 100, 3), dtype=np.float32)
        plate[:, :, 0] = 1.0  # Very weak blue
        plate[:, :, 1] = 200.0
        plate[:, :, 2] = 200.0
        gains = ISPEstimator.compute_auto_white_balance_gains(plate)
        assert 0.5 <= gains[0] <= 2.0
        assert 0.5 <= gains[2] <= 2.0

    def test_green_channel_always_unity(self):
        plate = make_warm_plate()
        gains = ISPEstimator.compute_auto_white_balance_gains(plate)
        assert gains[1] == pytest.approx(1.0)


class TestBlurMap:
    def test_shape(self):
        plate = make_neutral_plate()
        bmap = ISPEstimator.compute_blur_map(plate, grid_size=8)
        assert bmap.shape == (8, 8)
        assert bmap.dtype == np.float32

    def test_uniform_plate_low_variance(self):
        plate = make_neutral_plate()
        bmap = ISPEstimator.compute_blur_map(plate, grid_size=4)
        assert bmap.max() < 1.0

    def test_textured_half_higher(self):
        plate = make_textured_plate()
        bmap = ISPEstimator.compute_blur_map(plate, grid_size=8)
        left_mean = bmap[:, :4].mean()
        right_mean = bmap[:, 4:].mean()
        assert left_mean > right_mean * 2

    def test_all_non_negative(self):
        rng = np.random.RandomState(42)
        plate = rng.uniform(0, 255, (240, 320, 3)).astype(np.float32)
        bmap = ISPEstimator.compute_blur_map(plate, grid_size=4)
        assert np.all(bmap >= 0)


class TestApplyAutoExposure:
    def test_identity_lut_no_change(self):
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        lut = np.arange(256, dtype=np.uint8)
        result = ISPEstimator.apply_auto_exposure(frame, lut)
        np.testing.assert_array_equal(result, frame)

    def test_output_dtype_shape(self):
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        lut = np.arange(256, dtype=np.uint8)
        result = ISPEstimator.apply_auto_exposure(frame, lut)
        assert result.dtype == np.uint8
        assert result.shape == frame.shape

    def test_brightening_lut(self):
        frame = np.full((100, 100, 3), 80, dtype=np.uint8)
        indices = np.arange(256, dtype=np.float64)
        lut = ((indices / 255.0) ** 0.5 * 255.0).clip(0, 255).astype(np.uint8)
        result = ISPEstimator.apply_auto_exposure(frame, lut)
        assert result.mean() > frame.mean()


class TestApplyAutoWhiteBalance:
    def test_unity_gains_no_change(self):
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        gains = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        result = ISPEstimator.apply_auto_white_balance(frame, gains)
        np.testing.assert_array_equal(result, frame)

    def test_boost_blue(self):
        frame = np.full((100, 100, 3), 100, dtype=np.uint8)
        gains = np.array([1.5, 1.0, 1.0], dtype=np.float32)
        result = ISPEstimator.apply_auto_white_balance(frame, gains)
        assert result[0, 0, 0] > 100
        assert result[0, 0, 1] == 100
        assert result[0, 0, 2] == 100

    def test_clamped_to_255(self):
        frame = np.full((100, 100, 3), 200, dtype=np.uint8)
        gains = np.array([2.0, 1.0, 1.0], dtype=np.float32)
        result = ISPEstimator.apply_auto_white_balance(frame, gains)
        assert result[0, 0, 0] == 255


class TestApplyAutoFocus:
    def test_uniform_blur_map_no_change(self):
        """All cells equal variance → alpha=0 everywhere → identity."""
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        blur_map = np.ones((8, 8), dtype=np.float32) * 500.0
        result = ISPEstimator.apply_auto_focus(frame, blur_map)
        np.testing.assert_array_equal(result, frame)

    def test_zero_blur_map_no_change(self):
        """All-zeros blur map → early return → identity."""
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        blur_map = np.zeros((8, 8), dtype=np.float32)
        result = ISPEstimator.apply_auto_focus(frame, blur_map)
        np.testing.assert_array_equal(result, frame)

    def test_output_dtype_shape(self):
        rng = np.random.RandomState(42)
        frame = rng.randint(0, 256, (240, 320, 3)).astype(np.uint8)
        blur_map = rng.uniform(0, 1000, (8, 8)).astype(np.float32)
        result = ISPEstimator.apply_auto_focus(frame, blur_map)
        assert result.dtype == np.uint8
        assert result.shape == frame.shape

    def test_blurry_region_sharpened(self):
        """Left half is blurry in both frame and blur_map → should get sharpened."""
        frame = make_half_blur_frame()
        blur_map = make_half_blur_map()
        result = ISPEstimator.apply_auto_focus(frame, blur_map)
        # Measure Laplacian variance of left half before and after
        left_before = cv2.Laplacian(
            cv2.cvtColor(frame[:, :160], cv2.COLOR_BGR2GRAY), cv2.CV_64F
        ).var()
        left_after = cv2.Laplacian(
            cv2.cvtColor(result[:, :160], cv2.COLOR_BGR2GRAY), cv2.CV_64F
        ).var()
        assert left_after > left_before

    def test_sharp_region_unchanged(self):
        """Right half is sharp in blur_map → alpha near 0 → minimal change."""
        frame = make_half_blur_frame()
        blur_map = make_half_blur_map()
        result = ISPEstimator.apply_auto_focus(frame, blur_map)
        right_before = frame[:, 160:]
        right_after = result[:, 160:]
        # Mean absolute difference should be very small
        mad = np.abs(right_after.astype(float) - right_before.astype(float)).mean()
        assert mad < 1.0

    def test_output_clamped_no_overflow(self):
        """Near-255 input with max sharpening → no overflow."""
        frame = np.full((100, 100, 3), 250, dtype=np.uint8)
        # One cell very low (blurry) to trigger sharpening
        blur_map = np.ones((8, 8), dtype=np.float32) * 1000.0
        blur_map[0, 0] = 1.0
        result = ISPEstimator.apply_auto_focus(frame, blur_map)
        assert result.max() <= 255
        assert result.min() >= 0

    def test_non_grid_aligned_image_size(self):
        """Non-standard frame size → no crash, correct output shape."""
        frame = np.full((237, 317, 3), 128, dtype=np.uint8)
        blur_map = np.ones((8, 8), dtype=np.float32) * 500.0
        blur_map[2, 3] = 10.0  # One blurry cell
        result = ISPEstimator.apply_auto_focus(frame, blur_map)
        assert result.shape == (237, 317, 3)
        assert result.dtype == np.uint8


class TestApplyFull:
    def test_identity_params_no_change(self):
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        params = ISPParams(
            auto_exposure_lut=np.arange(256, dtype=np.uint8),
            auto_white_balance_gains=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            blur_map=np.ones((8, 8), dtype=np.float32),
        )
        result = ISPEstimator.apply(frame, params)
        np.testing.assert_array_equal(result, frame)


class TestEstimateIntegration:
    def test_returns_isp_params(self):
        plate = make_neutral_plate()
        estimator = ISPEstimator()
        params = estimator.estimate(plate)
        assert isinstance(params, ISPParams)
        assert params.auto_exposure_lut.shape == (256,)
        assert params.auto_white_balance_gains.shape == (3,)
        assert params.blur_map.shape == (8, 8)

    def test_roundtrip_apply(self):
        plate = make_dark_plate()
        estimator = ISPEstimator()
        params = estimator.estimate(plate)
        frame = plate.clip(0, 255).astype(np.uint8)
        corrected = ISPEstimator.apply(frame, params)
        assert corrected.dtype == np.uint8
        assert corrected.shape == frame.shape
        assert corrected.mean() > frame.mean()


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        lut = np.arange(256, dtype=np.uint8)
        gains = np.array([1.2, 1.0, 0.9], dtype=np.float32)
        blur = np.random.RandomState(42).rand(8, 8).astype(np.float32)
        params = ISPParams(auto_exposure_lut=lut, auto_white_balance_gains=gains, blur_map=blur)

        save_isp_params(params, tmp_path / "isp")
        loaded = load_isp_params(tmp_path / "isp")
        assert loaded is not None
        assert loaded == params

    def test_load_missing_returns_none(self, tmp_path):
        result = load_isp_params(tmp_path / "nonexistent")
        assert result is None

    def test_url_hash_deterministic(self):
        h1 = url_hash("https://example.com/stream1")
        h2 = url_hash("https://example.com/stream1")
        assert h1 == h2

    def test_url_hash_different_urls(self):
        h1 = url_hash("https://example.com/stream1")
        h2 = url_hash("https://example.com/stream2")
        assert h1 != h2


class TestISPToReinhard:
    def test_identity_params(self):
        """Identity ISPParams -> exposure~1, white_point~1, gains=1."""
        identity = ISPParams(
            auto_exposure_lut=np.arange(256, dtype=np.uint8),
            auto_white_balance_gains=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            blur_map=np.zeros((8, 8), dtype=np.float32),
        )
        result = isp_to_reinhard(identity)
        assert abs(result["exposure"] - 1.0) < 0.1
        assert abs(result["white_point"] - 1.0) < 0.1
        assert result["gamma"] == 1.0
        assert result["gain_b"] == 1.0
        assert result["gain_g"] == 1.0
        assert result["gain_r"] == 1.0

    def test_dark_scene_high_exposure(self):
        """Dark scene (gamma < 1) -> exposure > 1."""
        dark_plate = np.full((240, 320, 3), 50.0, dtype=np.float32)
        params = ISPEstimator().estimate(dark_plate)
        result = isp_to_reinhard(params)
        assert result["exposure"] > 1.0

    def test_bright_scene_low_exposure(self):
        """Bright scene (gamma > 1) -> exposure < 1."""
        bright_plate = np.full((240, 320, 3), 200.0, dtype=np.float32)
        params = ISPEstimator().estimate(bright_plate)
        result = isp_to_reinhard(params)
        assert result["exposure"] < 1.0

    def test_awb_gains_pass_through(self):
        """AWB gains from ISPParams pass through unchanged."""
        params = ISPParams(
            auto_exposure_lut=np.arange(256, dtype=np.uint8),
            auto_white_balance_gains=np.array([0.9, 1.0, 1.1], dtype=np.float32),
            blur_map=np.zeros((8, 8), dtype=np.float32),
        )
        result = isp_to_reinhard(params)
        assert result["gain_b"] == pytest.approx(0.9)
        assert result["gain_g"] == pytest.approx(1.0)
        assert result["gain_r"] == pytest.approx(1.1)

    def test_exposure_clamped(self):
        """Exposure stays within [0.2, 5.0]."""
        extreme = ISPParams(
            auto_exposure_lut=np.full(256, 255, dtype=np.uint8),
            auto_white_balance_gains=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            blur_map=np.zeros((8, 8), dtype=np.float32),
        )
        result = isp_to_reinhard(extreme)
        assert 0.2 <= result["exposure"] <= 5.0
        assert 1.0 <= result["white_point"] <= 5.0
