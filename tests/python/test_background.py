import numpy as np
import pytest

from src.python.core.stabilizer import BackgroundModel


class TestConstruction:
    def test_default_params(self):
        model = BackgroundModel()
        assert model.alpha == pytest.approx(0.05)
        assert model.threshold == 15
        assert model.warmup_frames == 60

    def test_custom_params(self):
        model = BackgroundModel(alpha=0.05, threshold=40, warmup_frames=30)
        assert model.alpha == pytest.approx(0.05)
        assert model.threshold == 40
        assert model.warmup_frames == 30


class TestInitialState:
    def test_no_background_before_first_frame(self):
        model = BackgroundModel()
        assert model.background is None

    def test_frame_count_starts_zero(self):
        model = BackgroundModel()
        assert model.frame_count == 0

    def test_not_warmed_up_initially(self):
        model = BackgroundModel()
        assert model.is_warmed_up is False


class TestFirstFrame:
    def test_first_frame_becomes_background(self):
        model = BackgroundModel()
        frame = np.full((4, 4, 3), 100, dtype=np.uint8)
        model.update(frame)
        np.testing.assert_array_equal(model.background, frame)

    def test_first_frame_increments_count(self):
        model = BackgroundModel()
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        model.update(frame)
        assert model.frame_count == 1

    def test_first_frame_returns_all_static_mask(self):
        model = BackgroundModel()
        frame = np.full((4, 4, 3), 100, dtype=np.uint8)
        mask = model.update(frame)
        # First frame: background == frame, so no motion
        assert mask.shape == (4, 4)
        assert mask.dtype == np.uint8
        assert np.all(mask == 0)


class TestWarmup:
    def test_warmup_uses_faster_alpha(self):
        """During warmup, effective alpha should be higher than configured alpha."""
        model = BackgroundModel(alpha=0.02, warmup_frames=60)
        # Start with black background
        frame_black = np.zeros((4, 4, 3), dtype=np.uint8)
        model.update(frame_black)

        # Feed a white frame during warmup (frame_count=1, still warming up)
        frame_white = np.full((4, 4, 3), 255, dtype=np.uint8)
        model.update(frame_white)

        # With warmup acceleration, background should move toward white
        # faster than normal alpha=0.02 would allow
        # Normal alpha: 0 + 0.02*255 = 5.1 -> 5
        # Warmup should be noticeably higher
        avg_val = model.background.mean()
        assert avg_val > 10  # Much more than 0.02 * 255 = 5.1

    def test_becomes_warmed_up(self):
        model = BackgroundModel(warmup_frames=5)
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        for _ in range(5):
            model.update(frame)
        assert model.is_warmed_up is True

    def test_not_warmed_up_before_threshold(self):
        model = BackgroundModel(warmup_frames=5)
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        for _ in range(4):
            model.update(frame)
        assert model.is_warmed_up is False


class TestMotionMask:
    def test_static_scene_produces_empty_mask(self):
        model = BackgroundModel(warmup_frames=1)
        frame = np.full((8, 8, 3), 128, dtype=np.uint8)
        # Build background
        for _ in range(10):
            model.update(frame)
        # Same frame -> no motion
        mask = model.update(frame)
        assert np.all(mask == 0)

    def test_large_change_produces_motion(self):
        model = BackgroundModel(threshold=30, warmup_frames=1)
        bg_frame = np.full((32, 32, 3), 100, dtype=np.uint8)
        # Converge background
        for _ in range(100):
            model.update(bg_frame)

        # Insert a bright block (large diff > threshold, big enough to survive morphology)
        moving_frame = bg_frame.copy()
        moving_frame[8:24, 8:24] = 200  # 16x16 block, diff = 100
        mask = model.update(moving_frame)

        # The bright block region should be marked as motion
        assert np.any(mask[10:22, 10:22] == 255)
        # Corners should remain static
        assert np.all(mask[0, 0] == 0)
        assert np.all(mask[31, 31] == 0)

    def test_small_change_below_threshold_is_static(self):
        model = BackgroundModel(threshold=30, warmup_frames=1)
        bg_frame = np.full((8, 8, 3), 100, dtype=np.uint8)
        for _ in range(100):
            model.update(bg_frame)

        # Small change well below threshold
        noisy_frame = bg_frame.copy()
        noisy_frame[2:6, 2:6] = 110  # diff = 10, below threshold=30
        mask = model.update(noisy_frame)
        assert np.all(mask == 0)

    def test_mask_is_binary(self):
        """Mask values should only be 0 or 255."""
        model = BackgroundModel(threshold=30, warmup_frames=1)
        bg_frame = np.full((16, 16, 3), 100, dtype=np.uint8)
        for _ in range(50):
            model.update(bg_frame)

        moving_frame = bg_frame.copy()
        moving_frame[4:12, 4:12] = 200
        mask = model.update(moving_frame)
        unique_vals = np.unique(mask)
        assert all(v in (0, 255) for v in unique_vals)


class TestMorphologicalCleanup:
    def test_isolated_noise_pixels_removed(self):
        """Single-pixel noise in the mask should be cleaned by morphological open."""
        model = BackgroundModel(threshold=30, warmup_frames=1)
        bg_frame = np.full((16, 16, 3), 100, dtype=np.uint8)
        for _ in range(100):
            model.update(bg_frame)

        # Create frame with scattered single-pixel noise
        noisy_frame = bg_frame.copy()
        # Isolated bright pixels (should be cleaned)
        noisy_frame[1, 1] = [200, 200, 200]
        noisy_frame[5, 10] = [200, 200, 200]
        noisy_frame[12, 3] = [200, 200, 200]
        mask = model.update(noisy_frame)

        # Isolated pixels should be removed by morphological cleanup
        assert np.sum(mask) == 0

    def test_solid_block_preserved(self):
        """A solid block of motion should survive morphological cleanup."""
        model = BackgroundModel(threshold=30, warmup_frames=1)
        bg_frame = np.full((16, 16, 3), 100, dtype=np.uint8)
        for _ in range(100):
            model.update(bg_frame)

        moving_frame = bg_frame.copy()
        moving_frame[4:12, 4:12] = 200  # 8x8 solid block
        mask = model.update(moving_frame)

        # Core of the block should still be detected
        assert np.any(mask[5:11, 5:11] == 255)


class TestConvergence:
    def test_background_converges_to_constant_input(self):
        """After many frames of the same input, background should closely match."""
        model = BackgroundModel(alpha=0.02, warmup_frames=10)
        target = np.full((8, 8, 3), 150, dtype=np.uint8)

        for _ in range(300):
            model.update(target)

        # Background should be very close to target
        diff = np.abs(model.background.astype(float) - target.astype(float))
        assert diff.max() < 2.0

    def test_background_tracks_slow_change(self):
        """Background should gradually follow slow illumination changes."""
        model = BackgroundModel(alpha=0.02, warmup_frames=5)

        # Start dark
        for _ in range(100):
            model.update(np.full((4, 4, 3), 50, dtype=np.uint8))
        bg_before = model.background.mean()

        # Shift to brighter
        for _ in range(200):
            model.update(np.full((4, 4, 3), 200, dtype=np.uint8))
        bg_after = model.background.mean()

        assert bg_after > bg_before + 50  # Significant shift toward 200
