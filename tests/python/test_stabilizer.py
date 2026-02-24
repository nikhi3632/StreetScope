"""Tests for FrameStabilizer (Subtract Dominant Motion)."""

import cv2
import numpy as np

from src.python.core.stabilizer import FrameStabilizer


def make_frame(h=288, w=512):
    """Create a test frame with trackable features (random pattern)."""
    rng = np.random.RandomState(42)
    frame = rng.randint(0, 256, (h, w, 3), dtype=np.uint8)
    # Add some high-contrast features for optical flow
    for _ in range(30):
        cx, cy = rng.randint(20, w - 20), rng.randint(20, h - 20)
        cv2.circle(frame, (cx, cy), 8, (255, 255, 255), -1)
    return frame


def shift_frame(frame, dx, dy):
    """Shift a frame by (dx, dy) pixels using affine warp."""
    h, w = frame.shape[:2]
    M = np.float64([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


class TestFrameStabilizerConstruction:
    def test_default_params(self):
        stab = FrameStabilizer()
        assert stab.is_initialized is False

    def test_custom_params(self):
        stab = FrameStabilizer(max_features=100, ransac_thresh=5.0)
        assert stab.is_initialized is False


class TestFrameStabilizerFirstFrame:
    def test_first_frame_passthrough(self):
        stab = FrameStabilizer()
        frame = make_frame()
        stabilized, warp = stab.stabilize(frame)
        assert stabilized.shape == frame.shape
        # Warp should be identity
        np.testing.assert_allclose(warp, np.eye(2, 3), atol=1e-10)
        assert stab.is_initialized is True


class TestFrameStabilizerStabilization:
    def test_cancels_pure_translation(self):
        """If camera shifts 5px right, stabilizer should cancel it."""
        stab = FrameStabilizer()
        frame0 = make_frame()
        stab.stabilize(frame0)  # Initialize

        # Simulate camera moving 5px right, 3px down
        frame1 = shift_frame(frame0, 5, 3)
        stabilized, warp = stab.stabilize(frame1)

        # The stabilized frame should be close to the original
        # (not pixel-perfect due to interpolation, but structurally similar)
        assert stabilized.shape == frame0.shape
        # The warp should have a negative translation component
        # (canceling the positive camera shift)
        assert warp[0, 2] < 0  # Negative x translation
        assert warp[1, 2] < 0  # Negative y translation

    def test_identity_when_no_motion(self):
        """Same frame twice → identity warp."""
        stab = FrameStabilizer()
        frame = make_frame()
        stab.stabilize(frame)
        _, warp = stab.stabilize(frame)
        # Translation components should be near zero
        assert abs(warp[0, 2]) < 1.0
        assert abs(warp[1, 2]) < 1.0

    def test_reset(self):
        stab = FrameStabilizer()
        frame = make_frame()
        stab.stabilize(frame)
        assert stab.is_initialized is True
        stab.reset()
        assert stab.is_initialized is False


class TestFrameStabilizerEdgeCases:
    def test_blank_frame(self):
        """All-black frame has no features → should return frame unchanged."""
        stab = FrameStabilizer()
        frame0 = make_frame()
        stab.stabilize(frame0)
        blank = np.zeros((288, 512, 3), dtype=np.uint8)
        stabilized, warp = stab.stabilize(blank)
        assert stabilized.shape == blank.shape

    def test_small_frame(self):
        stab = FrameStabilizer()
        frame = make_frame(h=64, w=64)
        stabilized, warp = stab.stabilize(frame)
        assert stabilized.shape == frame.shape
