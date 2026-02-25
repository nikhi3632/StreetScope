"""Tests for the fused process_frame pipeline."""

import numpy as np
import streetscope_simd as simd


class TestProcessFrameNoISP:
    """Fused pipeline without ISP (background model only)."""

    def test_returns_mask_and_display(self):
        frame = np.random.randint(0, 255, (8, 16, 3), dtype=np.uint8)
        bg = np.full((8, 16, 3), 128.0, dtype=np.float32)

        mask, display = simd.process_frame(frame, bg, alpha=0.5, threshold=15)

        assert mask.shape == (8, 16)
        assert mask.dtype == np.uint8
        assert display.shape == (8, 16, 3)
        assert display.dtype == np.uint8

    def test_display_equals_frame_when_no_isp(self):
        frame = np.random.randint(0, 255, (8, 16, 3), dtype=np.uint8)
        bg = np.full((8, 16, 3), 128.0, dtype=np.float32)

        mask, display = simd.process_frame(frame, bg, alpha=0.5, threshold=15)

        np.testing.assert_array_equal(display, frame)

    def test_background_modified_inplace(self):
        frame = np.full((8, 16, 3), 200, dtype=np.uint8)
        bg = np.full((8, 16, 3), 100.0, dtype=np.float32)
        bg_before = bg.copy()

        simd.process_frame(frame, bg, alpha=0.5, threshold=15)

        # Background should have moved toward frame value
        assert not np.array_equal(bg, bg_before)
        assert np.all(bg > bg_before)

    def test_mask_matches_separate_calls(self):
        frame = np.random.randint(0, 255, (8, 16, 3), dtype=np.uint8)
        bg_fused = np.random.rand(8, 16, 3).astype(np.float32) * 255
        bg_separate = bg_fused.copy()
        alpha = 0.3
        threshold = 20

        # Fused path
        mask_fused, _ = simd.process_frame(frame, bg_fused, alpha=alpha, threshold=threshold)

        # Separate path
        frame_f = frame.astype(np.float32)
        simd.accumulate_ema(frame_f, bg_separate, alpha=alpha)
        bg_u8 = bg_separate.clip(0, 255).astype(np.uint8)
        mask_separate = simd.subtract_background(frame, bg_u8, threshold=threshold)

        np.testing.assert_array_equal(mask_fused, mask_separate)
        np.testing.assert_allclose(bg_fused, bg_separate, atol=1e-5)


class TestProcessFrameWithISP:
    """Fused pipeline with ISP correction."""

    def test_isp_applied_when_lut_provided(self):
        frame = np.random.randint(50, 200, (8, 16, 3), dtype=np.uint8)
        bg = np.full((8, 16, 3), 128.0, dtype=np.float32)

        # Non-identity LUT: boost brightness
        lut = np.clip(np.arange(256) * 1.5, 0, 255).astype(np.uint8)
        alpha_map = np.full((8, 16), 0.5, dtype=np.float32)

        mask, display = simd.process_frame(
            frame,
            bg,
            alpha=0.1,
            threshold=15,
            lut=lut,
            gain_b=1.0,
            gain_g=1.0,
            gain_r=1.0,
            alpha_map=alpha_map,
            blur_ksize=5,
        )

        # Display should differ from input (ISP was applied)
        assert not np.array_equal(display, frame)

    def test_identity_isp_close_to_input(self):
        frame = np.random.randint(50, 200, (16, 16, 3), dtype=np.uint8)
        bg = np.full((16, 16, 3), 128.0, dtype=np.float32)

        # Identity LUT, unity gains, zero alpha (no sharpening)
        lut = np.arange(256, dtype=np.uint8)
        alpha_map = np.zeros((16, 16), dtype=np.float32)

        mask, display = simd.process_frame(
            frame,
            bg,
            alpha=0.1,
            threshold=15,
            lut=lut,
            gain_b=1.0,
            gain_g=1.0,
            gain_r=1.0,
            alpha_map=alpha_map,
            blur_ksize=5,
        )

        # With identity LUT, unity gains, zero alpha: display should equal frame
        np.testing.assert_array_equal(display, frame)

    def test_ae_awb_only_when_no_alpha_map(self):
        frame = np.random.randint(50, 200, (8, 16, 3), dtype=np.uint8)
        bg = np.full((8, 16, 3), 128.0, dtype=np.float32)

        lut = np.arange(256, dtype=np.uint8)

        # No alpha_map → skip AF, just AE+AWB
        mask, display = simd.process_frame(
            frame,
            bg,
            alpha=0.1,
            threshold=15,
            lut=lut,
            gain_b=1.0,
            gain_g=1.0,
            gain_r=1.0,
        )

        # Identity LUT + unity gains → display should equal frame
        np.testing.assert_array_equal(display, frame)
