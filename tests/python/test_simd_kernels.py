"""Tests comparing C++ SIMD kernels against Python BackgroundModel golden reference."""

import sys
from pathlib import Path

import numpy as np

# Add build directory to path so streetscope_simd can be imported
build_dir = Path(__file__).resolve().parent.parent.parent / "build"
if str(build_dir) not in sys.path:
    sys.path.insert(0, str(build_dir))

import streetscope_simd  # noqa: E402


class TestAccumulateEmaVsGolden:
    """Compare streetscope_simd.accumulate_ema against cv2.accumulateWeighted."""

    def test_single_frame_alpha_half(self):
        frame = np.full((8, 8, 3), 200.0, dtype=np.float32)
        bg_cpp = np.full((8, 8, 3), 100.0, dtype=np.float32)
        bg_golden = bg_cpp.copy()

        streetscope_simd.accumulate_ema(frame, bg_cpp, alpha=0.5)
        import cv2

        cv2.accumulateWeighted(frame, bg_golden, 0.5)

        np.testing.assert_allclose(bg_cpp, bg_golden, atol=1e-5)

    def test_multiple_frames(self):
        import cv2

        rng = np.random.RandomState(42)
        bg_cpp = rng.uniform(0, 255, (288, 512, 3)).astype(np.float32)
        bg_golden = bg_cpp.copy()

        for _ in range(20):
            frame = rng.uniform(0, 255, (288, 512, 3)).astype(np.float32)
            streetscope_simd.accumulate_ema(frame, bg_cpp, alpha=0.05)
            cv2.accumulateWeighted(frame, bg_golden, 0.05)

        np.testing.assert_allclose(bg_cpp, bg_golden, atol=1e-3)

    def test_warmup_alpha_sequence(self):
        """Simulate the warmup alpha ramp from BackgroundModel.effective_alpha."""
        import cv2

        rng = np.random.RandomState(99)
        bg_cpp = np.zeros((64, 64, 3), dtype=np.float32)
        bg_golden = bg_cpp.copy()

        warmup_frames = 60
        alpha_base = 0.05
        warmup_alpha = 0.5

        for i in range(warmup_frames):
            frame = rng.uniform(0, 255, (64, 64, 3)).astype(np.float32)
            if i == 0:
                alpha = 1.0
            else:
                progress = i / warmup_frames
                alpha = alpha_base + (warmup_alpha - alpha_base) * (1.0 - progress)

            streetscope_simd.accumulate_ema(frame, bg_cpp, alpha=alpha)
            cv2.accumulateWeighted(frame, bg_golden, alpha)

        np.testing.assert_allclose(bg_cpp, bg_golden, atol=1e-3)


class TestSubtractBackgroundVsGolden:
    """Compare streetscope_simd.subtract_background against Python absdiff+threshold."""

    def python_subtract(self, frame, background, threshold):
        """Replicate the max-across-channels absdiff logic."""
        diff = np.abs(frame.astype(np.int16) - background.astype(np.int16))
        max_diff = diff.max(axis=2).astype(np.uint8)
        mask = np.where(max_diff > threshold, np.uint8(255), np.uint8(0))
        return mask

    def test_identical_frames(self):
        frame = np.full((32, 32, 3), 128, dtype=np.uint8)
        bg = frame.copy()

        mask_cpp = streetscope_simd.subtract_background(frame, bg, threshold=15)
        mask_golden = self.python_subtract(frame, bg, 15)

        np.testing.assert_array_equal(mask_cpp, mask_golden)

    def test_large_diff(self):
        frame = np.full((32, 32, 3), 200, dtype=np.uint8)
        bg = np.full((32, 32, 3), 100, dtype=np.uint8)

        mask_cpp = streetscope_simd.subtract_background(frame, bg, threshold=30)
        mask_golden = self.python_subtract(frame, bg, 30)

        np.testing.assert_array_equal(mask_cpp, mask_golden)

    def test_mixed_with_random_data(self):
        rng = np.random.RandomState(42)
        frame = rng.randint(0, 256, (288, 512, 3), dtype=np.uint8)
        bg = rng.randint(0, 256, (288, 512, 3), dtype=np.uint8)

        mask_cpp = streetscope_simd.subtract_background(frame, bg, threshold=25)
        mask_golden = self.python_subtract(frame, bg, 25)

        np.testing.assert_array_equal(mask_cpp, mask_golden)

    def test_threshold_boundary(self):
        """Pixel with diff exactly at threshold should be static."""
        frame = np.array([[[130, 100, 100]]], dtype=np.uint8)
        bg = np.array([[[100, 100, 100]]], dtype=np.uint8)

        mask_cpp = streetscope_simd.subtract_background(frame, bg, threshold=30)
        mask_golden = self.python_subtract(frame, bg, 30)

        np.testing.assert_array_equal(mask_cpp, mask_golden)


class TestScalarBindings:
    """Verify scalar C++ bindings produce same results as NEON."""

    def test_accumulate_ema_scalar_matches_neon(self):
        rng = np.random.RandomState(77)
        frame = rng.uniform(0, 255, (64, 64, 3)).astype(np.float32)
        bg_scalar = rng.uniform(0, 255, (64, 64, 3)).astype(np.float32)
        bg_neon = bg_scalar.copy()

        streetscope_simd.accumulate_ema_scalar(frame, bg_scalar, alpha=0.1)
        streetscope_simd.accumulate_ema(frame, bg_neon, alpha=0.1)

        np.testing.assert_allclose(bg_scalar, bg_neon, atol=1e-5)

    def test_subtract_background_scalar_matches_neon(self):
        rng = np.random.RandomState(88)
        frame = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        bg = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)

        mask_scalar = streetscope_simd.subtract_background_scalar(frame, bg, threshold=20)
        mask_neon = streetscope_simd.subtract_background(frame, bg, threshold=20)

        np.testing.assert_array_equal(mask_scalar, mask_neon)


class TestBackgroundModelSIMDIntegration:
    """Compare BackgroundModel with SIMD vs OpenCV on the same frame sequence."""

    def test_background_plates_converge(self):
        """EMA is identical between SIMD and OpenCV — backgrounds must match."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src" / "python"))
        from core.stabilizer import BackgroundModel

        rng = np.random.RandomState(42)
        model_simd = BackgroundModel(alpha=0.05, threshold=15, use_simd=True)
        model_cv = BackgroundModel(alpha=0.05, threshold=15, use_simd=False)

        for _ in range(30):
            frame = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            model_simd.update(frame)
            model_cv.update(frame)

        np.testing.assert_allclose(
            model_simd.background_plate,
            model_cv.background_plate,
            atol=1e-3,
        )

    def test_masks_mostly_agree(self):
        """Masks may differ slightly (max-channel vs grayscale), but >95% agreement."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src" / "python"))
        from core.stabilizer import BackgroundModel

        rng = np.random.RandomState(42)
        model_simd = BackgroundModel(alpha=0.05, threshold=15, use_simd=True)
        model_cv = BackgroundModel(alpha=0.05, threshold=15, use_simd=False)

        # Warm up backgrounds
        for _ in range(60):
            frame = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            model_simd.update(frame)
            model_cv.update(frame)

        # Compare masks on new frames
        agreements = []
        for _ in range(20):
            frame = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            mask_simd, _ = model_simd.update(frame)
            mask_cv, _ = model_cv.update(frame)
            agreement = np.mean(mask_simd == mask_cv)
            agreements.append(agreement)

        avg_agreement = np.mean(agreements)
        assert avg_agreement > 0.95, f"Mask agreement {avg_agreement:.1%} < 95%"


class TestApplyAEAWBVsGolden:
    """Compare SIMD AE+AWB against Python ISP reference.

    Note: SIMD applies LUT to BGR directly, Python applies to Y channel in
    YCrCb space. For uniform-color frames they match closely. For colorful
    frames there will be per-pixel differences (tolerance=5).
    """

    def test_identity_no_change(self):
        frame = np.full((32, 32, 3), 128, dtype=np.uint8)
        lut = np.arange(256, dtype=np.uint8)
        result = streetscope_simd.apply_ae_awb(frame, lut, 1.0, 1.0, 1.0)
        np.testing.assert_array_equal(result, frame)

    def test_gains_only(self):
        frame = np.full((32, 32, 3), 100, dtype=np.uint8)
        lut = np.arange(256, dtype=np.uint8)
        result = streetscope_simd.apply_ae_awb(frame, lut, 1.5, 1.0, 0.5)
        assert result[0, 0, 0] == 150  # B * 1.5
        assert result[0, 0, 1] == 100  # G * 1.0
        assert result[0, 0, 2] == 50  # R * 0.5

    def test_lut_and_gains(self):
        rng = np.random.RandomState(42)
        frame = rng.randint(0, 256, (288, 512, 3), dtype=np.uint8)
        indices = np.arange(256, dtype=np.float64)
        lut = ((indices / 255.0) ** 0.5 * 255.0).clip(0, 255).astype(np.uint8)
        result = streetscope_simd.apply_ae_awb(frame, lut, 0.8, 1.0, 1.3)
        assert result.dtype == np.uint8
        assert result.shape == frame.shape
        assert not np.array_equal(result, frame)

    def test_scalar_matches_neon(self):
        rng = np.random.RandomState(55)
        frame = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        lut = np.arange(256, dtype=np.uint8)
        r_scalar = streetscope_simd.apply_ae_awb_scalar(frame, lut, 1.3, 1.0, 0.7)
        r_neon = streetscope_simd.apply_ae_awb(frame, lut, 1.3, 1.0, 0.7)
        np.testing.assert_allclose(r_scalar, r_neon, atol=1)

    def test_clamp_at_255(self):
        frame = np.full((8, 8, 3), 200, dtype=np.uint8)
        lut = np.arange(256, dtype=np.uint8)
        result = streetscope_simd.apply_ae_awb(frame, lut, 2.0, 2.0, 2.0)
        assert result.max() == 255


GOLDEN_DIR = Path(__file__).parent / "golden" / "process_frame"


class TestProcessFrame:
    """process_frame correctness: determinism, convergence, range, regression.

    process_frame is what FrameLoop calls on every frame in production.
    Tests verify observable properties, not internal kernel decomposition.
    """

    def test_determinism(self):
        """Same input, same output. Every time."""
        rng = np.random.RandomState(42)
        h, w = 64, 64
        frame = rng.randint(0, 256, (h, w, 3), dtype=np.uint8)
        lut = np.arange(256, dtype=np.uint8)
        alpha_map = np.full((h, w), 0.5, dtype=np.float32)

        bg1 = rng.uniform(0, 255, (h, w, 3)).astype(np.float32)
        bg2 = bg1.copy()

        mask1, display1 = streetscope_simd.process_frame(
            frame,
            bg1,
            alpha=0.05,
            threshold=15,
            lut=lut,
            gain_b=0.8,
            gain_g=1.0,
            gain_r=1.3,
            alpha_map=alpha_map,
        )
        mask2, display2 = streetscope_simd.process_frame(
            frame,
            bg2,
            alpha=0.05,
            threshold=15,
            lut=lut,
            gain_b=0.8,
            gain_g=1.0,
            gain_r=1.3,
            alpha_map=alpha_map,
        )

        np.testing.assert_array_equal(bg1, bg2)
        np.testing.assert_array_equal(mask1, mask2)
        np.testing.assert_array_equal(display1, display2)

    def test_convergence(self):
        """Background plate stabilizes after 60 frames of a steady scene."""
        rng = np.random.RandomState(42)
        h, w = 64, 64
        scene = rng.randint(100, 200, (h, w, 3), dtype=np.uint8)
        bg = np.zeros((h, w, 3), dtype=np.float32)

        for i in range(60):
            noise = rng.randint(-5, 6, (h, w, 3), dtype=np.int16)
            frame = np.clip(scene.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            alpha = 1.0 if i == 0 else 0.05
            streetscope_simd.process_frame(frame, bg, alpha=alpha, threshold=15)

        bg_u8 = bg.clip(0, 255).astype(np.uint8)
        diff = np.abs(bg_u8.astype(np.float32) - scene.astype(np.float32))
        assert diff.mean() < 10, f"Background didn't converge: mean diff {diff.mean():.1f}"

    def test_range(self):
        """All outputs stay in [0, 255]. Mask is binary. Background is finite."""
        rng = np.random.RandomState(42)
        h, w = 64, 64
        bg = np.zeros((h, w, 3), dtype=np.float32)

        indices = np.arange(256, dtype=np.float64)
        lut = ((indices / 255.0) ** 0.5 * 255.0).clip(0, 255).astype(np.uint8)
        alpha_map = rng.uniform(0, 1.5, (h, w)).astype(np.float32)

        for i in range(30):
            frame = rng.randint(0, 256, (h, w, 3), dtype=np.uint8)
            alpha = 1.0 if i == 0 else 0.05
            mask, display = streetscope_simd.process_frame(
                frame,
                bg,
                alpha=alpha,
                threshold=15,
                lut=lut,
                gain_b=1.5,
                gain_g=1.0,
                gain_r=1.5,
                alpha_map=alpha_map,
            )
            assert mask.dtype == np.uint8
            assert display.dtype == np.uint8
            assert set(np.unique(mask)).issubset({0, 255})
            assert np.all(np.isfinite(bg))

    def test_identity_isp_passthrough(self):
        """With no lut (identity ISP + zero alpha), display output equals the input frame."""
        rng = np.random.RandomState(42)
        frame = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        bg = rng.uniform(0, 255, (64, 64, 3)).astype(np.float32)

        _, display = streetscope_simd.process_frame(
            frame,
            bg,
            alpha=0.05,
            threshold=15,
        )
        np.testing.assert_array_equal(display, frame)

    def test_regression_golden(self):
        """Compare against saved known-good outputs."""
        rng = np.random.RandomState(12345)
        h, w = 64, 64
        bg = np.zeros((h, w, 3), dtype=np.float32)

        indices = np.arange(256, dtype=np.float64)
        lut = ((indices / 255.0) ** 0.5 * 255.0).clip(0, 255).astype(np.uint8)
        alpha_map = rng.uniform(0, 1.5, (h, w)).astype(np.float32)

        for i in range(10):
            frame = rng.randint(0, 256, (h, w, 3), dtype=np.uint8)
            alpha = 1.0 if i == 0 else 0.05
            mask, display = streetscope_simd.process_frame(
                frame,
                bg,
                alpha=alpha,
                threshold=15,
                lut=lut,
                gain_b=0.8,
                gain_g=1.0,
                gain_r=1.3,
                alpha_map=alpha_map,
            )
            golden_mask = np.load(GOLDEN_DIR / f"mask_{i:02d}.npy")
            golden_display = np.load(GOLDEN_DIR / f"display_{i:02d}.npy")
            np.testing.assert_array_equal(mask, golden_mask, err_msg=f"Mask diverged at frame {i}")
            np.testing.assert_array_equal(
                display, golden_display, err_msg=f"Display diverged at frame {i}"
            )

        golden_bg = np.load(GOLDEN_DIR / "background.npy")
        np.testing.assert_array_equal(bg, golden_bg, err_msg="Background plate diverged")


class TestApplyAFBlendVsGolden:
    """Compare SIMD AF blend against Python ISP reference."""

    def test_zero_alpha_identity(self):
        frame = np.full((32, 32, 3), 128, dtype=np.uint8)
        blurred = np.full((32, 32, 3), 100, dtype=np.uint8)
        alpha_map = np.zeros((32, 32), dtype=np.float32)
        result = streetscope_simd.apply_af_blend(frame, blurred, alpha_map)
        np.testing.assert_array_equal(result, frame)

    def test_positive_alpha_sharpens(self):
        frame = np.full((8, 8, 3), 150, dtype=np.uint8)
        blurred = np.full((8, 8, 3), 100, dtype=np.uint8)
        alpha_map = np.ones((8, 8), dtype=np.float32)
        result = streetscope_simd.apply_af_blend(frame, blurred, alpha_map)
        np.testing.assert_array_equal(result, np.full_like(frame, 200))

    def test_scalar_matches_neon(self):
        rng = np.random.RandomState(66)
        frame = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        blurred = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        alpha_map = rng.uniform(0, 1.5, (64, 64)).astype(np.float32)
        r_scalar = streetscope_simd.apply_af_blend_scalar(frame, blurred, alpha_map)
        r_neon = streetscope_simd.apply_af_blend(frame, blurred, alpha_map)
        np.testing.assert_allclose(r_scalar, r_neon, atol=1)

    def test_vs_python_isp(self):
        """Compare SIMD AF blend against Python ISPEstimator.apply_auto_focus."""
        import sys as _sys

        import cv2

        _sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from src.python.isp.estimator import AUTO_FOCUS_KSIZE, ISPEstimator  # noqa: E402

        rng = np.random.RandomState(77)
        frame = rng.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        blur_map = rng.uniform(10, 1000, (8, 8)).astype(np.float32)

        # Python reference
        py_result = ISPEstimator.apply_auto_focus(frame, blur_map)

        # SIMD path: replicate the alpha_map computation from Python
        bmax = float(blur_map.max())
        normalized = blur_map / bmax
        alpha_grid = ((1.0 - normalized) * 1.5).astype(np.float32)
        h, w = frame.shape[:2]
        alpha_map = cv2.resize(alpha_grid, (w, h), interpolation=cv2.INTER_LINEAR)
        blurred = cv2.GaussianBlur(frame, (AUTO_FOCUS_KSIZE, AUTO_FOCUS_KSIZE), 0)

        simd_result = streetscope_simd.apply_af_blend(frame, blurred, alpha_map)
        np.testing.assert_allclose(simd_result, py_result, atol=1)
