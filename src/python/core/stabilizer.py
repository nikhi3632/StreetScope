"""Frame stabilization and background modeling.

Provides:
- FrameStabilizer: removes global camera motion via sparse LK optical flow
- BackgroundModel: EMA background plate with binary motion mask output
"""

import cv2
import numpy as np


class FrameStabilizer:
    """Removes global camera motion (shake, vibration) from a video stream.

    Uses cv2.goodFeaturesToTrack to find salient points in the previous frame,
    cv2.calcOpticalFlowPyrLK to track them into the current frame, and
    cv2.estimateAffinePartial2D (4-DOF: translation + rotation + uniform
    scale) with RANSAC to robustly estimate the camera warp.  The inverse
    warp is applied to align the current frame with the previous one.
    """

    # -----------------------------------------------------------------------
    # Tuning constants — empirical
    #
    # max_features    200    Max corners for LK. 200 is enough for outdoor traffic
    #                        scenes; more adds compute, fewer risks losing lock.
    # quality_level  0.01    Corner quality threshold (fraction of strongest corner).
    #                        1% catches subtle features; lower = more noise corners.
    # min_distance     10    Min pixel spacing between features. Prevents clustering
    #                        on high-texture regions (e.g. bridge cables).
    # win_size     (21,21)   LK search window. 21px handles ~10px inter-frame motion
    #                        at 15fps. Larger = slower but tolerates more shake.
    # max_level         3    Pyramid levels for LK. 3 levels = 8x downscale, handles
    #                        large motions. Standard OpenCV default.
    # ransac_thresh   3.0    Max reprojection error (pixels) for RANSAC inlier.
    #                        3px tolerates sub-pixel flow noise.
    # min_inlier_ratio 0.5   Require 50% inliers for valid affine. Below this,
    #                        too many outliers (moving objects dominate frame).
    # blockSize         7    Sobel window for corner detection. Odd, ≥3. 7 smooths
    #                        noise while detecting structural corners.
    # LK criteria  (30, 0.01)  Max 30 iterations or 0.01px convergence. Standard.
    # -----------------------------------------------------------------------
    def __init__(
        self,
        max_features: int = 200,
        quality_level: float = 0.01,
        min_distance: int = 10,
        win_size: tuple[int, int] = (21, 21),
        max_level: int = 3,
        ransac_thresh: float = 3.0,
        min_inlier_ratio: float = 0.5,
    ) -> None:
        self.prev_gray: np.ndarray | None = None
        self.feature_params = dict(
            maxCorners=max_features,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=7,
        )
        self.lk_params = dict(
            winSize=win_size,
            maxLevel=max_level,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )
        self.ransac_thresh = ransac_thresh
        self.min_inlier_ratio = min_inlier_ratio

    @property
    def is_initialized(self) -> bool:
        """True after the first frame has been processed."""
        return self.prev_gray is not None

    def stabilize(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Stabilize a frame by removing estimated global camera motion.

        Args:
            frame: BGR uint8 image (H, W, 3).

        Returns:
            (stabilized_frame, warp_matrix) where *warp_matrix* is the 2x3
            affine that was applied to the current frame (maps raw frame
            coordinates -> stabilized coordinates).  On the very first call
            or when estimation fails, the identity transform is returned and
            the frame is passed through unchanged.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]
        identity = np.eye(2, 3, dtype=np.float64)

        if self.prev_gray is None:
            self.prev_gray = gray
            return frame.copy(), identity

        # Detect features in previous frame
        pts_prev = cv2.goodFeaturesToTrack(self.prev_gray, **self.feature_params)

        if pts_prev is None or len(pts_prev) < 4:
            self.prev_gray = gray
            return frame.copy(), identity

        # Track features into current frame
        pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            pts_prev,
            None,
            **self.lk_params,
        )

        good = status.flatten() == 1
        if np.sum(good) < 4:
            self.prev_gray = gray
            return frame.copy(), identity

        pts_p = pts_prev[good].reshape(-1, 2)
        pts_c = pts_curr[good].reshape(-1, 2)

        # Estimate 4-DOF affine (translation + rotation + uniform scale)
        # M maps previous-frame coords -> current-frame coords (camera motion)
        M, inliers = cv2.estimateAffinePartial2D(
            pts_p,
            pts_c,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_thresh,
        )

        if M is None:
            self.prev_gray = gray
            return frame.copy(), identity

        # Check inlier ratio
        if inliers is not None:
            ratio = np.sum(inliers) / len(inliers)
            if ratio < self.min_inlier_ratio:
                self.prev_gray = gray
                return frame.copy(), identity

        # Invert M to get the warp that cancels camera motion
        M_full = np.vstack([M, [0, 0, 1]])
        try:
            M_inv_full = np.linalg.inv(M_full)
        except np.linalg.LinAlgError:
            self.prev_gray = gray
            return frame.copy(), identity

        M_inv = M_inv_full[:2]

        stabilized = cv2.warpAffine(
            frame,
            M_inv,
            (w, h),
            borderMode=cv2.BORDER_REPLICATE,
        )

        # Use the *original* (unstabilized) gray for next frame's features
        self.prev_gray = gray
        return stabilized, M_inv

    def reset(self) -> None:
        """Reset internal state (e.g. on stream reconnect)."""
        self.prev_gray = None


class BackgroundModel:
    """Exponential moving average background model with motion mask output.

    Maintains a running background plate via EMA. Produces a binary motion
    mask by thresholding the absolute difference between current frame and
    background. Morphological open/close cleans salt-and-pepper noise.

    During warmup (first `warmup_frames` frames), a faster effective alpha
    accelerates convergence so the background is usable quickly.

    When ``use_simd=True`` (default), EMA and subtraction are performed by
    C++ NEON kernels via pybind11.  Falls back to OpenCV if the native
    module is unavailable.
    """

    # -----------------------------------------------------------------------
    # Tuning constants — empirical
    #
    # alpha          0.05    EMA learning rate. 5% per frame = ~20 frame half-life.
    #                        Slow enough for stable background, fast enough to adapt
    #                        to lighting changes over seconds.
    # threshold        15    Absolute pixel difference to classify as motion.
    #                        15/255 ≈ 6% change. Tuned for 8-bit traffic cameras;
    #                        too low = noise triggers motion, too high = misses slow cars.
    # warmup_frames    60    ~4 seconds at 15fps. Background needs enough frames
    #                        for EMA to converge. 60 frames at alpha=0.5 warmup
    #                        gives >99.9% convergence.
    # warmup_alpha    0.5    50% per frame during warmup. Aggressive learning to
    #                        build initial background fast, then switch to 5%.
    # kernel_size       5    5x5 morphological cleanup for salt-and-pepper noise
    #                        in the motion mask. Standard; 3x3 leaves too much noise.
    # -----------------------------------------------------------------------
    def __init__(
        self,
        alpha: float = 0.05,
        threshold: int = 15,
        warmup_frames: int = 60,
        warmup_alpha: float = 0.5,
        kernel_size: int = 5,
        use_simd: bool = True,
    ) -> None:
        self.alpha = alpha
        self.threshold = threshold
        self.warmup_frames = warmup_frames
        self.warmup_alpha = warmup_alpha

        self.background_plate: np.ndarray | None = None
        self.frame_count: int = 0
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        self._simd = None
        if use_simd:
            try:
                import streetscope_simd

                self._simd = streetscope_simd
            except ImportError:
                pass

    @property
    def background(self) -> np.ndarray | None:
        """Current background plate as uint8 BGR, or None before first frame."""
        if self.background_plate is None:
            return None
        return self.background_plate.astype(np.uint8)

    @property
    def is_warmed_up(self) -> bool:
        return self.frame_count >= self.warmup_frames

    def effective_alpha(self) -> float:
        """Higher alpha during warmup for fast convergence."""
        if self.frame_count == 0:
            return 1.0  # First frame: adopt entirely
        if not self.is_warmed_up:
            # Linear ramp from warmup_alpha down to self.alpha over warmup period
            progress = self.frame_count / self.warmup_frames
            return self.alpha + (self.warmup_alpha - self.alpha) * (1.0 - progress)
        return self.alpha

    def update(
        self, frame: np.ndarray, isp_params=None, alpha_map: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update background and return motion mask + display frame.

        When isp_params is provided and SIMD is available, runs the full
        pipeline (EMA + subtract + AE+AWB + AF blend) in a single C++ call.

        Args:
            frame: BGR uint8 image (H, W, 3).
            isp_params: ISPParams from estimator, or None to skip ISP.
            alpha_map: Per-pixel AF alpha (H, W) float32, or None to skip AF.

        Returns:
            (mask, display) where mask is (H, W) uint8 and display is (H, W, 3) uint8.
        """
        alpha = self.effective_alpha()

        if self.background_plate is None:
            self.background_plate = frame.astype(np.float32).copy()
            self.frame_count = 1
            h, w = frame.shape[:2]
            return np.zeros((h, w), dtype=np.uint8), frame.copy()

        if self._simd is not None:
            if isp_params is not None:
                gains = isp_params.auto_white_balance_gains
                mask, display = self._simd.process_frame(
                    frame,
                    self.background_plate,
                    alpha=alpha,
                    threshold=self.threshold,
                    lut=isp_params.auto_exposure_lut,
                    gain_b=float(gains[0]),
                    gain_g=float(gains[1]),
                    gain_r=float(gains[2]),
                    alpha_map=alpha_map,
                    blur_ksize=5,
                )
            else:
                mask, display = self._simd.process_frame(
                    frame,
                    self.background_plate,
                    alpha=alpha,
                    threshold=self.threshold,
                )
        else:
            # OpenCV path (golden reference)
            frame_f = frame.astype(np.float32)
            cv2.accumulateWeighted(frame_f, self.background_plate, alpha)
            bg_uint8 = self.background_plate.astype(np.uint8)
            diff = cv2.absdiff(frame, bg_uint8)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_diff, self.threshold, 255, cv2.THRESH_BINARY)
            display = frame.copy()

        self.frame_count += 1

        # Morphological cleanup: open removes small noise, close fills small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        return mask, display
