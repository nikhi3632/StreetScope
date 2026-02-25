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

    def update(self, frame: np.ndarray) -> np.ndarray:
        """Update background and return binary motion mask.

        Args:
            frame: BGR uint8 image (H, W, 3).

        Returns:
            Binary mask (H, W) uint8. 255 = motion, 0 = static.
        """
        frame_f = frame.astype(np.float32)
        alpha = self.effective_alpha()

        if self.background_plate is None:
            self.background_plate = frame_f.copy()
            self.frame_count = 1
            h, w = frame.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)

        if self._simd is not None:
            # NEON path: C++ SIMD kernels via pybind11
            self._simd.accumulate_ema(frame_f, self.background_plate, alpha=alpha)
            self.frame_count += 1
            bg_uint8 = self.background_plate.astype(np.uint8)
            mask = self._simd.subtract_background(frame, bg_uint8, threshold=self.threshold)
        else:
            # OpenCV path (golden reference)
            cv2.accumulateWeighted(frame_f, self.background_plate, alpha)
            self.frame_count += 1
            bg_uint8 = self.background_plate.astype(np.uint8)
            diff = cv2.absdiff(frame, bg_uint8)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_diff, self.threshold, 255, cv2.THRESH_BINARY)

        # Morphological cleanup: open removes small noise, close fills small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        return mask
