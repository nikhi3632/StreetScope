"""Background plate estimation and motion mask via temporal accumulation."""

import cv2
import numpy as np


class BackgroundModel:
    """Exponential moving average background model with motion mask output.

    Maintains a running background plate via EMA. Produces a binary motion
    mask by thresholding the absolute difference between current frame and
    background. Morphological open/close cleans salt-and-pepper noise.

    During warmup (first `warmup_frames` frames), a faster effective alpha
    accelerates convergence so the background is usable quickly.
    """

    def __init__(
        self,
        alpha: float = 0.02,
        threshold: int = 30,
        warmup_frames: int = 60,
        warmup_alpha: float = 0.5,
        kernel_size: int = 5,
    ) -> None:
        self.alpha = alpha
        self.threshold = threshold
        self.warmup_frames = warmup_frames
        self.warmup_alpha = warmup_alpha

        self._background: np.ndarray | None = None
        self._frame_count: int = 0
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    @property
    def background(self) -> np.ndarray | None:
        """Current background plate as uint8 BGR, or None before first frame."""
        if self._background is None:
            return None
        return self._background.astype(np.uint8)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def is_warmed_up(self) -> bool:
        return self._frame_count >= self.warmup_frames

    def _effective_alpha(self) -> float:
        """Higher alpha during warmup for fast convergence."""
        if self._frame_count == 0:
            return 1.0  # First frame: adopt entirely
        if not self.is_warmed_up:
            # Linear ramp from warmup_alpha down to self.alpha over warmup period
            progress = self._frame_count / self.warmup_frames
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
        alpha = self._effective_alpha()

        if self._background is None:
            self._background = frame_f.copy()
            self._frame_count = 1
            h, w = frame.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)

        # EMA update: bg = (1 - alpha) * bg + alpha * frame
        cv2.accumulateWeighted(frame_f, self._background, alpha)
        self._frame_count += 1

        # Absolute diff on uint8 for thresholding
        bg_uint8 = self._background.astype(np.uint8)
        diff = cv2.absdiff(frame, bg_uint8)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # Threshold to binary mask
        _, mask = cv2.threshold(gray_diff, self.threshold, 255, cv2.THRESH_BINARY)

        # Morphological cleanup: open removes small noise, close fills small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        return mask
