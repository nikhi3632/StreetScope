"""LK hybrid vehicle tracker.

Inverse Composition Affine optical flow tracking, template correction
via YOLO re-detection, and PCA appearance basis for illumination robustness.

YOLO answers "what is it?", LK answers "where did it go?".
"""

import time
from dataclasses import dataclass

import cv2
import numpy as np

from src.python.perception.detector import Detection

# ---------------------------------------------------------------------------
# IC Affine precomputation
# ---------------------------------------------------------------------------


def precompute_ic(template: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Precompute steepest-descent images and inverse Hessian for IC Affine.

    The Inverse Composition formulation precomputes everything that depends
    only on the *template*, so per-frame updates are cheap.

    Args:
        template: Grayscale float64 image (H, W).

    Returns:
        (sd_images, hessian_inv)
        sd_images:   (N, 6) steepest-descent images.
        hessian_inv: (6, 6) inverse of the Gauss-Newton Hessian.
        Returns (None, None) if the Hessian is singular.
    """
    th, tw = template.shape[:2]

    # Gradient of template
    Tx = cv2.Sobel(template, cv2.CV_64F, 1, 0, ksize=3) / 8.0
    Ty = cv2.Sobel(template, cv2.CV_64F, 0, 1, ksize=3) / 8.0

    # Pixel coordinate grids (template space)
    x_coords, y_coords = np.meshgrid(
        np.arange(tw, dtype=np.float64),
        np.arange(th, dtype=np.float64),
    )
    x_flat = x_coords.ravel()
    y_flat = y_coords.ravel()
    Tx_flat = Tx.ravel()
    Ty_flat = Ty.ravel()

    # Steepest descent: SD[i] = [∇T] @ ∂W/∂p  evaluated at identity warp.
    #   Affine warp: W = [[1+p0, p2, p4],
    #                      [p1, 1+p3, p5]]
    #   ∂W/∂p  = [[x, 0, y, 0, 1, 0],
    #              [0, x, 0, y, 0, 1]]
    N = len(x_flat)
    sd = np.empty((N, 6), dtype=np.float64)
    sd[:, 0] = Tx_flat * x_flat
    sd[:, 1] = Ty_flat * x_flat
    sd[:, 2] = Tx_flat * y_flat
    sd[:, 3] = Ty_flat * y_flat
    sd[:, 4] = Tx_flat
    sd[:, 5] = Ty_flat

    # Hessian: H = SDᵀ SD
    H = sd.T @ sd  # (6, 6)
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None, None

    return sd, H_inv


# ---------------------------------------------------------------------------
# IC Affine tracking step
# ---------------------------------------------------------------------------


def ic_affine_step(
    template: np.ndarray,
    image: np.ndarray,
    origin: np.ndarray,
    p: np.ndarray,
    sd: np.ndarray,
    H_inv: np.ndarray,
    num_iters: int = 20,
    threshold: float = 0.05,
    appearance_basis: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Run one IC Affine update: refine warp *p* to align template in image.

    Args:
        template:  Grayscale float64 (H, W).
        image:     Current full frame, grayscale float64.
        origin:    (2,) float64 — top-left (x, y) anchor of the track in
                   the image coordinate system.
        p:         (6,) current affine warp parameters (modified in-place).
        sd:        (N, 6) precomputed steepest-descent images.
        H_inv:     (6, 6) precomputed inverse Hessian.
        num_iters: Maximum Gauss-Newton iterations.
        threshold: Convergence threshold on ‖Δp‖.
        appearance_basis: Optional (K, N) PCA basis. If provided, the
                   appearance variation is projected out of the error image
                   before computing the parameter update.

    Returns:
        (p, ncc)  where *p* is the updated warp parameters and *ncc* is
        the normalised cross-correlation between template and warped image.
    """
    th, tw = template.shape[:2]
    ih, iw = image.shape[:2]

    x_coords, y_coords = np.meshgrid(
        np.arange(tw, dtype=np.float64),
        np.arange(th, dtype=np.float64),
    )
    x_flat = x_coords.ravel()
    y_flat = y_coords.ravel()
    ones = np.ones_like(x_flat)
    coords = np.vstack([x_flat, y_flat, ones])  # (3, N)

    template_flat = template.ravel()

    for _ in range(num_iters):
        # Current warp matrix
        W = np.array(
            [
                [1 + p[0], p[2], p[4]],
                [p[1], 1 + p[3], p[5]],
            ],
            dtype=np.float64,
        )

        # Warp template coordinates into image space
        warped_coords = W @ coords  # (2, N)
        wx = warped_coords[0] + origin[0]
        wy = warped_coords[1] + origin[1]

        # Bounds check — if warp goes mostly off-frame, bail
        if np.median(wx) < 0 or np.median(wx) >= iw or np.median(wy) < 0 or np.median(wy) >= ih:
            break

        # Sample image at warped locations
        map_x = wx.reshape(th, tw).astype(np.float32)
        map_y = wy.reshape(th, tw).astype(np.float32)
        warped_img = cv2.remap(
            image,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        warped_flat = warped_img.ravel()

        # Error image: I(W(x; p)) − T(x)
        # (Baker & Matthews IC formulation — sign must match the
        #  inverse compositional update W ← W ∘ W(Δp)⁻¹)
        error = warped_flat - template_flat

        # Project out appearance variation if basis is available
        if appearance_basis is not None:
            coeffs = appearance_basis @ error  # (K,)
            error = error - appearance_basis.T @ coeffs  # (N,)

        # Parameter update
        dp = H_inv @ (sd.T @ error)  # (6,)

        # Inverse compositional update: W ← W ∘ W(Δp)⁻¹
        dW = np.array(
            [
                [1 + dp[0], dp[2], dp[4]],
                [dp[1], 1 + dp[3], dp[5]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        try:
            dW_inv = np.linalg.inv(dW)
        except np.linalg.LinAlgError:
            break

        W_full = np.vstack([W, [0, 0, 1]])
        W_new = W_full @ dW_inv

        p[0] = W_new[0, 0] - 1
        p[1] = W_new[1, 0]
        p[2] = W_new[0, 1]
        p[3] = W_new[1, 1] - 1
        p[4] = W_new[0, 2]
        p[5] = W_new[1, 2]

        if np.linalg.norm(dp) < threshold:
            break

    # ── Compute NCC for quality assessment ─────────────────────────────
    W = np.array(
        [
            [1 + p[0], p[2], p[4]],
            [p[1], 1 + p[3], p[5]],
        ],
        dtype=np.float64,
    )
    warped_coords = W @ coords
    wx = warped_coords[0] + origin[0]
    wy = warped_coords[1] + origin[1]

    map_x = wx.reshape(th, tw).astype(np.float32)
    map_y = wy.reshape(th, tw).astype(np.float32)
    warped_img = cv2.remap(
        image,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    warped_flat = warped_img.ravel()

    # NCC ∈ [−1, 1]
    t_std = np.std(template_flat)
    w_std = np.std(warped_flat)
    if t_std < 1e-6 or w_std < 1e-6:
        ncc = 0.0
    else:
        ncc = float(np.corrcoef(template_flat, warped_flat)[0, 1])

    return p, ncc


# ---------------------------------------------------------------------------
# Bounding box from warp parameters
# ---------------------------------------------------------------------------


def warp_bbox(
    origin: np.ndarray,
    template_size: tuple[int, int],
    p: np.ndarray,
) -> tuple[int, int, int, int]:
    """Compute the axis-aligned bounding box of the warped template.

    Args:
        origin: (2,) top-left anchor (x, y).
        template_size: (width, height) of the template.
        p: (6,) affine warp parameters.

    Returns:
        (x1, y1, x2, y2) bounding box in image coordinates.
    """
    tw, th = template_size
    W = np.array(
        [
            [1 + p[0], p[2], p[4]],
            [p[1], 1 + p[3], p[5]],
        ],
        dtype=np.float64,
    )

    # Four corners of the template
    corners = np.array(
        [
            [0, 0, 1],
            [tw, 0, 1],
            [tw, th, 1],
            [0, th, 1],
        ],
        dtype=np.float64,
    ).T  # (3, 4)

    warped = W @ corners  # (2, 4)
    warped[0] += origin[0]
    warped[1] += origin[1]

    x1 = int(round(warped[0].min()))
    y1 = int(round(warped[1].min()))
    x2 = int(round(warped[0].max()))
    y2 = int(round(warped[1].max()))
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Template correction
# ---------------------------------------------------------------------------


def correct_template(
    new_crop: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Prepare a fresh reference template from a YOLO re-detection crop.

    Resets the warp to identity and recomputes the IC precomputation
    quantities on the new template.

    Args:
        new_crop: Grayscale float64 image (H, W) — the YOLO detection crop.

    Returns:
        (template, sd, H_inv).  sd and H_inv may be None if the Hessian
        is singular (degenerate template).
    """
    template = new_crop.copy()
    sd, H_inv = precompute_ic(template)
    return template, sd, H_inv


# ---------------------------------------------------------------------------
# Appearance basis (PCA)
# ---------------------------------------------------------------------------


def build_appearance_basis(
    samples: list[np.ndarray],
    n_components: int = 4,
) -> np.ndarray | None:
    """Build a PCA appearance basis from collected error images.

    Args:
        samples: List of flattened error images (each shape (N,)).
                 All must have the same length.
        n_components: Number of principal components to keep.

    Returns:
        (K, N) matrix of basis vectors, or None if insufficient samples.
    """
    if len(samples) < max(n_components + 1, 5):
        return None

    # Stack into (M, N) matrix
    data = np.array(samples, dtype=np.float64)
    # Center
    mean = data.mean(axis=0)
    centered = data - mean

    # SVD — we only need the top-k left singular vectors
    try:
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    # Keep top n_components rows of Vt  →  (K, N)
    k = min(n_components, Vt.shape[0])
    basis = Vt[:k]
    return basis


# ---------------------------------------------------------------------------
# Tracked object (public output)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackedObject:
    """A tracked detection with persistent ID and trajectory trail."""

    track_id: int
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    trail: list[tuple[float, float]]  # Recent centroids (oldest first)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


# ---------------------------------------------------------------------------
# Per-object tracking state
# ---------------------------------------------------------------------------


class Track:
    """Internal per-object tracking state."""

    __slots__ = (
        "track_id",
        "class_id",
        "class_name",
        "det_confidence",
        "template",
        "origin",
        "size",
        "warp_params",
        "sd",
        "H_inv",
        "appearance_samples",
        "appearance_basis",
        "ncc",
        "frames_without_detection",
        "trail",
        "trail_duration",
    )

    def __init__(
        self,
        track_id: int,
        template: np.ndarray,
        origin: np.ndarray,
        size: tuple[int, int],
        class_id: int,
        class_name: str,
        det_confidence: float,
        trail_duration: float,
    ) -> None:
        self.track_id = track_id
        self.class_id = class_id
        self.class_name = class_name
        self.det_confidence = det_confidence

        self.template = template  # grayscale float64 (H, W)
        self.origin = origin  # (x, y) float64
        self.size = size  # (width, height)
        self.warp_params = np.zeros(6, dtype=np.float64)

        sd, H_inv = precompute_ic(template)
        self.sd = sd
        self.H_inv = H_inv

        self.appearance_samples: list[np.ndarray] = []
        self.appearance_basis: np.ndarray | None = None

        self.ncc: float = 1.0
        self.frames_without_detection: int = 0

        self.trail: list[tuple[float, float, float]] = []  # (time, cx, cy)
        self.trail_duration = trail_duration

    @property
    def is_valid(self) -> bool:
        """False if IC precomputation failed (degenerate template)."""
        return self.sd is not None and self.H_inv is not None

    def predict(self, image_gray: np.ndarray) -> tuple[int, int, int, int]:
        """Run IC Affine to refine position in *image_gray*.

        Includes sanity checks: if IC Affine diverges (area change > 3x,
        or center moves more than the template diagonal), the update is
        rejected and the previous bbox is returned with NCC = 0.

        Returns the updated bounding box (x1, y1, x2, y2).
        """
        if not self.is_valid:
            return self.current_bbox()

        prev_bbox = self.current_bbox()
        prev_p = self.warp_params.copy()

        self.warp_params, self.ncc = ic_affine_step(
            self.template,
            image_gray,
            self.origin,
            self.warp_params,
            self.sd,
            self.H_inv,
            num_iters=20,
            threshold=0.05,
            appearance_basis=self.appearance_basis,
        )

        new_bbox = self.current_bbox()

        # Sanity: reject if area changed drastically
        tw, th = self.size
        orig_area = tw * th
        new_w = new_bbox[2] - new_bbox[0]
        new_h = new_bbox[3] - new_bbox[1]
        new_area = max(1, new_w * new_h)
        area_ratio = new_area / max(1, orig_area)

        # Sanity: reject if center moved more than template diagonal
        diag = (tw**2 + th**2) ** 0.5
        prev_cx = (prev_bbox[0] + prev_bbox[2]) / 2
        prev_cy = (prev_bbox[1] + prev_bbox[3]) / 2
        new_cx = (new_bbox[0] + new_bbox[2]) / 2
        new_cy = (new_bbox[1] + new_bbox[3]) / 2
        displacement = ((new_cx - prev_cx) ** 2 + (new_cy - prev_cy) ** 2) ** 0.5

        if area_ratio < 0.25 or area_ratio > 4.0 or displacement > diag:
            # Diverged — revert warp, mark as low quality
            self.warp_params = prev_p
            self.ncc = 0.0
            return prev_bbox

        return new_bbox

    def correct(
        self, image_gray: np.ndarray, bbox: tuple[int, int, int, int], det_confidence: float
    ) -> None:
        """Template correction from a YOLO re-detection.

        Resets the warp to identity using the fresh detection crop.
        """
        x1, y1, x2, y2 = bbox
        ih, iw = image_gray.shape[:2]

        # Clamp
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(iw, x2)
        y2 = min(ih, y2)
        if x2 <= x1 or y2 <= y1:
            return

        crop = image_gray[y1:y2, x1:x2].astype(np.float64)

        # Collect appearance sample (error before correction)
        if self.is_valid and self.template.shape == crop.shape:
            err = (self.template - crop).ravel()
            self.appearance_samples.append(err)
            # Rebuild basis periodically
            if len(self.appearance_samples) % 5 == 0:
                basis = build_appearance_basis(
                    self.appearance_samples,
                    n_components=4,
                )
                if basis is not None:
                    self.appearance_basis = basis
            # Cap stored samples
            if len(self.appearance_samples) > 50:
                self.appearance_samples = self.appearance_samples[-30:]

        # Reset template and warp
        new_template, new_sd, new_H_inv = correct_template(crop)
        # If template size changed, old appearance samples are invalid
        if self.template.shape != new_template.shape:
            self.appearance_samples.clear()
            self.appearance_basis = None
        self.template = new_template
        self.sd = new_sd
        self.H_inv = new_H_inv
        self.origin = np.array([x1, y1], dtype=np.float64)
        self.size = (x2 - x1, y2 - y1)
        self.warp_params = np.zeros(6, dtype=np.float64)
        self.det_confidence = det_confidence
        self.frames_without_detection = 0

    def append_trail(self, bbox: tuple[int, int, int, int]) -> None:
        """Record the current centroid and prune old trail entries."""
        now = time.monotonic()
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self.trail.append((now, cx, cy))
        cutoff = now - self.trail_duration
        self.trail = [(t, x, y) for t, x, y in self.trail if t >= cutoff]

    def get_trail_points(self) -> list[tuple[float, float]]:
        """Return trail as list of (cx, cy) — oldest first."""
        return [(x, y) for _, x, y in self.trail]

    def current_bbox(self) -> tuple[int, int, int, int]:
        return warp_bbox(self.origin, self.size, self.warp_params)


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection-over-Union of two axis-aligned bounding boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class Tracker:
    """LK hybrid tracker.

    Combines YOLO detections (re-detection / correction) with IC Affine
    optical flow tracking, PCA appearance basis, and template correction.
    """

    # -----------------------------------------------------------------------
    # Tuning constants — empirical
    #
    # trail_duration         2.0    Seconds of trail history to display. 2s at 15fps
    #                               = 30 points. Longer trails clutter; shorter lose context.
    # match_iou             0.15    IoU threshold for greedy detection-to-track matching.
    #                               15% is very loose — allows matching even when LK
    #                               drifts significantly between detections. Tight (>0.3)
    #                               would break matching for fast-moving small vehicles.
    # ncc_kill              0.1     NCC below 10% = template completely lost. Track is
    #                               marked low-quality and eligible for pruning.
    # max_frames_without_det  30    ~2 seconds at 15fps. Tracks survive occlusion for
    #                               2s via LK alone, then die. Longer risks ghost tracks.
    #
    # IC Affine (ic_affine_step):
    # num_iters              20     Max Gauss-Newton iterations. 20 is conservative;
    #                               most converge in 5-10. Cost is cheap per iteration.
    # threshold            0.05     Convergence on ||delta_p||. 0.05px = sub-pixel.
    #
    # Appearance basis:
    # n_components           4      PCA components. 4 captures major illumination
    #                               variation without overfitting to noise.
    # sample_cap            50      Max appearance samples per track. 50 covers
    #                               ~3.3s of observation at 15fps.
    # rebuild_interval       5      Rebuild PCA basis every 5 new samples.
    #
    # Track validation:
    # min_det_size           4      Detections < 4x4px are too small to track reliably.
    # area_ratio_bounds [0.25, 4.0] Sanity check: tracked area can shrink to 25% or
    #                               grow to 400% of original. Beyond = diverged.
    # -----------------------------------------------------------------------
    def __init__(
        self,
        frame_rate: int = 15,
        trail_duration: float = 2.0,
        match_iou: float = 0.15,
        ncc_kill: float = 0.1,
        max_frames_without_det: int = 30,
    ) -> None:
        self.trail_duration = trail_duration
        self.match_iou = match_iou
        self.ncc_kill = ncc_kill
        self.max_lost = max_frames_without_det
        self.frame_rate = frame_rate

        self.tracks: list[Track] = []
        self.next_id: int = 1
        self.prev_gray: np.ndarray | None = None

    def update(
        self,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> list[TrackedObject]:
        """Process one frame: predict with LK, associate with YOLO, correct.

        Args:
            frame: BGR uint8 image (stabilized).
            detections: YOLO detections for this frame.

        Returns:
            List of TrackedObject with persistent IDs and trails.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
        ih, iw = gray.shape[:2]

        # ── 1. LK predict ──────────────────────────────────────────────
        predicted_bboxes: list[tuple[int, int, int, int]] = []
        for track in self.tracks:
            bbox = track.predict(gray)
            # Clamp to image
            bbox = (
                max(0, min(bbox[0], iw)),
                max(0, min(bbox[1], ih)),
                max(0, min(bbox[2], iw)),
                max(0, min(bbox[3], ih)),
            )
            predicted_bboxes.append(bbox)

        # ── 2. Associate detections with tracks (greedy IoU) ───────────
        det_bboxes = [d.bbox for d in detections]
        n_tracks = len(self.tracks)
        n_dets = len(detections)

        matched_track: set[int] = set()
        matched_det: set[int] = set()
        match_pairs: list[tuple[int, int]] = []

        if n_tracks > 0 and n_dets > 0:
            iou_matrix = np.zeros((n_tracks, n_dets), dtype=np.float64)
            for ti in range(n_tracks):
                for di in range(n_dets):
                    iou_matrix[ti, di] = iou(predicted_bboxes[ti], det_bboxes[di])

            while True:
                idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                best_iou = iou_matrix[idx]
                if best_iou < self.match_iou:
                    break
                ti, di = int(idx[0]), int(idx[1])
                matched_track.add(ti)
                matched_det.add(di)
                match_pairs.append((ti, di))
                iou_matrix[ti, :] = -1
                iou_matrix[:, di] = -1

        # Apply template correction for matched pairs
        for ti, di in match_pairs:
            det = detections[di]
            self.tracks[ti].correct(gray, det.bbox, det.confidence)

        # ── 4. Increment lost counter for unmatched tracks ─────────────
        for ti in range(n_tracks):
            if ti not in matched_track:
                self.tracks[ti].frames_without_detection += 1

        # ── 5. Birth new tracks from unmatched detections ──────────────
        for di in range(n_dets):
            if di not in matched_det:
                det = detections[di]
                self.birth_track(gray, det)

        # ── 6. Kill dead tracks ────────────────────────────────────────
        alive: list[Track] = []
        for track in self.tracks:
            if track.ncc < self.ncc_kill and track.frames_without_detection > 5:
                continue
            if track.frames_without_detection > self.max_lost:
                continue
            bbox = track.current_bbox()
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w <= 0 or h <= 0:
                continue
            alive.append(track)
        self.tracks = alive

        # ── 7. Build output ────────────────────────────────────────────
        results: list[TrackedObject] = []
        for track in self.tracks:
            bbox = track.current_bbox()
            # Clamp
            bbox = (
                max(0, min(bbox[0], iw)),
                max(0, min(bbox[1], ih)),
                max(0, min(bbox[2], iw)),
                max(0, min(bbox[3], ih)),
            )
            track.append_trail(bbox)

            results.append(
                TrackedObject(
                    track_id=track.track_id,
                    bbox=bbox,
                    confidence=track.det_confidence,
                    class_id=track.class_id,
                    class_name=track.class_name,
                    trail=track.get_trail_points(),
                )
            )

        self.prev_gray = gray
        return results

    def birth_track(self, gray: np.ndarray, det: Detection) -> None:
        """Create a new track from a YOLO detection."""
        x1, y1, x2, y2 = det.bbox
        ih, iw = gray.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(iw, x2)
        y2 = min(ih, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return  # Too small to track

        crop = gray[y1:y2, x1:x2].astype(np.float64)
        origin = np.array([x1, y1], dtype=np.float64)
        size = (x2 - x1, y2 - y1)

        track = Track(
            track_id=self.next_id,
            template=crop,
            origin=origin,
            size=size,
            class_id=det.class_id,
            class_name=det.class_name,
            det_confidence=det.confidence,
            trail_duration=self.trail_duration,
        )

        if not track.is_valid:
            return  # Degenerate template (e.g. flat patch)

        self.next_id += 1
        self.tracks.append(track)
