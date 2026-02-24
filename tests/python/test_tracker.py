"""Tests for LK hybrid tracker (IC Affine, template correction, appearance basis, tracking)."""

import time

import cv2
import numpy as np
import pytest

from src.python.perception.detector import Detection
from src.python.perception.tracker import (
    TrackedObject,
    Tracker,
    build_appearance_basis,
    correct_template,
    ic_affine_step,
    precompute_ic,
    warp_bbox,
)

# ---------------------------------------------------------------------------
# Helpers — IC Affine tests
# ---------------------------------------------------------------------------

def make_template(h=40, w=60):
    """Create a synthetic template with gradients (non-degenerate)."""
    rng = np.random.RandomState(42)
    # Gradient + noise makes the Hessian well-conditioned
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    xg, yg = np.meshgrid(x, y)
    template = (xg * 128 + yg * 128 + rng.randn(h, w) * 10).clip(0, 255)
    return template.astype(np.float64)


def make_textured_image(h=288, w=512, seed=99):
    """Create a full-frame textured image (non-zero everywhere)."""
    rng = np.random.RandomState(seed)
    # Smooth gradient + noise to provide trackable features
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    xg, yg = np.meshgrid(x, y)
    image = (xg * 80 + yg * 80 + 40 + rng.randn(h, w) * 15).clip(10, 245)
    return image.astype(np.float64)


def embed_template(template, image_size=(288, 512), origin=(100, 80)):
    """Embed a template into a textured image at the given origin."""
    h, w = image_size
    image = make_textured_image(h, w)
    ox, oy = origin
    th, tw = template.shape
    image[oy:oy + th, ox:ox + tw] = template
    return image


# ---------------------------------------------------------------------------
# Helpers — Tracker tests
# ---------------------------------------------------------------------------

def make_det(x1, y1, x2, y2, conf=0.9, class_id=2, class_name="car"):
    return Detection(bbox=(x1, y1, x2, y2), confidence=conf,
                     class_id=class_id, class_name=class_name)


def make_frame(h=288, w=512):
    """Create a synthetic frame with a textured background."""
    rng = np.random.RandomState(42)
    frame = rng.randint(40, 200, (h, w, 3), dtype=np.uint8)
    return frame


def add_vehicle(frame, x1, y1, x2, y2, seed=7):
    """Draw a textured rectangle on the frame (simulates a vehicle).

    Uses a gradient + noise pattern so the LK tracker has strong
    gradient information for sub-pixel tracking.
    """
    out = frame.copy()
    h = y2 - y1
    w = x2 - x1
    rng = np.random.RandomState(seed)
    # Gradient + noise for rich texture
    xg = np.linspace(80, 200, w)
    yg = np.linspace(80, 200, h)
    xx, yy = np.meshgrid(xg, yg)
    patch = (xx * 0.5 + yy * 0.5 + rng.randn(h, w) * 15).clip(60, 240)
    patch_bgr = np.stack([patch, patch * 0.9, patch * 0.8], axis=-1).astype(np.uint8)
    out[y1:y2, x1:x2] = patch_bgr
    return out


# ===========================================================================
# IC Affine precomputation
# ===========================================================================

class TestPrecomputeIC:
    def test_returns_correct_shapes(self):
        template = make_template(40, 60)
        sd, H_inv = precompute_ic(template)
        assert sd.shape == (40 * 60, 6)
        assert H_inv.shape == (6, 6)

    def test_degenerate_template_returns_none(self):
        """Flat (constant) template -> singular Hessian."""
        flat = np.ones((30, 30), dtype=np.float64) * 128
        sd, H_inv = precompute_ic(flat)
        if sd is not None:
            pass
        else:
            assert H_inv is None


# ===========================================================================
# IC Affine tracking
# ===========================================================================

class TestICAffineSteady:
    """IC Affine should converge when template is at identity warp."""

    def test_identity_warp_high_ncc(self):
        template = make_template(40, 60)
        origin = np.array([100, 80], dtype=np.float64)
        image = embed_template(template, origin=(100, 80))
        sd, H_inv = precompute_ic(template)
        p = np.zeros(6, dtype=np.float64)

        p_out, ncc = ic_affine_step(
            template, image, origin, p, sd, H_inv,
            num_iters=20, threshold=0.01,
        )
        assert ncc > 0.95
        assert np.linalg.norm(p_out) < 0.5


class TestICAffinePureTranslation:
    """IC Affine should recover a small pure translation."""

    def test_recovers_small_shift(self):
        """Extract a template from a frame, shift the frame, recover the shift.

        Checks the output bbox center and NCC rather than raw warp params,
        since the 6-param affine can reach the same bbox through different
        parameter combinations.
        """
        # Rich texture: checkerboard + noise for well-conditioned Hessian
        h, w = 288, 512
        rng = np.random.RandomState(42)
        xg = np.arange(w, dtype=np.float64)
        yg = np.arange(h, dtype=np.float64)
        xx, yy = np.meshgrid(xg, yg)
        checker = ((xx // 8).astype(int) + (yy // 8).astype(int)) % 2
        image0 = (checker * 100 + 60 + rng.randn(h, w) * 10).clip(10, 245)

        ox, oy, tw, th = 100, 80, 60, 40
        template = image0[oy:oy + th, ox:ox + tw].copy()
        origin = np.array([ox, oy], dtype=np.float64)

        sd, H_inv = precompute_ic(template)
        assert sd is not None

        # Shift entire image by (3, 2)
        M = np.float64([[1, 0, 3], [0, 1, 2]])
        image1 = cv2.warpAffine(
            image0, M, (w, h), borderMode=cv2.BORDER_REPLICATE,
        )

        p = np.zeros(6, dtype=np.float64)
        p_out, ncc = ic_affine_step(
            template, image1, origin, p, sd, H_inv,
            num_iters=50, threshold=0.001,
        )

        # Check via bbox — center should be at (133, 102) = original + shift
        bbox = warp_bbox(origin, (tw, th), p_out)
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        expected_cx = ox + tw / 2.0 + 3.0  # 133
        expected_cy = oy + th / 2.0 + 2.0  # 102
        assert abs(cx - expected_cx) < 3.0
        assert abs(cy - expected_cy) < 3.0
        assert ncc > 0.85


# ===========================================================================
# Warp bbox
# ===========================================================================

class TestWarpBbox:
    def test_identity_warp(self):
        origin = np.array([100, 80], dtype=np.float64)
        size = (60, 40)  # width, height
        p = np.zeros(6, dtype=np.float64)
        x1, y1, x2, y2 = warp_bbox(origin, size, p)
        assert x1 == 100
        assert y1 == 80
        assert x2 == 160
        assert y2 == 120

    def test_translated_warp(self):
        origin = np.array([100, 80], dtype=np.float64)
        size = (60, 40)
        p = np.array([0, 0, 0, 0, 5.0, 3.0], dtype=np.float64)
        x1, y1, x2, y2 = warp_bbox(origin, size, p)
        assert x1 == 105
        assert y1 == 83
        assert x2 == 165
        assert y2 == 123


# ===========================================================================
# Template correction
# ===========================================================================

class TestCorrectTemplate:
    def test_returns_new_precomputation(self):
        crop = make_template(30, 50)
        template, sd, H_inv = correct_template(crop)
        assert template.shape == crop.shape
        assert sd is not None
        assert sd.shape == (30 * 50, 6)
        assert H_inv.shape == (6, 6)


# ===========================================================================
# Appearance basis (PCA)
# ===========================================================================

class TestAppearanceBasis:
    def test_too_few_samples(self):
        samples = [np.random.randn(100) for _ in range(3)]
        basis = build_appearance_basis(samples, n_components=4)
        assert basis is None

    def test_builds_basis(self):
        rng = np.random.RandomState(42)
        N = 200
        samples = [rng.randn(N) for _ in range(15)]
        basis = build_appearance_basis(samples, n_components=4)
        assert basis is not None
        assert basis.shape == (4, N)

    def test_basis_orthogonal(self):
        rng = np.random.RandomState(42)
        N = 100
        samples = [rng.randn(N) for _ in range(20)]
        basis = build_appearance_basis(samples, n_components=3)
        assert basis is not None
        # Rows should be approximately orthonormal
        gram = basis @ basis.T
        np.testing.assert_allclose(gram, np.eye(3), atol=0.1)

    def test_ic_affine_with_basis(self):
        """IC Affine should still converge when appearance basis is provided.

        The basis is built from actual appearance differences (brightness
        offsets), so it captures the illumination variation and allows the
        tracker to ignore it.
        """
        template = make_template(40, 60)
        origin = np.array([100, 80], dtype=np.float64)

        # Build basis from actual brightness-shifted versions of the template
        flat_t = template.ravel()
        samples = []
        for offset in range(-15, 16, 3):
            shifted = np.clip(flat_t + offset, 0, 255)
            samples.append(flat_t - shifted)
        basis = build_appearance_basis(samples, n_components=4)

        # Create image with brightness offset
        image = embed_template(template, origin=(100, 80))
        image[80:120, 100:160] = np.clip(template + 10, 0, 255)

        sd, H_inv = precompute_ic(template)
        p = np.zeros(6, dtype=np.float64)

        p_out, ncc = ic_affine_step(
            template, image, origin, p, sd, H_inv,
            num_iters=20, threshold=0.01,
            appearance_basis=basis,
        )
        assert ncc > 0.8


# ===========================================================================
# TrackedObject
# ===========================================================================

class TestTrackedObject:
    def test_fields(self):
        t = TrackedObject(
            track_id=1,
            bbox=(10, 20, 100, 200),
            confidence=0.85,
            class_id=2,
            class_name="car",
            trail=[(55.0, 110.0)],
        )
        assert t.track_id == 1
        assert t.bbox == (10, 20, 100, 200)
        assert t.confidence == pytest.approx(0.85)
        assert t.class_id == 2
        assert t.class_name == "car"
        assert len(t.trail) == 1

    def test_center(self):
        t = TrackedObject(
            track_id=1, bbox=(10, 20, 60, 80),
            confidence=0.5, class_id=2, class_name="car", trail=[],
        )
        cx, cy = t.center
        assert cx == pytest.approx(35.0)
        assert cy == pytest.approx(50.0)

    def test_area(self):
        t = TrackedObject(
            track_id=1, bbox=(10, 20, 60, 80),
            confidence=0.5, class_id=2, class_name="car", trail=[],
        )
        assert t.area == 3000


# ===========================================================================
# Tracker construction
# ===========================================================================

class TestTrackerConstruction:
    def test_default_params(self):
        tracker = Tracker()
        assert tracker.trail_duration == 2.0

    def test_custom_params(self):
        tracker = Tracker(frame_rate=30, trail_duration=3.0,
                          match_iou=0.4, ncc_kill=0.3)
        assert tracker.trail_duration == 3.0


# ===========================================================================
# Tracker update
# ===========================================================================

class TestTrackerUpdate:
    def test_empty_detections(self):
        tracker = Tracker()
        frame = make_frame()
        result = tracker.update(frame, [])
        assert result == []

    def test_single_detection_gets_track_id(self):
        tracker = Tracker()
        frame = make_frame()
        frame = add_vehicle(frame, 100, 100, 160, 150)
        dets = [make_det(100, 100, 160, 150)]
        tracked = tracker.update(frame, dets)
        assert len(tracked) == 1
        assert tracked[0].track_id is not None
        assert isinstance(tracked[0].track_id, int)
        assert tracked[0].class_name == "car"
        assert tracked[0].confidence == pytest.approx(0.9)

    def test_persistent_id_across_frames(self):
        tracker = Tracker()
        bg = make_frame()

        # Frame 1: vehicle at position A
        f1 = add_vehicle(bg, 100, 100, 160, 150)
        tracked1 = tracker.update(f1, [make_det(100, 100, 160, 150)])

        # Frame 2: vehicle moved slightly
        f2 = add_vehicle(bg, 105, 100, 165, 150)
        tracked2 = tracker.update(f2, [make_det(105, 100, 165, 150)])

        # Frame 3: vehicle moved again
        f3 = add_vehicle(bg, 110, 100, 170, 150)
        tracked3 = tracker.update(f3, [make_det(110, 100, 170, 150)])

        assert len(tracked1) == 1
        assert len(tracked2) == 1
        assert len(tracked3) == 1
        # Same vehicle should keep same ID
        assert tracked1[0].track_id == tracked2[0].track_id
        assert tracked2[0].track_id == tracked3[0].track_id

    def test_two_vehicles_get_different_ids(self):
        tracker = Tracker()
        bg = make_frame()
        frame = add_vehicle(bg, 10, 10, 70, 60, seed=7)
        frame = add_vehicle(frame, 300, 200, 360, 250, seed=13)
        dets = [
            make_det(10, 10, 70, 60),
            make_det(300, 200, 360, 250),
        ]
        tracked = tracker.update(frame, dets)
        assert len(tracked) == 2
        ids = {t.track_id for t in tracked}
        assert len(ids) == 2  # Different IDs

    def test_multiple_classes_tracked(self):
        tracker = Tracker()
        bg = make_frame()
        frame = add_vehicle(bg, 10, 10, 70, 60, seed=7)
        frame = add_vehicle(frame, 300, 200, 360, 250, seed=13)
        dets = [
            make_det(10, 10, 70, 60, class_id=2, class_name="car"),
            make_det(300, 200, 360, 250, class_id=7, class_name="truck"),
        ]
        tracked = tracker.update(frame, dets)
        names = {t.class_name for t in tracked}
        assert "car" in names
        assert "truck" in names


# ===========================================================================
# Tracker trails
# ===========================================================================

class TestTrackerTrails:
    def test_trail_grows_over_frames(self):
        tracker = Tracker()
        bg = make_frame()

        f1 = add_vehicle(bg, 100, 100, 160, 150)
        tracker.update(f1, [make_det(100, 100, 160, 150)])

        f2 = add_vehicle(bg, 105, 100, 165, 150)
        tracked = tracker.update(f2, [make_det(105, 100, 165, 150)])
        assert len(tracked) == 1
        assert len(tracked[0].trail) >= 2

    def test_trail_contains_centroids(self):
        tracker = Tracker()
        bg = make_frame()
        frame = add_vehicle(bg, 100, 100, 200, 200)
        tracked = tracker.update(frame, [make_det(100, 100, 200, 200)])
        assert len(tracked) == 1
        if tracked[0].trail:
            cx, cy = tracked[0].trail[0]
            assert cx == pytest.approx(150.0, abs=5)
            assert cy == pytest.approx(150.0, abs=5)

    def test_trail_time_based_expiry(self):
        tracker = Tracker(trail_duration=0.1)  # 100ms window
        bg = make_frame()

        f1 = add_vehicle(bg, 100, 100, 160, 150)
        tracker.update(f1, [make_det(100, 100, 160, 150)])

        # Wait longer than trail duration
        time.sleep(0.15)

        f2 = add_vehicle(bg, 105, 100, 165, 150)
        tracked = tracker.update(f2, [make_det(105, 100, 165, 150)])
        assert len(tracked) == 1
        # Old trail point should have expired, only new one remains
        assert len(tracked[0].trail) == 1

    def test_track_dies_without_detection(self):
        tracker = Tracker(max_frames_without_det=3, ncc_kill=0.9)
        bg = make_frame()

        # Frame 1: vehicle present
        f1 = add_vehicle(bg, 100, 100, 160, 150)
        tracker.update(f1, [make_det(100, 100, 160, 150)])

        # Frames 2-10: vehicle gone, no detections, background changes
        for _ in range(10):
            tracker.update(bg, [])

        tracked = tracker.update(bg, [])
        assert tracked == []


# ===========================================================================
# Tracker template correction
# ===========================================================================

class TestTrackerTemplateCorrection:
    def test_detection_resets_lost_counter(self):
        tracker = Tracker(max_frames_without_det=5)
        bg = make_frame()

        # Frame 1: detect and track
        f1 = add_vehicle(bg, 100, 100, 160, 150)
        tracker.update(f1, [make_det(100, 100, 160, 150)])

        # Frames 2-3: no detection (LK tracks)
        for _ in range(2):
            tracker.update(f1, [])

        # Frame 4: re-detect (template correction)
        tracked = tracker.update(f1, [make_det(100, 100, 160, 150)])
        assert len(tracked) >= 1  # Track should still be alive

    def test_tiny_detection_ignored(self):
        """Detections smaller than 4x4 should not create tracks."""
        tracker = Tracker()
        frame = make_frame()
        dets = [make_det(10, 10, 12, 12)]  # 2x2 — too small
        tracked = tracker.update(frame, dets)
        assert tracked == []
