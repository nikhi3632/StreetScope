"""Integration tests: PlanExecutor vs process_frame equivalence."""

import sys
from pathlib import Path

import numpy as np
import pytest

build_dir = Path(__file__).resolve().parent.parent.parent / "build"
if str(build_dir) not in sys.path:
    sys.path.insert(0, str(build_dir))

# Add src to path for plan package
src_dir = Path(__file__).resolve().parent.parent.parent / "src" / "python"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import streetscope_simd  # noqa: E402
from plan.serializer import build_plan  # noqa: E402


@pytest.fixture(autouse=True)
def seed_rng():
    np.random.seed(42)


def test_identity_isp_matches_process_frame():
    """Identity ISP + zero alpha: display should equal frame."""
    h, w = 16, 8
    frame = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    bg_ref = np.random.uniform(0, 255, (h * w * 3,)).astype(np.float32)
    bg_plan = bg_ref.copy()

    # Reference: process_frame with identity ISP (no lut = identity default)
    mask_ref, display_ref = streetscope_simd.process_frame(frame, bg_ref, alpha=0.1, threshold=15)

    # Plan executor
    plan_bytes = build_plan(w, h)
    executor = streetscope_simd.PlanExecutor(plan_bytes)

    assert executor.width == w
    assert executor.height == h
    assert executor.arena_size > 0
    assert executor.num_stages == 7

    mask_plan, display_plan, _ = executor.run_frame(frame, bg_plan, alpha=0.1, threshold=15)

    np.testing.assert_array_equal(mask_plan, mask_ref)
    np.testing.assert_array_equal(display_plan, display_ref)
    np.testing.assert_allclose(bg_plan, bg_ref, atol=1e-5)


def test_full_isp_matches_process_frame():
    """Full ISP with non-trivial alpha map."""
    h, w = 16, 8
    frame = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    bg_ref = np.random.uniform(0, 255, (h * w * 3,)).astype(np.float32)
    bg_plan = bg_ref.copy()

    lut = np.arange(256, dtype=np.uint8)
    alpha_map = np.full(h * w, 0.5, dtype=np.float32)

    mask_ref, display_ref = streetscope_simd.process_frame(
        frame,
        bg_ref,
        alpha=0.1,
        threshold=15,
        lut=lut,
        gain_b=1.0,
        gain_g=1.0,
        gain_r=1.0,
        alpha_map=alpha_map,
    )

    plan_bytes = build_plan(w, h)
    executor = streetscope_simd.PlanExecutor(plan_bytes)

    mask_plan, display_plan, _ = executor.run_frame(
        frame,
        bg_plan,
        alpha=0.1,
        threshold=15,
        lut=lut,
        gain_b=1.0,
        gain_g=1.0,
        gain_r=1.0,
        alpha_map=alpha_map,
    )

    np.testing.assert_array_equal(mask_plan, mask_ref)
    np.testing.assert_array_equal(display_plan, display_ref)
    np.testing.assert_allclose(bg_plan, bg_ref, atol=1e-5)


def test_stage_timing():
    h, w = 16, 8
    frame = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    bg = np.random.uniform(0, 255, (h * w * 3,)).astype(np.float32)

    plan_bytes = build_plan(w, h)
    executor = streetscope_simd.PlanExecutor(plan_bytes)

    _, _, times = executor.run_frame(frame, bg, alpha=0.1, threshold=15, timing=True)

    assert times is not None
    assert len(times) == executor.num_stages
    for t in times:
        assert t > 0
