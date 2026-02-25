"""Benchmark: Python/OpenCV vs scalar C++ vs NEON for the full ISP pipeline.

Measures four kernel tiers (Python, scalar, NEON) and three orchestration
tiers (separate pybind11 calls, fused process_frame, FrameLoop estimate).
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

build_dir = Path(__file__).resolve().parent.parent / "build"
if str(build_dir) not in sys.path:
    sys.path.insert(0, str(build_dir))

import streetscope_simd  # noqa: E402


def bench(name, fn, warmup=10, iterations=100):
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = (time.perf_counter() - start) / iterations * 1000
    print(f"  {name:45s} {elapsed:8.3f} ms")
    return elapsed


def main():
    h, w = 288, 512
    print(f"Frame size: {w}x{h} ({w * h * 3:,} bytes)\n")

    frame_f32 = np.random.uniform(0, 255, (h, w, 3)).astype(np.float32)
    bg_f32 = np.random.uniform(0, 255, (h, w, 3)).astype(np.float32)
    frame_u8 = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    bg_u8 = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

    # ── EMA Accumulator ──────────────────────────────────────
    print("=== EMA Accumulator ===")

    bg_py = bg_f32.copy()
    t_py = bench(
        "Python cv2.accumulateWeighted", lambda: cv2.accumulateWeighted(frame_f32, bg_py, 0.05)
    )

    bg_scalar = bg_f32.copy()
    t_scalar = bench(
        "C++ scalar",
        lambda: streetscope_simd.accumulate_ema_scalar(frame_f32, bg_scalar, alpha=0.05),
    )

    bg_neon = bg_f32.copy()
    t_neon = bench(
        "C++ NEON", lambda: streetscope_simd.accumulate_ema(frame_f32, bg_neon, alpha=0.05)
    )

    print(f"  {'Speedup scalar vs Python:':45s} {t_py / t_scalar:.1f}x")
    print(f"  {'Speedup NEON vs scalar:':45s} {t_scalar / t_neon:.1f}x")
    print()

    # ── Background Subtractor ────────────────────────────────
    print("=== Background Subtractor ===")

    t_py2 = bench(
        "Python absdiff+cvtColor+threshold",
        lambda: cv2.threshold(
            cv2.cvtColor(cv2.absdiff(frame_u8, bg_u8), cv2.COLOR_BGR2GRAY),
            15,
            255,
            cv2.THRESH_BINARY,
        )[1],
    )

    t_scalar2 = bench(
        "C++ scalar",
        lambda: streetscope_simd.subtract_background_scalar(frame_u8, bg_u8, threshold=15),
    )

    t_neon2 = bench(
        "C++ NEON", lambda: streetscope_simd.subtract_background(frame_u8, bg_u8, threshold=15)
    )

    print(f"  {'Speedup scalar vs Python:':45s} {t_py2 / t_scalar2:.1f}x")
    print(f"  {'Speedup NEON vs scalar:':45s} {t_scalar2 / t_neon2:.1f}x")
    print()

    # ── Fused AE+AWB ──────────────────────────────────────────
    print("=== Fused AE+AWB (Auto Exposure + Auto White Balance) ===")

    indices = np.arange(256, dtype=np.float64)
    lut = ((indices / 255.0) ** 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    gain_b, gain_g, gain_r = 0.8, 1.0, 1.3

    def python_ae_awb():
        ycrcb = cv2.cvtColor(frame_u8, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.LUT(ycrcb[:, :, 0], lut)
        corrected = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        gains = np.array([gain_b, gain_g, gain_r], dtype=np.float32)
        return (
            (corrected.astype(np.float32) * gains[np.newaxis, np.newaxis, :])
            .clip(0, 255)
            .astype(np.uint8)
        )

    t_py3 = bench("Python AE(YCrCb) + AWB (two passes)", python_ae_awb)
    t_scalar3 = bench(
        "C++ scalar (fused, BGR LUT)",
        lambda: streetscope_simd.apply_ae_awb_scalar(frame_u8, lut, gain_b, gain_g, gain_r),
    )
    t_neon3 = bench(
        "C++ NEON (fused, BGR LUT)",
        lambda: streetscope_simd.apply_ae_awb(frame_u8, lut, gain_b, gain_g, gain_r),
    )

    print(f"  {'Speedup scalar vs Python:':45s} {t_py3 / t_scalar3:.1f}x")
    print(f"  {'Speedup NEON vs scalar:':45s} {t_scalar3 / t_neon3:.1f}x")
    print()

    # ── AF Detail Blend ───────────────────────────────────────
    print("=== AF Detail Blend (Auto Focus sharpening) ===")

    blurred_u8 = cv2.GaussianBlur(frame_u8, (5, 5), 0)
    alpha_map = np.random.uniform(0, 1.5, (h, w)).astype(np.float32)

    def python_af_blend():
        detail = frame_u8.astype(np.float32) - blurred_u8.astype(np.float32)
        sharpened = frame_u8.astype(np.float32) + alpha_map[:, :, np.newaxis] * detail
        return sharpened.clip(0, 255).astype(np.uint8)

    t_py4 = bench("Python AF blend (numpy)", python_af_blend)
    t_scalar4 = bench(
        "C++ scalar",
        lambda: streetscope_simd.apply_af_blend_scalar(frame_u8, blurred_u8, alpha_map),
    )
    t_neon4 = bench(
        "C++ NEON", lambda: streetscope_simd.apply_af_blend(frame_u8, blurred_u8, alpha_map)
    )

    print(f"  {'Speedup scalar vs Python:':45s} {t_py4 / t_scalar4:.1f}x")
    print(f"  {'Speedup NEON vs scalar:':45s} {t_scalar4 / t_neon4:.1f}x")
    print()

    # ── Orchestration: separate → fused → FrameLoop ──────────
    print("=== Orchestration (pybind11 crossing overhead) ===")

    bg_separate = bg_f32.copy()
    bg_fused = bg_f32.copy()

    def separate_calls():
        streetscope_simd.accumulate_ema(frame_f32, bg_separate, alpha=0.05)
        bg_u8_tmp = bg_separate.clip(0, 255).astype(np.uint8)
        streetscope_simd.subtract_background(frame_u8, bg_u8_tmp, threshold=15)
        corrected = streetscope_simd.apply_ae_awb(frame_u8, lut, gain_b, gain_g, gain_r)
        blurred_tmp = cv2.GaussianBlur(corrected, (5, 5), 0)
        streetscope_simd.apply_af_blend(corrected, blurred_tmp, alpha_map)

    def fused_call():
        streetscope_simd.process_frame(
            frame_u8,
            bg_fused,
            alpha=0.05,
            threshold=15,
            lut=lut,
            gain_b=gain_b,
            gain_g=gain_g,
            gain_r=gain_r,
            alpha_map=alpha_map,
            blur_ksize=5,
        )

    t_separate = bench("separate calls (4 crossings + numpy)", separate_calls)
    t_fused = bench("process_frame  (1 crossing)", fused_call)

    # The difference between separate and fused is 3 extra crossings plus
    # numpy marshaling (clip/astype/GaussianBlur between calls). Estimate
    # per-crossing overhead from this, then subtract from fused to estimate
    # what FrameLoop achieves with zero crossings on the compute path.
    overhead_3x = t_separate - t_fused
    per_crossing = overhead_3x / 3.0
    t_frameloop_est = max(0.0, t_fused - per_crossing)

    print(f"  {'FrameLoop estimate (0 crossings)':45s} ~{t_frameloop_est:7.3f} ms")
    print()
    print(f"  {'Overhead per crossing:':45s} ~{per_crossing:7.3f} ms")
    print(f"  {'Fused saves vs separate:':45s} {overhead_3x:8.3f} ms")
    print(f"  {'FrameLoop saves vs fused:':45s} ~{per_crossing:7.3f} ms")
    print(f"  {'FrameLoop saves vs separate:':45s} ~{t_separate - t_frameloop_est:7.3f} ms")
    print()

    # ── Summary ───────────────────────────────────────────────
    print("=== Per-Frame Budget (NEON, 512x288) ===")
    print(f"  {'EMA accumulator:':45s} {t_neon:8.3f} ms")
    print(f"  {'Background subtract:':45s} {t_neon2:8.3f} ms")
    print(f"  {'AE+AWB:':45s} {t_neon3:8.3f} ms")
    print(f"  {'AF blend:':45s} {t_neon4:8.3f} ms")
    kernel_total = t_neon + t_neon2 + t_neon3 + t_neon4
    print(f"  {'─' * 45}{'─' * 9}")
    print(f"  {'Kernel total:':45s} {kernel_total:8.3f} ms")
    print(f"  {'process_frame (fused, via pybind11):':45s} {t_fused:8.3f} ms")
    print(f"  {'FrameLoop (fused, pure C++):':45s} ~{t_frameloop_est:7.3f} ms")
    print()
    target_ms = 1000.0 / 15.0
    print(f"  Target: {target_ms:.1f} ms/frame (15 fps)")
    print(f"  Headroom (FrameLoop): ~{target_ms - t_frameloop_est:.1f} ms for decode + tracking")
    print()


if __name__ == "__main__":
    main()
