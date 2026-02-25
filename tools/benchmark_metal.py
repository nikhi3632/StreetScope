#!/usr/bin/env python3
"""Benchmark: Python ISP vs NEON ISP vs Metal tone mapping.

Runs all paths on synthetic frames and reports latency comparison.
Also tests GPU/NE concurrency by dispatching Metal + CoreML simultaneously.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N_WARMUP = 5
N_ITERATIONS = 50
FRAME_H, FRAME_W = 288, 512
MODEL_PATH = "models/yolo11s.mlpackage"


def make_isp_params():
    """Create realistic ISP parameters for benchmarking."""
    # Gamma LUT: gamma=0.8 (brighten)
    lut = np.array([int(255 * (i / 255.0) ** 0.8) for i in range(256)], dtype=np.uint8)
    gains = np.array([0.95, 1.0, 1.05], dtype=np.float32)  # B, G, R
    return lut, gains


def benchmark_python_isp():
    """Python ISP: cv2.LUT + channel multiply."""
    import cv2

    lut, gains = make_isp_params()
    frames = [
        np.random.randint(0, 255, (FRAME_H, FRAME_W, 3), dtype=np.uint8)
        for _ in range(N_ITERATIONS)
    ]

    # Warmup
    for i in range(N_WARMUP):
        ycrcb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.LUT(ycrcb[:, :, 0], lut)
        cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    times = []
    for frame in frames:
        t0 = time.monotonic()
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.LUT(ycrcb[:, :, 0], lut)
        result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        result = (result.astype(np.float32) * gains).clip(0, 255).astype(np.uint8)
        times.append((time.monotonic() - t0) * 1000)

    return times


def benchmark_neon_isp():
    """NEON ISP via streetscope_simd."""
    import streetscope_simd

    lut, gains = make_isp_params()
    frames = [
        np.random.randint(0, 255, (FRAME_H, FRAME_W, 3), dtype=np.uint8)
        for _ in range(N_ITERATIONS)
    ]

    # Warmup
    for i in range(N_WARMUP):
        streetscope_simd.apply_ae_awb(
            frames[i], lut, float(gains[0]), float(gains[1]), float(gains[2])
        )

    times = []
    for frame in frames:
        t0 = time.monotonic()
        streetscope_simd.apply_ae_awb(frame, lut, float(gains[0]), float(gains[1]), float(gains[2]))
        times.append((time.monotonic() - t0) * 1000)

    return times


def benchmark_metal_bgr():
    """Metal tone mapping via MetalToneMapper (BGR path)."""
    from streetscope_pipeline import MetalToneMapper

    mapper = MetalToneMapper()
    frames = [
        np.random.randint(0, 255, (FRAME_H, FRAME_W, 3), dtype=np.uint8)
        for _ in range(N_ITERATIONS)
    ]

    # Warmup
    for i in range(N_WARMUP):
        mapper.tone_map(frames[i], exposure=1.5, white_point=2.0, gamma=2.2)

    times = []
    for frame in frames:
        t0 = time.monotonic()
        mapper.tone_map(frame, exposure=1.5, white_point=2.0, gamma=2.2)
        times.append((time.monotonic() - t0) * 1000)

    return times


def benchmark_metal_zerocopy():
    """Metal tone mapping via MetalToneMapper (CVPixelBuffer zero-copy path)."""
    from streetscope_pipeline import MetalToneMapper, create_test_pixelbuffer

    mapper = MetalToneMapper()
    pbs = [create_test_pixelbuffer(FRAME_W, FRAME_H) for _ in range(N_ITERATIONS)]

    # Warmup
    for i in range(N_WARMUP):
        mapper.tone_map_pixelbuffer(pbs[i], exposure=1.5, white_point=2.0, gamma=2.2)

    times = []
    for pb in pbs:
        t0 = time.monotonic()
        mapper.tone_map_pixelbuffer(pb, exposure=1.5, white_point=2.0, gamma=2.2)
        times.append((time.monotonic() - t0) * 1000)

    return times


def benchmark_concurrency():
    """Metal + CoreML dispatched together vs sequentially."""
    from streetscope_pipeline import CoreMLDetector, MetalToneMapper

    mapper = MetalToneMapper()
    detector = CoreMLDetector(MODEL_PATH, conf_threshold=0.25)

    frames = [
        np.random.randint(0, 255, (FRAME_H, FRAME_W, 3), dtype=np.uint8)
        for _ in range(N_ITERATIONS)
    ]

    # Warmup both
    for i in range(N_WARMUP):
        mapper.tone_map(frames[i], exposure=1.5, white_point=2.0, gamma=2.2)
        detector.detect(frames[i])

    # Sequential: Metal then CoreML
    seq_times = []
    for frame in frames:
        t0 = time.monotonic()
        mapper.tone_map(frame, exposure=1.5, white_point=2.0, gamma=2.2)
        detector.detect(frame, vehicles_only=True)
        seq_times.append((time.monotonic() - t0) * 1000)

    # Note: True async concurrency requires C++ level dispatch.
    # From Python, both calls are blocking. The C++ test_metal_tone_map.mm
    # validates actual GPU/NE overlap. Here we just report sequential baseline
    # for comparison with the C++ concurrency measurement.

    return seq_times


def report(name, times):
    arr = np.array(times)
    print(f"  {name}:")
    print(
        f"    mean={arr.mean():.2f}ms  median={np.median(arr):.2f}ms  "
        f"p95={np.percentile(arr, 95):.2f}ms  min={arr.min():.2f}ms  max={arr.max():.2f}ms"
    )


def main():
    print(f"Benchmark: {N_ITERATIONS} iterations on {FRAME_W}x{FRAME_H} frames")
    print()

    print("Running Python ISP (cv2.LUT + gains)...")
    py_times = benchmark_python_isp()
    report("Python ISP", py_times)
    print()

    try:
        print("Running NEON ISP (streetscope_simd.apply_ae_awb)...")
        neon_times = benchmark_neon_isp()
        report("NEON ISP", neon_times)
        print()
    except ImportError:
        print("NEON ISP not available (streetscope_simd not built)")
        neon_times = None
        print()

    try:
        print("Running Metal Tone Map (BGR path)...")
        metal_times = benchmark_metal_bgr()
        report("Metal BGR", metal_times)
        print()

        print("Running Metal Tone Map (CVPixelBuffer zero-copy)...")
        zc_times = benchmark_metal_zerocopy()
        report("Metal ZeroCopy", zc_times)
        print()

        py_mean = np.mean(py_times)
        metal_mean = np.mean(metal_times)
        zc_mean = np.mean(zc_times)

        print("Comparison:")
        print(f"  Metal BGR vs Python ISP:    {py_mean / metal_mean:.2f}x")
        print(f"  Metal ZeroCopy vs Python:   {py_mean / zc_mean:.2f}x")
        if neon_times is not None:
            neon_mean = np.mean(neon_times)
            print(f"  NEON ISP vs Python ISP:     {py_mean / neon_mean:.2f}x")
            print(f"  Metal BGR vs NEON ISP:      {neon_mean / metal_mean:.2f}x")
        print()

        if os.path.isdir(MODEL_PATH):
            print("Running Sequential (Metal + CoreML)...")
            seq_times = benchmark_concurrency()
            report("Sequential (Metal+YOLO)", seq_times)
            print()
            print(
                "Note: True GPU/NE concurrency requires C++ async dispatch.\n"
                "The C++ test (test_metal_tone_map) validates actual overlap."
            )
        else:
            print(f"Model not found at {MODEL_PATH} — skipping concurrency benchmark")

    except ImportError:
        print("Metal tone mapper not available (streetscope_pipeline not built)")
    except Exception as e:
        print(f"Metal benchmark failed: {e}")


if __name__ == "__main__":
    main()
