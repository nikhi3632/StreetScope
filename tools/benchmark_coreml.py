#!/usr/bin/env python3
"""Benchmark: Python coremltools vs native C++ CoreMLDetector.

Runs both detectors on synthetic frames and reports latency comparison.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = "models/yolo11s.mlpackage"
N_WARMUP = 5
N_ITERATIONS = 50
FRAME_H, FRAME_W = 288, 512


def benchmark_python():
    from src.python.perception.detector import YoloDetector

    detector = YoloDetector(MODEL_PATH, conf_threshold=0.25)

    # Warmup
    dummy = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    for _ in range(N_WARMUP):
        detector.detect(dummy)

    # Random frames for realistic memory access patterns
    frames = [
        np.random.randint(0, 255, (FRAME_H, FRAME_W, 3), dtype=np.uint8)
        for _ in range(N_ITERATIONS)
    ]

    times = []
    for frame in frames:
        t0 = time.monotonic()
        detector.detect(frame, vehicles_only=True)
        times.append((time.monotonic() - t0) * 1000)

    return times


def benchmark_native():
    from streetscope_pipeline import CoreMLDetector

    detector = CoreMLDetector(MODEL_PATH, conf_threshold=0.25)

    # Warmup
    dummy = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    for _ in range(N_WARMUP):
        detector.detect(dummy)

    frames = [
        np.random.randint(0, 255, (FRAME_H, FRAME_W, 3), dtype=np.uint8)
        for _ in range(N_ITERATIONS)
    ]

    times = []
    for frame in frames:
        t0 = time.monotonic()
        detector.detect(frame, vehicles_only=True)
        times.append((time.monotonic() - t0) * 1000)

    return times


def benchmark_zero_copy():
    from streetscope_pipeline import CoreMLDetector, create_test_pixelbuffer

    detector = CoreMLDetector(MODEL_PATH, conf_threshold=0.25)

    # Pre-allocate IOSurface-backed CVPixelBuffer capsules (auto-released by GC)
    pbs = [create_test_pixelbuffer(FRAME_W, FRAME_H) for _ in range(N_ITERATIONS)]

    # Warmup
    for i in range(N_WARMUP):
        detector.detect_pixelbuffer(pbs[i], vehicles_only=True)

    times = []
    for pb in pbs:
        t0 = time.monotonic()
        detector.detect_pixelbuffer(pb, vehicles_only=True)
        times.append((time.monotonic() - t0) * 1000)

    return times


def report(name, times):
    arr = np.array(times)
    print(f"  {name}:")
    print(
        f"    mean={arr.mean():.1f}ms  median={np.median(arr):.1f}ms  "
        f"p95={np.percentile(arr, 95):.1f}ms  min={arr.min():.1f}ms  max={arr.max():.1f}ms"
    )


def main():
    print(f"Benchmark: {N_ITERATIONS} iterations on {FRAME_W}x{FRAME_H} frames")
    print(f"Model: {MODEL_PATH}")
    print()

    print("Running Python (coremltools)...")
    py_times = benchmark_python()
    report("Python", py_times)
    print()

    try:
        print("Running Native (C++ CoreML, BGR path)...")
        native_times = benchmark_native()
        report("Native BGR", native_times)
        print()

        print("Running Native (C++ CoreML, zero-copy CVPixelBuffer)...")
        zc_times = benchmark_zero_copy()
        report("Native ZeroCopy", zc_times)
        print()

        py_mean = np.mean(py_times)
        native_mean = np.mean(native_times)
        zc_mean = np.mean(zc_times)

        print(
            f"Speedup vs Python:  BGR={py_mean / native_mean:.2f}x  ZeroCopy={py_mean / zc_mean:.2f}x"
        )
        print(f"ZeroCopy vs BGR:    {native_mean / zc_mean:.2f}x")
    except ImportError:
        print("Native detector not available (streetscope_pipeline not built)")
    except Exception as e:
        print(f"Native detector failed: {e}")


if __name__ == "__main__":
    main()
