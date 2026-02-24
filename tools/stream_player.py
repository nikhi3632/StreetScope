#!/usr/bin/env python3
"""Live stream player with real-time metrics overlay.

Usage:
    python tools/stream_player.py --url URL
    python tools/stream_player.py --url URL --duration 600  # 10 minutes

Press 'q' or Escape in the video window, close the window, or Ctrl+C to quit.
"""

import argparse
import logging
import os
import signal
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.python.bootstrap.stream_discovery import probe_stream
from src.python.core.stream import (
    StreamError,
    StreamMetrics,
    decode_frames,
)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

WINDOW_NAME = "StreetScope"

# Signal flag for clean shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    global shutdown_requested
    shutdown_requested = True


def draw_metrics_overlay(frame: np.ndarray, fm, sm: StreamMetrics, mem_mb: float) -> np.ndarray:
    """Draw real-time metrics on the frame."""
    display = frame.copy()
    h, w = display.shape[:2]

    scale = max(2, 640 // w)
    display = cv2.resize(display, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    lines = [
        f"Frame: {fm.frame_number}",
        f"Decode: {fm.decode_latency_ms:.1f} ms",
        f"Interval: {fm.arrival_interval_ms:.1f} ms",
        f"Avg FPS: {sm.effective_fps:.1f}",
        f"Avg decode: {sm.avg_decode_ms:.1f} ms",
        f"Memory: {mem_mb:.0f} MB",
    ]

    y = 20
    for line in lines:
        cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                     0.5, (0, 255, 0), 1, cv2.LINE_AA)
        y += 20

    return display


def get_memory_mb() -> float:
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    return 0.0


def print_summary(sm: StreamMetrics) -> None:
    print("\nSession summary:")
    print(f"  Frames decoded: {sm.frames_decoded}")
    if sm.frames_decoded > 0:
        print(f"  Avg decode latency: {sm.avg_decode_ms:.2f} ms")
        print(f"  Avg frame interval: {sm.avg_interval_ms:.2f} ms")
        if sm.min_interval_ms != float("inf"):
            print(f"  Min/Max interval: {sm.min_interval_ms:.1f} / {sm.max_interval_ms:.1f} ms")
        print(f"  Effective FPS: {sm.effective_fps:.1f}")
    if sm.dropped_frames > 0:
        print(f"  Dropped frames: {sm.dropped_frames}")
    print(f"  Final memory: {get_memory_mb():.0f} MB")


def run(url: str, duration: int = 0) -> None:
    global shutdown_requested
    shutdown_requested = False

    print(f"Probing stream: {url}")
    profile = probe_stream(url)
    print(f"  {profile.width}x{profile.height} @ {profile.frame_rate} fps")
    print(f"  Codec: {profile.codec}, Bitrate: {profile.bitrate_kbps} kbps")
    print(f"  Frame budget: {profile.frame_budget_ms:.1f} ms")
    print()
    print("Quit: 'q'/Escape in window | close window | Ctrl+C")
    print()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    sm = StreamMetrics()
    start_time = time.monotonic()
    stop_reason = "unknown"

    try:
        for frame, fm in decode_frames(url, realtime=True):
            if shutdown_requested:
                stop_reason = "signal"
                break

            sm.update(fm)
            display = draw_metrics_overlay(frame, fm, sm, get_memory_mb())
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                stop_reason = "user quit"
                break

            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    stop_reason = "window closed"
                    break
            except cv2.error:
                stop_reason = "window closed"
                break

            if duration > 0:
                elapsed = time.monotonic() - start_time
                if elapsed >= duration:
                    stop_reason = f"duration limit ({duration}s)"
                    break

        else:
            stop_reason = "stream ended"

    except StreamError as e:
        stop_reason = f"stream error: {e}"
        logging.getLogger(__name__).error("Stream error: %s", e)

    except KeyboardInterrupt:
        stop_reason = "interrupted"

    finally:
        cv2.destroyAllWindows()
        # Process remaining events so the window actually closes on macOS
        for _ in range(5):
            cv2.waitKey(1)

    print(f"\nStopped: {stop_reason}")
    print_summary(sm)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Live HLS stream player with metrics")
    parser.add_argument("--url", required=True, help="HLS stream URL")
    parser.add_argument("--duration", type=int, default=0,
                        help="Duration in seconds (0 = unlimited)")
    args = parser.parse_args()

    # Install signal handlers for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        run(args.url, args.duration)
    except Exception as e:
        logging.getLogger(__name__).error("Fatal: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
