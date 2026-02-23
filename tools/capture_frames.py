#!/usr/bin/env python3
"""Capture frames from an HLS stream and save to disk.

Usage:
    python tools/capture_frames.py --url URL --count 200 --output data/frames
    python tools/capture_frames.py --url URL --count 200 --interval 5  # every 5th frame
"""

import argparse
import json
import logging
import os
import signal
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.python.bootstrap.stream_discovery import probe_stream
from src.python.core.stream import StreamError, StreamMetrics, decode_frames

_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True


def run(url: str, count: int, interval: int, output: str) -> None:
    global _shutdown_requested
    _shutdown_requested = False

    os.makedirs(output, exist_ok=True)

    print(f"Probing stream: {url}")
    profile = probe_stream(url)
    print(f"  {profile.width}x{profile.height} @ {profile.frame_rate} fps, "
          f"{profile.codec}, {profile.bitrate_kbps} kbps")
    print(f"  Frame budget: {profile.frame_budget_ms:.1f} ms")
    print()

    profile_path = os.path.join(output, "stream_profile.json")
    with open(profile_path, "w") as f:
        json.dump({
            "url": url,
            "width": profile.width,
            "height": profile.height,
            "frame_rate": profile.frame_rate,
            "codec": profile.codec,
            "pixel_format": profile.pixel_format,
            "bitrate_kbps": profile.bitrate_kbps,
            "color_matrix": profile.color_matrix,
            "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)

    saved = 0
    total_frames = 0
    metrics = StreamMetrics()
    stop_reason = "complete"

    print(f"Capturing {count} frames (every {interval} frame(s))...")
    print("Press Ctrl+C to stop early.\n")

    try:
        for frame, fm in decode_frames(url):
            if _shutdown_requested:
                stop_reason = "interrupted"
                break

            metrics.update(fm)
            total_frames += 1

            if total_frames % interval == 0:
                filename = f"frame_{saved:05d}.png"
                filepath = os.path.join(output, filename)
                cv2.imwrite(filepath, frame)
                saved += 1

                if saved % 20 == 0:
                    print(f"  Saved {saved}/{count} frames "
                          f"(avg decode: {metrics.avg_decode_ms:.1f} ms)")

            if saved >= count:
                break

    except StreamError as e:
        stop_reason = f"stream error: {e}"
        logging.getLogger(__name__).error("Stream error: %s", e)

    except KeyboardInterrupt:
        stop_reason = "interrupted"

    print(f"\n{stop_reason.capitalize()}. Saved {saved} frames to {output}/")
    print(f"  Total decoded: {total_frames}")
    if metrics.frames_decoded > 0:
        print(f"  Avg decode latency: {metrics.avg_decode_ms:.2f} ms")
        print(f"  Avg frame interval: {metrics.avg_interval_ms:.2f} ms")
        print(f"  Effective FPS: {metrics.effective_fps:.1f}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Capture frames from HLS stream")
    parser.add_argument("--url", required=True, help="HLS stream URL")
    parser.add_argument("--count", type=int, default=200, help="Number of frames to capture")
    parser.add_argument("--interval", type=int, default=1, help="Save every Nth frame")
    parser.add_argument("--output", default="data/frames", help="Output directory")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        run(args.url, args.count, args.interval, args.output)
    except Exception as e:
        logging.getLogger(__name__).error("Fatal: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
