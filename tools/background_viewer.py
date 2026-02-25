#!/usr/bin/env python3
"""Live background plate and motion mask viewer.

Displays three panels side-by-side: live frame | background plate | motion mask.
Shows warmup progress and per-frame metrics.

Usage:
    python tools/background_viewer.py --url URL
    python tools/background_viewer.py --url URL --alpha 0.03 --threshold 25

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
from src.python.core.stabilizer import BackgroundModel
from src.python.core.stream import FrameGrabber, StreamError, StreamMetrics

WINDOW_NAME = "StreetScope - Background Viewer"

shutdown_requested = False


def signal_handler(signum, frame):
    global shutdown_requested
    shutdown_requested = True


def build_display(
    frame: np.ndarray, bg_model: BackgroundModel, mask: np.ndarray, fm, sm: StreamMetrics
) -> np.ndarray:
    """Compose three-panel display with metrics overlay."""
    h, w = frame.shape[:2]

    # Background panel (gray placeholder until first frame)
    if bg_model.background is not None:
        bg_panel = bg_model.background
    else:
        bg_panel = np.full_like(frame, 64)

    # Motion mask as 3-channel for stacking
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Stack horizontally: live | background | mask
    combined = np.hstack([frame, bg_panel, mask_color])

    # Scale up for visibility
    ch, cw = combined.shape[:2]
    scale = max(1, 640 // w)
    display = cv2.resize(combined, (cw * scale, ch * scale), interpolation=cv2.INTER_NEAREST)

    # Labels (top-right of each panel)
    panel_w = w * scale
    labels = ["Live", "Background", "Motion Mask"]
    for i, label in enumerate(labels):
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        x = (i + 1) * panel_w - tw - 5
        cv2.putText(
            display, label, (x, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA
        )

    # Metrics at bottom
    dh = display.shape[0]
    warmup_pct = min(100, bg_model.frame_count / bg_model.warmup_frames * 100)
    motion_pct = np.count_nonzero(mask) / mask.size * 100 if mask.size > 0 else 0

    lines = [
        f"Frame: {fm.frame_number}  FPS: {sm.effective_fps:.1f}  "
        f"Warmup: {warmup_pct:.0f}%  Motion: {motion_pct:.1f}%",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            display,
            line,
            (5, dh - 8 - i * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return display


def run(url: str, alpha: float, threshold: int, warmup: int, duration: int) -> None:
    global shutdown_requested
    shutdown_requested = False

    print(f"Probing stream: {url}")
    profile = probe_stream(url)
    print(f"  {profile.width}x{profile.height} @ {profile.frame_rate} fps")
    print(f"  Codec: {profile.codec}, Bitrate: {profile.bitrate_kbps} kbps")
    print()

    bg_model = BackgroundModel(alpha=alpha, threshold=threshold, warmup_frames=warmup)
    print(f"Background model: alpha={alpha}, threshold={threshold}, warmup={warmup} frames")
    print("Quit: 'q'/Escape in window | close window | Ctrl+C")
    print()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    start_time = time.monotonic()
    stop_reason = "unknown"

    with FrameGrabber(url, realtime=True) as grabber:
        grabber.start()

        last_frame = None
        last_mask = None
        last_fm = None
        last_frame_num = -1

        try:
            while True:
                if shutdown_requested:
                    stop_reason = "signal"
                    break

                err = grabber.error
                if err is not None:
                    raise err

                result = grabber.latest()
                if result is not None:
                    frame, fm = result

                    # Only process genuinely new frames
                    if fm.frame_number != last_frame_num:
                        last_frame_num = fm.frame_number
                        last_frame = frame
                        last_fm = fm

                        # Background update (synchronous, < 1ms)
                        mask, _ = bg_model.update(frame)
                        last_mask = mask

                        sm = grabber.metrics
                        display = build_display(
                            last_frame,
                            bg_model,
                            last_mask
                            if last_mask is not None
                            else np.zeros(last_frame.shape[:2], dtype=np.uint8),
                            last_fm,
                            sm,
                        )
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

                if duration > 0 and (time.monotonic() - start_time) >= duration:
                    stop_reason = f"duration limit ({duration}s)"
                    break

        except StreamError as e:
            stop_reason = f"stream error: {e}"
            logging.getLogger(__name__).error("Stream error: %s", e)

        except KeyboardInterrupt:
            stop_reason = "interrupted"

        finally:
            cv2.destroyAllWindows()
            for _ in range(5):
                cv2.waitKey(1)

    sm = grabber.metrics
    print(f"\nStopped: {stop_reason}")
    print(f"  Frames processed: {bg_model.frame_count}")
    print(f"  Warmed up: {bg_model.is_warmed_up}")
    if sm.frames_decoded > 0:
        print(f"  Avg decode latency: {sm.avg_decode_ms:.2f} ms")
        print(f"  Effective FPS: {sm.effective_fps:.1f}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Background plate and motion mask viewer")
    parser.add_argument("--url", required=True, help="HLS stream URL")
    parser.add_argument("--alpha", type=float, default=0.05, help="EMA learning rate")
    parser.add_argument("--threshold", type=int, default=15, help="Motion threshold (0-255)")
    parser.add_argument("--warmup", type=int, default=60, help="Warmup frames")
    parser.add_argument(
        "--duration", type=int, default=0, help="Duration in seconds (0 = unlimited)"
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        run(args.url, args.alpha, args.threshold, args.warmup, args.duration)
    except Exception as e:
        logging.getLogger(__name__).error("Fatal: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
