#!/usr/bin/env python3
"""Live YOLO detection viewer with optional motion mask cross-validation.

Displays bounding boxes, class labels, confidence scores, and inference latency.
Optionally overlays the motion mask and flags agreement/disagreement.

Usage:
    python tools/detection_viewer.py --url URL
    python tools/detection_viewer.py --url URL --show-mask
    python tools/detection_viewer.py --url URL --vehicles-only --conf 0.3

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
from src.python.core.stream import FrameGrabber, StreamError
from src.python.perception.detector import YoloDetector
from src.python.perception.tracker import Tracker

WINDOW_NAME = "StreetScope - Detection"
DEFAULT_MODEL_PATH = "models/yolo11s.mlpackage"

shutdown_requested = False


def signal_handler(signum, frame):
    global shutdown_requested
    shutdown_requested = True


# Colors per class for visual distinction
CLASS_COLORS = {
    "car": (0, 255, 0),
    "truck": (0, 165, 255),
    "bus": (0, 255, 255),
    "motorcycle": (255, 0, 255),
    "bicycle": (255, 255, 0),
}
DEFAULT_COLOR = (0, 200, 0)


def draw_tracked(frame: np.ndarray, tracked_objects, infer_ms: float) -> np.ndarray:
    """Draw bounding boxes, track IDs, labels, trails, and inference time on frame."""
    display = frame.copy()

    for obj in tracked_objects:
        x1, y1, x2, y2 = obj.bbox
        color = CLASS_COLORS.get(obj.class_name, DEFAULT_COLOR)
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

        label = f"#{obj.track_id} {obj.class_name} {obj.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(display, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(
            display, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA
        )

        # Draw trajectory trail as fading polyline
        if len(obj.trail) >= 2:
            for j in range(1, len(obj.trail)):
                alpha = j / len(obj.trail)
                thickness = max(1, int(alpha * 2))
                c = tuple(int(v * alpha) for v in color)
                pt1 = (int(obj.trail[j - 1][0]), int(obj.trail[j - 1][1]))
                pt2 = (int(obj.trail[j][0]), int(obj.trail[j][1]))
                cv2.line(display, pt1, pt2, c, thickness, cv2.LINE_AA)

    cv2.putText(
        display,
        f"YOLO: {infer_ms:.1f}ms  Tracks: {len(tracked_objects)}",
        (5, display.shape[0] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    return display


def run(
    url: str, model_path: str, vehicles_only: bool, conf: float, show_mask: bool, duration: int
) -> None:
    global shutdown_requested
    shutdown_requested = False

    print(f"Probing stream: {url}")
    profile = probe_stream(url)
    print(f"  {profile.width}x{profile.height} @ {profile.frame_rate} fps")
    print()

    bg_model = BackgroundModel() if show_mask else None
    if show_mask:
        print("Motion mask cross-validation enabled")

    print("Quit: 'q'/Escape in window | close window | Ctrl+C")
    print()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    start_time = time.monotonic()
    stop_reason = "unknown"
    total_detections = 0
    frames_displayed = 0

    detector = YoloDetector(model_path, conf_threshold=conf)
    # Warmup
    dummy = np.zeros((profile.height, profile.width, 3), dtype=np.uint8)
    detector.detect(dummy)
    print(f"Model: {model_path} (conf={conf}, warmed up)")

    tracker = Tracker(frame_rate=int(profile.frame_rate))

    with FrameGrabber(url, realtime=True) as grabber:
        grabber.start()

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

                        # Optional: background update (synchronous)
                        if bg_model is not None:
                            mask, _ = bg_model.update(frame)

                        # Synchronous detection
                        t0 = time.monotonic()
                        dets = detector.detect(frame, vehicles_only=vehicles_only)
                        infer_ms = (time.monotonic() - t0) * 1000
                        total_detections += len(dets)
                        frames_displayed += 1

                        # Track detections
                        tracked = tracker.update(dets)

                        # Draw tracked objects
                        display = draw_tracked(frame, tracked, infer_ms)

                        # Scale up for visibility
                        h, w = display.shape[:2]
                        scale = max(2, 640 // w)
                        display = cv2.resize(
                            display, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST
                        )

                        # Optional: side-by-side with motion mask
                        if bg_model is not None:
                            mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                            mask_display = cv2.resize(
                                mask_color, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST
                            )
                            display = np.hstack([display, mask_display])

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
    if sm.frames_decoded > 0:
        print(f"  Frames: {sm.frames_decoded}")
        print(f"  Total detections: {total_detections}")
        if frames_displayed > 0:
            print(f"  Avg detections/frame: {total_detections / frames_displayed:.1f}")
        print(f"  Effective FPS: {sm.effective_fps:.1f}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Live YOLO detection viewer")
    parser.add_argument("--url", required=True, help="HLS stream URL")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help=f"Core ML model path (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument("--vehicles-only", action="store_true", help="Only show vehicle detections")
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--show-mask",
        action="store_true",
        help="Show motion mask side-by-side for cross-validation",
    )
    parser.add_argument(
        "--duration", type=int, default=0, help="Duration in seconds (0 = unlimited)"
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        run(args.url, args.model, args.vehicles_only, args.conf, args.show_mask, args.duration)
    except Exception as e:
        logging.getLogger(__name__).error("Fatal: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
