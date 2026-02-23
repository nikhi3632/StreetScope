#!/usr/bin/env python3
"""Combined pipeline viewer: detection + background + motion mask.

Displays three panels side-by-side:
  1. Live frame with YOLO detection boxes, class labels, confidence
  2. Background plate
  3. Motion mask

Bottom bar shows FPS, inference latency, warmup %, motion %, detection count.

Usage:
    python tools/pipeline_viewer.py --url URL
    python tools/pipeline_viewer.py --url URL --vehicles-only --conf 0.3

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
from src.python.core.background import BackgroundModel
from src.python.core.stream import FrameGrabber, StreamError, StreamMetrics
from src.python.perception.detector import YoloDetector

WINDOW_NAME = "StreetScope - Pipeline"
DEFAULT_MODEL_PATH = "models/yolov8n.onnx"

_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True


CLASS_COLORS = {
    "car": (0, 255, 0),
    "truck": (0, 165, 255),
    "bus": (0, 255, 255),
    "motorcycle": (255, 0, 255),
    "bicycle": (255, 255, 0),
}
DEFAULT_COLOR = (0, 200, 0)


def draw_detections(frame: np.ndarray, detections) -> np.ndarray:
    display = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = CLASS_COLORS.get(det.class_name, DEFAULT_COLOR)
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(display, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(display, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return display


def build_display(det_frame: np.ndarray, bg_model: BackgroundModel,
                  mask: np.ndarray, dets, infer_ms: float,
                  sm: StreamMetrics) -> np.ndarray:
    h, w = det_frame.shape[:2]

    # Background panel
    if bg_model.background is not None:
        bg_panel = bg_model.background
    else:
        bg_panel = np.full_like(det_frame, 64)

    # Motion mask as BGR
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Stack: detected frame | background | mask
    combined = np.hstack([det_frame, bg_panel, mask_color])

    # Scale to fit ~1280px wide (3 panels)
    ch, cw = combined.shape[:2]
    target_w = 1280
    scale = target_w / cw
    display = cv2.resize(combined, (int(cw * scale), int(ch * scale)),
                         interpolation=cv2.INTER_LINEAR)

    # Panel labels
    panel_w = int(w * scale)
    labels = ["Detection", "Background", "Motion Mask"]
    for i, label in enumerate(labels):
        x = i * panel_w + 5
        cv2.putText(display, label, (x, 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 0), 1, cv2.LINE_AA)

    # Metrics bar at bottom
    dh = display.shape[0]
    warmup_pct = min(100, bg_model.frame_count / bg_model.warmup_frames * 100)
    motion_pct = np.count_nonzero(mask) / mask.size * 100 if mask.size > 0 else 0

    line = (f"FPS: {sm.effective_fps:.1f}  "
            f"YOLO: {infer_ms:.1f}ms  "
            f"Det: {len(dets)}  "
            f"Warmup: {warmup_pct:.0f}%  "
            f"Motion: {motion_pct:.1f}%")
    cv2.putText(display, line, (5, dh - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

    return display


def run(url: str, model_path: str, vehicles_only: bool, conf: float,
        duration: int) -> None:
    global _shutdown_requested
    _shutdown_requested = False

    print(f"Probing stream: {url}")
    profile = probe_stream(url)
    print(f"  {profile.width}x{profile.height} @ {profile.frame_rate} fps")
    print()

    bg_model = BackgroundModel()

    try:
        import psutil
        process = psutil.Process(os.getpid())
        has_psutil = True
    except ImportError:
        has_psutil = False

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    start_time = time.monotonic()
    stop_reason = "unknown"
    total_detections = 0
    infer_times: list[float] = []
    motion_pcts: list[float] = []
    frames_displayed = 0

    detector = YoloDetector(model_path, conf_threshold=conf)
    # Warmup: first ONNX Runtime inference triggers JIT compilation
    dummy = np.zeros((profile.height, profile.width, 3), dtype=np.uint8)
    detector.detect(dummy)
    print(f"Model: {model_path} (conf={conf}, warmed up)")
    print("Quit: 'q'/Escape in window | close window | Ctrl+C")
    print()

    with FrameGrabber(url, realtime=True) as grabber:
        grabber.start()

        last_frame_num = -1

        try:
            while True:
                if _shutdown_requested:
                    stop_reason = "signal"
                    break

                # Check for stream error
                err = grabber.error
                if err is not None:
                    raise err

                # Get latest decoded frame (non-blocking)
                result = grabber.latest()
                if result is not None:
                    frame, fm = result

                    # Only process genuinely new frames
                    if fm.frame_number != last_frame_num:
                        last_frame_num = fm.frame_number

                        # Background + motion mask (< 1ms)
                        mask = bg_model.update(frame)
                        motion_pcts.append(np.count_nonzero(mask) / mask.size * 100)

                        # Synchronous detection (accurate boxes on this exact frame)
                        t0 = time.monotonic()
                        dets = detector.detect(frame, vehicles_only=vehicles_only)
                        infer_ms = (time.monotonic() - t0) * 1000
                        infer_times.append(infer_ms)
                        total_detections += len(dets)
                        frames_displayed += 1

                        # Compose and display
                        det_frame = draw_detections(frame, dets)
                        sm = grabber.metrics
                        display = build_display(det_frame, bg_model, mask, dets, infer_ms, sm)
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
    elapsed = time.monotonic() - start_time
    mem_mb = process.memory_info().rss / (1024 * 1024) if has_psutil else 0

    print(f"\nStopped: {stop_reason}")
    print(f"  Runtime: {elapsed:.1f}s")
    if sm.frames_decoded > 0:
        print()
        print("  Stream:")
        print(f"    Frames decoded:    {sm.frames_decoded}")
        print(f"    Effective FPS:     {sm.effective_fps:.1f}")
        print(f"    Decode latency:    {sm.avg_decode_ms:.2f} ms avg")
        if sm.min_interval_ms != float("inf"):
            print(f"    Frame interval:    {sm.avg_interval_ms:.1f} ms avg"
                  f"  ({sm.min_interval_ms:.1f} / {sm.max_interval_ms:.1f} min/max)")
        if sm.dropped_frames > 0:
            print(f"    Dropped frames:    {sm.dropped_frames}")
        print()
        print("  Detection:")
        print(f"    Total detections:  {total_detections}")
        if frames_displayed > 0:
            print(f"    Avg per frame:     {total_detections / frames_displayed:.1f}")
        if infer_times:
            print(f"    YOLO latency:      {sum(infer_times) / len(infer_times):.1f} ms avg"
                  f"  ({min(infer_times):.1f} / {max(infer_times):.1f} min/max)")
        print()
        print("  Background:")
        print(f"    Warmed up:         {bg_model.is_warmed_up}")
        print(f"    Frames ingested:   {bg_model.frame_count}")
        if motion_pcts:
            print(f"    Motion:            {sum(motion_pcts) / len(motion_pcts):.1f}% avg"
                  f"  ({min(motion_pcts):.1f} / {max(motion_pcts):.1f} min/max)")
        print()
        print("  System:")
        print(f"    Memory:            {mem_mb:.0f} MB")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Combined pipeline viewer")
    parser.add_argument("--url", required=True, help="HLS stream URL")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH,
                        help=f"ONNX model path (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--vehicles-only", action="store_true",
                        help="Only show vehicle detections")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--duration", type=int, default=0,
                        help="Duration in seconds (0 = unlimited)")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        run(args.url, args.model, args.vehicles_only, args.conf, args.duration)
    except Exception as e:
        logging.getLogger(__name__).error("Fatal: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
