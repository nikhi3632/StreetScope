#!/usr/bin/env python3
"""Combined pipeline viewer: stabilization + background + ISP + detection + LK tracking.

Displays two panels side-by-side:
  1. Original — raw camera frame (unprocessed)
  2. Combined — ISP-corrected stabilized frame with tracked objects and trails

The full pipeline: stabilize -> background model -> ISP (AE+AWB+AF) ->
YOLO detect -> motion filter -> LK track. ISP corrections are computed once
after background warmup and applied to the display path only. Detection and
tracking run on the raw stabilized frame (perception path).

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
from src.python.core.stabilizer import BackgroundModel, FrameStabilizer
from src.python.core.stream import FrameGrabber, StreamError, StreamMetrics
from src.python.isp.estimator import ISPEstimator, ISPParams
from src.python.perception.detector import YoloDetector
from src.python.perception.tracker import Tracker

# Try to import SIMD module for accelerated ISP
_simd_module = None
try:
    import streetscope_simd

    _simd_module = streetscope_simd
except ImportError:
    pass

WINDOW_NAME = "StreetScope - Pipeline"
DEFAULT_MODEL_PATH = "models/yolo11s.mlpackage"

shutdown_requested = False


def signal_handler(signum, frame):
    global shutdown_requested
    logging.getLogger(__name__).debug(
        "Signal %d at line %s", signum, frame.f_lineno if frame else "?"
    )
    shutdown_requested = True


CLASS_COLORS = {
    "car": (0, 255, 0),
    "truck": (0, 165, 255),
    "bus": (0, 255, 255),
    "motorcycle": (255, 0, 255),
    "bicycle": (255, 255, 0),
}
DEFAULT_COLOR = (0, 200, 0)


def filter_by_motion(detections, mask: np.ndarray, min_overlap: float = 0.05):
    """Keep only detections that overlap sufficiently with the motion mask.

    A detection is considered moving if at least `min_overlap` fraction of
    its bounding box pixels are non-zero in the motion mask. This filters
    out stationary objects like parked cars.
    """
    moving = []
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        h, w = mask.shape[:2]
        x1c = max(0, min(x1, w))
        y1c = max(0, min(y1, h))
        x2c = max(0, min(x2, w))
        y2c = max(0, min(y2, h))
        area = (x2c - x1c) * (y2c - y1c)
        if area <= 0:
            continue
        motion_pixels = np.count_nonzero(mask[y1c:y2c, x1c:x2c])
        if motion_pixels / area >= min_overlap:
            moving.append(det)
    return moving


def draw_combined(frame: np.ndarray, tracked_objects) -> np.ndarray:
    """Draw tracked objects with bounding boxes and IDs."""
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
    return display


def build_display(
    original: np.ndarray,
    combined: np.ndarray,
    infer_ms: float,
    sm: StreamMetrics,
    num_tracks: int,
    num_dets: int,
    motion_pct: float,
    isp_active: bool = False,
) -> np.ndarray:
    w = original.shape[1]

    # Two panels: Original | Combined
    display = np.hstack([original, combined])

    # Scale to fit ~1280px wide (2 panels)
    dh, dw = display.shape[:2]
    target_w = 1280
    scale = target_w / dw
    display = cv2.resize(
        display, (int(dw * scale), int(dh * scale)), interpolation=cv2.INTER_LINEAR
    )

    # Panel labels (top-right of each panel)
    panel_w = int(w * scale)
    labels = ["Original", "Pipeline"]
    for i, label in enumerate(labels):
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        x = (i + 1) * panel_w - tw - 5
        cv2.putText(
            display, label, (x, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA
        )

    # Metrics bar at bottom
    out_h = display.shape[0]
    isp_tag = "ISP: on" if isp_active else "ISP: warmup"
    line = (
        f"FPS: {sm.effective_fps:.1f}  "
        f"YOLO: {infer_ms:.1f}ms  "
        f"Det: {num_dets}  Tracks: {num_tracks}  "
        f"Motion: {motion_pct:.1f}%  {isp_tag}"
    )
    cv2.putText(
        display, line, (5, out_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA
    )

    return display


def run(url: str, model_path: str, vehicles_only: bool, conf: float, duration: int) -> None:
    global shutdown_requested
    shutdown_requested = False

    print(f"Probing stream: {url}")
    profile = probe_stream(url)
    print(f"  {profile.width}x{profile.height} @ {profile.frame_rate} fps")
    print()

    try:
        import psutil

        process = psutil.Process(os.getpid())
        has_psutil = True
    except ImportError:
        has_psutil = False

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    start_time = time.monotonic()
    stop_reason = "unknown"
    total_raw_detections = 0
    total_detections = 0
    infer_times: list[float] = []
    frames_displayed = 0

    # Tracking quality metrics
    all_confidences: list[float] = []
    track_first_seen: dict[int, float] = {}
    track_last_seen: dict[int, float] = {}
    prev_track_ids: set[int] = set()
    continuity_scores: list[float] = []

    stabilizer = FrameStabilizer()
    bg_model = BackgroundModel()
    detector = YoloDetector(model_path, conf_threshold=conf)
    # Warmup: first Core ML inference triggers compilation
    dummy = np.zeros((profile.height, profile.width, 3), dtype=np.uint8)
    detector.detect(dummy)
    tracker = Tracker(frame_rate=int(profile.frame_rate))
    estimator = ISPEstimator()
    isp_params: ISPParams | None = None
    total_tracks = 0

    print(f"Model: {model_path} (conf={conf}, warmed up)")
    print("Pipeline: detect → track")
    print("Quit: 'q'/Escape in window | close window | Ctrl+C")
    print()

    with FrameGrabber(url, realtime=True) as grabber:
        grabber.start()

        last_frame_num = -1

        try:
            while True:
                if shutdown_requested:
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

                        # Stabilize frame (cancel camera shake)
                        stabilized, _ = stabilizer.stabilize(frame)

                        # Background model on stabilized frame
                        mask = bg_model.update(stabilized)
                        motion_pct = (
                            np.count_nonzero(mask) / mask.size * 100 if mask.size > 0 else 0
                        )

                        # Synchronous detection on stabilized frame
                        t0 = time.monotonic()
                        raw_dets = detector.detect(stabilized, vehicles_only=vehicles_only)
                        infer_ms = (time.monotonic() - t0) * 1000
                        infer_times.append(infer_ms)
                        frames_displayed += 1

                        # Compute ISP params once after warmup
                        if isp_params is None and bg_model.is_warmed_up:
                            isp_params = estimator.estimate(bg_model.background_plate)

                        # Filter out stationary detections using motion mask
                        total_raw_detections += len(raw_dets)
                        if bg_model.is_warmed_up:
                            dets = filter_by_motion(raw_dets, mask)
                        else:
                            dets = raw_dets  # Pass all during warmup
                        total_detections += len(dets)

                        # LK tracking on stabilized frame
                        tracked = tracker.update(stabilized, dets)
                        total_tracks = max(
                            total_tracks, max((t.track_id for t in tracked), default=0)
                        )

                        # Collect quality metrics
                        now = time.monotonic()
                        current_ids = {t.track_id for t in tracked}
                        for t in tracked:
                            all_confidences.append(t.confidence)
                            if t.track_id not in track_first_seen:
                                track_first_seen[t.track_id] = now
                            track_last_seen[t.track_id] = now
                        if prev_track_ids and current_ids:
                            persisted = len(prev_track_ids & current_ids)
                            continuity_scores.append(persisted / len(prev_track_ids))
                        prev_track_ids = current_ids

                        # Panel 1: raw original frame
                        # Panel 2: ISP-corrected stabilized + tracked objects
                        display_frame = stabilized
                        if isp_params is not None:
                            if _simd_module is not None:
                                display_frame = ISPEstimator.apply_simd(
                                    stabilized, isp_params, _simd_module
                                )
                            else:
                                display_frame = ISPEstimator.apply(stabilized, isp_params)
                        combined_frame = draw_combined(display_frame, tracked)
                        sm = grabber.metrics
                        display = build_display(
                            frame,
                            combined_frame,
                            infer_ms,
                            sm,
                            num_tracks=len(tracked),
                            num_dets=len(dets),
                            motion_pct=motion_pct,
                            isp_active=isp_params is not None,
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
            print(
                f"    Frame interval:    {sm.avg_interval_ms:.1f} ms avg "
                f" ({sm.min_interval_ms:.1f} / {sm.max_interval_ms:.1f} min/max)"
            )
        if sm.dropped_frames > 0:
            print(f"    Dropped frames:    {sm.dropped_frames}")
        print()
        print("  Detection:")
        print(f"    Raw detections:    {total_raw_detections}")
        print(f"    After motion filt: {total_detections}")
        filtered = total_raw_detections - total_detections
        if total_raw_detections > 0:
            print(
                f"    Filtered out:      {filtered}"
                f" ({filtered * 100 // total_raw_detections}% stationary)"
            )
        if frames_displayed > 0:
            print(f"    Avg per frame:     {total_detections / frames_displayed:.1f}")
        if infer_times:
            print(
                f"    YOLO latency:      {sum(infer_times) / len(infer_times):.1f} ms avg "
                f" ({min(infer_times):.1f} / {max(infer_times):.1f} min/max)"
            )
        print()
        print("  Tracking:")
        print(f"    Unique tracks:     {total_tracks}")
        if track_first_seen:
            durations = [track_last_seen[tid] - track_first_seen[tid] for tid in track_first_seen]
            avg_dur = sum(durations) / len(durations)
            long_tracks = sum(1 for d in durations if d >= 1.0)
            print(f"    Avg track duration: {avg_dur:.1f}s")
            print(
                f"    Tracks > 1s:       {long_tracks}/{len(durations)}"
                f" ({long_tracks * 100 // len(durations)}%)"
            )
        if continuity_scores:
            avg_cont = sum(continuity_scores) / len(continuity_scores)
            print(f"    Frame continuity:  {avg_cont:.0%}")
        if all_confidences:
            sorted_conf = sorted(all_confidences)
            avg_conf = sum(sorted_conf) / len(sorted_conf)
            median_conf = sorted_conf[len(sorted_conf) // 2]
            print(f"    Avg confidence:    {avg_conf:.2f}")
            print(f"    Median confidence: {median_conf:.2f}")
        print()
        print("  ISP 3A:")
        if isp_params is not None:
            lut = isp_params.auto_exposure_lut
            gains = isp_params.auto_white_balance_gains
            bmap = isp_params.blur_map
            print("    Auto Exposure (tone mapping):")
            print(f"      Gamma LUT range: {lut.min()}-{lut.max()}")
            print("    Auto White Balance (color correction):")
            print(f"      Gains [B,G,R]:   [{gains[0]:.3f}, {gains[1]:.3f}, {gains[2]:.3f}]")
            print("    Auto Focus (sharpening):")
            print(f"      Blur map range:  {bmap.min():.0f} / {bmap.max():.0f} min/max")
            print(f"    SIMD accelerated:  {'yes' if _simd_module is not None else 'no'}")
        else:
            print("    Not computed (warmup incomplete)")
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
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help=f"Core ML model path (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument("--vehicles-only", action="store_true", help="Only show vehicle detections")
    parser.add_argument(
        "--conf", type=float, default=0.20, help="Confidence threshold (default: 0.20)"
    )
    parser.add_argument(
        "--duration", type=int, default=0, help="Duration in seconds (0 = unlimited)"
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        run(args.url, args.model, args.vehicles_only, args.conf, args.duration)
    except Exception as e:
        logging.getLogger(__name__).error("Fatal: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
