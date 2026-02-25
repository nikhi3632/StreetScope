#!/usr/bin/env python3
"""Live ISP 3A correction viewer.

Displays six panels (3x2 grid):
  Top:    Original | Auto Exposure | Auto White Balance
  Bottom: Auto Focus | Full ISP | Blur Map (heatmap)

Computes quality score and ISP parameters once background plate has warmed up.

Usage:
    python tools/isp_viewer.py --url URL
    python tools/isp_viewer.py --url URL --target-mean 140

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
from src.python.isp.estimator import ISPEstimator, ISPParams
from src.python.isp.quality import QualityAssessor, QualityScore

WINDOW_NAME = "StreetScope - ISP Viewer"

shutdown_requested = False


def signal_handler(signum, frame):
    global shutdown_requested
    shutdown_requested = True


def build_blur_heatmap(blur_map: np.ndarray, h: int, w: int) -> np.ndarray:
    """Render the blur map as a JET colormap heatmap at frame resolution."""
    bmax = float(blur_map.max())
    if bmax < 1e-6:
        return np.zeros((h, w, 3), dtype=np.uint8)
    normalized = (blur_map / bmax * 255).clip(0, 255).astype(np.uint8)
    upscaled = cv2.resize(normalized, (w, h), interpolation=cv2.INTER_NEAREST)
    return cv2.applyColorMap(upscaled, cv2.COLORMAP_JET)


def build_display(
    frame: np.ndarray,
    isp_params: ISPParams | None,
    quality: QualityScore | None,
    bg_model: BackgroundModel,
    sm: StreamMetrics,
) -> np.ndarray:
    """Compose six-panel (3x2) display with metrics overlay."""
    h, w = frame.shape[:2]

    if isp_params is not None:
        ae_frame = ISPEstimator.apply_auto_exposure(frame, isp_params.auto_exposure_lut)
        awb_frame = ISPEstimator.apply_auto_white_balance(
            frame, isp_params.auto_white_balance_gains
        )
        af_frame = ISPEstimator.apply_auto_focus(frame, isp_params.blur_map)
        full_frame = ISPEstimator.apply(frame, isp_params)
        blur_heat = build_blur_heatmap(isp_params.blur_map, h, w)
    else:
        ae_frame = frame.copy()
        awb_frame = frame.copy()
        af_frame = frame.copy()
        full_frame = frame.copy()
        blur_heat = np.zeros_like(frame)

    # 3x2 grid
    top = np.hstack([frame, ae_frame, awb_frame])
    bottom = np.hstack([af_frame, full_frame, blur_heat])
    combined = np.vstack([top, bottom])

    # Scale to fit ~1280px wide (3 columns)
    ch, cw = combined.shape[:2]
    target_w = 1280
    scale = target_w / cw
    display = cv2.resize(
        combined, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_LINEAR
    )

    # Panel labels
    panel_w = int(w * scale)
    panel_h = int(h * scale)
    labels = [
        (5, 15, "Original"),
        (panel_w + 5, 15, "Auto Exposure"),
        (panel_w * 2 + 5, 15, "Auto White Balance"),
        (5, panel_h + 15, "Auto Focus"),
        (panel_w + 5, panel_h + 15, "Full ISP"),
        (panel_w * 2 + 5, panel_h + 15, "Blur Map"),
    ]
    for x, y, label in labels:
        cv2.putText(
            display, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA
        )

    # Metrics at bottom
    dh = display.shape[0]
    warmup_pct = min(100, bg_model.frame_count / bg_model.warmup_frames * 100)

    if isp_params is not None and quality is not None:
        info = (
            f"FPS: {sm.effective_fps:.1f}  "
            f"Noise: {quality.noise_level:.1f}  "
            f"Block: {quality.blocking_severity:.2f}  "
            f"Blur: {quality.blur_level:.0f}  "
            f"DR: {quality.dynamic_range:.2f}  "
            f"WB: [{isp_params.auto_white_balance_gains[0]:.2f}, "
            f"{isp_params.auto_white_balance_gains[1]:.2f}, "
            f"{isp_params.auto_white_balance_gains[2]:.2f}]"
        )
    else:
        info = f"FPS: {sm.effective_fps:.1f}  Warmup: {warmup_pct:.0f}%"

    cv2.putText(
        display, info, (5, dh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA
    )

    return display


def run(url: str, target_mean: float, duration: int) -> None:
    global shutdown_requested
    shutdown_requested = False

    print(f"Probing stream: {url}")
    profile = probe_stream(url)
    print(f"  {profile.width}x{profile.height} @ {profile.frame_rate} fps")
    print(f"  Codec: {profile.codec}, Bitrate: {profile.bitrate_kbps} kbps")
    print()

    bg_model = BackgroundModel()
    assessor = QualityAssessor()
    estimator = ISPEstimator(target_mean=target_mean)

    isp_params: ISPParams | None = None
    quality: QualityScore | None = None

    print(f"ISP target mean: {target_mean}")
    print("Waiting for background warmup before computing ISP params...")
    print("Quit: 'q'/Escape in window | close window | Ctrl+C")
    print()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    start_time = time.monotonic()
    stop_reason = "unknown"

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

                    if fm.frame_number != last_frame_num:
                        last_frame_num = fm.frame_number
                        bg_model.update(frame)

                        # Compute ISP params once after warmup
                        if isp_params is None and bg_model.is_warmed_up:
                            quality = assessor.assess(bg_model.background_plate)
                            isp_params = estimator.estimate(bg_model.background_plate)
                            print("ISP parameters computed:")
                            print(f"  Noise: {quality.noise_level:.2f}")
                            print(f"  Blocking: {quality.blocking_severity:.2f}")
                            print(f"  Blur: {quality.blur_level:.1f}")
                            print(f"  Dynamic range: {quality.dynamic_range:.2f}")
                            print(f"  White balance gains: {isp_params.auto_white_balance_gains}")
                            print()

                        sm = grabber.metrics
                        display = build_display(
                            frame,
                            isp_params,
                            quality,
                            bg_model,
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

    print(f"\nStopped: {stop_reason}")
    print(f"  Frames processed: {bg_model.frame_count}")
    print(f"  ISP computed: {isp_params is not None}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="ISP 3A correction viewer")
    parser.add_argument("--url", required=True, help="HLS stream URL")
    parser.add_argument(
        "--target-mean",
        type=float,
        default=128.0,
        help="Tone correction target mean luminance (default 128)",
    )
    parser.add_argument(
        "--duration", type=int, default=0, help="Duration in seconds (0 = unlimited)"
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        run(args.url, args.target_mean, args.duration)
    except Exception as e:
        logging.getLogger(__name__).error("Fatal: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
