"""HLS stream ingest via OpenCV. Decodes frames and yields them with timing metadata."""

import logging
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class StreamError(Exception):
    """Base exception for stream errors."""


class StreamOpenError(StreamError):
    """Failed to open stream."""


class StreamReadError(StreamError):
    """Failed to read frame from stream."""


@dataclass
class FrameMetrics:
    """Per-frame timing and sequence metadata."""

    frame_number: int
    decode_latency_ms: float
    arrival_interval_ms: float
    timestamp_s: float


@dataclass
class StreamMetrics:
    """Accumulated stream-level metrics."""

    frames_decoded: int = 0
    total_decode_ms: float = 0.0
    min_interval_ms: float = float("inf")
    max_interval_ms: float = 0.0
    dropped_frames: int = 0
    intervals: list[float] = field(default_factory=list)

    def update(self, fm: FrameMetrics) -> None:
        self.frames_decoded += 1
        self.total_decode_ms += fm.decode_latency_ms
        if fm.arrival_interval_ms > 0:
            self.intervals.append(fm.arrival_interval_ms)
            self.min_interval_ms = min(self.min_interval_ms, fm.arrival_interval_ms)
            self.max_interval_ms = max(self.max_interval_ms, fm.arrival_interval_ms)

    @property
    def avg_decode_ms(self) -> float:
        if self.frames_decoded == 0:
            return 0.0
        return self.total_decode_ms / self.frames_decoded

    @property
    def avg_interval_ms(self) -> float:
        if not self.intervals:
            return 0.0
        return sum(self.intervals) / len(self.intervals)

    @property
    def effective_fps(self) -> float:
        avg = self.avg_interval_ms
        if avg <= 0:
            return 0.0
        return 1000.0 / avg


DEFAULT_FPS = 15.0
RETRY_SLEEP_S = 0.01


def decode_frames(url: str, realtime: bool = False, max_consecutive_failures: int = 30):
    """Generator that decodes frames from an HLS stream via OpenCV.

    Yields (numpy_bgr_frame, FrameMetrics) tuples.

    The generator cleans up the VideoCapture on exit regardless of how it
    terminates — normal exhaustion, GeneratorExit from the caller, or
    exception.

    Args:
        url: HLS stream URL.
        realtime: If True, pace output at the stream's frame rate.
        max_consecutive_failures: Number of consecutive read failures before
            raising StreamReadError. HLS streams can have transient gaps
            between segments.
    """
    logger.info("Opening stream: %s", url)
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise StreamOpenError(f"Failed to open stream: {url}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    target_interval_s = 1.0 / fps if fps > 0 else 1.0 / DEFAULT_FPS
    logger.info("Stream opened: fps=%.1f, target_interval=%.1fms", fps, target_interval_s * 1000)

    last_wall = time.monotonic()
    frame_num = 0
    consecutive_failures = 0

    try:
        while True:
            decode_start = time.monotonic()
            ret, frame = cap.read()
            decode_end = time.monotonic()

            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    logger.error("Stream lost: %d consecutive read failures", consecutive_failures)
                    raise StreamReadError(
                        f"Stream read failed {consecutive_failures} consecutive times"
                    )
                logger.debug(
                    "Frame read failed (%d/%d), retrying",
                    consecutive_failures,
                    max_consecutive_failures,
                )
                time.sleep(RETRY_SLEEP_S)
                continue

            consecutive_failures = 0
            decode_ms = (decode_end - decode_start) * 1000.0

            if realtime and frame_num > 0:
                elapsed = time.monotonic() - last_wall
                sleep_s = target_interval_s - elapsed
                if sleep_s > 0.001:
                    time.sleep(sleep_s)

            now = time.monotonic()
            interval_ms = (now - last_wall) * 1000.0

            metrics = FrameMetrics(
                frame_number=frame_num,
                decode_latency_ms=decode_ms,
                arrival_interval_ms=interval_ms if frame_num > 0 else 0.0,
                timestamp_s=now,
            )

            last_wall = now
            frame_num += 1

            yield frame, metrics
    finally:
        logger.info("Releasing stream after %d frames", frame_num)
        cap.release()


class FrameGrabber:
    """Threaded frame decoder. Continuously decodes and holds the latest frame.

    Uses a single-slot "latest frame" pattern — no queue, no backpressure.
    The consumer calls latest() which never blocks and always returns the
    most recently decoded frame (or None if no frame yet).
    """

    def __init__(
        self, url: str, max_consecutive_failures: int = 30, realtime: bool = False
    ) -> None:
        self.url = url
        self.max_consecutive_failures = max_consecutive_failures
        self.realtime = realtime

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

        self.latest_frame: tuple[np.ndarray, FrameMetrics] | None = None
        self.error_value: BaseException | None = None
        self.stream_metrics = StreamMetrics()

    def start(self) -> None:
        """Start the decode thread."""
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Signal the decode thread to stop and wait for it."""
        self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def latest(self) -> tuple[np.ndarray, FrameMetrics] | None:
        """Return the most recently decoded (frame, metrics) or None."""
        with self.lock:
            return self.latest_frame

    @property
    def error(self) -> BaseException | None:
        """Return the fatal error if the decode thread crashed, else None."""
        with self.lock:
            return self.error_value

    @property
    def metrics(self) -> StreamMetrics:
        """Return accumulated stream metrics."""
        with self.lock:
            return self.stream_metrics

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()

    def run(self) -> None:
        """Decode loop — runs in a daemon thread."""
        try:
            for frame, fm in decode_frames(
                self.url,
                realtime=self.realtime,
                max_consecutive_failures=self.max_consecutive_failures,
            ):
                if self.stop_event.is_set():
                    break
                with self.lock:
                    self.latest_frame = (frame, fm)
                    self.stream_metrics.update(fm)
        except Exception as exc:
            with self.lock:
                self.error_value = exc
