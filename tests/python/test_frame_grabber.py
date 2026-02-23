import time

import numpy as np

from src.python.core.stream import FrameGrabber, FrameMetrics, StreamMetrics


class TestFrameGrabberConstruction:
    def test_accepts_url(self):
        fg = FrameGrabber("http://example.com/stream.m3u8")
        assert fg._url == "http://example.com/stream.m3u8"

    def test_default_max_consecutive_failures(self):
        fg = FrameGrabber("http://example.com/stream.m3u8")
        assert fg._max_consecutive_failures == 30


class TestFrameGrabberBeforeStart:
    def test_latest_returns_none_before_start(self):
        fg = FrameGrabber("http://example.com/stream.m3u8")
        assert fg.latest() is None

    def test_error_returns_none_before_start(self):
        fg = FrameGrabber("http://example.com/stream.m3u8")
        assert fg.error is None

    def test_metrics_returns_empty_before_start(self):
        fg = FrameGrabber("http://example.com/stream.m3u8")
        sm = fg.metrics
        assert isinstance(sm, StreamMetrics)
        assert sm.frames_decoded == 0


class TestFrameGrabberStop:
    def test_stop_before_start_is_safe(self):
        fg = FrameGrabber("http://example.com/stream.m3u8")
        fg.stop()  # Should not raise

    def test_stop_is_idempotent(self):
        fg = FrameGrabber("http://example.com/stream.m3u8")
        fg.stop()
        fg.stop()  # Should not raise


class TestFrameGrabberContextManager:
    def test_context_manager_calls_stop(self):
        fg = FrameGrabber("http://example.com/stream.m3u8")
        with fg:
            pass
        # Should have called stop without error
        assert fg._stop_event.is_set()


class TestFrameGrabberWithFakeStream:
    """Test FrameGrabber by monkey-patching decode_frames."""

    def test_latest_returns_frame_after_decode(self, monkeypatch):
        """After the decode thread produces a frame, latest() should return it."""
        fake_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        fake_metrics = FrameMetrics(
            frame_number=0,
            decode_latency_ms=1.0,
            arrival_interval_ms=0.0,
            timestamp_s=time.monotonic(),
        )

        def fake_decode_frames(url, realtime=False, max_consecutive_failures=30):
            yield fake_frame, fake_metrics

        monkeypatch.setattr("src.python.core.stream.decode_frames", fake_decode_frames)

        fg = FrameGrabber("http://fake")
        fg.start()
        time.sleep(0.1)  # Let thread run

        result = fg.latest()
        assert result is not None
        frame, fm = result
        assert frame.shape == (240, 320, 3)
        assert fm.frame_number == 0
        fg.stop()

    def test_latest_returns_most_recent_frame(self, monkeypatch):
        """When multiple frames are produced, latest() returns the last one."""
        frames_produced = []

        def fake_decode_frames(url, realtime=False, max_consecutive_failures=30):
            for i in range(5):
                frame = np.full((240, 320, 3), i, dtype=np.uint8)
                fm = FrameMetrics(
                    frame_number=i,
                    decode_latency_ms=1.0,
                    arrival_interval_ms=66.0 if i > 0 else 0.0,
                    timestamp_s=time.monotonic(),
                )
                frames_produced.append(i)
                yield frame, fm

        monkeypatch.setattr("src.python.core.stream.decode_frames", fake_decode_frames)

        fg = FrameGrabber("http://fake")
        fg.start()
        time.sleep(0.2)  # Let all frames through
        fg.stop()

        result = fg.latest()
        assert result is not None
        frame, fm = result
        # Should be the last frame (pixel value 4)
        assert frame[0, 0, 0] == 4
        assert fm.frame_number == 4

    def test_metrics_accumulate(self, monkeypatch):
        """StreamMetrics should accumulate across decoded frames."""

        def fake_decode_frames(url, realtime=False, max_consecutive_failures=30):
            for i in range(3):
                fm = FrameMetrics(
                    frame_number=i,
                    decode_latency_ms=2.0,
                    arrival_interval_ms=66.0 if i > 0 else 0.0,
                    timestamp_s=time.monotonic(),
                )
                yield np.zeros((240, 320, 3), dtype=np.uint8), fm

        monkeypatch.setattr("src.python.core.stream.decode_frames", fake_decode_frames)

        fg = FrameGrabber("http://fake")
        fg.start()
        time.sleep(0.2)
        fg.stop()

        sm = fg.metrics
        assert sm.frames_decoded == 3

    def test_error_propagation(self, monkeypatch):
        """If decode_frames raises, error property should capture it."""
        from src.python.core.stream import StreamReadError

        def fake_decode_frames(url, realtime=False, max_consecutive_failures=30):
            raise StreamReadError("connection lost")

        monkeypatch.setattr("src.python.core.stream.decode_frames", fake_decode_frames)

        fg = FrameGrabber("http://fake")
        fg.start()
        time.sleep(0.1)

        assert fg.error is not None
        assert isinstance(fg.error, StreamReadError)
        fg.stop()
