import pytest

from src.python.core.stream import (
    FrameMetrics,
    StreamError,
    StreamMetrics,
    StreamOpenError,
    StreamReadError,
)


class TestFrameMetrics:
    def test_stores_values(self):
        fm = FrameMetrics(
            frame_number=42,
            decode_latency_ms=1.5,
            arrival_interval_ms=66.0,
            timestamp_s=1000.0,
        )
        assert fm.frame_number == 42
        assert fm.decode_latency_ms == 1.5
        assert fm.arrival_interval_ms == 66.0


class TestStreamMetrics:
    def test_empty_metrics(self):
        sm = StreamMetrics()
        assert sm.frames_decoded == 0
        assert sm.avg_decode_ms == 0.0
        assert sm.avg_interval_ms == 0.0
        assert sm.effective_fps == 0.0

    def test_single_frame(self):
        sm = StreamMetrics()
        fm = FrameMetrics(0, decode_latency_ms=2.0, arrival_interval_ms=0.0, timestamp_s=1.0)
        sm.update(fm)
        assert sm.frames_decoded == 1
        assert sm.avg_decode_ms == 2.0
        # First frame has interval 0, so not counted
        assert sm.avg_interval_ms == 0.0

    def test_multiple_frames(self):
        sm = StreamMetrics()
        sm.update(FrameMetrics(0, 2.0, 0.0, 1.0))
        sm.update(FrameMetrics(1, 3.0, 65.0, 1.065))
        sm.update(FrameMetrics(2, 1.0, 67.0, 1.132))
        assert sm.frames_decoded == 3
        assert sm.avg_decode_ms == pytest.approx(2.0, abs=0.01)
        assert sm.avg_interval_ms == pytest.approx(66.0, abs=0.01)
        assert sm.min_interval_ms == pytest.approx(65.0)
        assert sm.max_interval_ms == pytest.approx(67.0)

    def test_effective_fps(self):
        sm = StreamMetrics()
        sm.update(FrameMetrics(0, 1.0, 0.0, 0.0))
        sm.update(FrameMetrics(1, 1.0, 66.67, 0.067))
        sm.update(FrameMetrics(2, 1.0, 66.67, 0.133))
        assert sm.effective_fps == pytest.approx(15.0, abs=0.1)

    def test_dropped_frames_default_zero(self):
        sm = StreamMetrics()
        assert sm.dropped_frames == 0


class TestStreamErrors:
    def test_open_error_is_stream_error(self):
        assert issubclass(StreamOpenError, StreamError)

    def test_read_error_is_stream_error(self):
        assert issubclass(StreamReadError, StreamError)

    def test_stream_error_is_exception(self):
        assert issubclass(StreamError, Exception)
