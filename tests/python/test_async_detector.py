import time

import numpy as np

from src.python.perception.detector import AsyncDetector, Detection

MODEL_PATH = "models/yolo11s.mlpackage"


def poll_latest(det, timeout=10.0, interval=0.1):
    """Poll det.latest() until non-None or timeout (seconds)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = det.latest()
        if result is not None:
            return result
        time.sleep(interval)
    return None


class TestAsyncDetectorConstruction:
    def test_accepts_model_path(self):
        det = AsyncDetector(MODEL_PATH)
        assert det.model_path == MODEL_PATH

    def test_default_thresholds(self):
        det = AsyncDetector(MODEL_PATH)
        assert det.conf_threshold == 0.25
        assert det.iou_threshold == 0.45

    def test_vehicles_only_default_false(self):
        det = AsyncDetector(MODEL_PATH)
        assert det.vehicles_only is False


class TestAsyncDetectorBeforeStart:
    def test_latest_returns_none_before_start(self):
        det = AsyncDetector(MODEL_PATH)
        assert det.latest() is None


class TestAsyncDetectorStop:
    def test_stop_before_start_is_safe(self):
        det = AsyncDetector(MODEL_PATH)
        det.stop()  # Should not raise

    def test_stop_is_idempotent(self):
        det = AsyncDetector(MODEL_PATH)
        det.stop()
        det.stop()


class TestAsyncDetectorContextManager:
    def test_context_manager_calls_stop(self):
        det = AsyncDetector(MODEL_PATH)
        with det:
            pass
        assert det.stop_event.is_set()


class TestAsyncDetectorInference:
    def test_submit_and_get_result(self):
        det = AsyncDetector(MODEL_PATH)
        det.start()

        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        det.submit(frame)

        result = poll_latest(det)
        assert result is not None
        dets, infer_ms = result
        assert isinstance(dets, list)
        assert infer_ms > 0
        det.stop()

    def test_latest_returns_detections_list(self):
        det = AsyncDetector(MODEL_PATH)
        det.start()

        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        det.submit(frame)

        result = poll_latest(det)
        assert result is not None
        dets, _ = result
        for d in dets:
            assert isinstance(d, Detection)
        det.stop()

    def test_vehicles_only_filter(self):
        det = AsyncDetector(MODEL_PATH, vehicles_only=True)
        det.start()

        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        det.submit(frame)

        result = poll_latest(det)
        if result is not None:
            dets, _ = result
            from src.python.perception.detector import COCO_VEHICLE_CLASSES

            for d in dets:
                assert d.class_id in COCO_VEHICLE_CLASSES
        det.stop()

    def test_submit_overwrites_pending(self):
        """Submitting a new frame while busy should replace the pending frame."""
        det = AsyncDetector(MODEL_PATH)
        det.start()

        # Submit two frames rapidly — second should overwrite first in pending
        frame1 = np.zeros((240, 320, 3), dtype=np.uint8)
        frame2 = np.full((240, 320, 3), 128, dtype=np.uint8)
        det.submit(frame1)
        det.submit(frame2)  # Overwrites pending

        # Should not crash, result should be valid
        result = poll_latest(det)
        assert result is not None
        det.stop()
