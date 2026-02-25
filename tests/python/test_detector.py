from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from src.python.perception.detector import (
    COCO_VEHICLE_CLASSES,
    Detection,
    YoloDetector,
    postprocess,
    preprocess,
)

MODEL_PATH = "models/yolo11s.mlpackage"


class TestDetection:
    def test_fields(self):
        d = Detection(
            bbox=(10, 20, 100, 200),
            confidence=0.85,
            class_id=2,
            class_name="car",
        )
        assert d.bbox == (10, 20, 100, 200)
        assert d.confidence == pytest.approx(0.85)
        assert d.class_id == 2
        assert d.class_name == "car"

    def test_frozen(self):
        d = Detection(bbox=(0, 0, 1, 1), confidence=0.5, class_id=0, class_name="person")
        with pytest.raises(FrozenInstanceError):
            d.confidence = 0.9

    def test_area(self):
        d = Detection(bbox=(10, 20, 60, 80), confidence=0.5, class_id=2, class_name="car")
        # area = (60-10) * (80-20) = 50 * 60 = 3000
        assert d.area == 3000

    def test_center(self):
        d = Detection(bbox=(10, 20, 60, 80), confidence=0.5, class_id=2, class_name="car")
        cx, cy = d.center
        assert cx == pytest.approx(35.0)
        assert cy == pytest.approx(50.0)


class TestPreprocess:
    def test_output_is_pil_image(self):
        from PIL import Image

        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        image, ratio, pad = preprocess(frame, input_size=640)
        assert isinstance(image, Image.Image)
        assert image.size == (640, 640)
        assert image.mode == "RGB"

    def test_preserves_aspect_ratio(self):
        """Letterboxing should preserve aspect ratio via padding."""
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        _, ratio, (pad_w, pad_h) = preprocess(frame, input_size=640)
        # 320x240 -> scale to fit 640x640 preserving aspect
        # scale = 640/320 = 2.0, scaled size = 640x480, pad_h = (640-480)/2 = 80
        assert ratio == pytest.approx(2.0)
        assert pad_w == pytest.approx(0.0, abs=1)
        assert pad_h == pytest.approx(80.0, abs=1)

    def test_square_input_no_padding(self):
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        _, ratio, (pad_w, pad_h) = preprocess(frame, input_size=640)
        assert ratio == pytest.approx(1.0)
        assert pad_w == pytest.approx(0.0, abs=1)
        assert pad_h == pytest.approx(0.0, abs=1)


class TestPostprocess:
    def make_output(self, cx, cy, w, h, class_id, score):
        """Build a synthetic YOLOv8 output tensor with one detection."""
        # output shape: (1, 84, 8400) — 4 bbox + 80 class scores
        output = np.zeros((1, 84, 8400), dtype=np.float32)
        output[0, 0, 0] = cx
        output[0, 1, 0] = cy
        output[0, 2, 0] = w
        output[0, 3, 0] = h
        output[0, 4 + class_id, 0] = score
        return output

    def test_single_detection(self):
        # Place a car (class 2) at center of 640x640
        output = self.make_output(320, 320, 100, 80, class_id=2, score=0.9)
        dets = postprocess(
            output,
            conf_threshold=0.5,
            iou_threshold=0.5,
            ratio=1.0,
            pad=(0, 0),
            orig_shape=(640, 640),
        )
        assert len(dets) == 1
        assert dets[0].class_id == 2
        assert dets[0].class_name == "car"
        assert dets[0].confidence >= 0.5

    def test_low_confidence_filtered(self):
        output = self.make_output(320, 320, 100, 80, class_id=2, score=0.1)
        dets = postprocess(
            output,
            conf_threshold=0.5,
            iou_threshold=0.5,
            ratio=1.0,
            pad=(0, 0),
            orig_shape=(640, 640),
        )
        assert len(dets) == 0

    def test_empty_output(self):
        output = np.zeros((1, 84, 8400), dtype=np.float32)
        dets = postprocess(
            output,
            conf_threshold=0.5,
            iou_threshold=0.5,
            ratio=1.0,
            pad=(0, 0),
            orig_shape=(640, 640),
        )
        assert len(dets) == 0

    def test_bbox_rescaled_to_original(self):
        """Detections should be in original image coordinates, not 640x640."""
        # Simulating 320x240 input: ratio=2.0, pad=(0, 80)
        # Detection at (320, 320) in model space -> (160, 120) in original
        output = self.make_output(320, 320, 100, 80, class_id=2, score=0.9)
        dets = postprocess(
            output,
            conf_threshold=0.5,
            iou_threshold=0.5,
            ratio=2.0,
            pad=(0, 80),
            orig_shape=(240, 320),
        )
        assert len(dets) == 1
        x1, y1, x2, y2 = dets[0].bbox
        # Center should be near (160, 120) in original coords
        cx, cy = dets[0].center
        assert cx == pytest.approx(160.0, abs=5)
        assert cy == pytest.approx(120.0, abs=5)

    def test_bbox_clipped_to_image(self):
        """Bounding boxes should not extend beyond image boundaries."""
        # Detection near edge
        output = self.make_output(10, 10, 100, 100, class_id=2, score=0.9)
        dets = postprocess(
            output,
            conf_threshold=0.5,
            iou_threshold=0.5,
            ratio=1.0,
            pad=(0, 0),
            orig_shape=(640, 640),
        )
        if len(dets) > 0:
            x1, y1, x2, y2 = dets[0].bbox
            assert x1 >= 0
            assert y1 >= 0
            assert x2 <= 640
            assert y2 <= 640


class TestCocoVehicleClasses:
    def test_contains_common_vehicles(self):
        assert 2 in COCO_VEHICLE_CLASSES  # car
        assert 5 in COCO_VEHICLE_CLASSES  # bus
        assert 7 in COCO_VEHICLE_CLASSES  # truck

    def test_names(self):
        assert COCO_VEHICLE_CLASSES[2] == "car"
        assert COCO_VEHICLE_CLASSES[5] == "bus"
        assert COCO_VEHICLE_CLASSES[7] == "truck"


class TestYoloDetector:
    @pytest.fixture(scope="class")
    def detector(self):
        return YoloDetector(MODEL_PATH)

    def test_loads_model(self, detector):
        assert detector.input_size == 640

    def test_detect_returns_list(self, detector):
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        dets = detector.detect(frame)
        assert isinstance(dets, list)

    def test_detect_on_black_frame(self, detector):
        """Black frame should produce few or no detections."""
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        dets = detector.detect(frame)
        assert isinstance(dets, list)
        # No vehicles in a black frame
        vehicle_dets = [d for d in dets if d.class_id in COCO_VEHICLE_CLASSES]
        assert len(vehicle_dets) == 0

    def test_detections_have_valid_fields(self, detector):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dets = detector.detect(frame)
        for d in dets:
            assert 0 <= d.confidence <= 1.0
            assert 0 <= d.class_id < 80
            x1, y1, x2, y2 = d.bbox
            assert x1 < x2
            assert y1 < y2

    def test_vehicles_only_filter(self, detector):
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        dets = detector.detect(frame, vehicles_only=True)
        for d in dets:
            assert d.class_id in COCO_VEHICLE_CLASSES
