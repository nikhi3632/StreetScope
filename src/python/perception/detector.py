"""YOLOv8 object detection via ONNX Runtime."""

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

# COCO class IDs for vehicles
COCO_VEHICLE_CLASSES: dict[int, str] = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Full COCO 80-class names
COCO_NAMES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


@dataclass(frozen=True)
class Detection:
    """A single object detection."""

    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def preprocess(frame: np.ndarray, input_size: int = 640) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Letterbox resize and normalize for YOLOv8.

    Args:
        frame: BGR uint8 image (H, W, 3).
        input_size: Model input dimension (square).

    Returns:
        (blob, ratio, (pad_w, pad_h))
        blob: float32 NCHW tensor [1, 3, input_size, input_size] in [0, 1].
        ratio: scale factor applied.
        (pad_w, pad_h): letterbox padding added.
    """
    h, w = frame.shape[:2]
    ratio = input_size / max(h, w)
    new_w, new_h = int(round(w * ratio)), int(round(h * ratio))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Letterbox: center the resized image on a gray canvas
    pad_w = (input_size - new_w) / 2.0
    pad_h = (input_size - new_h) / 2.0
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # Ensure exact size (rounding can be off by 1)
    if padded.shape[0] != input_size or padded.shape[1] != input_size:
        padded = cv2.resize(padded, (input_size, input_size))

    # BGR -> RGB, HWC -> CHW, normalize to [0, 1], add batch dim
    blob = padded[:, :, ::-1].astype(np.float32) / 255.0
    blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[np.newaxis])
    return blob, ratio, (pad_w, pad_h)


def postprocess(
    output: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
    ratio: float,
    pad: tuple[float, float],
    orig_shape: tuple[int, int],
) -> list[Detection]:
    """Convert raw YOLOv8 output to Detection list.

    Args:
        output: Raw model output, shape (1, 84, 8400).
        conf_threshold: Minimum class confidence.
        iou_threshold: NMS IoU threshold.
        ratio: Scale ratio from preprocessing.
        pad: (pad_w, pad_h) from preprocessing.
        orig_shape: (height, width) of the original image.

    Returns:
        List of Detection objects in original image coordinates.
    """
    # Transpose to (8400, 84) for easier indexing
    predictions = output[0].T  # (8400, 84)

    # Extract class scores and find best class per candidate
    class_scores = predictions[:, 4:]  # (8400, 80)
    max_scores = class_scores.max(axis=1)  # (8400,)
    class_ids = class_scores.argmax(axis=1)  # (8400,)

    # Filter by confidence
    mask = max_scores >= conf_threshold
    if not np.any(mask):
        return []

    filtered = predictions[mask]
    scores = max_scores[mask]
    cids = class_ids[mask]

    # Convert cx, cy, w, h -> x1, y1, x2, y2
    cx, cy, w, h = filtered[:, 0], filtered[:, 1], filtered[:, 2], filtered[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    # NMS via OpenCV
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), scores.tolist(), conf_threshold, iou_threshold,
    )
    if len(indices) == 0:
        return []

    orig_h, orig_w = orig_shape
    pad_w, pad_h = pad
    detections: list[Detection] = []

    for i in indices.flatten():
        bx1, by1, bx2, by2 = boxes[i]

        # Remove padding and rescale to original image coordinates
        bx1 = (bx1 - pad_w) / ratio
        by1 = (by1 - pad_h) / ratio
        bx2 = (bx2 - pad_w) / ratio
        by2 = (by2 - pad_h) / ratio

        # Clip to image bounds
        bx1 = max(0, int(round(bx1)))
        by1 = max(0, int(round(by1)))
        bx2 = min(orig_w, int(round(bx2)))
        by2 = min(orig_h, int(round(by2)))

        if bx2 <= bx1 or by2 <= by1:
            continue

        cid = int(cids[i])
        detections.append(Detection(
            bbox=(bx1, by1, bx2, by2),
            confidence=float(scores[i]),
            class_id=cid,
            class_name=COCO_NAMES[cid] if cid < len(COCO_NAMES) else "unknown",
        ))

    return detections


class YoloDetector:
    """YOLOv8 detector using ONNX Runtime."""

    def __init__(self, model_path: str, conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45) -> None:
        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # ERROR only, suppress CoreML partition warnings
        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CoreMLExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        inp_shape = self.session.get_inputs()[0].shape
        self.input_size: int = inp_shape[2]  # 640
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def detect(self, frame: np.ndarray, vehicles_only: bool = False) -> list[Detection]:
        """Run detection on a BGR uint8 frame.

        Args:
            frame: BGR uint8 image (H, W, 3).
            vehicles_only: If True, only return vehicle detections.

        Returns:
            List of Detection objects.
        """
        blob, ratio, pad = preprocess(frame, self.input_size)
        output = self.session.run(None, {self.input_name: blob})[0]
        orig_shape = (frame.shape[0], frame.shape[1])

        dets = postprocess(output, self.conf_threshold, self.iou_threshold,
                           ratio, pad, orig_shape)

        if vehicles_only:
            dets = [d for d in dets if d.class_id in COCO_VEHICLE_CLASSES]

        return dets


class AsyncDetector:
    """Threaded YOLO detector. Accepts frames and produces detections asynchronously.

    Uses a submit/latest pattern — submit() overwrites the pending frame
    (latest-wins), the background thread picks it up and runs inference,
    and latest() returns the most recent result without blocking.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        vehicles_only: bool = False,
    ) -> None:
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.vehicles_only = vehicles_only

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.frame_event = threading.Event()
        self.thread: threading.Thread | None = None

        self.pending_frame: np.ndarray | None = None
        self.latest_result: tuple[list[Detection], float] | None = None

    def start(self) -> None:
        """Start the inference thread. Loads the ONNX model."""
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Signal the inference thread to stop and wait for it."""
        self.stop_event.set()
        self.frame_event.set()  # Unblock if waiting
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def submit(self, frame: np.ndarray) -> None:
        """Submit a frame for detection. Non-blocking, overwrites any pending frame."""
        with self.lock:
            self.pending_frame = frame
        self.frame_event.set()

    def latest(self) -> tuple[list[Detection], float] | None:
        """Return the most recent (detections, infer_ms) or None."""
        with self.lock:
            return self.latest_result

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()

    def run(self) -> None:
        """Inference loop — runs in a daemon thread."""
        detector = YoloDetector(
            self.model_path,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
        )

        while not self.stop_event.is_set():
            self.frame_event.wait()
            self.frame_event.clear()

            if self.stop_event.is_set():
                break

            with self.lock:
                frame = self.pending_frame
                self.pending_frame = None

            if frame is None:
                continue

            t0 = time.monotonic()
            dets = detector.detect(frame, vehicles_only=self.vehicles_only)
            infer_ms = (time.monotonic() - t0) * 1000.0

            with self.lock:
                self.latest_result = (dets, infer_ms)
