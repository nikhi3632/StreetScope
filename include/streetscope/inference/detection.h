#pragma once
#include <cstdint>
#include <vector>

namespace streetscope {

struct Detection {
    int x1, y1, x2, y2;   // bbox in original image coords
    float confidence;
    int class_id;

    int width() const { return x2 - x1; }
    int height() const { return y2 - y1; }
    int area() const { return width() * height(); }
};

struct DetectionResult {
    std::vector<Detection> detections;
    double inference_ms = 0.0;
    int64_t frame_number = -1;

    DetectionResult() = default;
    DetectionResult(DetectionResult&&) = default;
    DetectionResult& operator=(DetectionResult&&) = default;
    DetectionResult(const DetectionResult&) = delete;
    DetectionResult& operator=(const DetectionResult&) = delete;
};

/// COCO 80-class name lookup. Returns "unknown" if class_id >= 80 or < 0.
const char* coco_class_name(int class_id);

/// Vehicle class IDs: {1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck}
bool is_vehicle_class(int class_id);

}  // namespace streetscope
