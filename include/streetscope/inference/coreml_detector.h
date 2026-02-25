#pragma once
#include <streetscope/inference/detection.h>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace streetscope {

class CoreMLDetector {
public:
    explicit CoreMLDetector(const std::string& model_path,
                           float conf_threshold = 0.25f,
                           float iou_threshold = 0.45f);
    ~CoreMLDetector();

    CoreMLDetector(const CoreMLDetector&) = delete;
    CoreMLDetector& operator=(const CoreMLDetector&) = delete;
    CoreMLDetector(CoreMLDetector&&) = delete;
    CoreMLDetector& operator=(CoreMLDetector&&) = delete;

    /// Sync: letterbox + infer + postprocess. Blocks until complete.
    std::vector<Detection> detect(const uint8_t* bgr, int w, int h,
                                  bool vehicles_only = false);

    /// Async: submit frame for background inference.
    void submit(const uint8_t* bgr, int w, int h,
                bool vehicles_only = false);

    /// Async: poll for result. Returns nullopt if not ready.
    std::optional<DetectionResult> try_get_result();

    float conf_threshold() const;
    float iou_threshold() const;
    int input_size() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace streetscope
