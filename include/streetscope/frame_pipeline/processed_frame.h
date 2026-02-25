#pragma once
#include <cstdint>
#include <vector>

namespace streetscope {

struct ProcessedFrame {
    std::vector<uint8_t> bgr_data;      // Raw decoded frame (H*W*3) — for YOLO
    std::vector<uint8_t> mask_data;     // Motion mask (H*W) — for visualization
    std::vector<uint8_t> display_data;  // ISP-corrected display (H*W*3)
    int width = 0;
    int height = 0;
    int64_t frame_number = 0;
    double timestamp_s = 0.0;
    double media_pts_s = 0.0;

    ProcessedFrame() = default;
    ProcessedFrame(ProcessedFrame&&) = default;
    ProcessedFrame& operator=(ProcessedFrame&&) = default;
    ProcessedFrame(const ProcessedFrame&) = delete;
    ProcessedFrame& operator=(const ProcessedFrame&) = delete;
};

}  // namespace streetscope
