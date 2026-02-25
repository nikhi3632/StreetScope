#pragma once
#include <cstdint>
#include <vector>

namespace streetscope {

struct DecodedFrame {
    std::vector<uint8_t> bgr_data;  // BGR interleaved, H*W*3 bytes
    int width = 0;
    int height = 0;
    int64_t frame_number = 0;
    double timestamp_s = 0.0;     // Wall clock (steady_clock)
    double media_pts_s = 0.0;     // Media presentation timestamp (from decoder)

    // Move-only (prevent accidental copies of ~440KB frame data)
    DecodedFrame() = default;
    DecodedFrame(DecodedFrame&&) = default;
    DecodedFrame& operator=(DecodedFrame&&) = default;
    DecodedFrame(const DecodedFrame&) = delete;
    DecodedFrame& operator=(const DecodedFrame&) = delete;
};

}  // namespace streetscope
