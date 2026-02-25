#pragma once
#include <cstdint>
#include <vector>

#if __APPLE__
#include <CoreVideo/CVPixelBuffer.h>
#endif

namespace streetscope {

struct DecodedFrame {
    std::vector<uint8_t> bgr_data;  // BGR interleaved, H*W*3 bytes

#if __APPLE__
    CVPixelBufferRef pixel_buffer = nullptr;  // IOSurface-backed, from decoder
#endif

    int width = 0;
    int height = 0;
    int64_t frame_number = 0;
    double timestamp_s = 0.0;     // Wall clock (steady_clock)
    double media_pts_s = 0.0;     // Media presentation timestamp (from decoder)

    // Move-only (prevent accidental copies of ~440KB frame data)
    DecodedFrame() = default;
    ~DecodedFrame();
    DecodedFrame(DecodedFrame&& other) noexcept;
    DecodedFrame& operator=(DecodedFrame&& other) noexcept;
    DecodedFrame(const DecodedFrame&) = delete;
    DecodedFrame& operator=(const DecodedFrame&) = delete;
};

}  // namespace streetscope
