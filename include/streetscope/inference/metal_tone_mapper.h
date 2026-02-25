#pragma once
#include <cstdint>
#include <memory>
#include <vector>

namespace streetscope {

class MetalToneMapper {
public:
    struct Params {
        float exposure = 1.0f;
        float white_point = 1.0f;
        float gamma = 2.2f;
        float gain_b = 1.0f;
        float gain_g = 1.0f;
        float gain_r = 1.0f;
    };

    MetalToneMapper();
    ~MetalToneMapper();

    MetalToneMapper(const MetalToneMapper&) = delete;
    MetalToneMapper& operator=(const MetalToneMapper&) = delete;
    MetalToneMapper(MetalToneMapper&&) = delete;
    MetalToneMapper& operator=(MetalToneMapper&&) = delete;

    /// BGR CPU bytes -> BGR CPU bytes. Uploads to GPU, dispatches, reads back.
    std::vector<uint8_t> tone_map(const uint8_t* bgr, int w, int h,
                                  const Params& params);

    /// Zero-copy: CVPixelBuffer -> CVPixelBuffer (void* = CVPixelBufferRef).
    /// Input must be BGRA, IOSurface-backed. Caller owns returned buffer.
    void* tone_map_pixelbuffer(void* cv_pixel_buffer, const Params& params);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace streetscope
