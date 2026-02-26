#pragma once
#include <cstdint>
#include <memory>
#include <vector>

namespace streetscope {

/// GPU tone mapping via Metal compute shader (Extended Reinhard with fused AWB).
///
/// Params are scene-dependent — see tone_map.h for derivation notes.
/// The shader assumes sRGB input from camera feeds. Set gamma=1.0 for
/// camera sources (already gamma-encoded). Use gamma=2.2 only for linear input.
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

    /// Fused GPU path: MPS Lanczos upscale + Reinhard tone map in a single
    /// command buffer. BGR bytes at (w,h) -> BGR bytes at (w*scale, h*scale).
    /// One CPU-to-GPU upload, two GPU operations, one readback. The upscaled
    /// intermediate stays GPU-resident. scale=1 delegates to tone_map().
    std::vector<uint8_t> upscale_and_tone_map(const uint8_t* bgr, int w, int h,
                                               int scale, const Params& params);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace streetscope
