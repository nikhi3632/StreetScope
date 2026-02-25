#pragma once
#include <cstdint>

namespace streetscope::simd {

struct ToneMapParams {
    float exposure = 1.0f;
    float white_point = 1.0f;
    float gamma = 2.2f;
    float gain_b = 1.0f;
    float gain_g = 1.0f;
    float gain_r = 1.0f;
};

/// Scalar Extended Reinhard tone mapping with fused AWB.
/// BGR in → BGR out. Reference implementation for testing Metal shader.
void tone_map_scalar(const uint8_t* bgr_in, uint8_t* bgr_out,
                     int width, int height, const ToneMapParams& params);

} // namespace streetscope::simd
