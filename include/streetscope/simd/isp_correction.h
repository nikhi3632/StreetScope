#pragma once
#include <cstdint>

namespace streetscope::simd {

struct AEAWBConfig {
    uint8_t lut[256];
    float gain_b;
    float gain_g;
    float gain_r;
    int width;
    int height;
};

struct AFBlendConfig {
    int num_pixels;
};

void apply_ae_awb_scalar(const uint8_t* input, uint8_t* output, const AEAWBConfig& config);
void apply_ae_awb_neon(const uint8_t* input, uint8_t* output, const AEAWBConfig& config);

void apply_af_blend_scalar(const uint8_t* frame, const float* alpha_map,
                           const uint8_t* blurred, uint8_t* output,
                           const AFBlendConfig& config);
void apply_af_blend_neon(const uint8_t* frame, const float* alpha_map,
                         const uint8_t* blurred, uint8_t* output,
                         const AFBlendConfig& config);

} // namespace streetscope::simd
