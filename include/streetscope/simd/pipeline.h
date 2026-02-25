#pragma once
#include <cstdint>
#include <streetscope/simd/isp_correction.h>

namespace streetscope::simd {

struct PipelineConfig {
    float ema_alpha;
    uint8_t motion_threshold;
    int width;
    int height;

    bool apply_isp;
    AEAWBConfig ae_awb;
    int af_blur_ksize;  // 5 for 5x5 box blur, 0 to skip AF
};

/// Fused per-frame pipeline: EMA -> subtract -> (optional) AE+AWB -> blur -> AF blend.
void process_frame(
    const uint8_t* frame_u8,
    float* background_f32,
    uint8_t* mask_out,
    uint8_t* display_out,
    const float* af_alpha_map,
    const PipelineConfig& config
);

/// Separable box blur (internal, exposed for testing).
void box_blur_5x5(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height
);

} // namespace streetscope::simd
