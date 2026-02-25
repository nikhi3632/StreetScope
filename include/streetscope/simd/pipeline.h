#pragma once
#include <cstdint>
#include <streetscope/simd/isp_correction.h>

// Tuning constants — empirical
//
// ema_alpha          Perception path: EMA learning rate for background accumulator.
//                    Typical: 0.05 (5% per frame). Set from Python config.
// motion_threshold   Perception path: absolute pixel difference for motion mask.
//                    Typical: 15 (6% of 255). Set from Python config.
// af_blur_ksize      Display path: box blur kernel size for auto focus. 5 = 5x5.
//                    0 disables AF. Odd values only.
// apply_isp          Display path: when true, applies gamma LUT + AWB gains.
//                    Disabled when Metal handles display correction.
//
// Internal (pipeline.cpp):
// blur_radius      2    Fixed 5x5 kernel (2*radius+1 = 5). Balances smoothing vs cost.
// div_magic    13108    Fixed-point reciprocal of 5: (x * 13108) >> 16 ≈ x / 5.
//                       Avoids integer division in the NEON inner loop.

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
