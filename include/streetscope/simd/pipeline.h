#pragma once
#include <cstdint>
#include <streetscope/simd/isp_correction.h>

// Tuning constants — empirical
//
// ema_alpha          Perception path: EMA learning rate for background accumulator.
//                    Typical: 0.05 (5% per frame). Set from Python config.
// motion_threshold   Perception path: absolute pixel difference for motion mask.
//                    Typical: 15 (6% of 255). Set from Python config.
// ae_awb             ISP: AE (gamma LUT) + AWB (gain correction). Always runs.
//                    Use identity LUT + unity gains for pass-through.
// af_alpha_map       ISP: AF depth-of-field blend. Always runs (passed to process_frame).
//                    Use all-zeros alpha map for fully sharp (no blur applied).
//
// ISP = AE + AWB + AF. All three always run. Fixed 5x5 box blur kernel.
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

    AEAWBConfig ae_awb;
};

/// Fused per-frame pipeline: EMA -> subtract -> ISP (AE+AWB -> AF blur+blend).
void process_frame(
    const uint8_t* frame_u8,
    float* background_f32,
    uint8_t* mask_out,
    uint8_t* display_out,
    const float* af_alpha_map,
    const PipelineConfig& config
);

/// Convert uint8 BGR to float32 (NEON widening).
void convert_u8_to_f32(const uint8_t* input, float* output, int count);

/// Convert float32 to uint8 with saturation (NEON narrowing).
void convert_f32_to_u8(const float* input, uint8_t* output, int count);

/// Separable box blur (internal, exposed for testing).
void box_blur_5x5(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height
);

/// Separable box blur with external temp buffer.
void box_blur_5x5(
    const uint8_t* input,
    uint8_t* output,
    uint8_t* temp,
    int width,
    int height
);

} // namespace streetscope::simd
