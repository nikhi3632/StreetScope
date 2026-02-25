#pragma once
#include <cstdint>

namespace streetscope::simd {

// Tuning constants — scene-dependent, derived at runtime from ISPEstimator
//
// exposure      Scene-dependent. Derived as 0.5 / mean_luminance to push
//               scene toward mid-gray. Default 1.0 = no change.
// white_point   Scene-dependent. Derived from 99th percentile luminance
//               scaled by exposure. Controls highlight roll-off ceiling.
//               Default 1.0 = minimal compression.
// gamma         Display encoding. 1.0 for sRGB input (camera already encodes
//               gamma). 2.2 only if input is linear (rare for camera feeds).
// gain_b/g/r    AWB per-channel gains from gray-world estimator. Fused into
//               the tone map dispatch — no separate AWB pass needed.
//
// Rec. 709 luminance weights (in shader): 0.2126, 0.7152, 0.0722
//               ITU standard. Not tunable.
// luma epsilon  1e-6 — skip Reinhard for near-black pixels to avoid div-by-zero.
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
