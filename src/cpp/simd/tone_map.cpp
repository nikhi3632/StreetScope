#include <streetscope/simd/tone_map.h>
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace streetscope::simd {

void tone_map_scalar(const uint8_t* bgr_in, uint8_t* bgr_out,
                     int width, int height, const ToneMapParams& params) {
    const int num_pixels = width * height;
    const float inv_gamma = 1.0f / params.gamma;
    const float white_sq = params.white_point * params.white_point;

    for (int i = 0; i < num_pixels; i++) {
        const auto base = static_cast<ptrdiff_t>(i) * 3;

        // Read BGR, convert to linear RGB [0,1]
        float r = static_cast<float>(bgr_in[base + 2]) / 255.0f;
        float g = static_cast<float>(bgr_in[base + 1]) / 255.0f;
        float b = static_cast<float>(bgr_in[base])     / 255.0f;

        // Apply AWB gains
        r *= params.gain_r;
        g *= params.gain_g;
        b *= params.gain_b;

        // Apply exposure
        r *= params.exposure;
        g *= params.exposure;
        b *= params.exposure;

        // Compute luminance (Rec. 709)
        float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;

        // Extended Reinhard: L_mapped = L * (1 + L/white^2) / (1 + L)
        if (lum > 1e-6f) {
            float lum_mapped = lum * (1.0f + lum / white_sq) / (1.0f + lum);
            float scale = lum_mapped / lum;
            r *= scale;
            g *= scale;
            b *= scale;
        }

        // Clamp to [0,1] and apply display gamma
        r = std::pow(std::min(std::max(r, 0.0f), 1.0f), inv_gamma);
        g = std::pow(std::min(std::max(g, 0.0f), 1.0f), inv_gamma);
        b = std::pow(std::min(std::max(b, 0.0f), 1.0f), inv_gamma);

        // Write BGR
        bgr_out[base]     = static_cast<uint8_t>(std::lroundf(b * 255.0f));
        bgr_out[base + 1] = static_cast<uint8_t>(std::lroundf(g * 255.0f));
        bgr_out[base + 2] = static_cast<uint8_t>(std::lroundf(r * 255.0f));
    }
}

} // namespace streetscope::simd
