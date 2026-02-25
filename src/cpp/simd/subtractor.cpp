#include <streetscope/simd/subtractor.h>
#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace streetscope::simd {

void subtract_background_scalar(const uint8_t* frame, const uint8_t* background, uint8_t* mask, const SubtractorConfig& config) {
    const int num_pixels = config.width * config.height;
    for (int i = 0; i < num_pixels; i++) {
        const ptrdiff_t base = static_cast<ptrdiff_t>(i) * 3;
        int diff_b = std::abs(static_cast<int>(frame[base])     - static_cast<int>(background[base]));
        int diff_g = std::abs(static_cast<int>(frame[base + 1]) - static_cast<int>(background[base + 1]));
        int diff_r = std::abs(static_cast<int>(frame[base + 2]) - static_cast<int>(background[base + 2]));
        int max_diff = std::max({diff_b, diff_g, diff_r});
        mask[i] = (max_diff > config.threshold) ? 255 : 0;
    }
}

void subtract_background_neon(const uint8_t* frame, const uint8_t* background, uint8_t* mask, const SubtractorConfig& config) {
    const int num_pixels = config.width * config.height;
    const uint8x16_t thresh_vec = vdupq_n_u8(config.threshold);

    int i = 0;
    // Process 16 pixels per iteration
    // Each pixel is 3 bytes (BGR), so vld3q_u8 loads 48 bytes and deinterleaves
    for (; i + 15 < num_pixels; i += 16) {
        // Deinterleave BGR: loads 48 bytes, splits into B[16], G[16], R[16]
        uint8x16x3_t fr = vld3q_u8(frame + static_cast<ptrdiff_t>(i) * 3);
        uint8x16x3_t bg = vld3q_u8(background + static_cast<ptrdiff_t>(i) * 3);

        // Absolute difference per channel
        uint8x16_t diff_b = vabdq_u8(fr.val[0], bg.val[0]);
        uint8x16_t diff_g = vabdq_u8(fr.val[1], bg.val[1]);
        uint8x16_t diff_r = vabdq_u8(fr.val[2], bg.val[2]);

        // Max across channels
        uint8x16_t max_diff = vmaxq_u8(vmaxq_u8(diff_b, diff_g), diff_r);

        // Compare: max_diff > threshold -> 0xFF, else 0x00
        uint8x16_t result = vcgtq_u8(max_diff, thresh_vec);

        vst1q_u8(mask + i, result);
    }

    // Scalar tail
    for (; i < num_pixels; i++) {
        const ptrdiff_t base = static_cast<ptrdiff_t>(i) * 3;
        int diff_b = std::abs(static_cast<int>(frame[base])     - static_cast<int>(background[base]));
        int diff_g = std::abs(static_cast<int>(frame[base + 1]) - static_cast<int>(background[base + 1]));
        int diff_r = std::abs(static_cast<int>(frame[base + 2]) - static_cast<int>(background[base + 2]));
        int max_diff = std::max({diff_b, diff_g, diff_r});
        mask[i] = (max_diff > config.threshold) ? 255 : 0;
    }
}

} // namespace streetscope::simd
