#include <streetscope/simd/isp_correction.h>
#include <arm_neon.h>
#include <algorithm>
#include <cstddef>

namespace streetscope::simd {

void apply_ae_awb_scalar(const uint8_t* input, uint8_t* output, const AEAWBConfig& config) {
    const int num_pixels = config.width * config.height;
    for (int i = 0; i < num_pixels; i++) {
        const ptrdiff_t base = static_cast<ptrdiff_t>(i) * 3;
        // LUT lookup for each BGR channel, then apply per-channel gain
        float b = static_cast<float>(config.lut[input[base]])     * config.gain_b;
        float g = static_cast<float>(config.lut[input[base + 1]]) * config.gain_g;
        float r = static_cast<float>(config.lut[input[base + 2]]) * config.gain_r;
        output[base]     = static_cast<uint8_t>(std::min(std::max(b, 0.0f), 255.0f));
        output[base + 1] = static_cast<uint8_t>(std::min(std::max(g, 0.0f), 255.0f));
        output[base + 2] = static_cast<uint8_t>(std::min(std::max(r, 0.0f), 255.0f));
    }
}

void apply_ae_awb_neon(const uint8_t* input, uint8_t* output, const AEAWBConfig& config) {
    const int num_pixels = config.width * config.height;
    const float32x4_t gain_b_vec = vdupq_n_f32(config.gain_b);
    const float32x4_t gain_g_vec = vdupq_n_f32(config.gain_g);
    const float32x4_t gain_r_vec = vdupq_n_f32(config.gain_r);

    int i = 0;
    for (; i + 15 < num_pixels; i += 16) {
        const ptrdiff_t base = static_cast<ptrdiff_t>(i) * 3;

        // Step 1: Scalar LUT lookup for 16 pixels (48 bytes) into temp buffer
        uint8_t lut_buf[48];
        for (int j = 0; j < 48; j++) {
            lut_buf[j] = config.lut[input[base + j]];
        }

        // Step 2: Deinterleave BGR from LUT results
        uint8x16x3_t bgr = vld3q_u8(lut_buf);

        // Step 3: For each channel, widen u8->f32, multiply by gain, narrow back
        // Process B channel
        uint8x8_t b_lo8 = vget_low_u8(bgr.val[0]);
        uint8x8_t b_hi8 = vget_high_u8(bgr.val[0]);
        uint16x8_t b_lo16 = vmovl_u8(b_lo8);
        uint16x8_t b_hi16 = vmovl_u8(b_hi8);

        // Group 0: elements 0-3
        float32x4_t b_f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(b_lo16)));
        b_f0 = vmulq_f32(b_f0, gain_b_vec);
        uint32x4_t b_u0 = vcvtq_u32_f32(b_f0);
        // Group 1: elements 4-7
        float32x4_t b_f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(b_lo16)));
        b_f1 = vmulq_f32(b_f1, gain_b_vec);
        uint32x4_t b_u1 = vcvtq_u32_f32(b_f1);
        // Group 2: elements 8-11
        float32x4_t b_f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(b_hi16)));
        b_f2 = vmulq_f32(b_f2, gain_b_vec);
        uint32x4_t b_u2 = vcvtq_u32_f32(b_f2);
        // Group 3: elements 12-15
        float32x4_t b_f3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(b_hi16)));
        b_f3 = vmulq_f32(b_f3, gain_b_vec);
        uint32x4_t b_u3 = vcvtq_u32_f32(b_f3);
        // Narrow back: u32 -> u16 (saturating) -> u8 (saturating)
        uint16x8_t b_n16_lo = vcombine_u16(vqmovn_u32(b_u0), vqmovn_u32(b_u1));
        uint16x8_t b_n16_hi = vcombine_u16(vqmovn_u32(b_u2), vqmovn_u32(b_u3));
        uint8x16_t b_out = vcombine_u8(vqmovn_u16(b_n16_lo), vqmovn_u16(b_n16_hi));

        // Process G channel
        uint8x8_t g_lo8 = vget_low_u8(bgr.val[1]);
        uint8x8_t g_hi8 = vget_high_u8(bgr.val[1]);
        uint16x8_t g_lo16 = vmovl_u8(g_lo8);
        uint16x8_t g_hi16 = vmovl_u8(g_hi8);

        float32x4_t g_f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(g_lo16)));
        g_f0 = vmulq_f32(g_f0, gain_g_vec);
        uint32x4_t g_u0 = vcvtq_u32_f32(g_f0);
        float32x4_t g_f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(g_lo16)));
        g_f1 = vmulq_f32(g_f1, gain_g_vec);
        uint32x4_t g_u1 = vcvtq_u32_f32(g_f1);
        float32x4_t g_f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(g_hi16)));
        g_f2 = vmulq_f32(g_f2, gain_g_vec);
        uint32x4_t g_u2 = vcvtq_u32_f32(g_f2);
        float32x4_t g_f3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(g_hi16)));
        g_f3 = vmulq_f32(g_f3, gain_g_vec);
        uint32x4_t g_u3 = vcvtq_u32_f32(g_f3);
        uint16x8_t g_n16_lo = vcombine_u16(vqmovn_u32(g_u0), vqmovn_u32(g_u1));
        uint16x8_t g_n16_hi = vcombine_u16(vqmovn_u32(g_u2), vqmovn_u32(g_u3));
        uint8x16_t g_out = vcombine_u8(vqmovn_u16(g_n16_lo), vqmovn_u16(g_n16_hi));

        // Process R channel
        uint8x8_t r_lo8 = vget_low_u8(bgr.val[2]);
        uint8x8_t r_hi8 = vget_high_u8(bgr.val[2]);
        uint16x8_t r_lo16 = vmovl_u8(r_lo8);
        uint16x8_t r_hi16 = vmovl_u8(r_hi8);

        float32x4_t r_f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(r_lo16)));
        r_f0 = vmulq_f32(r_f0, gain_r_vec);
        uint32x4_t r_u0 = vcvtq_u32_f32(r_f0);
        float32x4_t r_f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(r_lo16)));
        r_f1 = vmulq_f32(r_f1, gain_r_vec);
        uint32x4_t r_u1 = vcvtq_u32_f32(r_f1);
        float32x4_t r_f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(r_hi16)));
        r_f2 = vmulq_f32(r_f2, gain_r_vec);
        uint32x4_t r_u2 = vcvtq_u32_f32(r_f2);
        float32x4_t r_f3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(r_hi16)));
        r_f3 = vmulq_f32(r_f3, gain_r_vec);
        uint32x4_t r_u3 = vcvtq_u32_f32(r_f3);
        uint16x8_t r_n16_lo = vcombine_u16(vqmovn_u32(r_u0), vqmovn_u32(r_u1));
        uint16x8_t r_n16_hi = vcombine_u16(vqmovn_u32(r_u2), vqmovn_u32(r_u3));
        uint8x16_t r_out = vcombine_u8(vqmovn_u16(r_n16_lo), vqmovn_u16(r_n16_hi));

        // Step 4: Interleave and store
        uint8x16x3_t result = {{b_out, g_out, r_out}};
        vst3q_u8(output + base, result);
    }

    // Scalar tail for remaining pixels
    for (; i < num_pixels; i++) {
        const ptrdiff_t base = static_cast<ptrdiff_t>(i) * 3;
        float b = static_cast<float>(config.lut[input[base]])     * config.gain_b;
        float g = static_cast<float>(config.lut[input[base + 1]]) * config.gain_g;
        float r = static_cast<float>(config.lut[input[base + 2]]) * config.gain_r;
        output[base]     = static_cast<uint8_t>(std::min(std::max(b, 0.0f), 255.0f));
        output[base + 1] = static_cast<uint8_t>(std::min(std::max(g, 0.0f), 255.0f));
        output[base + 2] = static_cast<uint8_t>(std::min(std::max(r, 0.0f), 255.0f));
    }
}

void apply_af_blend_scalar(const uint8_t* frame, const float* alpha_map,
                           const uint8_t* blurred, uint8_t* output,
                           const AFBlendConfig& config) {
    const int num_pixels = config.num_pixels;
    for (int i = 0; i < num_pixels; i++) {
        const ptrdiff_t base = static_cast<ptrdiff_t>(i) * 3;
        float alpha = alpha_map[i];
        for (int c = 0; c < 3; c++) {
            auto f = static_cast<float>(frame[base + c]);
            auto b = static_cast<float>(blurred[base + c]);
            float detail = f - b;
            float result = f + alpha * detail;
            output[base + c] = static_cast<uint8_t>(
                std::min(std::max(result, 0.0f), 255.0f));
        }
    }
}

void apply_af_blend_neon(const uint8_t* frame, const float* alpha_map,
                         const uint8_t* blurred, uint8_t* output,
                         const AFBlendConfig& config) {
    const int num_pixels = config.num_pixels;
    const float32x4_t zero = vdupq_n_f32(0.0f);

    int i = 0;
    // Process 16 pixels per iteration via vld3q_u8 deinterleave
    for (; i + 15 < num_pixels; i += 16) {
        const ptrdiff_t base = static_cast<ptrdiff_t>(i) * 3;

        // Deinterleave BGR: 16 pixels → B[16], G[16], R[16]
        uint8x16x3_t fr = vld3q_u8(frame + base);
        uint8x16x3_t bl = vld3q_u8(blurred + base);

        // Load 16 alpha values (4 groups of 4)
        float32x4_t a0 = vld1q_f32(alpha_map + i);
        float32x4_t a1 = vld1q_f32(alpha_map + i + 4);
        float32x4_t a2 = vld1q_f32(alpha_map + i + 8);
        float32x4_t a3 = vld1q_f32(alpha_map + i + 12);

        // Process each channel: result = frame + alpha * (frame - blurred)
        // After deinterleave, alpha[n] maps to element [n] of each channel vector
        for (int ch = 0; ch < 3; ch++) {
            uint16x8_t fr_lo16 = vmovl_u8(vget_low_u8(fr.val[ch]));
            uint16x8_t fr_hi16 = vmovl_u8(vget_high_u8(fr.val[ch]));
            uint16x8_t bl_lo16 = vmovl_u8(vget_low_u8(bl.val[ch]));
            uint16x8_t bl_hi16 = vmovl_u8(vget_high_u8(bl.val[ch]));

            // Group 0: pixels 0-3
            float32x4_t ff0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(fr_lo16)));
            float32x4_t bb0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(bl_lo16)));
            float32x4_t r0 = vmlaq_f32(ff0, a0, vsubq_f32(ff0, bb0));
            r0 = vmaxq_f32(r0, zero);

            // Group 1: pixels 4-7
            float32x4_t ff1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(fr_lo16)));
            float32x4_t bb1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(bl_lo16)));
            float32x4_t r1 = vmlaq_f32(ff1, a1, vsubq_f32(ff1, bb1));
            r1 = vmaxq_f32(r1, zero);

            // Group 2: pixels 8-11
            float32x4_t ff2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(fr_hi16)));
            float32x4_t bb2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(bl_hi16)));
            float32x4_t r2 = vmlaq_f32(ff2, a2, vsubq_f32(ff2, bb2));
            r2 = vmaxq_f32(r2, zero);

            // Group 3: pixels 12-15
            float32x4_t ff3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(fr_hi16)));
            float32x4_t bb3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(bl_hi16)));
            float32x4_t r3 = vmlaq_f32(ff3, a3, vsubq_f32(ff3, bb3));
            r3 = vmaxq_f32(r3, zero);

            // Narrow: f32 → u32 → u16 (saturating) → u8 (saturating)
            uint16x8_t n_lo = vcombine_u16(vqmovn_u32(vcvtq_u32_f32(r0)),
                                           vqmovn_u32(vcvtq_u32_f32(r1)));
            uint16x8_t n_hi = vcombine_u16(vqmovn_u32(vcvtq_u32_f32(r2)),
                                           vqmovn_u32(vcvtq_u32_f32(r3)));
            fr.val[ch] = vcombine_u8(vqmovn_u16(n_lo), vqmovn_u16(n_hi));
        }

        vst3q_u8(output + base, fr);
    }

    // Scalar tail for remaining pixels
    for (; i < num_pixels; i++) {
        const ptrdiff_t base = static_cast<ptrdiff_t>(i) * 3;
        float alpha_val = alpha_map[i];
        for (int c = 0; c < 3; c++) {
            auto f = static_cast<float>(frame[base + c]);
            auto b = static_cast<float>(blurred[base + c]);
            float result = f + alpha_val * (f - b);
            output[base + c] = static_cast<uint8_t>(
                std::min(std::max(result, 0.0f), 255.0f));
        }
    }
}

} // namespace streetscope::simd
