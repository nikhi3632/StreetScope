#include <streetscope/simd/pipeline.h>
#include <streetscope/simd/accumulator.h>
#include <streetscope/simd/subtractor.h>
#include <streetscope/simd/isp_correction.h>
#include <arm_neon.h>
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

namespace streetscope::simd {

void box_blur_5x5(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height
) {
    const int channels = 3;
    const int stride = width * channels;
    const int radius = 2;
    const int pixel_stride = channels;

    std::vector<uint8_t> temp(static_cast<size_t>(width) * height * channels);

    // Divide-by-5 constant: (sum * 13108) >> 16 is exact for sum in [0, 1275]
    const uint16x4_t div5 = vdup_n_u16(13108);

    // --- Horizontal pass ---
    for (int y = 0; y < height; y++) {
        const uint8_t* row_in = input + static_cast<ptrdiff_t>(y) * stride;
        uint8_t* row_out = temp.data() + static_cast<ptrdiff_t>(y) * stride;

        // Scalar: first 2 pixels (boundary)
        for (int b = 0; b < radius * pixel_stride; b++) {
            int x = b / channels;
            int c = b % channels;
            int sum = 0;
            int count = 0;
            for (int dx = -radius; dx <= radius; dx++) {
                int nx = x + dx;
                if (nx >= 0 && nx < width) {
                    sum += row_in[nx * channels + c];
                    count++;
                }
            }
            row_out[b] = static_cast<uint8_t>(sum / count);
        }

        // NEON: interior pixels where all 5 neighbors exist
        const int neon_start = radius * pixel_stride;
        const int neon_end = (width - radius) * pixel_stride;
        int i = neon_start;
        for (; i + 15 < neon_end; i += 16) {
            // Load 16 bytes at 5 shifted positions (pixel stride = 3)
            uint8x16_t s0 = vld1q_u8(row_in + i - 6);
            uint8x16_t s1 = vld1q_u8(row_in + i - 3);
            uint8x16_t s2 = vld1q_u8(row_in + i);
            uint8x16_t s3 = vld1q_u8(row_in + i + 3);
            uint8x16_t s4 = vld1q_u8(row_in + i + 6);

            // Sum low 8 bytes in u16
            uint16x8_t lo = vaddl_u8(vget_low_u8(s0), vget_low_u8(s1));
            lo = vaddw_u8(lo, vget_low_u8(s2));
            lo = vaddw_u8(lo, vget_low_u8(s3));
            lo = vaddw_u8(lo, vget_low_u8(s4));

            // Sum high 8 bytes in u16
            uint16x8_t hi = vaddl_u8(vget_high_u8(s0), vget_high_u8(s1));
            hi = vaddw_u8(hi, vget_high_u8(s2));
            hi = vaddw_u8(hi, vget_high_u8(s3));
            hi = vaddw_u8(hi, vget_high_u8(s4));

            // Divide by 5: (sum * 13108) >> 16
            uint16x8_t d_lo = vcombine_u16(
                vshrn_n_u32(vmull_u16(vget_low_u16(lo), div5), 16),
                vshrn_n_u32(vmull_u16(vget_high_u16(lo), div5), 16));
            uint16x8_t d_hi = vcombine_u16(
                vshrn_n_u32(vmull_u16(vget_low_u16(hi), div5), 16),
                vshrn_n_u32(vmull_u16(vget_high_u16(hi), div5), 16));

            vst1q_u8(row_out + i, vcombine_u8(vmovn_u16(d_lo), vmovn_u16(d_hi)));
        }

        // Scalar: remaining interior + last 2 pixels
        for (; i < stride; i++) {
            int x = i / channels;
            int c = i % channels;
            int sum = 0;
            int count = 0;
            for (int dx = -radius; dx <= radius; dx++) {
                int nx = x + dx;
                if (nx >= 0 && nx < width) {
                    sum += row_in[nx * channels + c];
                    count++;
                }
            }
            row_out[i] = static_cast<uint8_t>(sum / count);
        }
    }

    // --- Vertical pass ---
    for (int y = 0; y < height; y++) {
        uint8_t* row_out = output + static_cast<ptrdiff_t>(y) * stride;

        if (y < radius || y >= height - radius) {
            // Scalar: border rows
            for (int b = 0; b < stride; b++) {
                int x = b / channels;
                int c = b % channels;
                int sum = 0;
                int count = 0;
                for (int dy = -radius; dy <= radius; dy++) {
                    int ny = y + dy;
                    if (ny >= 0 && ny < height) {
                        sum += temp[static_cast<ptrdiff_t>(ny) * stride + static_cast<ptrdiff_t>(x) * channels + c];
                        count++;
                    }
                }
                row_out[b] = static_cast<uint8_t>(sum / count);
            }
        } else {
            // NEON: interior rows — sum 5 rows at same byte offset
            const uint8_t* r0 = temp.data() + static_cast<ptrdiff_t>(y - 2) * stride;
            const uint8_t* r1 = temp.data() + static_cast<ptrdiff_t>(y - 1) * stride;
            const uint8_t* r2 = temp.data() + static_cast<ptrdiff_t>(y) * stride;
            const uint8_t* r3 = temp.data() + static_cast<ptrdiff_t>(y + 1) * stride;
            const uint8_t* r4 = temp.data() + static_cast<ptrdiff_t>(y + 2) * stride;

            int i = 0;
            for (; i + 15 < stride; i += 16) {
                uint16x8_t lo = vaddl_u8(vget_low_u8(vld1q_u8(r0 + i)),
                                         vget_low_u8(vld1q_u8(r1 + i)));
                lo = vaddw_u8(lo, vget_low_u8(vld1q_u8(r2 + i)));
                lo = vaddw_u8(lo, vget_low_u8(vld1q_u8(r3 + i)));
                lo = vaddw_u8(lo, vget_low_u8(vld1q_u8(r4 + i)));

                uint16x8_t hi = vaddl_u8(vget_high_u8(vld1q_u8(r0 + i)),
                                         vget_high_u8(vld1q_u8(r1 + i)));
                hi = vaddw_u8(hi, vget_high_u8(vld1q_u8(r2 + i)));
                hi = vaddw_u8(hi, vget_high_u8(vld1q_u8(r3 + i)));
                hi = vaddw_u8(hi, vget_high_u8(vld1q_u8(r4 + i)));

                uint16x8_t d_lo = vcombine_u16(
                    vshrn_n_u32(vmull_u16(vget_low_u16(lo), div5), 16),
                    vshrn_n_u32(vmull_u16(vget_high_u16(lo), div5), 16));
                uint16x8_t d_hi = vcombine_u16(
                    vshrn_n_u32(vmull_u16(vget_low_u16(hi), div5), 16),
                    vshrn_n_u32(vmull_u16(vget_high_u16(hi), div5), 16));

                vst1q_u8(row_out + i, vcombine_u8(vmovn_u16(d_lo), vmovn_u16(d_hi)));
            }

            // Scalar tail
            for (; i < stride; i++) {
                int sum = r0[i] + r1[i] + r2[i] + r3[i] + r4[i];
                row_out[i] = static_cast<uint8_t>(sum / 5);
            }
        }
    }
}

void process_frame(
    const uint8_t* frame_u8,
    float* background_f32,
    uint8_t* mask_out,
    uint8_t* display_out,
    const float* af_alpha_map,
    const PipelineConfig& config
) {
    const int pixels = config.width * config.height;
    const int bytes = pixels * 3;

    // 1. Convert frame uint8 -> float32 (NEON widening)
    std::vector<float> frame_f32(bytes);
    {
        int i = 0;
        for (; i + 15 < bytes; i += 16) {
            uint8x16_t v = vld1q_u8(frame_u8 + i);
            uint16x8_t lo16 = vmovl_u8(vget_low_u8(v));
            uint16x8_t hi16 = vmovl_u8(vget_high_u8(v));
            vst1q_f32(frame_f32.data() + i,      vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16))));
            vst1q_f32(frame_f32.data() + i + 4,  vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo16))));
            vst1q_f32(frame_f32.data() + i + 8,  vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16))));
            vst1q_f32(frame_f32.data() + i + 12, vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi16))));
        }
        for (; i < bytes; i++) {
            frame_f32[i] = static_cast<float>(frame_u8[i]);
        }
    }

    // 2. EMA accumulate
    AccumulatorConfig acc_cfg{config.ema_alpha, bytes};
    accumulate_ema_neon(frame_f32.data(), background_f32, acc_cfg);

    // 3. Convert background float32 -> uint8 (NEON narrowing with saturation)
    std::vector<uint8_t> bg_u8(bytes);
    {
        int i = 0;
        for (; i + 15 < bytes; i += 16) {
            uint32x4_t u0 = vcvtq_u32_f32(vld1q_f32(background_f32 + i));
            uint32x4_t u1 = vcvtq_u32_f32(vld1q_f32(background_f32 + i + 4));
            uint32x4_t u2 = vcvtq_u32_f32(vld1q_f32(background_f32 + i + 8));
            uint32x4_t u3 = vcvtq_u32_f32(vld1q_f32(background_f32 + i + 12));
            uint16x8_t n_lo = vcombine_u16(vqmovn_u32(u0), vqmovn_u32(u1));
            uint16x8_t n_hi = vcombine_u16(vqmovn_u32(u2), vqmovn_u32(u3));
            vst1q_u8(bg_u8.data() + i, vcombine_u8(vqmovn_u16(n_lo), vqmovn_u16(n_hi)));
        }
        for (; i < bytes; i++) {
            float v = background_f32[i];
            bg_u8[i] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, v)));
        }
    }

    // 4. Subtract background -> motion mask
    SubtractorConfig sub_cfg{config.motion_threshold, config.width, config.height};
    subtract_background_neon(frame_u8, bg_u8.data(), mask_out, sub_cfg);

    // 5. ISP correction (optional)
    if (config.apply_isp) {
        // AE+AWB
        std::vector<uint8_t> corrected(bytes);
        apply_ae_awb_neon(frame_u8, corrected.data(), config.ae_awb);

        // AF blend (optional, requires alpha_map and blur_ksize > 0)
        if (af_alpha_map != nullptr && config.af_blur_ksize > 0) {
            std::vector<uint8_t> blurred(bytes);
            box_blur_5x5(corrected.data(), blurred.data(), config.width, config.height);

            AFBlendConfig af_cfg{pixels};
            apply_af_blend_neon(corrected.data(), af_alpha_map, blurred.data(), display_out, af_cfg);
        } else {
            std::memcpy(display_out, corrected.data(), bytes);
        }
    } else {
        std::memcpy(display_out, frame_u8, bytes);
    }
}

} // namespace streetscope::simd
