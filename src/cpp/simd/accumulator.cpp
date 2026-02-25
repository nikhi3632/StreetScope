#include <streetscope/simd/accumulator.h>
#include <arm_neon.h>

namespace streetscope::simd {

void accumulate_ema_scalar(const float* frame, float* background, const AccumulatorConfig& config) {
    const float one_minus_alpha = 1.0f - config.alpha;
    for (int i = 0; i < config.num_elements; i++) {
        background[i] = one_minus_alpha * background[i] + config.alpha * frame[i];
    }
}

void accumulate_ema_neon(const float* frame, float* background, const AccumulatorConfig& config) {
    const int n = config.num_elements;
    const float32x4_t alpha_vec = vdupq_n_f32(config.alpha);
    const float32x4_t one_minus_alpha_vec = vdupq_n_f32(1.0f - config.alpha);

    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t bg = vld1q_f32(background + i);
        float32x4_t fr = vld1q_f32(frame + i);
        bg = vmlaq_f32(vmulq_f32(one_minus_alpha_vec, bg), alpha_vec, fr);
        vst1q_f32(background + i, bg);
    }
    // Scalar tail for remaining elements
    const float one_minus_alpha = 1.0f - config.alpha;
    for (; i < n; i++) {
        background[i] = one_minus_alpha * background[i] + config.alpha * frame[i];
    }
}

} // namespace streetscope::simd
