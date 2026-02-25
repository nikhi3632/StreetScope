#pragma once
#include <cstdint>

namespace streetscope::simd {

struct AccumulatorConfig {
    float alpha;
    int num_elements;
};

void accumulate_ema_scalar(const float* frame, float* background, const AccumulatorConfig& config);
void accumulate_ema_neon(const float* frame, float* background, const AccumulatorConfig& config);

} // namespace streetscope::simd
