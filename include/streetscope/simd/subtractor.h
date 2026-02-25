#pragma once
#include <cstdint>

namespace streetscope::simd {

struct SubtractorConfig {
    uint8_t threshold;
    int width;
    int height;
};

void subtract_background_scalar(const uint8_t* frame, const uint8_t* background, uint8_t* mask, const SubtractorConfig& config);
void subtract_background_neon(const uint8_t* frame, const uint8_t* background, uint8_t* mask, const SubtractorConfig& config);

} // namespace streetscope::simd
