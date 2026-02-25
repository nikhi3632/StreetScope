#pragma once
#include <cstdint>

namespace streetscope {

struct LetterboxInfo {
    float ratio;          // Scale factor applied
    float pad_w, pad_h;   // Padding in model-space pixels
    int new_w, new_h;     // Scaled image dimensions before padding
};

/// Compute letterbox parameters. Must match Python exactly:
///   ratio = input_size / max(h, w)
///   new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
///   pad_w = (input_size - new_w) / 2.0
///   pad_h = (input_size - new_h) / 2.0
LetterboxInfo compute_letterbox(int src_w, int src_h, int input_size = 640);

/// Apply letterbox: bilinear resize src BGR -> fill dst BGRA at input_size x input_size.
/// dst must be input_size * input_size * 4 bytes (BGRA interleaved).
/// Padding value: (114, 114, 114, 255) BGRA.
/// BGR->BGRA conversion: copies B,G,R from src, sets A=255.
void apply_letterbox(
    const uint8_t* src_bgr, int src_w, int src_h,
    uint8_t* dst_bgra, int input_size,
    const LetterboxInfo& info
);

}  // namespace streetscope
