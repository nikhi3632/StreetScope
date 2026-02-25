#include "streetscope/inference/letterbox.h"
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace streetscope {

LetterboxInfo compute_letterbox(int src_w, int src_h, int input_size) {
    LetterboxInfo info{};
    auto max_dim = static_cast<float>(std::max(src_w, src_h));
    info.ratio = static_cast<float>(input_size) / max_dim;
    info.new_w = static_cast<int>(std::round(static_cast<float>(src_w) * info.ratio));
    info.new_h = static_cast<int>(std::round(static_cast<float>(src_h) * info.ratio));
    info.pad_w = static_cast<float>(input_size - info.new_w) / 2.0f;
    info.pad_h = static_cast<float>(input_size - info.new_h) / 2.0f;
    return info;
}

void apply_letterbox(
    const uint8_t* src_bgr, int src_w, int src_h,
    uint8_t* dst_bgra, int input_size,
    const LetterboxInfo& info
) {
    // Wrap raw pointers as cv::Mat (no copy)
    cv::Mat src(src_h, src_w, CV_8UC3, const_cast<uint8_t*>(src_bgr));

    // Resize to letterbox dimensions
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(info.new_w, info.new_h), 0, 0, cv::INTER_LINEAR);

    // Pad to input_size x input_size (matches Python rounding: round(pad - 0.1))
    int top = static_cast<int>(std::round(info.pad_h - 0.1f));
    int bottom = static_cast<int>(std::round(info.pad_h + 0.1f));
    int left = static_cast<int>(std::round(info.pad_w - 0.1f));
    int right = static_cast<int>(std::round(info.pad_w + 0.1f));

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    // Ensure exact size (rounding can be off by 1)
    if (padded.rows != input_size || padded.cols != input_size) {
        cv::resize(padded, padded, cv::Size(input_size, input_size));
    }

    // BGR -> BGRA into the destination buffer
    cv::Mat dst(input_size, input_size, CV_8UC4, dst_bgra);
    cv::cvtColor(padded, dst, cv::COLOR_BGR2BGRA);
}

}  // namespace streetscope
