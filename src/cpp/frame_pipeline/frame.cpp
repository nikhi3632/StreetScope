#include <streetscope/frame_pipeline/frame.h>
#include <utility>

namespace streetscope {

DecodedFrame::~DecodedFrame() {
#if __APPLE__
    if (pixel_buffer) CVPixelBufferRelease(pixel_buffer);
#endif
}

DecodedFrame::DecodedFrame(DecodedFrame&& other) noexcept
    : bgr_data(std::move(other.bgr_data)),
#if __APPLE__
      pixel_buffer(other.pixel_buffer),
#endif
      width(other.width),
      height(other.height),
      frame_number(other.frame_number),
      timestamp_s(other.timestamp_s),
      media_pts_s(other.media_pts_s) {
#if __APPLE__
    other.pixel_buffer = nullptr;
#endif
}

DecodedFrame& DecodedFrame::operator=(DecodedFrame&& other) noexcept {
    if (this != &other) {
#if __APPLE__
        if (pixel_buffer) CVPixelBufferRelease(pixel_buffer);
#endif
        bgr_data = std::move(other.bgr_data);
#if __APPLE__
        pixel_buffer = other.pixel_buffer;
        other.pixel_buffer = nullptr;
#endif
        width = other.width;
        height = other.height;
        frame_number = other.frame_number;
        timestamp_s = other.timestamp_s;
        media_pts_s = other.media_pts_s;
    }
    return *this;
}

}  // namespace streetscope
