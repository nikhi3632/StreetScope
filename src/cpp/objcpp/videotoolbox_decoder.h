#pragma once
#include <streetscope/frame_pipeline/frame.h>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace streetscope {

class VideoToolboxDecoder {
public:
    explicit VideoToolboxDecoder(const std::string& url);
    ~VideoToolboxDecoder();

    VideoToolboxDecoder(const VideoToolboxDecoder&) = delete;
    VideoToolboxDecoder& operator=(const VideoToolboxDecoder&) = delete;
    VideoToolboxDecoder(VideoToolboxDecoder&&) = delete;
    VideoToolboxDecoder& operator=(VideoToolboxDecoder&&) = delete;

    void start();
    void stop();

    /// Non-blocking: returns nullopt if no new frame is ready.
    std::optional<DecodedFrame> try_get_frame();

    bool is_running() const;
    int64_t frames_decoded() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Pump the main thread's run loop for the given duration. Must be called from
/// the main thread. Required for AVPlayer's internal networking when there is
/// no Cocoa event loop (e.g. standalone Python scripts). In pipeline_viewer,
/// cv2.waitKey(1) serves this purpose.
void pump_main_runloop(double seconds = 0.01);

}  // namespace streetscope
