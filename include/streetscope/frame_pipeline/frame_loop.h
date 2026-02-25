#pragma once
#include <streetscope/frame_pipeline/processed_frame.h>
#include <streetscope/frame_pipeline/ring_buffer.h>
#include <streetscope/simd/pipeline.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace streetscope {

class VideoToolboxDecoder;

/// Config snapshot pushed from Python. Contains ISP parameters that change
/// per-frame based on Python analysis of the scene.
struct FrameLoopConfig {
    simd::PipelineConfig pipeline{};
    std::vector<float> alpha_map;  // H*W float32. Empty = no AF.
};

/// C++ frame processing loop: decode -> process_frame() -> output.
///
/// Runs a dedicated thread that pulls DecodedFrame from VideoToolboxDecoder,
/// calls process_frame() with zero marshaling, and pushes ProcessedFrame to
/// an output ring buffer. Python reads results via try_get_result().
///
/// Thread model:
///   - Decoder poll thread (owned by VideoToolboxDecoder): BGRA->BGR -> decoder ring
///   - FrameLoop thread (this class): decoder ring -> process_frame -> output ring
///   - Python main thread: output ring -> YOLO/tracker/display
class FrameLoop {
public:
    explicit FrameLoop(const std::string& url);
    ~FrameLoop();

    FrameLoop(const FrameLoop&) = delete;
    FrameLoop& operator=(const FrameLoop&) = delete;
    FrameLoop(FrameLoop&&) = delete;
    FrameLoop& operator=(FrameLoop&&) = delete;

    void start();
    void stop();

    /// Non-blocking: returns next processed frame, or nullopt if none ready.
    std::optional<ProcessedFrame> try_get_result();

    /// Thread-safe config update. Python pushes new ISP params here.
    void update_config(FrameLoopConfig config);

    bool is_running() const;
    int64_t frames_processed() const;

private:
    void run();
    void ensure_buffers(int width, int height);

    std::unique_ptr<VideoToolboxDecoder> decoder_;

    static constexpr std::size_t kOutputRingSize = 4;
    RingBuffer<ProcessedFrame, kOutputRingSize> output_ring_;

    // EMA background plate — persistent state, allocated on first frame
    std::vector<float> background_;
    int width_ = 0;
    int height_ = 0;

    // Config — mutex-protected, snapshot per frame
    std::mutex config_mutex_;
    FrameLoopConfig config_;
    bool config_set_ = false;

    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_requested_{false};
    std::atomic<int64_t> frame_count_{0};
};

}  // namespace streetscope
