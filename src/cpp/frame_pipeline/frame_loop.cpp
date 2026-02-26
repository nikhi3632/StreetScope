#include <streetscope/frame_pipeline/frame_loop.h>
#include <videotoolbox_decoder.h>

#include <chrono>
#include <cstddef>
#include <thread>

namespace streetscope {

FrameLoop::FrameLoop(const std::string& url)
    : decoder_(std::make_unique<VideoToolboxDecoder>(url)) {}

FrameLoop::~FrameLoop() {
    if (running_.load(std::memory_order_relaxed)) {
        stop();
    }
}

void FrameLoop::start() {
    if (running_.load(std::memory_order_relaxed)) return;
    stop_requested_.store(false, std::memory_order_relaxed);
    decoder_->start();
    running_.store(true, std::memory_order_release);
    thread_ = std::thread(&FrameLoop::run, this);
}

void FrameLoop::stop() {
    stop_requested_.store(true, std::memory_order_relaxed);
    if (thread_.joinable()) {
        thread_.join();
    }
    decoder_->stop();
    running_.store(false, std::memory_order_release);
}

std::optional<ProcessedFrame> FrameLoop::try_get_result() {
    return output_ring_.try_pop();
}

void FrameLoop::update_config(FrameLoopConfig config) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    config_ = std::move(config);
    config_set_ = true;
}

bool FrameLoop::is_running() const {
    return running_.load(std::memory_order_acquire);
}

int64_t FrameLoop::frames_processed() const {
    return frame_count_.load(std::memory_order_relaxed);
}

void FrameLoop::ensure_buffers(int width, int height) {
    if (width == width_ && height == height_) return;
    const auto bytes = static_cast<std::size_t>(width) * height * 3;
    background_.resize(bytes, 0.0f);
    width_ = width;
    height_ = height;
}

void FrameLoop::run() {
    bool first_frame = true;

    while (!stop_requested_.load(std::memory_order_relaxed)) {
        auto decoded = decoder_->try_get_frame();
        if (!decoded) {
            std::this_thread::sleep_for(std::chrono::microseconds(500));
            continue;
        }

        ensure_buffers(decoded->width, decoded->height);

        const int w = decoded->width;
        const int h = decoded->height;
        const int pixels = w * h;
        const auto bytes = static_cast<std::size_t>(pixels) * 3;

        // First frame: initialize background plate from frame values
        if (first_frame) {
            for (std::size_t i = 0; i < bytes; i++) {
                background_[i] = static_cast<float>(decoded->bgr_data[i]);
            }
            first_frame = false;
        }

        // Snapshot config under lock
        simd::PipelineConfig pcfg{};
        const float* alpha_ptr = nullptr;
        std::vector<float> alpha_copy;

        {
            std::lock_guard<std::mutex> lock(config_mutex_);
            if (config_set_) {
                pcfg = config_.pipeline;
                if (!config_.alpha_map.empty()) {
                    alpha_copy = config_.alpha_map;
                    alpha_ptr = alpha_copy.data();
                }
            } else {
                pcfg.ema_alpha = 0.05f;
                pcfg.motion_threshold = 15;
                // ISP defaults: identity LUT + unity gains
                for (int i = 0; i < 256; i++) pcfg.ae_awb.lut[i] = static_cast<uint8_t>(i);
                pcfg.ae_awb.gain_b = 1.0f;
                pcfg.ae_awb.gain_g = 1.0f;
                pcfg.ae_awb.gain_r = 1.0f;
            }
        }

        // Ensure dimensions match current frame
        pcfg.width = w;
        pcfg.height = h;
        pcfg.ae_awb.width = w;
        pcfg.ae_awb.height = h;

        // AF always runs; use all-zeros alpha map when not provided
        if (alpha_ptr == nullptr) {
            alpha_copy.resize(static_cast<size_t>(pixels), 0.0f);
            alpha_ptr = alpha_copy.data();
        }

        // Allocate output
        ProcessedFrame result;
        result.width = w;
        result.height = h;
        result.frame_number = decoded->frame_number;
        result.timestamp_s = decoded->timestamp_s;
        result.media_pts_s = decoded->media_pts_s;
        result.mask_data.resize(static_cast<std::size_t>(pixels));
        result.display_data.resize(bytes);

        // THE CALL — zero marshaling, pure C++
        simd::process_frame(
            decoded->bgr_data.data(),
            background_.data(),
            result.mask_data.data(),
            result.display_data.data(),
            alpha_ptr,
            pcfg
        );

        // Steal decoded frame's BGR data (zero-copy move)
        result.bgr_data = std::move(decoded->bgr_data);

        output_ring_.try_push(std::move(result));
        frame_count_.fetch_add(1, std::memory_order_relaxed);
    }
}

}  // namespace streetscope
