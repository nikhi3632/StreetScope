#include "videotoolbox_decoder.h"
#include <streetscope/frame_pipeline/ring_buffer.h>

#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <QuartzCore/CABase.h>

#include <arm_neon.h>
#include <atomic>
#include <chrono>
#include <thread>

namespace streetscope {

static constexpr std::size_t kRingSize = 4;

/// Convert BGRA pixel buffer to BGR, handling row stride padding.
static void bgra_to_bgr(const uint8_t* src, uint8_t* dst,
                         int width, int height, size_t bytes_per_row) {
    for (int y = 0; y < height; y++) {
        const uint8_t* row = src + static_cast<size_t>(y) * bytes_per_row;
        uint8_t* out = dst + static_cast<size_t>(y) * width * 3;
        auto pixels = static_cast<size_t>(width);
        size_t i = 0;

        // NEON: 16 pixels per iteration (64 bytes BGRA -> 48 bytes BGR)
        for (; i + 15 < pixels; i += 16) {
            uint8x16x4_t bgra = vld4q_u8(row + i * 4);
            uint8x16x3_t bgr = {{bgra.val[0], bgra.val[1], bgra.val[2]}};
            vst3q_u8(out + i * 3, bgr);
        }

        // Scalar tail
        for (; i < pixels; i++) {
            out[i * 3 + 0] = row[i * 4 + 0];
            out[i * 3 + 1] = row[i * 4 + 1];
            out[i * 3 + 2] = row[i * 4 + 2];
        }
    }
}

struct VideoToolboxDecoder::Impl {
    std::string url;
    RingBuffer<DecodedFrame, kRingSize> ring;
    std::thread poll_thread;
    std::atomic<bool> running{false};
    std::atomic<bool> stop_requested{false};
    std::atomic<int64_t> frame_count{0};

    // ObjC objects — created on main queue, accessed from poll thread
    // (safe: created before poll starts, destroyed after poll joins)
    AVPlayer* player = nil;
    AVPlayerItem* player_item = nil;
    AVPlayerItemVideoOutput* video_output = nil;

    explicit Impl(const std::string& u) : url(u) {}

    void start() {
        if (running.load(std::memory_order_relaxed)) return;

        stop_requested.store(false, std::memory_order_relaxed);

        // Set up AVPlayer on the main queue. AVPlayer's internal networking
        // (HLS manifest fetch, segment downloads) dispatches callbacks to the
        // main dispatch queue. The caller's main thread must pump this queue
        // (cv2.waitKey does this in pipeline_viewer, pump_main_runloop() for
        // standalone use).
        dispatch_async(dispatch_get_main_queue(), ^{
            @autoreleasepool {
                NSURL* ns_url = [NSURL URLWithString:
                    [NSString stringWithUTF8String:url.c_str()]];

                NSDictionary* attrs = @{
                    (NSString*)kCVPixelBufferPixelFormatTypeKey:
                        @(kCVPixelFormatType_32BGRA)
                };

                video_output = [[AVPlayerItemVideoOutput alloc]
                    initWithPixelBufferAttributes:attrs];

                player_item = [AVPlayerItem playerItemWithURL:ns_url];
                [player_item addOutput:video_output];

                player = [AVPlayer playerWithPlayerItem:player_item];
                player.automaticallyWaitsToMinimizeStalling = YES;
                [player play];
            }
        });

        running.store(true, std::memory_order_release);
        poll_thread = std::thread(&Impl::poll_loop, this);
    }

    void stop() {
        stop_requested.store(true, std::memory_order_relaxed);
        if (poll_thread.joinable()) {
            poll_thread.join();
        }

        // Tear down AVPlayer on the main queue (matches creation thread)
        dispatch_async(dispatch_get_main_queue(), ^{
            @autoreleasepool {
                [player pause];
                if (player_item && video_output) {
                    [player_item removeOutput:video_output];
                }
                player = nil;
                player_item = nil;
                video_output = nil;
            }
        });

        running.store(false, std::memory_order_release);
    }

    void poll_loop() {
        while (!stop_requested.load(std::memory_order_relaxed)) {
            @autoreleasepool {
                // Player may not be set up yet (dispatch_async is async)
                if (video_output == nil) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                }

                // Map host clock to media time. Using itemTimeForHostTime
                // instead of currentTime because currentTime depends on
                // the player's internal render pipeline which may not be
                // active in a headless (no display) context.
                CMTime time = [video_output itemTimeForHostTime:CACurrentMediaTime()];
                if ([video_output hasNewPixelBufferForItemTime:time]) {
                    CMTime display_time = kCMTimeZero;
                    CVPixelBufferRef pb =
                        [video_output copyPixelBufferForItemTime:time
                                            itemTimeForDisplay:&display_time];
                    if (pb != nullptr) {
                        double pts = CMTIME_IS_VALID(display_time)
                            ? CMTimeGetSeconds(display_time)
                            : -1.0;
                        process_pixel_buffer(pb, pts);
                        CVPixelBufferRelease(pb);
                    }
                }
            }
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
    }

    void process_pixel_buffer(CVPixelBufferRef pb, double media_pts) {
        CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);

        auto width = static_cast<int>(CVPixelBufferGetWidth(pb));
        auto height = static_cast<int>(CVPixelBufferGetHeight(pb));
        size_t bytes_per_row = CVPixelBufferGetBytesPerRow(pb);
        const auto* src = static_cast<const uint8_t*>(
            CVPixelBufferGetBaseAddress(pb));

        DecodedFrame frame;
        frame.width = width;
        frame.height = height;
        frame.frame_number = frame_count.fetch_add(1, std::memory_order_relaxed);
        frame.timestamp_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        frame.media_pts_s = media_pts;
        frame.bgr_data.resize(static_cast<size_t>(width) * height * 3);

        bgra_to_bgr(src, frame.bgr_data.data(), width, height, bytes_per_row);

        CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);

        ring.try_push(std::move(frame));  // Drop if full
    }
};

VideoToolboxDecoder::VideoToolboxDecoder(const std::string& url)
    : impl_(std::make_unique<Impl>(url)) {}

VideoToolboxDecoder::~VideoToolboxDecoder() {
    if (impl_->running.load(std::memory_order_relaxed)) {
        impl_->stop();
    }
}

void VideoToolboxDecoder::start() { impl_->start(); }
void VideoToolboxDecoder::stop() { impl_->stop(); }

std::optional<DecodedFrame> VideoToolboxDecoder::try_get_frame() {
    return impl_->ring.try_pop();
}

bool VideoToolboxDecoder::is_running() const {
    return impl_->running.load(std::memory_order_acquire);
}

int64_t VideoToolboxDecoder::frames_decoded() const {
    return impl_->frame_count.load(std::memory_order_relaxed);
}

void pump_main_runloop(double seconds) {
    CFRunLoopRunInMode(kCFRunLoopDefaultMode, seconds, false);
}

}  // namespace streetscope
