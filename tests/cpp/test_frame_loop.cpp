#include <streetscope/frame_pipeline/frame_loop.h>
#include <streetscope/frame_pipeline/processed_frame.h>
#include <streetscope/frame_pipeline/ring_buffer.h>

#include <gtest/gtest.h>

#include <atomic>

#include <mutex>
#include <thread>
#include <vector>

using streetscope::FrameLoop;
using streetscope::FrameLoopConfig;
using streetscope::ProcessedFrame;
using streetscope::RingBuffer;

// --- ProcessedFrame tests ---

TEST(ProcessedFrame, MoveSemantics) {
    ProcessedFrame f;
    f.width = 640;
    f.height = 480;
    f.frame_number = 42;
    f.timestamp_s = 1.5;
    f.bgr_data.resize(static_cast<size_t>(640) * 480 * 3, 0xAB);
    f.mask_data.resize(static_cast<size_t>(640) * 480, 0xFF);
    f.display_data.resize(static_cast<size_t>(640) * 480 * 3, 0xCD);

    ProcessedFrame moved = std::move(f);
    EXPECT_EQ(moved.width, 640);
    EXPECT_EQ(moved.height, 480);
    EXPECT_EQ(moved.frame_number, 42);
    EXPECT_DOUBLE_EQ(moved.timestamp_s, 1.5);
    EXPECT_EQ(moved.bgr_data.size(), 640u * 480 * 3);
    EXPECT_EQ(moved.mask_data.size(), 640u * 480);
    EXPECT_EQ(moved.display_data.size(), 640u * 480 * 3);
    EXPECT_EQ(moved.bgr_data[0], 0xAB);
    EXPECT_EQ(moved.mask_data[0], 0xFF);
    EXPECT_EQ(moved.display_data[0], 0xCD);
}

TEST(ProcessedFrame, RingBufferRoundTrip) {
    RingBuffer<ProcessedFrame, 4> rb;

    ProcessedFrame f;
    f.width = 320;
    f.height = 240;
    f.frame_number = 7;
    f.timestamp_s = 0.5;
    f.bgr_data.resize(static_cast<size_t>(320) * 240 * 3, 100);
    f.mask_data.resize(static_cast<size_t>(320) * 240, 255);
    f.display_data.resize(static_cast<size_t>(320) * 240 * 3, 200);

    EXPECT_TRUE(rb.try_push(std::move(f)));

    auto popped = rb.try_pop();
    ASSERT_TRUE(popped.has_value());
    EXPECT_EQ(popped->width, 320);
    EXPECT_EQ(popped->height, 240);
    EXPECT_EQ(popped->frame_number, 7);
    EXPECT_EQ(popped->bgr_data.size(), 320u * 240 * 3);
    EXPECT_EQ(popped->mask_data.size(), 320u * 240);
    EXPECT_EQ(popped->display_data.size(), 320u * 240 * 3);
    EXPECT_EQ(popped->bgr_data[0], 100);
    EXPECT_EQ(popped->mask_data[0], 255);
    EXPECT_EQ(popped->display_data[0], 200);
}

// --- FrameLoopConfig tests ---

TEST(FrameLoopConfig, MoveWithAlphaMap) {
    FrameLoopConfig cfg;
    cfg.pipeline.ema_alpha = 0.05f;
    cfg.pipeline.motion_threshold = 15;
    cfg.alpha_map.resize(static_cast<size_t>(640) * 480, 0.5f);

    FrameLoopConfig moved = std::move(cfg);
    EXPECT_FLOAT_EQ(moved.pipeline.ema_alpha, 0.05f);
    EXPECT_EQ(moved.pipeline.motion_threshold, 15);
    EXPECT_EQ(moved.alpha_map.size(), 640u * 480);
    EXPECT_FLOAT_EQ(moved.alpha_map[0], 0.5f);
}

TEST(FrameLoopConfig, ConcurrentAccess) {
    // Stress test: writer pushes configs, reader snapshots under mutex.
    // TSan acid test for the mutex pattern used in FrameLoop::run().
    std::mutex mu;
    FrameLoopConfig shared_config;
    std::atomic<bool> done{false};
    std::atomic<int> reads{0};

    std::thread writer([&] {
        for (int i = 0; i < 10000 && !done.load(std::memory_order_relaxed); i++) {
            FrameLoopConfig cfg;
            cfg.pipeline.ema_alpha = static_cast<float>(i) * 0.001f;
            cfg.alpha_map.resize(100, static_cast<float>(i));
            std::lock_guard<std::mutex> lock(mu);
            shared_config = std::move(cfg);
        }
        done.store(true, std::memory_order_relaxed);
    });

    std::thread reader([&] {
        while (!done.load(std::memory_order_relaxed)) {
            FrameLoopConfig snapshot;
            {
                std::lock_guard<std::mutex> lock(mu);
                // Copy pipeline config (small, trivially copyable)
                snapshot.pipeline = shared_config.pipeline;
                // Copy alpha map if present
                if (!shared_config.alpha_map.empty()) {
                    snapshot.alpha_map = shared_config.alpha_map;
                }
            }
            // Use snapshot outside lock (simulates process_frame duration)
            volatile float v = snapshot.pipeline.ema_alpha;
            (void)v;
            reads.fetch_add(1, std::memory_order_relaxed);
        }
    });

    writer.join();
    reader.join();
    EXPECT_GT(reads.load(), 0);
}

// --- FrameLoop lifecycle tests ---

TEST(FrameLoop, ConstructionDefaults) {
    FrameLoop loop("http://invalid.example.com/test.m3u8");
    EXPECT_FALSE(loop.is_running());
    EXPECT_EQ(loop.frames_processed(), 0);
    // Destructor should be clean without calling start()
}

TEST(FrameLoop, ConfigUpdateBeforeStart) {
    FrameLoop loop("http://invalid.example.com/test.m3u8");

    FrameLoopConfig cfg;
    cfg.pipeline.ema_alpha = 0.1f;
    cfg.pipeline.motion_threshold = 20;
    loop.update_config(std::move(cfg));

    // Should not crash. Config stored for when processing starts.
    EXPECT_FALSE(loop.is_running());
}

TEST(FrameLoop, TryGetResultBeforeStart) {
    FrameLoop loop("http://invalid.example.com/test.m3u8");
    auto result = loop.try_get_result();
    EXPECT_FALSE(result.has_value());
}
