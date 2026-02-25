#include <streetscope/frame_pipeline/ring_buffer.h>
#include <streetscope/frame_pipeline/frame.h>
#include <gtest/gtest.h>
#include <atomic>
#include <thread>
#include <vector>

// Detect Valgrind at runtime (header only available in Docker/Linux)
#ifdef __has_include
#if __has_include(<valgrind/valgrind.h>)
#include <valgrind/valgrind.h>
#endif
#endif
#ifndef RUNNING_ON_VALGRIND
#define RUNNING_ON_VALGRIND 0
#endif

using streetscope::RingBuffer;
using streetscope::DecodedFrame;

// --- Basic operations (single-threaded) ---

TEST(RingBuffer, PushPopSingle) {
    RingBuffer<int, 4> rb;
    EXPECT_TRUE(rb.try_push(42));
    auto val = rb.try_pop();
    ASSERT_TRUE(val.has_value());
    EXPECT_EQ(val.value(), 42);
}

TEST(RingBuffer, EmptyPopReturnsNullopt) {
    RingBuffer<int, 4> rb;
    EXPECT_FALSE(rb.try_pop().has_value());
    EXPECT_TRUE(rb.empty());
}

TEST(RingBuffer, FullPushReturnsFalse) {
    RingBuffer<int, 4> rb;  // capacity = 3
    EXPECT_TRUE(rb.try_push(1));
    EXPECT_TRUE(rb.try_push(2));
    EXPECT_TRUE(rb.try_push(3));
    EXPECT_FALSE(rb.try_push(4));  // full
}

TEST(RingBuffer, FIFOOrder) {
    RingBuffer<int, 8> rb;  // capacity = 7
    for (int i = 0; i < 7; i++) {
        EXPECT_TRUE(rb.try_push(int{i}));
    }
    for (int i = 0; i < 7; i++) {
        auto val = rb.try_pop();
        ASSERT_TRUE(val.has_value());
        EXPECT_EQ(val.value(), i);
    }
    EXPECT_TRUE(rb.empty());
}

TEST(RingBuffer, WrapAround) {
    RingBuffer<int, 4> rb;  // capacity = 3

    // Fill and drain twice to force wraparound past index 3
    for (int round = 0; round < 3; round++) {
        for (int i = 0; i < 3; i++) {
            EXPECT_TRUE(rb.try_push(int{round * 10 + i}));
        }
        for (int i = 0; i < 3; i++) {
            auto val = rb.try_pop();
            ASSERT_TRUE(val.has_value());
            EXPECT_EQ(val.value(), round * 10 + i);
        }
    }
}

TEST(RingBuffer, Capacity) {
    RingBuffer<int, 8> rb;
    EXPECT_EQ(rb.capacity(), 7u);

    RingBuffer<int, 4> rb2;
    EXPECT_EQ(rb2.capacity(), 3u);
}

// --- Move semantics with DecodedFrame ---

TEST(RingBuffer, MoveSemantics) {
    RingBuffer<DecodedFrame, 4> rb;

    const size_t data_size = static_cast<size_t>(512) * 288 * 3;

    DecodedFrame f;
    f.width = 512;
    f.height = 288;
    f.frame_number = 42;
    f.timestamp_s = 1.5;
    f.bgr_data.resize(data_size, 0xAB);

    EXPECT_TRUE(rb.try_push(std::move(f)));
    // f is moved-from; don't access it

    auto popped = rb.try_pop();
    ASSERT_TRUE(popped.has_value());
    DecodedFrame& result = popped.value();
    EXPECT_EQ(result.width, 512);
    EXPECT_EQ(result.height, 288);
    EXPECT_EQ(result.frame_number, 42);
    EXPECT_DOUBLE_EQ(result.timestamp_s, 1.5);
    EXPECT_EQ(result.bgr_data.size(), data_size);
    EXPECT_EQ(result.bgr_data[0], 0xAB);
}

// --- Threaded SPSC stress test ---

TEST(RingBuffer, ProducerConsumerThreaded) {
    // Fewer iterations under Valgrind (atomics are ~100x slower)
    const int kItems = RUNNING_ON_VALGRIND ? 1000 : 100000;
    RingBuffer<int, 1024> rb;
    std::atomic<int> consumed{0};
    std::vector<int> received;
    received.reserve(kItems);

    // Consumer thread
    std::thread consumer([&] {
        while (consumed.load(std::memory_order_relaxed) < kItems) {
            auto val = rb.try_pop();
            if (val.has_value()) {
                received.push_back(val.value());
                consumed.fetch_add(1, std::memory_order_relaxed);
            }
        }
    });

    // Producer: push all items
    int pushed = 0;
    while (pushed < kItems) {
        if (rb.try_push(int{pushed})) {
            pushed++;
        }
        // Spin if full — producer waits for consumer to drain
    }

    consumer.join();

    // Verify: all items received exactly once, in order
    ASSERT_EQ(static_cast<int>(received.size()), kItems);
    for (int i = 0; i < kItems; i++) {
        EXPECT_EQ(received[i], i) << "Mismatch at index " << i;
    }
}
