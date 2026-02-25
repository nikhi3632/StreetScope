#pragma once
#include <array>
#include <atomic>
#include <cstddef>
#include <optional>

namespace streetscope {

/// Lock-free single-producer single-consumer ring buffer.
///
/// N must be a power of 2. Usable capacity is N-1 (one slot distinguishes
/// full from empty). Producer calls try_push(); consumer calls try_pop().
/// No locks, no syscalls — just atomic load/store with acquire/release.
template <typename T, std::size_t N>
class RingBuffer {
    static_assert(N > 1, "Ring buffer capacity must be > 1");
    static_assert((N & (N - 1)) == 0, "N must be a power of 2");

public:
    RingBuffer() = default;

    /// Push an item. Returns false if the buffer is full (caller should drop).
    bool try_push(T&& item) {
        const auto h = head_.load(std::memory_order_relaxed);
        const auto next = (h + 1) & kMask;
        if (next == tail_.load(std::memory_order_acquire)) {
            return false;  // full
        }
        slots_[h] = std::move(item);
        head_.store(next, std::memory_order_release);
        return true;
    }

    /// Pop an item. Returns std::nullopt if the buffer is empty.
    std::optional<T> try_pop() {
        const auto t = tail_.load(std::memory_order_relaxed);
        if (t == head_.load(std::memory_order_acquire)) {
            return std::nullopt;  // empty
        }
        T item = std::move(slots_[t]);
        tail_.store((t + 1) & kMask, std::memory_order_release);
        return item;
    }

    bool empty() const {
        return head_.load(std::memory_order_acquire) ==
               tail_.load(std::memory_order_acquire);
    }

    /// Maximum number of items the buffer can hold (N-1).
    static constexpr std::size_t capacity() { return N - 1; }

private:
    static constexpr std::size_t kMask = N - 1;

    alignas(64) std::atomic<std::size_t> head_{0};
    alignas(64) std::atomic<std::size_t> tail_{0};
    std::array<T, N> slots_{};
};

}  // namespace streetscope
