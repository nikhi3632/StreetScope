#include <streetscope/plan_executor/plan_executor.h>
#include <streetscope/simd/accumulator.h>
#include <streetscope/simd/subtractor.h>
#include <streetscope/simd/isp_correction.h>
#include <streetscope/simd/pipeline.h>

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <time.h>
#else
#include <chrono>
#endif

namespace streetscope {

// ---------------------------------------------------------------------------
// Helper: high-resolution timestamp (nanoseconds)
// ---------------------------------------------------------------------------
static inline uint64_t now_ns() {
#ifdef __APPLE__
    return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
#else
    using namespace std::chrono;
    return static_cast<uint64_t>(
        duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
#endif
}

// ---------------------------------------------------------------------------
// Constructor — parse and validate the binary plan
// ---------------------------------------------------------------------------
PlanExecutor::PlanExecutor(const uint8_t* plan_data, size_t plan_size)
    : arena_(nullptr, std::free)
{
    // 1. Validate minimum size
    if (plan_size < sizeof(PlanHeader)) {
        throw std::invalid_argument(
            "Plan data too small: " + std::to_string(plan_size)
            + " bytes, need at least " + std::to_string(sizeof(PlanHeader)));
    }

    // 2. Read header
    PlanHeader header{};
    std::memcpy(&header, plan_data, sizeof(PlanHeader));

    if (header.magic != kPlanMagic) {
        throw std::invalid_argument(
            "Bad plan magic: 0x" + std::to_string(header.magic)
            + ", expected 0x" + std::to_string(kPlanMagic));
    }
    if (header.version != kPlanVersion) {
        throw std::invalid_argument(
            "Unsupported plan version: " + std::to_string(header.version)
            + ", expected " + std::to_string(kPlanVersion));
    }

    // 3. Validate total size
    const size_t expected_size =
        sizeof(PlanHeader)
        + static_cast<size_t>(header.num_stages) * sizeof(StageDesc)
        + static_cast<size_t>(header.num_buffers) * sizeof(BufferDesc);

    if (plan_size < expected_size) {
        throw std::invalid_argument(
            "Plan data truncated: " + std::to_string(plan_size)
            + " bytes, expected " + std::to_string(expected_size));
    }

    // 4. Copy stages and buffers
    stages_.resize(header.num_stages);
    buffers_.resize(header.num_buffers);

    const uint8_t* cursor = plan_data + sizeof(PlanHeader);

    std::memcpy(stages_.data(), cursor,
                header.num_stages * sizeof(StageDesc));
    cursor += header.num_stages * sizeof(StageDesc);

    std::memcpy(buffers_.data(), cursor,
                header.num_buffers * sizeof(BufferDesc));

    // 5 & 6. Validate arena buffer alignment and bounds
    for (int i = 0; i < header.num_buffers; ++i) {
        const auto& buf = buffers_[static_cast<size_t>(i)];
        if (buf.size_bytes == 0) continue;  // external buffer

        if ((buf.offset & 0xF) != 0) {
            throw std::invalid_argument(
                "Buffer " + std::to_string(i) + " offset "
                + std::to_string(buf.offset) + " is not 16-byte aligned");
        }
        if (buf.offset + buf.size_bytes > header.arena_size_bytes) {
            throw std::invalid_argument(
                "Buffer " + std::to_string(i) + " exceeds arena: offset "
                + std::to_string(buf.offset) + " + size "
                + std::to_string(buf.size_bytes) + " > arena "
                + std::to_string(header.arena_size_bytes));
        }
    }

    // 7. Allocate arena (16-byte aligned)
    arena_size_ = header.arena_size_bytes;
    if (arena_size_ > 0) {
        // aligned_alloc on macOS requires size to be a multiple of alignment
        const size_t alloc_size = (static_cast<size_t>(arena_size_) + 15) & ~static_cast<size_t>(15);
        void* p = std::aligned_alloc(16, alloc_size);
        if (!p) {
            throw std::invalid_argument(
                "Failed to allocate arena of " + std::to_string(alloc_size) + " bytes");
        }
        arena_.reset(static_cast<uint8_t*>(p));
    }

    // 8. Store dimensions
    width_ = static_cast<int>(header.width);
    height_ = static_cast<int>(header.height);
}

// ---------------------------------------------------------------------------
// run_frame — dispatch stages linearly
// ---------------------------------------------------------------------------
void PlanExecutor::run_frame(
    const uint8_t* frame_u8,
    float* background_f32,
    uint8_t* mask_out,
    uint8_t* display_out,
    const float* af_alpha_map,
    const simd::PipelineConfig& config,
    uint64_t* stage_times_ns)
{
    // 1. Build pointer table
    const size_t num_bufs = buffers_.size();
    std::vector<void*> ptrs(num_bufs);

    ptrs[kBufFrameU8]       = const_cast<uint8_t*>(frame_u8);
    ptrs[kBufBackgroundF32] = background_f32;
    ptrs[kBufMaskOut]       = mask_out;
    ptrs[kBufDisplayOut]    = display_out;
    ptrs[kBufAfAlphaMap]    = const_cast<float*>(af_alpha_map);

    for (size_t i = kBufArenaStart; i < num_bufs; ++i) {
        ptrs[i] = arena_.get() + buffers_[i].offset;
    }

    // 2. Walk stages
    const int n_stages = static_cast<int>(stages_.size());
    for (int s = 0; s < n_stages; ++s) {
        uint64_t t0 = 0;
        if (stage_times_ns) t0 = now_ns();

        const auto& st = stages_[static_cast<size_t>(s)];
        switch (st.op) {
        case StageOp::kU8ToF32: {
            auto* in  = static_cast<const uint8_t*>(ptrs[st.inputs[0]]);
            auto* out = static_cast<float*>(ptrs[st.outputs[0]]);
            simd::convert_u8_to_f32(in, out, width_ * height_ * 3);
            break;
        }
        case StageOp::kEmaAccumulate: {
            auto* frame_f = static_cast<const float*>(ptrs[st.inputs[0]]);
            auto* bg      = static_cast<float*>(ptrs[st.inputs[1]]);
            simd::AccumulatorConfig cfg{config.ema_alpha, width_ * height_ * 3};
            simd::accumulate_ema_neon(frame_f, bg, cfg);
            break;
        }
        case StageOp::kF32ToU8: {
            auto* in  = static_cast<const float*>(ptrs[st.inputs[0]]);
            auto* out = static_cast<uint8_t*>(ptrs[st.outputs[0]]);
            simd::convert_f32_to_u8(in, out, width_ * height_ * 3);
            break;
        }
        case StageOp::kBackgroundSubtract: {
            auto* frame = static_cast<const uint8_t*>(ptrs[st.inputs[0]]);
            auto* bg    = static_cast<const uint8_t*>(ptrs[st.inputs[1]]);
            auto* mask  = static_cast<uint8_t*>(ptrs[st.outputs[0]]);
            simd::SubtractorConfig cfg{config.motion_threshold, width_, height_};
            simd::subtract_background_neon(frame, bg, mask, cfg);
            break;
        }
        case StageOp::kAeAwb: {
            auto* in  = static_cast<const uint8_t*>(ptrs[st.inputs[0]]);
            auto* out = static_cast<uint8_t*>(ptrs[st.outputs[0]]);
            simd::apply_ae_awb_neon(in, out, config.ae_awb);
            break;
        }
        case StageOp::kBoxBlur: {
            auto* in   = static_cast<const uint8_t*>(ptrs[st.inputs[0]]);
            auto* temp = static_cast<uint8_t*>(ptrs[st.inputs[1]]);
            auto* out  = static_cast<uint8_t*>(ptrs[st.outputs[0]]);
            simd::box_blur_5x5(in, out, temp, width_, height_);
            break;
        }
        case StageOp::kAfBlend: {
            auto* frame   = static_cast<const uint8_t*>(ptrs[st.inputs[0]]);
            auto* alpha   = static_cast<const float*>(ptrs[st.inputs[1]]);
            auto* blurred = static_cast<const uint8_t*>(ptrs[st.inputs[2]]);
            auto* out     = static_cast<uint8_t*>(ptrs[st.outputs[0]]);
            simd::AFBlendConfig cfg{width_ * height_};
            simd::apply_af_blend_neon(frame, alpha, blurred, out, cfg);
            break;
        }
        case StageOp::kMemcpy: {
            auto* in  = static_cast<const uint8_t*>(ptrs[st.inputs[0]]);
            auto* out = static_cast<uint8_t*>(ptrs[st.outputs[0]]);
            auto size = buffers_[st.inputs[0]].size_bytes;
            // For external buffers (size_bytes == 0), compute from dimensions
            if (size == 0) size = static_cast<uint32_t>(width_) * height_ * 3;
            std::memcpy(out, in, size);
            break;
        }
        }

        if (stage_times_ns) {
            stage_times_ns[s] = now_ns() - t0;
        }
    }
}

}  // namespace streetscope
