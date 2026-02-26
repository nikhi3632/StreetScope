#pragma once
#include <streetscope/plan_executor/plan.h>
#include <streetscope/simd/pipeline.h>
#include <cstdint>
#include <memory>
#include <vector>

namespace streetscope {

class PlanExecutor {
public:
    explicit PlanExecutor(const uint8_t* plan_data, size_t plan_size);

    void run_frame(
        const uint8_t* frame_u8,
        float* background_f32,
        uint8_t* mask_out,
        uint8_t* display_out,
        const float* af_alpha_map,
        const simd::PipelineConfig& config,
        uint64_t* stage_times_ns = nullptr
    );

    int width() const { return width_; }
    int height() const { return height_; }
    size_t arena_size() const { return arena_size_; }
    int num_stages() const { return static_cast<int>(stages_.size()); }

private:
    std::vector<StageDesc> stages_;
    std::vector<BufferDesc> buffers_;
    int width_ = 0;
    int height_ = 0;
    uint32_t arena_size_ = 0;
    std::unique_ptr<uint8_t, void(*)(void*)> arena_;
};

}  // namespace streetscope
