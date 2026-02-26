#include <streetscope/plan_executor/plan.h>
#include <streetscope/plan_executor/plan_executor.h>
#include <streetscope/simd/pipeline.h>
#include <streetscope/simd/accumulator.h>
#include <streetscope/simd/subtractor.h>
#include <streetscope/simd/isp_correction.h>
#include <gtest/gtest.h>
#include <vector>
#include <cstring>

using namespace streetscope;
using namespace streetscope::simd;

// ---------------------------------------------------------------------------
// Helper: build the full 7-stage plan (ISP = AE+AWB+AF, always)
// ---------------------------------------------------------------------------
static std::vector<uint8_t> build_test_plan(int width, int height) {
    const int pixels = width * height;
    const int bytes3 = pixels * 3;

    // Define buffers: 5 external + 5 arena temporaries = 10 total
    std::vector<BufferDesc> buffers;
    // External buffers (indices 0-4): offset=0, size=0
    for (int i = 0; i < 5; i++) {
        BufferDesc bd{};
        bd.offset = 0;
        bd.size_bytes = 0;
        bd.dtype = (i == 1 || i == 4) ? DType::kFloat32 : DType::kUint8;
        bd.padding[0] = bd.padding[1] = bd.padding[2] = 0;
        buffers.push_back(bd);
    }

    // Arena buffers with lifetime packing
    auto align16 = [](uint32_t s) -> uint32_t { return (s + 15) & ~15u; };

    uint32_t frame_f32_size = align16(static_cast<uint32_t>(bytes3) * 4);
    uint32_t u8_size = align16(static_cast<uint32_t>(bytes3));

    // Buffer 5: frame_f32 at offset 0 (stages 0-1)
    {
        BufferDesc bd{};
        bd.offset = 0;
        bd.size_bytes = frame_f32_size;
        bd.dtype = DType::kFloat32;
        bd.padding[0] = bd.padding[1] = bd.padding[2] = 0;
        buffers.push_back(bd);
    }
    // Buffer 6: bg_u8 at offset 0 (stages 2-3, reuses frame_f32 space)
    {
        BufferDesc bd{};
        bd.offset = 0;
        bd.size_bytes = u8_size;
        bd.dtype = DType::kUint8;
        bd.padding[0] = bd.padding[1] = bd.padding[2] = 0;
        buffers.push_back(bd);
    }
    // Buffer 7: corrected at offset 0 (stages 4-6, reuses frame_f32 space)
    {
        BufferDesc bd{};
        bd.offset = 0;
        bd.size_bytes = u8_size;
        bd.dtype = DType::kUint8;
        bd.padding[0] = bd.padding[1] = bd.padding[2] = 0;
        buffers.push_back(bd);
    }
    // Buffer 8: blurred at offset u8_size (stages 5-6)
    {
        BufferDesc bd{};
        bd.offset = u8_size;
        bd.size_bytes = u8_size;
        bd.dtype = DType::kUint8;
        bd.padding[0] = bd.padding[1] = bd.padding[2] = 0;
        buffers.push_back(bd);
    }
    // Buffer 9: blur_temp at offset u8_size * 2 (stage 5 only)
    {
        BufferDesc bd{};
        bd.offset = u8_size * 2;
        bd.size_bytes = u8_size;
        bd.dtype = DType::kUint8;
        bd.padding[0] = bd.padding[1] = bd.padding[2] = 0;
        buffers.push_back(bd);
    }

    uint32_t arena_size = std::max(frame_f32_size, u8_size * 3);

    // Define all 7 stages
    std::vector<StageDesc> stages;

    auto make_stage = [](StageOp op, uint8_t i0, uint8_t i1, uint8_t i2, uint8_t i3,
                         uint8_t o0, uint8_t o1) -> StageDesc {
        StageDesc sd{};
        sd.op = op;
        sd.inputs[0] = i0; sd.inputs[1] = i1;
        sd.inputs[2] = i2; sd.inputs[3] = i3;
        sd.outputs[0] = o0; sd.outputs[1] = o1;
        sd.padding = 0;
        return sd;
    };

    const uint8_t N = kBufferNone;

    // Stage 0: u8_to_f32: frame_u8(0) -> frame_f32(5)
    stages.push_back(make_stage(StageOp::kU8ToF32, 0, N, N, N, 5, N));
    // Stage 1: ema_accumulate: frame_f32(5), background(1)
    stages.push_back(make_stage(StageOp::kEmaAccumulate, 5, 1, N, N, N, N));
    // Stage 2: f32_to_u8: background(1) -> bg_u8(6)
    stages.push_back(make_stage(StageOp::kF32ToU8, 1, N, N, N, 6, N));
    // Stage 3: background_subtract: frame_u8(0), bg_u8(6) -> mask(2)
    stages.push_back(make_stage(StageOp::kBackgroundSubtract, 0, 6, N, N, 2, N));
    // Stage 4: ae_awb: frame_u8(0) -> corrected(7)
    stages.push_back(make_stage(StageOp::kAeAwb, 0, N, N, N, 7, N));
    // Stage 5: box_blur: corrected(7), blur_temp(9) -> blurred(8)
    stages.push_back(make_stage(StageOp::kBoxBlur, 7, 9, N, N, 8, N));
    // Stage 6: af_blend: corrected(7), af_alpha(4), blurred(8) -> display(3)
    stages.push_back(make_stage(StageOp::kAfBlend, 7, 4, 8, N, 3, N));

    // Serialize
    PlanHeader header{};
    header.magic = kPlanMagic;
    header.version = kPlanVersion;
    header.num_stages = static_cast<uint16_t>(stages.size());
    header.num_buffers = static_cast<uint16_t>(buffers.size());
    header.reserved = 0;
    header.arena_size_bytes = arena_size;
    header.width = static_cast<uint32_t>(width);
    header.height = static_cast<uint32_t>(height);

    size_t total = sizeof(PlanHeader)
                   + stages.size() * sizeof(StageDesc)
                   + buffers.size() * sizeof(BufferDesc);
    std::vector<uint8_t> plan(total);
    size_t offset = 0;
    std::memcpy(plan.data() + offset, &header, sizeof(header));
    offset += sizeof(header);
    std::memcpy(plan.data() + offset, stages.data(), stages.size() * sizeof(StageDesc));
    offset += stages.size() * sizeof(StageDesc);
    std::memcpy(plan.data() + offset, buffers.data(), buffers.size() * sizeof(BufferDesc));

    return plan;
}

// ---------------------------------------------------------------------------
// Helper: build a PipelineConfig with identity ISP
// ---------------------------------------------------------------------------
static PipelineConfig make_config(
    int width, int height,
    float ema_alpha = 0.5f, uint8_t threshold = 15
) {
    PipelineConfig config{};
    config.ema_alpha = ema_alpha;
    config.motion_threshold = threshold;
    config.width = width;
    config.height = height;

    for (int i = 0; i < 256; i++) {
        config.ae_awb.lut[i] = static_cast<uint8_t>(i);
    }
    config.ae_awb.gain_b = 1.0f;
    config.ae_awb.gain_g = 1.0f;
    config.ae_awb.gain_r = 1.0f;
    config.ae_awb.width = width;
    config.ae_awb.height = height;

    return config;
}

// ===========================================================================
// Test 1: FullISP_MatchesProcessFrame (identity ISP, zero alpha)
// ===========================================================================
class PlanExecutorIdentityTest : public ::testing::Test {
protected:
    static constexpr int kWidth = 16;
    static constexpr int kHeight = 8;
    static constexpr int kPixels = kWidth * kHeight;
    static constexpr int kBytes = kPixels * 3;
};

TEST_F(PlanExecutorIdentityTest, IdentityISP_MatchesProcessFrame) {
    auto plan_data = build_test_plan(kWidth, kHeight);

    std::vector<uint8_t> frame(kBytes);
    for (int i = 0; i < kBytes; i++) {
        frame[i] = static_cast<uint8_t>((i * 7 + 13) % 256);
    }

    std::vector<float> bg_ref(kBytes);
    std::vector<float> bg_plan(kBytes);
    for (int i = 0; i < kBytes; i++) {
        auto val = static_cast<float>((i * 3 + 50) % 256);
        bg_ref[i] = val;
        bg_plan[i] = val;
    }

    // Zero alpha map = fully sharp (AF is a no-op)
    std::vector<float> alpha_map(kPixels, 0.0f);

    PipelineConfig config = make_config(kWidth, kHeight);

    // Reference: process_frame
    std::vector<uint8_t> mask_ref(kPixels);
    std::vector<uint8_t> display_ref(kBytes);
    process_frame(
        frame.data(), bg_ref.data(),
        mask_ref.data(), display_ref.data(),
        alpha_map.data(), config
    );

    // PlanExecutor path
    PlanExecutor executor(plan_data.data(), plan_data.size());
    std::vector<uint8_t> mask_plan(kPixels);
    std::vector<uint8_t> display_plan(kBytes);
    executor.run_frame(
        frame.data(), bg_plan.data(),
        mask_plan.data(), display_plan.data(),
        alpha_map.data(), config
    );

    for (int i = 0; i < kPixels; i++) {
        EXPECT_EQ(mask_plan[i], mask_ref[i]) << "mask index " << i;
    }
    for (int i = 0; i < kBytes; i++) {
        EXPECT_EQ(display_plan[i], display_ref[i]) << "display index " << i;
    }
    for (int i = 0; i < kBytes; i++) {
        EXPECT_NEAR(bg_plan[i], bg_ref[i], 1e-5f) << "bg index " << i;
    }
}

// ===========================================================================
// Test 2: FullISP_WithAlpha_MatchesProcessFrame
// ===========================================================================
class PlanExecutorFullISPTest : public ::testing::Test {
protected:
    static constexpr int kWidth = 16;
    static constexpr int kHeight = 8;
    static constexpr int kPixels = kWidth * kHeight;
    static constexpr int kBytes = kPixels * 3;
};

TEST_F(PlanExecutorFullISPTest, FullISP_MatchesProcessFrame) {
    auto plan_data = build_test_plan(kWidth, kHeight);

    std::vector<uint8_t> frame(kBytes);
    for (int i = 0; i < kBytes; i++) {
        frame[i] = static_cast<uint8_t>((i * 11 + 7) % 256);
    }

    std::vector<float> bg_ref(kBytes, 128.0f);
    std::vector<float> bg_plan(kBytes, 128.0f);

    // Non-trivial alpha map
    std::vector<float> alpha_map(kPixels, 0.5f);

    PipelineConfig config = make_config(kWidth, kHeight, 0.1f, 20);

    // Reference
    std::vector<uint8_t> mask_ref(kPixels);
    std::vector<uint8_t> display_ref(kBytes);
    process_frame(
        frame.data(), bg_ref.data(),
        mask_ref.data(), display_ref.data(),
        alpha_map.data(), config
    );

    // PlanExecutor
    PlanExecutor executor(plan_data.data(), plan_data.size());
    std::vector<uint8_t> mask_plan(kPixels);
    std::vector<uint8_t> display_plan(kBytes);
    executor.run_frame(
        frame.data(), bg_plan.data(),
        mask_plan.data(), display_plan.data(),
        alpha_map.data(), config
    );

    for (int i = 0; i < kPixels; i++) {
        EXPECT_EQ(mask_plan[i], mask_ref[i]) << "mask index " << i;
    }
    for (int i = 0; i < kBytes; i++) {
        EXPECT_EQ(display_plan[i], display_ref[i]) << "display index " << i;
    }
    for (int i = 0; i < kBytes; i++) {
        EXPECT_NEAR(bg_plan[i], bg_ref[i], 1e-5f) << "bg index " << i;
    }
}

// ===========================================================================
// Test 3: ArenaAlignment
// ===========================================================================
class PlanExecutorArenaTest : public ::testing::Test {};

TEST_F(PlanExecutorArenaTest, ArenaAlignment) {
    struct TestCase {
        int width;
        int height;
    };
    TestCase cases[] = {
        {16, 8},
        {512, 288},
        {100, 100},
    };

    for (const auto& tc : cases) {
        SCOPED_TRACE("size " + std::to_string(tc.width) + "x" + std::to_string(tc.height));

        auto plan_data = build_test_plan(tc.width, tc.height);

        // Parse header + buffers to verify offsets
        PlanHeader header{};
        std::memcpy(&header, plan_data.data(), sizeof(PlanHeader));

        const uint8_t* cursor = plan_data.data() + sizeof(PlanHeader)
                                + header.num_stages * sizeof(StageDesc);
        for (int i = 0; i < header.num_buffers; i++) {
            BufferDesc bd{};
            std::memcpy(&bd, cursor + i * sizeof(BufferDesc), sizeof(BufferDesc));
            if (bd.size_bytes > 0) {
                EXPECT_EQ(bd.offset % 16, 0u) << "buffer " << i << " offset not 16-aligned";
            }
        }

        // Verify arena_size via PlanExecutor
        PlanExecutor executor(plan_data.data(), plan_data.size());
        EXPECT_EQ(executor.arena_size() % 16, 0u) << "arena_size not 16-aligned";
    }
}

// ===========================================================================
// Test 4: StageTimingPopulated
// ===========================================================================
class PlanExecutorTimingTest : public ::testing::Test {
protected:
    static constexpr int kWidth = 16;
    static constexpr int kHeight = 8;
    static constexpr int kPixels = kWidth * kHeight;
    static constexpr int kBytes = kPixels * 3;
};

TEST_F(PlanExecutorTimingTest, StageTimingPopulated) {
    auto plan_data = build_test_plan(kWidth, kHeight);

    PlanExecutor executor(plan_data.data(), plan_data.size());

    std::vector<uint8_t> frame(kBytes, 100);
    std::vector<float> bg(kBytes, 100.0f);
    std::vector<uint8_t> mask(kPixels);
    std::vector<uint8_t> display(kBytes);
    std::vector<float> alpha_map(kPixels, 0.0f);

    PipelineConfig config = make_config(kWidth, kHeight);

    int num_stages = executor.num_stages();
    std::vector<uint64_t> stage_times(static_cast<size_t>(num_stages), 0);

    executor.run_frame(
        frame.data(), bg.data(),
        mask.data(), display.data(),
        alpha_map.data(), config,
        stage_times.data()
    );

    for (int i = 0; i < num_stages; i++) {
        EXPECT_GT(stage_times[static_cast<size_t>(i)], 0u) << "stage " << i << " timing is zero";
    }
}

// ===========================================================================
// Test 5: InvalidPlanThrows
// ===========================================================================
class PlanExecutorInvalidTest : public ::testing::Test {};

TEST_F(PlanExecutorInvalidTest, WrongMagicThrows) {
    auto plan_data = build_test_plan(16, 8);

    // Corrupt the magic number
    PlanHeader header{};
    std::memcpy(&header, plan_data.data(), sizeof(PlanHeader));
    header.magic = 0xDEADBEEF;
    std::memcpy(plan_data.data(), &header, sizeof(PlanHeader));

    EXPECT_THROW(
        PlanExecutor(plan_data.data(), plan_data.size()),
        std::invalid_argument
    );
}

TEST_F(PlanExecutorInvalidTest, TruncatedDataThrows) {
    // Plan data shorter than PlanHeader
    std::vector<uint8_t> tiny(4, 0);
    EXPECT_THROW(
        PlanExecutor(tiny.data(), tiny.size()),
        std::invalid_argument
    );
}

TEST_F(PlanExecutorInvalidTest, WrongVersionThrows) {
    auto plan_data = build_test_plan(16, 8);

    // Corrupt the version
    PlanHeader header{};
    std::memcpy(&header, plan_data.data(), sizeof(PlanHeader));
    header.version = 99;
    std::memcpy(plan_data.data(), &header, sizeof(PlanHeader));

    EXPECT_THROW(
        PlanExecutor(plan_data.data(), plan_data.size()),
        std::invalid_argument
    );
}
