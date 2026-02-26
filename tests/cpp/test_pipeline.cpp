#include <gtest/gtest.h>
#include <streetscope/simd/pipeline.h>
#include <streetscope/simd/accumulator.h>
#include <streetscope/simd/subtractor.h>
#include <streetscope/simd/isp_correction.h>
#include <vector>
#include <cstring>
#include <cmath>

using namespace streetscope::simd;

// Helper: build a PipelineConfig with identity ISP (pass-through)
static PipelineConfig make_identity_config(int width, int height, float ema_alpha, uint8_t threshold) {
    PipelineConfig config{};
    config.ema_alpha = ema_alpha;
    config.motion_threshold = threshold;
    config.width = width;
    config.height = height;
    for (int i = 0; i < 256; i++) config.ae_awb.lut[i] = static_cast<uint8_t>(i);
    config.ae_awb.gain_b = 1.0f;
    config.ae_awb.gain_g = 1.0f;
    config.ae_awb.gain_r = 1.0f;
    config.ae_awb.width = width;
    config.ae_awb.height = height;
    return config;
}

class PipelineIdentityISPTest : public ::testing::Test {
protected:
    static constexpr int kWidth = 16;
    static constexpr int kHeight = 8;
    static constexpr int kPixels = kWidth * kHeight;
    static constexpr int kBytes = kPixels * 3;
    static constexpr float kAlpha = 0.5f;
    static constexpr uint8_t kThreshold = 15;
};

TEST_F(PipelineIdentityISPTest, MaskMatchesSeparateKernels) {
    // Create a frame with some variation
    std::vector<uint8_t> frame(kBytes);
    for (int i = 0; i < kBytes; i++) {
        frame[i] = static_cast<uint8_t>((i * 7 + 13) % 256);
    }

    // Background starts different from frame
    std::vector<float> bg_fused(kBytes);
    std::vector<float> bg_separate(kBytes);
    for (int i = 0; i < kBytes; i++) {
        bg_fused[i] = static_cast<float>((i * 3 + 50) % 256);
        bg_separate[i] = bg_fused[i];
    }

    // Fused path: identity ISP + zero alpha map = display equals frame
    std::vector<uint8_t> mask_fused(kPixels);
    std::vector<uint8_t> display_fused(kBytes);
    std::vector<float> alpha_map(kPixels, 0.0f);

    PipelineConfig config = make_identity_config(kWidth, kHeight, kAlpha, kThreshold);

    process_frame(
        frame.data(), bg_fused.data(),
        mask_fused.data(), display_fused.data(),
        alpha_map.data(), config
    );

    // Separate path: same operations individually
    // 1. Convert frame to float, run EMA
    std::vector<float> frame_f(kBytes);
    for (int i = 0; i < kBytes; i++) {
        frame_f[i] = static_cast<float>(frame[i]);
    }
    AccumulatorConfig acc_cfg{kAlpha, kBytes};
    accumulate_ema_neon(frame_f.data(), bg_separate.data(), acc_cfg);

    // 2. Convert background to uint8, run subtract
    std::vector<uint8_t> bg_u8(kBytes);
    for (int i = 0; i < kBytes; i++) {
        bg_u8[i] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, bg_separate[i])));
    }
    std::vector<uint8_t> mask_separate(kPixels);
    SubtractorConfig sub_cfg{kThreshold, kWidth, kHeight};
    subtract_background_neon(frame.data(), bg_u8.data(), mask_separate.data(), sub_cfg);

    // Verify masks match
    for (int i = 0; i < kPixels; i++) {
        EXPECT_EQ(mask_fused[i], mask_separate[i]) << "mask index " << i;
    }

    // Verify backgrounds match
    for (int i = 0; i < kBytes; i++) {
        EXPECT_NEAR(bg_fused[i], bg_separate[i], 1e-5f) << "bg index " << i;
    }

    // Identity ISP + zero alpha → display equals frame
    EXPECT_EQ(std::memcmp(display_fused.data(), frame.data(), kBytes), 0);
}

class PipelineFullISPTest : public ::testing::Test {
protected:
    static constexpr int kWidth = 16;
    static constexpr int kHeight = 8;
    static constexpr int kPixels = kWidth * kHeight;
    static constexpr int kBytes = kPixels * 3;
};

TEST_F(PipelineFullISPTest, DisplayMatchesSeparateKernels) {
    std::vector<uint8_t> frame(kBytes);
    for (int i = 0; i < kBytes; i++) {
        frame[i] = static_cast<uint8_t>((i * 11 + 7) % 256);
    }

    std::vector<float> bg_fused(kBytes, 128.0f);
    std::vector<float> bg_separate(kBytes, 128.0f);

    PipelineConfig config = make_identity_config(kWidth, kHeight, 0.1f, 20);

    // Uniform alpha map (no sharpening variation)
    std::vector<float> alpha_map(kPixels, 0.5f);

    std::vector<uint8_t> mask_fused(kPixels);
    std::vector<uint8_t> display_fused(kBytes);

    process_frame(
        frame.data(), bg_fused.data(),
        mask_fused.data(), display_fused.data(),
        alpha_map.data(), config
    );

    // Separate path for comparison
    std::vector<float> frame_f(kBytes);
    for (int i = 0; i < kBytes; i++) {
        frame_f[i] = static_cast<float>(frame[i]);
    }
    AccumulatorConfig acc_cfg{0.1f, kBytes};
    accumulate_ema_neon(frame_f.data(), bg_separate.data(), acc_cfg);

    std::vector<uint8_t> bg_u8(kBytes);
    for (int i = 0; i < kBytes; i++) {
        bg_u8[i] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, bg_separate[i])));
    }
    std::vector<uint8_t> mask_separate(kPixels);
    SubtractorConfig sub_cfg{20, kWidth, kHeight};
    subtract_background_neon(frame.data(), bg_u8.data(), mask_separate.data(), sub_cfg);

    // AE+AWB with identity params
    AEAWBConfig ae_cfg{};
    for (int i = 0; i < 256; i++) ae_cfg.lut[i] = static_cast<uint8_t>(i);
    ae_cfg.gain_b = 1.0f;
    ae_cfg.gain_g = 1.0f;
    ae_cfg.gain_r = 1.0f;
    ae_cfg.width = kWidth;
    ae_cfg.height = kHeight;

    std::vector<uint8_t> corrected(kBytes);
    apply_ae_awb_neon(frame.data(), corrected.data(), ae_cfg);

    // Box blur the corrected frame
    std::vector<uint8_t> blurred(kBytes);
    box_blur_5x5(corrected.data(), blurred.data(), kWidth, kHeight);

    // AF blend
    std::vector<uint8_t> display_separate(kBytes);
    AFBlendConfig af_cfg{kPixels};
    apply_af_blend_neon(corrected.data(), alpha_map.data(), blurred.data(), display_separate.data(), af_cfg);

    // Verify masks match
    for (int i = 0; i < kPixels; i++) {
        EXPECT_EQ(mask_fused[i], mask_separate[i]) << "mask index " << i;
    }

    // Verify display matches (exact, same code path)
    for (int i = 0; i < kBytes; i++) {
        EXPECT_EQ(display_fused[i], display_separate[i]) << "display index " << i;
    }
}

class BoxBlurTest : public ::testing::Test {};

TEST_F(BoxBlurTest, UniformInputUnchanged) {
    static constexpr int kW = 16;
    static constexpr int kH = 8;
    static constexpr int kBytes = kW * kH * 3;

    // Uniform frame: blur should not change it
    std::vector<uint8_t> input(kBytes, 100);
    std::vector<uint8_t> output(kBytes, 0);

    box_blur_5x5(input.data(), output.data(), kW, kH);

    for (int i = 0; i < kBytes; i++) {
        EXPECT_EQ(output[i], 100) << "index " << i;
    }
}

TEST_F(BoxBlurTest, ReducesContrast) {
    static constexpr int kW = 16;
    static constexpr int kH = 16;
    static constexpr int kBytes = kW * kH * 3;

    // Checkerboard pattern: alternating 0 and 255
    std::vector<uint8_t> input(kBytes);
    for (int y = 0; y < kH; y++) {
        for (int x = 0; x < kW; x++) {
            uint8_t val = ((x + y) % 2 == 0) ? 0 : 255;
            int idx = (y * kW + x) * 3;
            input[idx] = input[idx + 1] = input[idx + 2] = val;
        }
    }

    std::vector<uint8_t> output(kBytes);
    box_blur_5x5(input.data(), output.data(), kW, kH);

    // Interior pixels should be closer to 128 than original 0/255
    int center_idx = (kH / 2 * kW + kW / 2) * 3;
    int diff_from_mean = std::abs(static_cast<int>(output[center_idx]) - 128);
    EXPECT_LT(diff_from_mean, 64) << "blur should reduce contrast";
}
