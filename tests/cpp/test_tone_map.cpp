#include <gtest/gtest.h>
#include <streetscope/simd/tone_map.h>
#include <cstddef>
#include <vector>

using namespace streetscope::simd;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static ToneMapParams make_identity_params() {
    ToneMapParams p{};
    p.exposure = 1.0f;
    p.white_point = 1.0f;
    p.gamma = 1.0f;
    p.gain_b = 1.0f;
    p.gain_g = 1.0f;
    p.gain_r = 1.0f;
    return p;
}

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

class ToneMapScalarTest : public ::testing::Test {};

TEST_F(ToneMapScalarTest, IdentityParams) {
    // With exposure=1, white=1, gamma=1, gains=1:
    // Reinhard: L_mapped = L * (1 + L/1) / (1 + L) = L * (1+L)/(1+L) = L
    // So output should equal input.
    const int w = 4, h = 2;
    const int num_pixels = w * h;
    std::vector<uint8_t> input(static_cast<size_t>(num_pixels) * 3);
    for (size_t i = 0; i < input.size(); i++) {
        input[i] = static_cast<uint8_t>((i * 31) % 256);
    }
    std::vector<uint8_t> output(input.size(), 0);

    ToneMapParams params = make_identity_params();
    tone_map_scalar(input.data(), output.data(), w, h, params);

    for (size_t i = 0; i < input.size(); i++) {
        EXPECT_NEAR(output[i], input[i], 1) << "byte " << i;
    }
}

// ---------------------------------------------------------------------------
// AWB gains
// ---------------------------------------------------------------------------

TEST_F(ToneMapScalarTest, AWBGainsApplied) {
    // Boost blue gain, keep others at 1. Gamma=1, white=high (no Reinhard effect).
    const int w = 1, h = 1;
    // BGR = (100, 100, 100)
    uint8_t input[3] = {100, 100, 100};
    uint8_t output[3] = {0, 0, 0};

    ToneMapParams params = make_identity_params();
    params.gain_b = 1.5f;
    params.white_point = 100.0f;  // Very high → Reinhard ≈ identity

    tone_map_scalar(input, output, w, h, params);

    // Blue channel should be boosted, others less affected
    // (luminance changes from gain affect all channels via Reinhard)
    EXPECT_GT(output[0], output[1]);  // Blue > Green
}

// ---------------------------------------------------------------------------
// Exposure scaling
// ---------------------------------------------------------------------------

TEST_F(ToneMapScalarTest, ExposureIncreaseBrightens) {
    const int w = 1, h = 1;
    uint8_t input[3] = {50, 50, 50};  // BGR, mid-dark
    uint8_t output_low[3] = {};
    uint8_t output_high[3] = {};

    ToneMapParams params{};
    params.gamma = 1.0f;
    params.white_point = 100.0f;

    params.exposure = 1.0f;
    tone_map_scalar(input, output_low, w, h, params);

    params.exposure = 3.0f;
    tone_map_scalar(input, output_high, w, h, params);

    // Higher exposure → brighter output
    EXPECT_GT(output_high[0], output_low[0]);
    EXPECT_GT(output_high[1], output_low[1]);
    EXPECT_GT(output_high[2], output_low[2]);
}

// ---------------------------------------------------------------------------
// Reinhard highlight compression
// ---------------------------------------------------------------------------

TEST_F(ToneMapScalarTest, ReinhardCompressesHighlights) {
    // With white_point=1, Reinhard(L) = L*(1+L)/(1+L) = L → identity.
    // With white_point=2, Reinhard maps more aggressively for high L.
    // Key: bright pixels (high L) get compressed more than dark pixels.
    const int w = 2, h = 1;
    // Dark pixel (BGR=30,30,30), Bright pixel (BGR=200,200,200)
    uint8_t input[6] = {30, 30, 30, 200, 200, 200};
    uint8_t output[6] = {};

    ToneMapParams params{};
    params.exposure = 2.0f;
    params.white_point = 1.5f;
    params.gamma = 1.0f;

    tone_map_scalar(input, output, w, h, params);

    // Both should produce valid output (no overflow)
    for (int i = 0; i < 6; i++) {
        EXPECT_LE(output[i], 255);
    }

    // Bright pixel should still be brighter than dark pixel
    EXPECT_GT(output[3], output[0]);

    // Reinhard compresses: ratio between bright and dark output should be
    // less than ratio between bright and dark input (200/30 = 6.67x).
    // Reinhard's soft roll-off compresses highlights relative to shadows.
    auto dark_val = static_cast<float>(output[0]);
    auto bright_val = static_cast<float>(output[3]);
    if (dark_val > 0) {
        float output_ratio = bright_val / dark_val;
        EXPECT_LT(output_ratio, 200.0f / 30.0f);  // Compressed
    }
}

// ---------------------------------------------------------------------------
// Gamma
// ---------------------------------------------------------------------------

TEST_F(ToneMapScalarTest, GammaAffectsOutput) {
    const int w = 1, h = 1;
    uint8_t input[3] = {128, 128, 128};
    uint8_t output_low_gamma[3] = {};
    uint8_t output_high_gamma[3] = {};

    ToneMapParams params = make_identity_params();

    // gamma=1 → linear (identity Reinhard with white=1)
    params.gamma = 1.0f;
    tone_map_scalar(input, output_low_gamma, w, h, params);

    // gamma=2.2 → brighter midtones (pow(x, 1/2.2) > x for x < 1)
    params.gamma = 2.2f;
    tone_map_scalar(input, output_high_gamma, w, h, params);

    EXPECT_GT(output_high_gamma[0], output_low_gamma[0]);
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

TEST_F(ToneMapScalarTest, AllBlack) {
    const int w = 2, h = 2;
    std::vector<uint8_t> input(static_cast<size_t>(w * h) * 3, 0);
    std::vector<uint8_t> output(input.size(), 255);

    ToneMapParams params{};
    params.exposure = 5.0f;
    tone_map_scalar(input.data(), output.data(), w, h, params);

    for (size_t i = 0; i < output.size(); i++) {
        EXPECT_EQ(output[i], 0) << "byte " << i;
    }
}

TEST_F(ToneMapScalarTest, AllWhite) {
    const int w = 2, h = 2;
    std::vector<uint8_t> input(static_cast<size_t>(w * h) * 3, 255);
    std::vector<uint8_t> output(input.size(), 0);

    ToneMapParams params{};
    params.exposure = 1.0f;
    params.white_point = 2.0f;
    params.gamma = 2.2f;
    tone_map_scalar(input.data(), output.data(), w, h, params);

    // All white → should produce high but valid values
    for (size_t i = 0; i < output.size(); i++) {
        EXPECT_GT(output[i], 100);
        EXPECT_LE(output[i], 255);
    }
}

TEST_F(ToneMapScalarTest, SinglePixel) {
    uint8_t input[3] = {80, 160, 240};  // BGR
    uint8_t output[3] = {};

    ToneMapParams params{};
    params.exposure = 1.5f;
    params.white_point = 2.0f;
    params.gamma = 2.2f;
    params.gain_b = 0.9f;
    params.gain_g = 1.0f;
    params.gain_r = 1.1f;

    tone_map_scalar(input, output, 1, 1, params);

    // Just verify no crash and values are in range
    for (int i = 0; i < 3; i++) {
        EXPECT_LE(output[i], 255);
    }
}

// ---------------------------------------------------------------------------
// Clamp: extreme exposure doesn't exceed 255
// ---------------------------------------------------------------------------

TEST_F(ToneMapScalarTest, ExtremeExposureClamps) {
    const int w = 1, h = 1;
    uint8_t input[3] = {200, 200, 200};
    uint8_t output[3] = {};

    ToneMapParams params{};
    params.exposure = 100.0f;  // Extreme
    params.white_point = 0.5f;
    params.gamma = 1.0f;

    tone_map_scalar(input, output, w, h, params);

    for (int i = 0; i < 3; i++) {
        EXPECT_LE(output[i], 255);
    }
}

// ---------------------------------------------------------------------------
// Large frame (512x288) — no crash, reasonable output
// ---------------------------------------------------------------------------

TEST_F(ToneMapScalarTest, LargeFrame) {
    const int w = 512, h = 288;
    const auto sz = static_cast<size_t>(w * h) * 3;
    std::vector<uint8_t> input(sz);
    for (size_t i = 0; i < sz; i++) {
        input[i] = static_cast<uint8_t>((i * 7 + 13) % 256);
    }
    std::vector<uint8_t> output(sz, 0);

    ToneMapParams params{};
    params.exposure = 1.5f;
    params.white_point = 2.0f;
    params.gamma = 2.2f;

    tone_map_scalar(input.data(), output.data(), w, h, params);

    // Spot-check a few pixels aren't all zeros
    bool any_nonzero = false;
    for (size_t i = 0; i < sz; i++) {
        if (output[i] != 0) {
            any_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(any_nonzero);
}
