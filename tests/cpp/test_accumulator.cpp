#include <gtest/gtest.h>
#include <streetscope/simd/accumulator.h>
#include <vector>
#include <cmath>

using namespace streetscope::simd;

class AccumulatorScalarTest : public ::testing::Test {
protected:
    static constexpr float kTol = 1e-6f;
};

TEST_F(AccumulatorScalarTest, AlphaZeroNoChange) {
    std::vector<float> frame = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> bg = {10.0f, 20.0f, 30.0f, 40.0f};
    std::vector<float> expected = bg;

    AccumulatorConfig config{0.0f, 4};
    accumulate_ema_scalar(frame.data(), bg.data(), config);

    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(bg[i], expected[i], kTol) << "index " << i;
    }
}

TEST_F(AccumulatorScalarTest, AlphaOneFullReplace) {
    std::vector<float> frame = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> bg = {10.0f, 20.0f, 30.0f, 40.0f};

    AccumulatorConfig config{1.0f, 4};
    accumulate_ema_scalar(frame.data(), bg.data(), config);

    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(bg[i], frame[i], kTol) << "index " << i;
    }
}

TEST_F(AccumulatorScalarTest, AlphaHalfBlend) {
    std::vector<float> frame = {0.0f, 100.0f};
    std::vector<float> bg = {100.0f, 0.0f};

    AccumulatorConfig config{0.5f, 2};
    accumulate_ema_scalar(frame.data(), bg.data(), config);

    EXPECT_NEAR(bg[0], 50.0f, kTol);
    EXPECT_NEAR(bg[1], 50.0f, kTol);
}

TEST_F(AccumulatorScalarTest, TypicalAlpha) {
    std::vector<float> frame = {200.0f};
    std::vector<float> bg = {100.0f};

    AccumulatorConfig config{0.05f, 1};
    accumulate_ema_scalar(frame.data(), bg.data(), config);

    EXPECT_NEAR(bg[0], 105.0f, kTol);
}

TEST_F(AccumulatorScalarTest, LargeArray) {
    const int n = 512 * 288 * 3;
    std::vector<float> frame(n, 200.0f);
    std::vector<float> bg(n, 100.0f);

    AccumulatorConfig config{0.1f, n};
    accumulate_ema_scalar(frame.data(), bg.data(), config);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(bg[i], 110.0f, kTol) << "index " << i;
    }
}

TEST_F(AccumulatorScalarTest, SingleElement) {
    std::vector<float> frame = {42.0f};
    std::vector<float> bg = {0.0f};

    AccumulatorConfig config{0.25f, 1};
    accumulate_ema_scalar(frame.data(), bg.data(), config);

    EXPECT_NEAR(bg[0], 10.5f, kTol);
}

class AccumulatorNeonTest : public ::testing::Test {
protected:
    // NEON vmlaq_f32 (fused multiply-add) has slightly different rounding
    // than separate scalar mul+add, so we allow a wider tolerance.
    static constexpr float kTol = 1e-4f;
};

TEST_F(AccumulatorNeonTest, MatchesScalarSmall) {
    std::vector<float> frame = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> bg_scalar = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
    std::vector<float> bg_neon = bg_scalar;

    AccumulatorConfig config{0.3f, 8};
    accumulate_ema_scalar(frame.data(), bg_scalar.data(), config);
    accumulate_ema_neon(frame.data(), bg_neon.data(), config);

    for (int i = 0; i < 8; i++) {
        EXPECT_NEAR(bg_neon[i], bg_scalar[i], kTol) << "index " << i;
    }
}

TEST_F(AccumulatorNeonTest, MatchesScalarNonMultipleOfFour) {
    std::vector<float> frame = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    std::vector<float> bg_scalar = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f};
    std::vector<float> bg_neon = bg_scalar;

    AccumulatorConfig config{0.15f, 7};
    accumulate_ema_scalar(frame.data(), bg_scalar.data(), config);
    accumulate_ema_neon(frame.data(), bg_neon.data(), config);

    for (int i = 0; i < 7; i++) {
        EXPECT_NEAR(bg_neon[i], bg_scalar[i], kTol) << "index " << i;
    }
}

TEST_F(AccumulatorNeonTest, MatchesScalarLargeArray) {
    const int n = 512 * 288 * 3;
    std::vector<float> frame(n);
    std::vector<float> bg_scalar(n);
    for (int i = 0; i < n; i++) {
        frame[i] = static_cast<float>(i % 256);
        bg_scalar[i] = static_cast<float>((i * 7) % 256);
    }
    std::vector<float> bg_neon = bg_scalar;

    AccumulatorConfig config{0.05f, n};
    accumulate_ema_scalar(frame.data(), bg_scalar.data(), config);
    accumulate_ema_neon(frame.data(), bg_neon.data(), config);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(bg_neon[i], bg_scalar[i], kTol) << "index " << i;
    }
}

TEST_F(AccumulatorNeonTest, MatchesScalarSingleElement) {
    std::vector<float> frame = {42.0f};
    std::vector<float> bg_scalar = {100.0f};
    std::vector<float> bg_neon = bg_scalar;

    AccumulatorConfig config{0.7f, 1};
    accumulate_ema_scalar(frame.data(), bg_scalar.data(), config);
    accumulate_ema_neon(frame.data(), bg_neon.data(), config);

    EXPECT_NEAR(bg_neon[0], bg_scalar[0], kTol);
}
