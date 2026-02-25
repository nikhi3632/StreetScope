#include <gtest/gtest.h>
#include <streetscope/simd/subtractor.h>
#include <cstddef>
#include <vector>

using namespace streetscope::simd;

class SubtractorScalarTest : public ::testing::Test {};

TEST_F(SubtractorScalarTest, IdenticalFramesAllStatic) {
    const int w = 4, h = 2;
    std::vector<uint8_t> frame(static_cast<size_t>(w) * h * 3, 128);
    std::vector<uint8_t> bg = frame;
    std::vector<uint8_t> mask(static_cast<size_t>(w) * h, 99);

    SubtractorConfig config{15, w, h};
    subtract_background_scalar(frame.data(), bg.data(), mask.data(), config);

    for (int i = 0; i < w * h; i++) {
        EXPECT_EQ(mask[i], 0) << "pixel " << i;
    }
}

TEST_F(SubtractorScalarTest, LargeDiffAllMotion) {
    const int w = 4, h = 2;
    std::vector<uint8_t> frame(static_cast<size_t>(w) * h * 3, 200);
    std::vector<uint8_t> bg(static_cast<size_t>(w) * h * 3, 100);
    std::vector<uint8_t> mask(static_cast<size_t>(w) * h, 0);

    SubtractorConfig config{30, w, h};
    subtract_background_scalar(frame.data(), bg.data(), mask.data(), config);

    for (int i = 0; i < w * h; i++) {
        EXPECT_EQ(mask[i], 255) << "pixel " << i;
    }
}

TEST_F(SubtractorScalarTest, BelowThresholdIsStatic) {
    const int w = 2, h = 2;
    std::vector<uint8_t> frame(static_cast<size_t>(w) * h * 3, 110);
    std::vector<uint8_t> bg(static_cast<size_t>(w) * h * 3, 100);
    std::vector<uint8_t> mask(static_cast<size_t>(w) * h, 99);

    SubtractorConfig config{30, w, h};
    subtract_background_scalar(frame.data(), bg.data(), mask.data(), config);

    for (int i = 0; i < w * h; i++) {
        EXPECT_EQ(mask[i], 0) << "pixel " << i;
    }
}

TEST_F(SubtractorScalarTest, ExactlyAtThresholdIsStatic) {
    const int w = 1, h = 1;
    std::vector<uint8_t> frame = {130, 130, 130};
    std::vector<uint8_t> bg = {100, 100, 100};
    std::vector<uint8_t> mask(1, 99);

    SubtractorConfig config{30, w, h};
    subtract_background_scalar(frame.data(), bg.data(), mask.data(), config);

    EXPECT_EQ(mask[0], 0);
}

TEST_F(SubtractorScalarTest, OneAboveThresholdIsMotion) {
    const int w = 1, h = 1;
    std::vector<uint8_t> frame = {131, 100, 100};
    std::vector<uint8_t> bg = {100, 100, 100};
    std::vector<uint8_t> mask(1, 0);

    SubtractorConfig config{30, w, h};
    subtract_background_scalar(frame.data(), bg.data(), mask.data(), config);

    EXPECT_EQ(mask[0], 255);
}

TEST_F(SubtractorScalarTest, MaxAcrossChannels) {
    const int w = 1, h = 1;
    std::vector<uint8_t> frame = {105, 110, 140};
    std::vector<uint8_t> bg = {100, 100, 100};
    std::vector<uint8_t> mask(1, 0);

    SubtractorConfig config{30, w, h};
    subtract_background_scalar(frame.data(), bg.data(), mask.data(), config);

    EXPECT_EQ(mask[0], 255);
}

TEST_F(SubtractorScalarTest, MixedPixels) {
    const int w = 2, h = 1;
    std::vector<uint8_t> frame = {200, 200, 200, 105, 105, 105};
    std::vector<uint8_t> bg = {100, 100, 100, 100, 100, 100};
    std::vector<uint8_t> mask(2, 0);

    SubtractorConfig config{30, w, h};
    subtract_background_scalar(frame.data(), bg.data(), mask.data(), config);

    EXPECT_EQ(mask[0], 255);
    EXPECT_EQ(mask[1], 0);
}

TEST_F(SubtractorScalarTest, BackgroundBrighterThanFrame) {
    const int w = 1, h = 1;
    std::vector<uint8_t> frame = {50, 50, 50};
    std::vector<uint8_t> bg = {100, 100, 100};
    std::vector<uint8_t> mask(1, 0);

    SubtractorConfig config{30, w, h};
    subtract_background_scalar(frame.data(), bg.data(), mask.data(), config);

    EXPECT_EQ(mask[0], 255);
}

class SubtractorNeonTest : public ::testing::Test {};

TEST_F(SubtractorNeonTest, MatchesScalarIdentical) {
    const int w = 8, h = 4;
    std::vector<uint8_t> frame(static_cast<size_t>(w) * h * 3, 128);
    std::vector<uint8_t> bg = frame;
    std::vector<uint8_t> mask_scalar(static_cast<size_t>(w) * h, 99);
    std::vector<uint8_t> mask_neon(static_cast<size_t>(w) * h, 99);

    SubtractorConfig config{15, w, h};
    subtract_background_scalar(frame.data(), bg.data(), mask_scalar.data(), config);
    subtract_background_neon(frame.data(), bg.data(), mask_neon.data(), config);

    for (int i = 0; i < w * h; i++) {
        EXPECT_EQ(mask_neon[i], mask_scalar[i]) << "pixel " << i;
    }
}

TEST_F(SubtractorNeonTest, MatchesScalarMixed) {
    const int w = 16, h = 8;
    const int num_pixels = w * h;
    std::vector<uint8_t> frame(static_cast<size_t>(num_pixels) * 3);
    std::vector<uint8_t> bg(static_cast<size_t>(num_pixels) * 3);
    for (size_t i = 0; i < static_cast<size_t>(num_pixels) * 3; i++) {
        frame[i] = static_cast<uint8_t>((i * 13) % 256);
        bg[i] = static_cast<uint8_t>((i * 7 + 50) % 256);
    }
    std::vector<uint8_t> mask_scalar(num_pixels, 0);
    std::vector<uint8_t> mask_neon(num_pixels, 0);

    SubtractorConfig config{30, w, h};
    subtract_background_scalar(frame.data(), bg.data(), mask_scalar.data(), config);
    subtract_background_neon(frame.data(), bg.data(), mask_neon.data(), config);

    for (int i = 0; i < num_pixels; i++) {
        EXPECT_EQ(mask_neon[i], mask_scalar[i]) << "pixel " << i;
    }
}

TEST_F(SubtractorNeonTest, MatchesScalarLargeFrame) {
    const int w = 512, h = 288;
    const int num_pixels = w * h;
    std::vector<uint8_t> frame(static_cast<size_t>(num_pixels) * 3);
    std::vector<uint8_t> bg(static_cast<size_t>(num_pixels) * 3);
    for (size_t i = 0; i < static_cast<size_t>(num_pixels) * 3; i++) {
        frame[i] = static_cast<uint8_t>(i % 256);
        bg[i] = static_cast<uint8_t>((i + 100) % 256);
    }
    std::vector<uint8_t> mask_scalar(num_pixels, 0);
    std::vector<uint8_t> mask_neon(num_pixels, 0);

    SubtractorConfig config{25, w, h};
    subtract_background_scalar(frame.data(), bg.data(), mask_scalar.data(), config);
    subtract_background_neon(frame.data(), bg.data(), mask_neon.data(), config);

    for (int i = 0; i < num_pixels; i++) {
        EXPECT_EQ(mask_neon[i], mask_scalar[i]) << "pixel " << i;
    }
}

TEST_F(SubtractorNeonTest, MatchesScalarNonMultipleOfSixteen) {
    const int w = 5, h = 3;
    const int num_pixels = w * h;
    std::vector<uint8_t> frame(static_cast<size_t>(num_pixels) * 3);
    std::vector<uint8_t> bg(static_cast<size_t>(num_pixels) * 3);
    for (size_t i = 0; i < static_cast<size_t>(num_pixels) * 3; i++) {
        frame[i] = static_cast<uint8_t>((i * 17) % 256);
        bg[i] = static_cast<uint8_t>((i * 3 + 80) % 256);
    }
    std::vector<uint8_t> mask_scalar(num_pixels, 0);
    std::vector<uint8_t> mask_neon(num_pixels, 0);

    SubtractorConfig config{20, w, h};
    subtract_background_scalar(frame.data(), bg.data(), mask_scalar.data(), config);
    subtract_background_neon(frame.data(), bg.data(), mask_neon.data(), config);

    for (int i = 0; i < num_pixels; i++) {
        EXPECT_EQ(mask_neon[i], mask_scalar[i]) << "pixel " << i;
    }
}
