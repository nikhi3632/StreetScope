#include <gtest/gtest.h>
#include <streetscope/simd/isp_correction.h>
#include <cstddef>
#include <vector>

using namespace streetscope::simd;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static AEAWBConfig make_identity_config() {
    AEAWBConfig config{};
    for (int i = 0; i < 256; i++) {
        config.lut[i] = static_cast<uint8_t>(i);
    }
    config.gain_b = 1.0f;
    config.gain_g = 1.0f;
    config.gain_r = 1.0f;
    config.width = 0;
    config.height = 0;
    return config;
}

// ---------------------------------------------------------------------------
// AEAWBScalarTest
// ---------------------------------------------------------------------------

class AEAWBScalarTest : public ::testing::Test {};

TEST_F(AEAWBScalarTest, IdentityLutUnityGains) {
    const int w = 4, h = 2;
    const int num_pixels = w * h;
    std::vector<uint8_t> input(static_cast<size_t>(num_pixels) * 3);
    for (size_t i = 0; i < input.size(); i++) {
        input[i] = static_cast<uint8_t>((i * 17) % 256);
    }
    std::vector<uint8_t> output(input.size(), 0);

    AEAWBConfig config = make_identity_config();
    config.width = w;
    config.height = h;
    apply_ae_awb_scalar(input.data(), output.data(), config);

    for (size_t i = 0; i < input.size(); i++) {
        EXPECT_EQ(output[i], input[i]) << "byte " << i;
    }
}

TEST_F(AEAWBScalarTest, LutOnlyNoGains) {
    const int w = 2, h = 1;
    // LUT that doubles values (capped at 255)
    AEAWBConfig config = make_identity_config();
    config.width = w;
    config.height = h;
    for (int i = 0; i < 256; i++) {
        config.lut[i] = static_cast<uint8_t>(std::min(i * 2, 255));
    }

    // Input: B=10, G=20, R=30, B=100, G=120, R=50
    std::vector<uint8_t> input = {10, 20, 30, 100, 120, 50};
    std::vector<uint8_t> output(6, 0);

    apply_ae_awb_scalar(input.data(), output.data(), config);

    EXPECT_EQ(output[0], 20);   // 10 * 2
    EXPECT_EQ(output[1], 40);   // 20 * 2
    EXPECT_EQ(output[2], 60);   // 30 * 2
    EXPECT_EQ(output[3], 200);  // 100 * 2
    EXPECT_EQ(output[4], 240);  // 120 * 2
    EXPECT_EQ(output[5], 100);  // 50 * 2
}

TEST_F(AEAWBScalarTest, GainsOnlyNoLut) {
    const int w = 1, h = 1;
    AEAWBConfig config = make_identity_config();
    config.width = w;
    config.height = h;
    config.gain_b = 1.5f;
    config.gain_g = 1.0f;
    config.gain_r = 0.5f;

    std::vector<uint8_t> input = {100, 100, 100};
    std::vector<uint8_t> output(3, 0);

    apply_ae_awb_scalar(input.data(), output.data(), config);

    EXPECT_EQ(output[0], 150);  // 100 * 1.5
    EXPECT_EQ(output[1], 100);  // 100 * 1.0
    EXPECT_EQ(output[2], 50);   // 100 * 0.5
}

TEST_F(AEAWBScalarTest, ClampAt255) {
    const int w = 1, h = 1;
    AEAWBConfig config = make_identity_config();
    config.width = w;
    config.height = h;
    config.gain_b = 2.0f;
    config.gain_g = 2.0f;
    config.gain_r = 2.0f;

    std::vector<uint8_t> input = {200, 200, 200};
    std::vector<uint8_t> output(3, 0);

    apply_ae_awb_scalar(input.data(), output.data(), config);

    EXPECT_EQ(output[0], 255);  // 200 * 2.0 = 400 -> clamped to 255
    EXPECT_EQ(output[1], 255);
    EXPECT_EQ(output[2], 255);
}

TEST_F(AEAWBScalarTest, LutAndGainsCombined) {
    const int w = 1, h = 1;
    // LUT adds 50 (capped at 255)
    AEAWBConfig config{};
    for (int i = 0; i < 256; i++) {
        config.lut[i] = static_cast<uint8_t>(std::min(i + 50, 255));
    }
    config.gain_b = 1.0f;
    config.gain_g = 1.5f;
    config.gain_r = 0.5f;
    config.width = w;
    config.height = h;

    // Input: B=100, G=100, R=100
    // After LUT: B=150, G=150, R=150
    // After gains: B=150*1.0=150, G=150*1.5=225, R=150*0.5=75
    std::vector<uint8_t> input = {100, 100, 100};
    std::vector<uint8_t> output(3, 0);

    apply_ae_awb_scalar(input.data(), output.data(), config);

    EXPECT_EQ(output[0], 150);
    EXPECT_EQ(output[1], 225);
    EXPECT_EQ(output[2], 75);
}

// ---------------------------------------------------------------------------
// AEAWBNeonTest
// ---------------------------------------------------------------------------

class AEAWBNeonTest : public ::testing::Test {};

TEST_F(AEAWBNeonTest, MatchesScalarSmall) {
    const int w = 8, h = 4;
    const int num_pixels = w * h;
    std::vector<uint8_t> input(static_cast<size_t>(num_pixels) * 3);
    for (size_t i = 0; i < input.size(); i++) {
        input[i] = static_cast<uint8_t>((i * 13 + 7) % 256);
    }
    std::vector<uint8_t> out_scalar(input.size(), 0);
    std::vector<uint8_t> out_neon(input.size(), 0);

    AEAWBConfig config = make_identity_config();
    config.width = w;
    config.height = h;
    config.gain_b = 1.2f;
    config.gain_g = 0.9f;
    config.gain_r = 1.1f;
    for (int i = 0; i < 256; i++) {
        config.lut[i] = static_cast<uint8_t>(std::min(i + 10, 255));
    }

    apply_ae_awb_scalar(input.data(), out_scalar.data(), config);
    apply_ae_awb_neon(input.data(), out_neon.data(), config);

    for (size_t i = 0; i < input.size(); i++) {
        EXPECT_NEAR(out_neon[i], out_scalar[i], 1) << "byte " << i;
    }
}

TEST_F(AEAWBNeonTest, MatchesScalarLarge) {
    const int w = 512, h = 288;
    const int num_pixels = w * h;
    std::vector<uint8_t> input(static_cast<size_t>(num_pixels) * 3);
    for (size_t i = 0; i < input.size(); i++) {
        input[i] = static_cast<uint8_t>((i * 7 + 31) % 256);
    }
    std::vector<uint8_t> out_scalar(input.size(), 0);
    std::vector<uint8_t> out_neon(input.size(), 0);

    AEAWBConfig config = make_identity_config();
    config.width = w;
    config.height = h;
    config.gain_b = 1.3f;
    config.gain_g = 0.8f;
    config.gain_r = 1.0f;
    for (int i = 0; i < 256; i++) {
        config.lut[i] = static_cast<uint8_t>(std::min(255 - i / 2, 255));
    }

    apply_ae_awb_scalar(input.data(), out_scalar.data(), config);
    apply_ae_awb_neon(input.data(), out_neon.data(), config);

    for (size_t i = 0; i < input.size(); i++) {
        EXPECT_NEAR(out_neon[i], out_scalar[i], 1) << "byte " << i;
    }
}

TEST_F(AEAWBNeonTest, MatchesScalarNonAligned) {
    const int w = 5, h = 3;
    const int num_pixels = w * h;  // 15 pixels, not multiple of 16
    std::vector<uint8_t> input(static_cast<size_t>(num_pixels) * 3);
    for (size_t i = 0; i < input.size(); i++) {
        input[i] = static_cast<uint8_t>((i * 11 + 3) % 256);
    }
    std::vector<uint8_t> out_scalar(input.size(), 0);
    std::vector<uint8_t> out_neon(input.size(), 0);

    AEAWBConfig config = make_identity_config();
    config.width = w;
    config.height = h;
    config.gain_b = 1.4f;
    config.gain_g = 1.1f;
    config.gain_r = 0.7f;

    apply_ae_awb_scalar(input.data(), out_scalar.data(), config);
    apply_ae_awb_neon(input.data(), out_neon.data(), config);

    for (size_t i = 0; i < input.size(); i++) {
        EXPECT_NEAR(out_neon[i], out_scalar[i], 1) << "byte " << i;
    }
}

// ---------------------------------------------------------------------------
// AFBlendScalarTest
// ---------------------------------------------------------------------------

class AFBlendScalarTest : public ::testing::Test {};

TEST_F(AFBlendScalarTest, ZeroAlphaIdentity) {
    const int num_pixels = 8;
    std::vector<uint8_t> frame(static_cast<size_t>(num_pixels) * 3);
    std::vector<uint8_t> blurred(static_cast<size_t>(num_pixels) * 3);
    for (size_t i = 0; i < frame.size(); i++) {
        frame[i] = static_cast<uint8_t>((i * 13 + 5) % 256);
        blurred[i] = static_cast<uint8_t>((i * 7 + 20) % 256);
    }
    std::vector<float> alpha(num_pixels, 0.0f);
    std::vector<uint8_t> output(frame.size(), 0);

    AFBlendConfig config{num_pixels};
    apply_af_blend_scalar(frame.data(), alpha.data(), blurred.data(), output.data(), config);

    for (size_t i = 0; i < frame.size(); i++) {
        EXPECT_EQ(output[i], frame[i]) << "byte " << i;
    }
}

TEST_F(AFBlendScalarTest, PositiveAlphaSharpens) {
    const int num_pixels = 1;
    std::vector<uint8_t> frame = {150, 150, 150};
    std::vector<uint8_t> blurred = {100, 100, 100};
    std::vector<float> alpha = {1.0f};
    std::vector<uint8_t> output(3, 0);

    AFBlendConfig config{num_pixels};
    apply_af_blend_scalar(frame.data(), alpha.data(), blurred.data(), output.data(), config);

    // result = 150 + 1.0 * (150 - 100) = 200
    for (int c = 0; c < 3; c++) {
        EXPECT_EQ(output[c], 200) << "channel " << c;
    }
}

TEST_F(AFBlendScalarTest, ClampNoOverflow) {
    const int num_pixels = 1;
    std::vector<uint8_t> frame = {250, 250, 250};
    std::vector<uint8_t> blurred = {200, 200, 200};
    std::vector<float> alpha = {1.5f};
    std::vector<uint8_t> output(3, 0);

    AFBlendConfig config{num_pixels};
    apply_af_blend_scalar(frame.data(), alpha.data(), blurred.data(), output.data(), config);

    // result = 250 + 1.5 * (250 - 200) = 250 + 75 = 325 -> clamped to 255
    for (int c = 0; c < 3; c++) {
        EXPECT_EQ(output[c], 255) << "channel " << c;
    }
}

TEST_F(AFBlendScalarTest, ClampNoUnderflow) {
    const int num_pixels = 1;
    std::vector<uint8_t> frame = {50, 50, 50};
    std::vector<uint8_t> blurred = {200, 200, 200};
    std::vector<float> alpha = {1.5f};
    std::vector<uint8_t> output(3, 0);

    AFBlendConfig config{num_pixels};
    apply_af_blend_scalar(frame.data(), alpha.data(), blurred.data(), output.data(), config);

    // result = 50 + 1.5 * (50 - 200) = 50 + (-225) = -175 -> clamped to 0
    for (int c = 0; c < 3; c++) {
        EXPECT_EQ(output[c], 0) << "channel " << c;
    }
}

TEST_F(AFBlendScalarTest, VaryingAlphaPerPixel) {
    const int num_pixels = 2;
    std::vector<uint8_t> frame = {150, 150, 150, 150, 150, 150};
    std::vector<uint8_t> blurred = {100, 100, 100, 100, 100, 100};
    std::vector<float> alpha = {0.0f, 1.0f};
    std::vector<uint8_t> output(6, 0);

    AFBlendConfig config{num_pixels};
    apply_af_blend_scalar(frame.data(), alpha.data(), blurred.data(), output.data(), config);

    // Pixel 0: alpha=0, identity
    for (int c = 0; c < 3; c++) {
        EXPECT_EQ(output[c], 150) << "pixel 0, channel " << c;
    }
    // Pixel 1: alpha=1.0, sharpened
    for (int c = 0; c < 3; c++) {
        EXPECT_EQ(output[3 + c], 200) << "pixel 1, channel " << c;
    }
}

// ---------------------------------------------------------------------------
// AFBlendNeonTest
// ---------------------------------------------------------------------------

class AFBlendNeonTest : public ::testing::Test {};

TEST_F(AFBlendNeonTest, MatchesScalarSmall) {
    const int w = 8, h = 4;
    const int num_pixels = w * h;
    std::vector<uint8_t> frame(static_cast<size_t>(num_pixels) * 3);
    std::vector<uint8_t> blurred(static_cast<size_t>(num_pixels) * 3);
    for (size_t i = 0; i < frame.size(); i++) {
        frame[i] = static_cast<uint8_t>((i * 11 + 3) % 256);
        blurred[i] = static_cast<uint8_t>((i * 7 + 50) % 256);
    }
    std::vector<float> alpha(num_pixels);
    for (int i = 0; i < num_pixels; i++) {
        alpha[i] = static_cast<float>(i % 5) * 0.3f;
    }
    std::vector<uint8_t> out_scalar(frame.size(), 0);
    std::vector<uint8_t> out_neon(frame.size(), 0);

    AFBlendConfig config{num_pixels};
    apply_af_blend_scalar(frame.data(), alpha.data(), blurred.data(), out_scalar.data(), config);
    apply_af_blend_neon(frame.data(), alpha.data(), blurred.data(), out_neon.data(), config);

    for (size_t i = 0; i < frame.size(); i++) {
        EXPECT_NEAR(out_neon[i], out_scalar[i], 1) << "byte " << i;
    }
}

TEST_F(AFBlendNeonTest, MatchesScalarLarge) {
    const int w = 512, h = 288;
    const int num_pixels = w * h;
    std::vector<uint8_t> frame(static_cast<size_t>(num_pixels) * 3);
    std::vector<uint8_t> blurred(static_cast<size_t>(num_pixels) * 3);
    for (size_t i = 0; i < frame.size(); i++) {
        frame[i] = static_cast<uint8_t>((i * 13 + 7) % 256);
        blurred[i] = static_cast<uint8_t>((i * 3 + 80) % 256);
    }
    std::vector<float> alpha(num_pixels);
    for (int i = 0; i < num_pixels; i++) {
        alpha[i] = static_cast<float>(i % 10) * 0.15f;
    }
    std::vector<uint8_t> out_scalar(frame.size(), 0);
    std::vector<uint8_t> out_neon(frame.size(), 0);

    AFBlendConfig config{num_pixels};
    apply_af_blend_scalar(frame.data(), alpha.data(), blurred.data(), out_scalar.data(), config);
    apply_af_blend_neon(frame.data(), alpha.data(), blurred.data(), out_neon.data(), config);

    for (size_t i = 0; i < frame.size(); i++) {
        EXPECT_NEAR(out_neon[i], out_scalar[i], 1) << "byte " << i;
    }
}
