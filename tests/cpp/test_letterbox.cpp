#include <gtest/gtest.h>
#include <streetscope/inference/letterbox.h>
#include <vector>
#include <cmath>

using namespace streetscope;

class ComputeLetterboxTest : public ::testing::Test {
protected:
    static constexpr float kTol = 1e-4f;
};

TEST_F(ComputeLetterboxTest, SquareInput) {
    auto info = compute_letterbox(640, 640, 640);
    EXPECT_NEAR(info.ratio, 1.0f, kTol);
    EXPECT_NEAR(info.pad_w, 0.0f, kTol);
    EXPECT_NEAR(info.pad_h, 0.0f, kTol);
    EXPECT_EQ(info.new_w, 640);
    EXPECT_EQ(info.new_h, 640);
}

TEST_F(ComputeLetterboxTest, WideInput) {
    // 512x288 -> ratio = 640/512 = 1.25
    auto info = compute_letterbox(512, 288, 640);
    EXPECT_NEAR(info.ratio, 1.25f, kTol);
    EXPECT_EQ(info.new_w, 640);
    EXPECT_EQ(info.new_h, 360);  // round(288 * 1.25) = 360
    EXPECT_NEAR(info.pad_w, 0.0f, kTol);
    EXPECT_NEAR(info.pad_h, 140.0f, kTol);
}

TEST_F(ComputeLetterboxTest, TallInput) {
    // 288x512 -> ratio = 640/512 = 1.25
    auto info = compute_letterbox(288, 512, 640);
    EXPECT_NEAR(info.ratio, 1.25f, kTol);
    EXPECT_EQ(info.new_w, 360);
    EXPECT_EQ(info.new_h, 640);
    EXPECT_NEAR(info.pad_w, 140.0f, kTol);
    EXPECT_NEAR(info.pad_h, 0.0f, kTol);
}

TEST_F(ComputeLetterboxTest, LargeInput) {
    // 1920x1080 -> ratio = 640/1920 = 0.33333...
    auto info = compute_letterbox(1920, 1080, 640);
    float expected_ratio = 640.0f / 1920.0f;
    EXPECT_NEAR(info.ratio, expected_ratio, kTol);

    // new_w = round(1920 * 0.33333) = round(640.0) = 640
    // new_h = round(1080 * 0.33333) = round(360.0) = 360
    EXPECT_EQ(info.new_w, 640);
    EXPECT_EQ(info.new_h, 360);

    // pad_w = (640 - 640) / 2.0 = 0
    // pad_h = (640 - 360) / 2.0 = 140
    EXPECT_NEAR(info.pad_w, 0.0f, kTol);
    EXPECT_NEAR(info.pad_h, 140.0f, kTol);
}

class ApplyLetterboxTest : public ::testing::Test {
protected:
    static constexpr int kInputSize = 640;
    static constexpr int kDstBytes = kInputSize * kInputSize * 4;
};

TEST_F(ApplyLetterboxTest, PaddingPixels) {
    // Create a small 4x4 black BGR image (all zeros).
    const int src_w = 4, src_h = 4;
    std::vector<uint8_t> src(static_cast<size_t>(src_w) * src_h * 3, 0);

    std::vector<uint8_t> dst(kDstBytes, 0);
    auto info = compute_letterbox(src_w, src_h, kInputSize);
    apply_letterbox(src.data(), src_w, src_h, dst.data(), kInputSize, info);

    // The source is square 4x4, ratio = 640/4 = 160, new_w = new_h = 640.
    // So entire image is content (no padding expected for square input).
    // Instead, check a known padding case: use a non-square input.

    // Use a wide image: 8x4
    const int wide_w = 8, wide_h = 4;
    std::vector<uint8_t> wide_src(static_cast<size_t>(wide_w) * wide_h * 3, 0);
    std::vector<uint8_t> wide_dst(kDstBytes, 0);
    auto wide_info = compute_letterbox(wide_w, wide_h, kInputSize);
    apply_letterbox(wide_src.data(), wide_w, wide_h, wide_dst.data(), kInputSize, wide_info);

    // For 8x4: ratio = 640/8 = 80, new_w = 640, new_h = 320
    // pad_h = (640 - 320) / 2 = 160
    // top = round(160 - 0.1) = round(159.9) = 160
    // Bottom padding starts at row 160 + 320 = 480.
    // Check top-left corner is padding.
    int top = static_cast<int>(std::round(wide_info.pad_h - 0.1f));
    ASSERT_GT(top, 0);

    // Verify a pixel in the top padding region.
    int pad_idx = (0 * kInputSize + 0) * 4;  // row 0, col 0
    EXPECT_EQ(wide_dst[pad_idx + 0], 114);  // B
    EXPECT_EQ(wide_dst[pad_idx + 1], 114);  // G
    EXPECT_EQ(wide_dst[pad_idx + 2], 114);  // R
    EXPECT_EQ(wide_dst[pad_idx + 3], 255);  // A

    // Verify a pixel in the bottom padding region.
    int bottom_start = top + wide_info.new_h;
    if (bottom_start < kInputSize) {
        int bot_idx = (bottom_start * kInputSize + 0) * 4;
        EXPECT_EQ(wide_dst[bot_idx + 0], 114);
        EXPECT_EQ(wide_dst[bot_idx + 1], 114);
        EXPECT_EQ(wide_dst[bot_idx + 2], 114);
        EXPECT_EQ(wide_dst[bot_idx + 3], 255);
    }
}

TEST_F(ApplyLetterboxTest, ContentPlacement) {
    // Create a 2x2 white BGR image.
    const int src_w = 2, src_h = 2;
    std::vector<uint8_t> src(static_cast<size_t>(src_w) * src_h * 3, 255);

    std::vector<uint8_t> dst(kDstBytes, 0);
    auto info = compute_letterbox(src_w, src_h, kInputSize);
    apply_letterbox(src.data(), src_w, src_h, dst.data(), kInputSize, info);

    // 2x2 square input -> ratio = 640/2 = 320, new_w = new_h = 640
    // pad_w = pad_h = 0, so content fills entire image.
    // All pixels should be white (255,255,255,255).
    EXPECT_EQ(info.new_w, 640);
    EXPECT_EQ(info.new_h, 640);

    // Check center pixel is white.
    int cx = kInputSize / 2;
    int cy = kInputSize / 2;
    int idx = (cy * kInputSize + cx) * 4;
    EXPECT_EQ(dst[idx + 0], 255);  // B
    EXPECT_EQ(dst[idx + 1], 255);  // G
    EXPECT_EQ(dst[idx + 2], 255);  // R
    EXPECT_EQ(dst[idx + 3], 255);  // A

    // Now test with a non-square image to verify offset placement.
    // Use 4x2 wide image, all white.
    const int wide_w = 4, wide_h = 2;
    std::vector<uint8_t> wide_src(static_cast<size_t>(wide_w) * wide_h * 3, 255);
    std::vector<uint8_t> wide_dst(kDstBytes, 0);
    auto wide_info = compute_letterbox(wide_w, wide_h, kInputSize);
    apply_letterbox(wide_src.data(), wide_w, wide_h, wide_dst.data(), kInputSize, wide_info);

    // 4x2: ratio = 640/4 = 160, new_w = 640, new_h = 320
    // pad_h = 160, top = round(160 - 0.1) = round(159.9) = 160
    int top = static_cast<int>(std::round(wide_info.pad_h - 0.1f));
    int left = static_cast<int>(std::round(wide_info.pad_w - 0.1f));

    // Check that a pixel within the content region is white.
    int content_y = top + wide_info.new_h / 2;
    int content_x = left + wide_info.new_w / 2;
    int content_idx = (content_y * kInputSize + content_x) * 4;
    EXPECT_EQ(wide_dst[content_idx + 0], 255);  // B
    EXPECT_EQ(wide_dst[content_idx + 1], 255);  // G
    EXPECT_EQ(wide_dst[content_idx + 2], 255);  // R
    EXPECT_EQ(wide_dst[content_idx + 3], 255);  // A

    // Check that a pixel in the top padding region is gray.
    if (top > 0) {
        int pad_y = top / 2;
        int pad_idx = (pad_y * kInputSize + content_x) * 4;
        EXPECT_EQ(wide_dst[pad_idx + 0], 114);  // B
        EXPECT_EQ(wide_dst[pad_idx + 1], 114);  // G
        EXPECT_EQ(wide_dst[pad_idx + 2], 114);  // R
        EXPECT_EQ(wide_dst[pad_idx + 3], 255);  // A
    }
}

TEST_F(ApplyLetterboxTest, BGRtoBGRA) {
    // Create a 3x3 BGR image with known values.
    const int src_w = 3, src_h = 3;
    std::vector<uint8_t> src(static_cast<size_t>(src_w) * src_h * 3);
    for (int i = 0; i < src_w * src_h; i++) {
        src[i * 3 + 0] = 10;   // B
        src[i * 3 + 1] = 20;   // G
        src[i * 3 + 2] = 30;   // R
    }

    std::vector<uint8_t> dst(kDstBytes, 0);
    auto info = compute_letterbox(src_w, src_h, kInputSize);
    apply_letterbox(src.data(), src_w, src_h, dst.data(), kInputSize, info);

    // Every pixel in the output should have alpha = 255 (both content and padding).
    for (int i = 0; i < kInputSize * kInputSize; i++) {
        EXPECT_EQ(dst[i * 4 + 3], 255) << "pixel " << i << " alpha != 255";
    }
}
