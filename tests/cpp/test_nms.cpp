#include <gtest/gtest.h>
#include "streetscope/inference/nms.h"

using namespace streetscope;

TEST(NmsTest, EmptyInput) {
    std::vector<Detection> candidates;
    auto result = nms(candidates, 0.5f);
    EXPECT_TRUE(result.empty());
}

TEST(NmsTest, SingleDetection) {
    std::vector<Detection> candidates = {
        {10, 20, 110, 120, 0.9f, 0}
    };
    auto result = nms(candidates, 0.5f);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].x1, 10);
    EXPECT_EQ(result[0].y1, 20);
    EXPECT_EQ(result[0].x2, 110);
    EXPECT_EQ(result[0].y2, 120);
    EXPECT_FLOAT_EQ(result[0].confidence, 0.9f);
    EXPECT_EQ(result[0].class_id, 0);
}

TEST(NmsTest, NonOverlapping) {
    std::vector<Detection> candidates = {
        {0, 0, 10, 10, 0.9f, 0},
        {20, 20, 30, 30, 0.8f, 0}
    };
    auto result = nms(candidates, 0.5f);
    ASSERT_EQ(result.size(), 2u);
    // Higher confidence first after sort
    EXPECT_FLOAT_EQ(result[0].confidence, 0.9f);
    EXPECT_FLOAT_EQ(result[1].confidence, 0.8f);
}

TEST(NmsTest, IdenticalBoxes) {
    std::vector<Detection> candidates = {
        {0, 0, 100, 100, 0.9f, 0},
        {0, 0, 100, 100, 0.7f, 0}
    };
    auto result = nms(candidates, 0.5f);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_FLOAT_EQ(result[0].confidence, 0.9f);
}

TEST(NmsTest, PartialOverlapBothSurvive) {
    // Box A: (0,0,100,100) area=10000
    // Box B: (50,50,150,150) area=10000
    // Intersection: (50,50,100,100) = 50*50 = 2500
    // Union: 10000 + 10000 - 2500 = 17500
    // IoU = 2500/17500 ~= 0.1429
    std::vector<Detection> candidates = {
        {0, 0, 100, 100, 0.9f, 0},
        {50, 50, 150, 150, 0.8f, 0}
    };
    auto result = nms(candidates, 0.5f);
    ASSERT_EQ(result.size(), 2u);
    EXPECT_FLOAT_EQ(result[0].confidence, 0.9f);
    EXPECT_FLOAT_EQ(result[1].confidence, 0.8f);
}

TEST(NmsTest, PartialOverlapOneSuppressed) {
    // Same boxes as above, IoU ~= 0.1429, threshold 0.1 means B is suppressed
    std::vector<Detection> candidates = {
        {0, 0, 100, 100, 0.9f, 0},
        {50, 50, 150, 150, 0.8f, 0}
    };
    auto result = nms(candidates, 0.1f);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_FLOAT_EQ(result[0].confidence, 0.9f);
}

TEST(NmsTest, ThreeBoxSuppression) {
    // A(0,0,100,100) conf=0.9
    // B(10,10,110,110) conf=0.8  -- heavily overlaps with A
    // C(200,200,300,300) conf=0.7 -- no overlap
    // A and B: intersection (10,10,100,100) = 90*90 = 8100
    //   union = 10000 + 10000 - 8100 = 11900
    //   IoU = 8100/11900 ~= 0.6807
    // At threshold 0.3, A suppresses B. C survives.
    std::vector<Detection> candidates = {
        {0, 0, 100, 100, 0.9f, 0},
        {10, 10, 110, 110, 0.8f, 0},
        {200, 200, 300, 300, 0.7f, 0}
    };
    auto result = nms(candidates, 0.3f);
    ASSERT_EQ(result.size(), 2u);
    EXPECT_FLOAT_EQ(result[0].confidence, 0.9f);
    EXPECT_EQ(result[0].x1, 0);
    EXPECT_FLOAT_EQ(result[1].confidence, 0.7f);
    EXPECT_EQ(result[1].x1, 200);
}

TEST(NmsTest, ConfidenceOrdering) {
    // When two identical boxes overlap, the lower-confidence one is suppressed
    std::vector<Detection> candidates = {
        {0, 0, 100, 100, 0.6f, 0},  // lower confidence added first
        {0, 0, 100, 100, 0.95f, 0}  // higher confidence added second
    };
    auto result = nms(candidates, 0.5f);
    ASSERT_EQ(result.size(), 1u);
    // The higher confidence detection must survive
    EXPECT_FLOAT_EQ(result[0].confidence, 0.95f);
}

TEST(NmsTest, ZeroAreaBox) {
    // Zero-width box: x1 == x2
    std::vector<Detection> candidates = {
        {50, 50, 50, 100, 0.9f, 0},  // width=0, area=0
        {0, 0, 100, 100, 0.8f, 0}
    };
    auto result = nms(candidates, 0.5f);
    // IoU should be 0 (zero-area box), so both survive
    ASSERT_EQ(result.size(), 2u);

    // Zero-height box: y1 == y2
    std::vector<Detection> candidates2 = {
        {0, 0, 100, 100, 0.85f, 0},
        {10, 50, 90, 50, 0.75f, 0}  // height=0, area=0
    };
    auto result2 = nms(candidates2, 0.5f);
    ASSERT_EQ(result2.size(), 2u);
}
