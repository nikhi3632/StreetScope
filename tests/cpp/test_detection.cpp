#include <gtest/gtest.h>
#include "streetscope/inference/detection.h"

using namespace streetscope;

TEST(DetectionTest, DetectionFields) {
    Detection d{10, 20, 110, 220, 0.95f, 2};

    EXPECT_EQ(d.x1, 10);
    EXPECT_EQ(d.y1, 20);
    EXPECT_EQ(d.x2, 110);
    EXPECT_EQ(d.y2, 220);
    EXPECT_FLOAT_EQ(d.confidence, 0.95f);
    EXPECT_EQ(d.class_id, 2);
    EXPECT_EQ(d.width(), 100);
    EXPECT_EQ(d.height(), 200);
    EXPECT_EQ(d.area(), 20000);
}

TEST(DetectionTest, CocoClassNameValid) {
    EXPECT_STREQ(coco_class_name(0), "person");
    EXPECT_STREQ(coco_class_name(2), "car");
    EXPECT_STREQ(coco_class_name(7), "truck");
    EXPECT_STREQ(coco_class_name(79), "toothbrush");
}

TEST(DetectionTest, CocoClassNameInvalid) {
    EXPECT_STREQ(coco_class_name(80), "unknown");
    EXPECT_STREQ(coco_class_name(-1), "unknown");
    EXPECT_STREQ(coco_class_name(999), "unknown");
}

TEST(DetectionTest, IsVehicleClass) {
    EXPECT_TRUE(is_vehicle_class(1));   // bicycle
    EXPECT_TRUE(is_vehicle_class(2));   // car
    EXPECT_TRUE(is_vehicle_class(3));   // motorcycle
    EXPECT_TRUE(is_vehicle_class(5));   // bus
    EXPECT_TRUE(is_vehicle_class(7));   // truck
}

TEST(DetectionTest, IsNotVehicleClass) {
    EXPECT_FALSE(is_vehicle_class(0));  // person
    EXPECT_FALSE(is_vehicle_class(4));  // airplane
    EXPECT_FALSE(is_vehicle_class(6));  // train
    EXPECT_FALSE(is_vehicle_class(8));  // boat
}

TEST(DetectionTest, DetectionResultMoveSemantics) {
    DetectionResult source;
    source.detections.push_back({0, 0, 10, 10, 0.9f, 0});
    source.detections.push_back({20, 20, 30, 30, 0.8f, 1});
    source.detections.push_back({40, 40, 50, 50, 0.7f, 2});
    source.inference_ms = 5.0;
    source.frame_number = 42;

    DetectionResult dest = std::move(source);

    EXPECT_EQ(dest.detections.size(), 3u);
    EXPECT_DOUBLE_EQ(dest.inference_ms, 5.0);
    EXPECT_EQ(dest.frame_number, 42);
}
