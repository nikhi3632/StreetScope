#include "streetscope/inference/detection.h"

namespace streetscope {

static const char* kCocoNames[80] = {
    "person",        "bicycle",       "car",           "motorcycle",
    "airplane",      "bus",           "train",         "truck",
    "boat",          "traffic light", "fire hydrant",  "stop sign",
    "parking meter", "bench",         "bird",          "cat",
    "dog",           "horse",         "sheep",         "cow",
    "elephant",      "bear",          "zebra",         "giraffe",
    "backpack",      "umbrella",      "handbag",       "tie",
    "suitcase",      "frisbee",       "skis",          "snowboard",
    "sports ball",   "kite",          "baseball bat",  "baseball glove",
    "skateboard",    "surfboard",     "tennis racket", "bottle",
    "wine glass",    "cup",           "fork",          "knife",
    "spoon",         "bowl",          "banana",        "apple",
    "sandwich",      "orange",        "broccoli",      "carrot",
    "hot dog",       "pizza",         "donut",         "cake",
    "chair",         "couch",         "potted plant",  "bed",
    "dining table",  "toilet",        "tv",            "laptop",
    "mouse",         "remote",        "keyboard",      "cell phone",
    "microwave",     "oven",          "toaster",       "sink",
    "refrigerator",  "book",          "clock",         "vase",
    "scissors",      "teddy bear",    "hair drier",    "toothbrush"
};

const char* coco_class_name(int class_id) {
    if (class_id < 0 || class_id >= 80) {
        return "unknown";
    }
    return kCocoNames[class_id];
}

bool is_vehicle_class(int class_id) {
    return class_id == 1 ||  // bicycle
           class_id == 2 ||  // car
           class_id == 3 ||  // motorcycle
           class_id == 5 ||  // bus
           class_id == 7;    // truck
}

}  // namespace streetscope
