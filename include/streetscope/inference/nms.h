#pragma once
#include <streetscope/inference/detection.h>
#include <vector>

namespace streetscope {

/// Greedy IoU-based Non-Maximum Suppression.
/// Sort candidates by confidence descending. For each, keep if IoU < threshold
/// with all already-kept detections.
/// Input candidates are consumed (may be reordered).
std::vector<Detection> nms(std::vector<Detection>& candidates, float iou_threshold);

}  // namespace streetscope
