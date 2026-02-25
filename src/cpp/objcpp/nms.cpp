#include <streetscope/inference/nms.h>
#include <algorithm>

namespace streetscope {

static float compute_iou(const Detection& a, const Detection& b) {
    int ix1 = std::max(a.x1, b.x1);
    int iy1 = std::max(a.y1, b.y1);
    int ix2 = std::min(a.x2, b.x2);
    int iy2 = std::min(a.y2, b.y2);
    int inter_w = std::max(0, ix2 - ix1);
    int inter_h = std::max(0, iy2 - iy1);
    int inter_area = inter_w * inter_h;
    int area_a = a.area();
    int area_b = b.area();
    int union_area = area_a + area_b - inter_area;
    if (union_area <= 0) return 0.0f;
    return static_cast<float>(inter_area) / static_cast<float>(union_area);
}

std::vector<Detection> nms(std::vector<Detection>& candidates, float iou_threshold) {
    // Sort by confidence descending
    std::sort(candidates.begin(), candidates.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<Detection> kept;
    for (auto& det : candidates) {
        bool suppress = false;
        for (const auto& k : kept) {
            if (compute_iou(det, k) > iou_threshold) {
                suppress = true;
                break;
            }
        }
        if (!suppress) {
            kept.push_back(det);
        }
    }
    return kept;
}

}  // namespace streetscope
