#include <streetscope/inference/coreml_detector.h>
#include <streetscope/inference/letterbox.h>
#include <streetscope/inference/nms.h>
#include <streetscope/frame_pipeline/ring_buffer.h>

#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>

#include <chrono>
#include <dispatch/dispatch.h>
#include <stdexcept>

namespace streetscope {

// ── CVPixelBuffer helpers ────────────────────────────────────

/// Create an IOSurface-backed 640x640 BGRA CVPixelBuffer (Neural Engine accessible).
static CVPixelBufferRef create_pixel_buffer(int size) {
    NSDictionary* attrs = @{
        (NSString*)kCVPixelBufferIOSurfacePropertiesKey: @{},
        (NSString*)kCVPixelBufferWidthKey: @(size),
        (NSString*)kCVPixelBufferHeightKey: @(size),
        (NSString*)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
    };
    CVPixelBufferRef pb = nullptr;
    CVReturn status = CVPixelBufferCreate(
        kCFAllocatorDefault,
        static_cast<size_t>(size), static_cast<size_t>(size),
        kCVPixelFormatType_32BGRA,
        (__bridge CFDictionaryRef)attrs,
        &pb
    );
    if (status != kCVReturnSuccess || pb == nullptr) {
        throw std::runtime_error("Failed to create CVPixelBuffer");
    }
    return pb;
}

// ── Impl ─────────────────────────────────────────────────────

struct CoreMLDetector::Impl {
    MLModel* model = nil;
    NSString* input_name = nil;
    NSString* output_name = nil;
    float conf_thresh;
    float iou_thresh;
    static constexpr int kInputSize = 640;

    // Double-buffered CVPixelBuffers for async (ping-pong)
    CVPixelBufferRef pixel_buffers[2] = {nullptr, nullptr};
    int current_buffer = 0;

    // Async inference
    dispatch_queue_t inference_queue;
    RingBuffer<DetectionResult, 2> result_ring;

    explicit Impl(const std::string& model_path, float conf, float iou)
        : conf_thresh(conf), iou_thresh(iou) {
        @autoreleasepool {
            // Load model
            NSString* path = [NSString stringWithUTF8String:model_path.c_str()];
            NSURL* url = [NSURL fileURLWithPath:path];

            // Compile if .mlpackage, load compiled .mlmodelc
            NSError* error = nil;
            NSURL* compiled_url = [MLModel compileModelAtURL:url error:&error];
            if (error != nil) {
                throw std::runtime_error(
                    std::string("Failed to compile Core ML model: ") +
                    [[error localizedDescription] UTF8String]);
            }

            MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
            config.computeUnits = MLComputeUnitsAll;

            model = [MLModel modelWithContentsOfURL:compiled_url
                                      configuration:config
                                              error:&error];
            if (error != nil) {
                throw std::runtime_error(
                    std::string("Failed to load Core ML model: ") +
                    [[error localizedDescription] UTF8String]);
            }

            // Extract input/output names from model description
            MLModelDescription* desc = model.modelDescription;
            input_name = desc.inputDescriptionsByName.allKeys.firstObject;
            output_name = desc.outputDescriptionsByName.allKeys.firstObject;

            if (input_name == nil || output_name == nil) {
                throw std::runtime_error("Model has no input or output");
            }

            // Pre-allocate two IOSurface-backed pixel buffers
            pixel_buffers[0] = create_pixel_buffer(kInputSize);
            pixel_buffers[1] = create_pixel_buffer(kInputSize);

            // Serial dispatch queue for async inference
            inference_queue = dispatch_queue_create(
                "com.streetscope.inference", DISPATCH_QUEUE_SERIAL);
        }
    }

    ~Impl() {
        if (pixel_buffers[0]) CVPixelBufferRelease(pixel_buffers[0]);
        if (pixel_buffers[1]) CVPixelBufferRelease(pixel_buffers[1]);
    }

    /// Fill a CVPixelBuffer with letterboxed BGRA data from BGR input.
    void fill_pixel_buffer(CVPixelBufferRef pb,
                           const uint8_t* bgr, int w, int h,
                           LetterboxInfo& info_out) {
        info_out = compute_letterbox(w, h, kInputSize);

        CVPixelBufferLockBaseAddress(pb, 0);
        auto* dst = static_cast<uint8_t*>(CVPixelBufferGetBaseAddress(pb));
        apply_letterbox(bgr, w, h, dst, kInputSize, info_out);
        CVPixelBufferUnlockBaseAddress(pb, 0);
    }

    /// Run inference on a CVPixelBuffer, return raw MLMultiArray.
    MLMultiArray* infer(CVPixelBufferRef pb) {
        @autoreleasepool {
            NSError* error = nil;
            MLFeatureValue* input_fv =
                [MLFeatureValue featureValueWithPixelBuffer:pb];

            NSDictionary* input_dict = @{input_name: input_fv};
            MLDictionaryFeatureProvider* provider =
                [[MLDictionaryFeatureProvider alloc] initWithDictionary:input_dict
                                                                 error:&error];
            if (error != nil) {
                throw std::runtime_error(
                    std::string("Failed to create feature provider: ") +
                    [[error localizedDescription] UTF8String]);
            }

            id<MLFeatureProvider> result =
                [model predictionFromFeatures:provider error:&error];
            if (error != nil) {
                throw std::runtime_error(
                    std::string("Inference failed: ") +
                    [[error localizedDescription] UTF8String]);
            }

            MLFeatureValue* output_fv =
                [result featureValueForName:output_name];
            return output_fv.multiArrayValue;
        }
    }

    /// Postprocess raw model output (1, 84, 8400) -> vector<Detection>.
    std::vector<Detection> postprocess(
        MLMultiArray* raw_output,
        const LetterboxInfo& info,
        int orig_w, int orig_h,
        bool vehicles_only
    ) {
        // Shape: (1, 84, 8400) — 84 = 4 bbox + 80 class scores
        // Datatype: Float32
        const float* data = static_cast<const float*>(raw_output.dataPointer);

        // Strides from MLMultiArray (in elements, not bytes)
        NSArray<NSNumber*>* strides = raw_output.strides;
        auto stride_dim1 = [strides[1] integerValue];  // stride for dim 1 (84)
        auto stride_dim2 = [strides[2] integerValue];  // stride for dim 2 (8400)
        auto n_candidates = [raw_output.shape[2] integerValue];

        std::vector<Detection> candidates;
        candidates.reserve(128);

        for (NSInteger i = 0; i < n_candidates; i++) {
            // Find best class score for this candidate
            float best_score = 0.0f;
            int best_class = 0;
            for (int c = 0; c < 80; c++) {
                float score = data[(4 + c) * stride_dim1 + i * stride_dim2];
                if (score > best_score) {
                    best_score = score;
                    best_class = c;
                }
            }

            if (best_score < conf_thresh) continue;
            if (vehicles_only && !is_vehicle_class(best_class)) continue;

            // Extract cx, cy, w, h in model space
            float cx = data[0 * stride_dim1 + i * stride_dim2];
            float cy = data[1 * stride_dim1 + i * stride_dim2];
            float bw = data[2 * stride_dim1 + i * stride_dim2];
            float bh = data[3 * stride_dim1 + i * stride_dim2];

            // cxcywh -> xyxy in model space
            float x1 = cx - bw / 2.0f;
            float y1 = cy - bh / 2.0f;
            float x2 = cx + bw / 2.0f;
            float y2 = cy + bh / 2.0f;

            // Inverse letterbox: remove padding and rescale to original coords
            x1 = (x1 - info.pad_w) / info.ratio;
            y1 = (y1 - info.pad_h) / info.ratio;
            x2 = (x2 - info.pad_w) / info.ratio;
            y2 = (y2 - info.pad_h) / info.ratio;

            // Clip to image bounds
            Detection det;
            det.x1 = std::max(0, static_cast<int>(std::round(x1)));
            det.y1 = std::max(0, static_cast<int>(std::round(y1)));
            det.x2 = std::min(orig_w, static_cast<int>(std::round(x2)));
            det.y2 = std::min(orig_h, static_cast<int>(std::round(y2)));
            det.confidence = best_score;
            det.class_id = best_class;

            if (det.x2 <= det.x1 || det.y2 <= det.y1) continue;

            candidates.push_back(det);
        }

        return nms(candidates, iou_thresh);
    }

    /// Full sync pipeline: letterbox -> infer -> postprocess.
    std::vector<Detection> detect(
        const uint8_t* bgr, int w, int h, bool vehicles_only
    ) {
        @autoreleasepool {
            CVPixelBufferRef pb = pixel_buffers[0];
            LetterboxInfo info{};
            fill_pixel_buffer(pb, bgr, w, h, info);
            MLMultiArray* output = infer(pb);
            return postprocess(output, info, w, h, vehicles_only);
        }
    }

    /// Async: submit frame for background inference.
    void submit(const uint8_t* bgr, int w, int h, bool vehicles_only) {
        // Copy input data (caller's buffer may be transient)
        auto input = std::make_shared<std::vector<uint8_t>>(
            bgr, bgr + static_cast<size_t>(w) * h * 3);
        int cap_w = w, cap_h = h;

        // Swap to the other pixel buffer (ping-pong)
        int buf_idx = current_buffer;
        current_buffer = 1 - current_buffer;

        dispatch_async(inference_queue, ^{
            @autoreleasepool {
                CVPixelBufferRef pb = pixel_buffers[buf_idx];
                LetterboxInfo info{};
                fill_pixel_buffer(pb, input->data(), cap_w, cap_h, info);

                auto t0 = std::chrono::steady_clock::now();
                MLMultiArray* output = infer(pb);
                auto t1 = std::chrono::steady_clock::now();

                auto dets = postprocess(output, info, cap_w, cap_h, vehicles_only);

                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                DetectionResult result;
                result.detections = std::move(dets);
                result.inference_ms = ms;
                result_ring.try_push(std::move(result));
            }
        });
    }
};

// ── Public API (forwarding to Impl) ──────────────────────────

CoreMLDetector::CoreMLDetector(
    const std::string& model_path, float conf_threshold, float iou_threshold
) : impl_(std::make_unique<Impl>(model_path, conf_threshold, iou_threshold)) {}

CoreMLDetector::~CoreMLDetector() = default;

std::vector<Detection> CoreMLDetector::detect(
    const uint8_t* bgr, int w, int h, bool vehicles_only
) {
    return impl_->detect(bgr, w, h, vehicles_only);
}

void CoreMLDetector::submit(
    const uint8_t* bgr, int w, int h, bool vehicles_only
) {
    impl_->submit(bgr, w, h, vehicles_only);
}

std::optional<DetectionResult> CoreMLDetector::try_get_result() {
    return impl_->result_ring.try_pop();
}

float CoreMLDetector::conf_threshold() const { return impl_->conf_thresh; }
float CoreMLDetector::iou_threshold() const { return impl_->iou_thresh; }
int CoreMLDetector::input_size() const { return Impl::kInputSize; }

}  // namespace streetscope
