#include <gtest/gtest.h>
#include <streetscope/inference/metal_tone_mapper.h>
#include <streetscope/simd/tone_map.h>

#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <Metal/Metal.h>

#include <chrono>
#include <cstddef>
#include <vector>

using streetscope::MetalToneMapper;
using streetscope::simd::ToneMapParams;
using streetscope::simd::tone_map_scalar;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create an IOSurface-backed BGRA CVPixelBuffer with a gradient pattern.
static CVPixelBufferRef create_test_pixelbuffer(int w, int h) {
    NSDictionary* attrs = @{
        (NSString*)kCVPixelBufferIOSurfacePropertiesKey: @{},
        (NSString*)kCVPixelBufferMetalCompatibilityKey: @YES,
    };
    CVPixelBufferRef pb = nullptr;
    CVReturn status = CVPixelBufferCreate(
        kCFAllocatorDefault,
        static_cast<size_t>(w), static_cast<size_t>(h),
        kCVPixelFormatType_32BGRA,
        (__bridge CFDictionaryRef)attrs, &pb);
    EXPECT_EQ(status, kCVReturnSuccess);

    CVPixelBufferLockBaseAddress(pb, 0);
    auto* base = static_cast<uint8_t*>(CVPixelBufferGetBaseAddress(pb));
    size_t row_bytes = CVPixelBufferGetBytesPerRow(pb);
    for (int y = 0; y < h; y++) {
        auto* row = base + y * row_bytes;
        for (int x = 0; x < w; x++) {
            row[x * 4 + 0] = static_cast<uint8_t>((x * 7 + y * 13) % 256);      // B
            row[x * 4 + 1] = static_cast<uint8_t>((x * 11 + y * 3) % 256);      // G
            row[x * 4 + 2] = static_cast<uint8_t>((x * 5 + y * 17) % 256);      // R
            row[x * 4 + 3] = 255;                                                 // A
        }
    }
    CVPixelBufferUnlockBaseAddress(pb, 0);
    return pb;
}

/// Extract BGR bytes from a BGRA CVPixelBuffer (for comparison with scalar).
static std::vector<uint8_t> extract_bgr(CVPixelBufferRef pb) {
    auto w = static_cast<int>(CVPixelBufferGetWidth(pb));
    auto h = static_cast<int>(CVPixelBufferGetHeight(pb));
    std::vector<uint8_t> bgr(static_cast<size_t>(w) * h * 3);

    CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
    auto* base = static_cast<const uint8_t*>(CVPixelBufferGetBaseAddress(pb));
    size_t row_bytes = CVPixelBufferGetBytesPerRow(pb);
    for (int y = 0; y < h; y++) {
        const auto* row = base + y * row_bytes;
        for (int x = 0; x < w; x++) {
            auto dst = static_cast<size_t>(y * w + x) * 3;
            bgr[dst]     = row[x * 4];      // B
            bgr[dst + 1] = row[x * 4 + 1];  // G
            bgr[dst + 2] = row[x * 4 + 2];  // R
        }
    }
    CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
    return bgr;
}

// ---------------------------------------------------------------------------
// Metal BGR vs Scalar
// ---------------------------------------------------------------------------

class MetalToneMapTest : public ::testing::Test {
protected:
    MetalToneMapper mapper;
};

TEST_F(MetalToneMapTest, BGRMatchesScalar) {
    const int w = 64, h = 48;
    const auto sz = static_cast<size_t>(w * h) * 3;
    std::vector<uint8_t> input(sz);
    for (size_t i = 0; i < sz; i++) {
        input[i] = static_cast<uint8_t>((i * 31 + 17) % 256);
    }

    MetalToneMapper::Params mparams{};
    mparams.exposure = 1.5f;
    mparams.white_point = 2.0f;
    mparams.gamma = 2.2f;
    mparams.gain_b = 0.95f;
    mparams.gain_g = 1.0f;
    mparams.gain_r = 1.05f;

    // Metal path
    auto metal_out = mapper.tone_map(input.data(), w, h, mparams);

    // Scalar path
    ToneMapParams sparams{};
    sparams.exposure = mparams.exposure;
    sparams.white_point = mparams.white_point;
    sparams.gamma = mparams.gamma;
    sparams.gain_b = mparams.gain_b;
    sparams.gain_g = mparams.gain_g;
    sparams.gain_r = mparams.gain_r;
    std::vector<uint8_t> scalar_out(sz);
    tone_map_scalar(input.data(), scalar_out.data(), w, h, sparams);

    ASSERT_EQ(metal_out.size(), scalar_out.size());
    int max_diff = 0;
    for (size_t i = 0; i < sz; i++) {
        int diff = std::abs(static_cast<int>(metal_out[i]) - static_cast<int>(scalar_out[i]));
        max_diff = std::max(max_diff, diff);
        EXPECT_NEAR(metal_out[i], scalar_out[i], 2) << "byte " << i;
    }
    // Log max difference for visibility
    EXPECT_LE(max_diff, 2) << "Max per-channel difference between Metal and scalar";
}

TEST_F(MetalToneMapTest, BGRIdentityParams) {
    const int w = 8, h = 4;
    const auto sz = static_cast<size_t>(w * h) * 3;
    std::vector<uint8_t> input(sz);
    for (size_t i = 0; i < sz; i++) {
        input[i] = static_cast<uint8_t>((i * 13) % 256);
    }

    // Identity: exposure=1, white=1, gamma=1, gains=1
    // Reinhard with white=1: L*(1+L)/(1+L) = L → identity
    MetalToneMapper::Params params{};
    params.exposure = 1.0f;
    params.white_point = 1.0f;
    params.gamma = 1.0f;

    auto out = mapper.tone_map(input.data(), w, h, params);
    ASSERT_EQ(out.size(), sz);
    for (size_t i = 0; i < sz; i++) {
        EXPECT_NEAR(out[i], input[i], 2) << "byte " << i;
    }
}

TEST_F(MetalToneMapTest, BGRLargeFrame) {
    const int w = 512, h = 288;
    const auto sz = static_cast<size_t>(w * h) * 3;
    std::vector<uint8_t> input(sz);
    for (size_t i = 0; i < sz; i++) {
        input[i] = static_cast<uint8_t>((i * 7) % 256);
    }

    MetalToneMapper::Params params{};
    params.exposure = 1.5f;
    params.white_point = 2.0f;
    params.gamma = 2.2f;

    auto metal_out = mapper.tone_map(input.data(), w, h, params);

    ToneMapParams sparams{};
    sparams.exposure = params.exposure;
    sparams.white_point = params.white_point;
    sparams.gamma = params.gamma;
    std::vector<uint8_t> scalar_out(sz);
    tone_map_scalar(input.data(), scalar_out.data(), w, h, sparams);

    int max_diff = 0;
    for (size_t i = 0; i < sz; i++) {
        int diff = std::abs(static_cast<int>(metal_out[i]) - static_cast<int>(scalar_out[i]));
        max_diff = std::max(max_diff, diff);
    }
    EXPECT_LE(max_diff, 2) << "512x288 frame: max diff between Metal and scalar";
}

// ---------------------------------------------------------------------------
// Metal BGRA (CVPixelBuffer) vs Scalar
// ---------------------------------------------------------------------------

TEST_F(MetalToneMapTest, PixelBufferMatchesScalar) {
    const int w = 64, h = 48;

    CVPixelBufferRef src_pb = create_test_pixelbuffer(w, h);
    ASSERT_NE(src_pb, nullptr);

    // Extract BGR from source for scalar reference
    std::vector<uint8_t> src_bgr = extract_bgr(src_pb);

    MetalToneMapper::Params mparams{};
    mparams.exposure = 1.8f;
    mparams.white_point = 1.5f;
    mparams.gamma = 2.2f;
    mparams.gain_b = 0.9f;
    mparams.gain_g = 1.0f;
    mparams.gain_r = 1.1f;

    // Metal pixelbuffer path
    auto* dst_raw = mapper.tone_map_pixelbuffer(src_pb, mparams);
    ASSERT_NE(dst_raw, nullptr);
    auto* dst_pb = static_cast<CVPixelBufferRef>(dst_raw);

    // Extract BGR from Metal output
    std::vector<uint8_t> metal_bgr = extract_bgr(dst_pb);

    // Scalar reference
    ToneMapParams sparams{};
    sparams.exposure = mparams.exposure;
    sparams.white_point = mparams.white_point;
    sparams.gamma = mparams.gamma;
    sparams.gain_b = mparams.gain_b;
    sparams.gain_g = mparams.gain_g;
    sparams.gain_r = mparams.gain_r;
    std::vector<uint8_t> scalar_bgr(src_bgr.size());
    tone_map_scalar(src_bgr.data(), scalar_bgr.data(), w, h, sparams);

    // Compare
    int max_diff = 0;
    for (size_t i = 0; i < metal_bgr.size(); i++) {
        int diff = std::abs(static_cast<int>(metal_bgr[i]) - static_cast<int>(scalar_bgr[i]));
        max_diff = std::max(max_diff, diff);
        EXPECT_NEAR(metal_bgr[i], scalar_bgr[i], 2) << "byte " << i;
    }
    EXPECT_LE(max_diff, 2) << "CVPixelBuffer path: max diff between Metal and scalar";

    CVPixelBufferRelease(dst_pb);
    CVPixelBufferRelease(src_pb);
}

// ---------------------------------------------------------------------------
// Fused MPS Lanczos + Reinhard
// ---------------------------------------------------------------------------

TEST_F(MetalToneMapTest, FusedUpscaleOutputDimensions) {
    const int w = 16, h = 12;
    const auto sz = static_cast<size_t>(w) * h * 3;
    std::vector<uint8_t> input(sz);
    for (size_t i = 0; i < sz; i++) {
        input[i] = static_cast<uint8_t>(i % 256);
    }

    MetalToneMapper::Params params{};
    for (int scale : {2, 3, 4}) {
        auto result = mapper.upscale_and_tone_map(input.data(), w, h, scale, params);
        EXPECT_EQ(result.size(), static_cast<size_t>(w * scale) * (h * scale) * 3)
            << "scale=" << scale;
    }
}

TEST_F(MetalToneMapTest, FusedScale1MatchesToneMap) {
    const int w = 64, h = 48;
    const auto sz = static_cast<size_t>(w) * h * 3;
    std::vector<uint8_t> input(sz);
    for (size_t i = 0; i < sz; i++) {
        input[i] = static_cast<uint8_t>((i * 13 + 7) % 256);
    }

    MetalToneMapper::Params params{};
    params.exposure = 1.5f;
    params.white_point = 2.0f;
    params.gamma = 2.2f;

    auto tone_only = mapper.tone_map(input.data(), w, h, params);
    auto fused_1x = mapper.upscale_and_tone_map(input.data(), w, h, 1, params);

    ASSERT_EQ(tone_only.size(), fused_1x.size());
    for (size_t i = 0; i < tone_only.size(); i++) {
        EXPECT_EQ(tone_only[i], fused_1x[i]) << "byte " << i;
    }
}

TEST_F(MetalToneMapTest, FusedToneMappingApplied) {
    const int w = 32, h = 24;
    const int scale = 2;
    const auto sz = static_cast<size_t>(w) * h * 3;
    std::vector<uint8_t> input(sz);
    for (size_t i = 0; i < sz; i++) {
        input[i] = static_cast<uint8_t>((i * 31 + 17) % 256);
    }

    MetalToneMapper::Params active{};
    active.exposure = 1.5f;
    active.white_point = 2.0f;
    active.gamma = 1.0f;
    active.gain_b = 0.95f;
    active.gain_r = 1.05f;

    MetalToneMapper::Params identity{};
    identity.exposure = 1.0f;
    identity.white_point = 1.0f;
    identity.gamma = 1.0f;

    auto fused = mapper.upscale_and_tone_map(input.data(), w, h, scale, active);
    auto neutral = mapper.upscale_and_tone_map(input.data(), w, h, scale, identity);

    ASSERT_EQ(fused.size(), neutral.size());

    int diff_count = 0;
    for (size_t i = 0; i < fused.size(); i++) {
        if (fused[i] != neutral[i]) diff_count++;
    }
    EXPECT_GT(diff_count, static_cast<int>(fused.size()) / 2)
        << "Tone mapping should modify most pixels";
}

TEST_F(MetalToneMapTest, FusedInvalidScale) {
    std::vector<uint8_t> input(8 * 4 * 3, 128);
    MetalToneMapper::Params params{};

    EXPECT_THROW(mapper.upscale_and_tone_map(input.data(), 8, 4, 0, params),
                 std::invalid_argument);
    EXPECT_THROW(mapper.upscale_and_tone_map(input.data(), 8, 4, -1, params),
                 std::invalid_argument);
}

TEST_F(MetalToneMapTest, FusedRealisticResolution) {
    const int w = 512, h = 288;
    const int scale = 2;
    const auto sz = static_cast<size_t>(w) * h * 3;
    std::vector<uint8_t> input(sz);
    for (size_t i = 0; i < sz; i++) {
        input[i] = static_cast<uint8_t>((i * 7) % 256);
    }

    MetalToneMapper::Params params{};
    params.exposure = 1.5f;
    params.white_point = 2.0f;
    params.gamma = 1.0f;
    params.gain_b = 0.95f;
    params.gain_r = 1.05f;

    auto result = mapper.upscale_and_tone_map(input.data(), w, h, scale, params);
    ASSERT_EQ(result.size(), static_cast<size_t>(w * scale) * (h * scale) * 3);

    // Sanity: not all zero or all saturated
    int zero_count = 0, max_count = 0;
    for (auto b : result) {
        if (b == 0) zero_count++;
        if (b == 255) max_count++;
    }
    EXPECT_LT(zero_count, static_cast<int>(result.size()) / 2);
    EXPECT_LT(max_count, static_cast<int>(result.size()) / 2);
}

// ---------------------------------------------------------------------------
// GPU / NE Concurrency Test
// ---------------------------------------------------------------------------
//
// Proves heterogeneous compute: Metal GPU and CoreML Neural Engine run in
// parallel. Uses raw Metal APIs for non-blocking GPU dispatch and loads the
// YOLO model with MLComputeUnitsCPUAndNeuralEngine to exclude the GPU —
// forcing inference onto the Neural Engine.
//

namespace {

// GPU-side params — must match Metal shader layout
struct GPUToneMapParams {
    float exposure;
    float white_point;
    float gamma;
    float gain_b;
    float gain_g;
    float gain_r;
    int width;
    int height;
};

// Same tone map shader used in MetalToneMapper — embedded here so the test
// can dispatch non-blocking without going through the synchronous API.
constexpr const char* kTestShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

struct ToneMapParams {
    float exposure;
    float white_point;
    float gamma;
    float gain_b;
    float gain_g;
    float gain_r;
    int width;
    int height;
};

static float3 apply_reinhard(float3 rgb, constant ToneMapParams& p) {
    rgb *= float3(p.gain_r, p.gain_g, p.gain_b);
    rgb *= p.exposure;
    float lum = dot(rgb, float3(0.2126f, 0.7152f, 0.0722f));
    if (lum > 1e-6f) {
        float ws = p.white_point * p.white_point;
        float lm = lum * (1.0f + lum / ws) / (1.0f + lum);
        rgb *= (lm / lum);
    }
    rgb = clamp(rgb, 0.0f, 1.0f);
    float inv_gamma = 1.0f / p.gamma;
    rgb = pow(rgb, float3(inv_gamma));
    return rgb;
}

kernel void tone_map_bgr(
    device const uchar* input  [[buffer(0)]],
    device uchar*       output [[buffer(1)]],
    constant ToneMapParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    int total = params.width * params.height;
    if (int(gid) >= total) return;
    uint base = gid * 3;
    float3 rgb = float3(
        float(input[base + 2]) / 255.0f,
        float(input[base + 1]) / 255.0f,
        float(input[base])     / 255.0f
    );
    rgb = apply_reinhard(rgb, params);
    output[base]     = uchar(rgb.z * 255.0f + 0.5f);
    output[base + 1] = uchar(rgb.y * 255.0f + 0.5f);
    output[base + 2] = uchar(rgb.x * 255.0f + 0.5f);
}
)";

/// Encode + commit Metal tone map work. Does NOT wait — returns the command
/// buffer so the caller can wait later.
static id<MTLCommandBuffer> dispatch_metal_nonblocking(
    id<MTLCommandQueue> queue,
    id<MTLComputePipelineState> pipeline,
    id<MTLBuffer> input_buf,
    id<MTLBuffer> output_buf,
    const GPUToneMapParams& params,
    int w, int h
) {
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input_buf offset:0 atIndex:0];
    [encoder setBuffer:output_buf offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];

    auto total_threads = static_cast<NSUInteger>(w) * h;
    NSUInteger thread_width = pipeline.maxTotalThreadsPerThreadgroup;
    if (thread_width > total_threads) thread_width = total_threads;
    MTLSize grid = MTLSizeMake(total_threads, 1, 1);
    MTLSize group = MTLSizeMake(thread_width, 1, 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
    [cmd commit];
    return cmd;  // caller calls [cmd waitUntilCompleted]
}

/// Run one YOLO prediction on a dummy input. Returns wall time in ms.
static double run_yolo_prediction(MLModel* model, NSString* input_name) {
    @autoreleasepool {
        // Create 640x640 BGRA CVPixelBuffer as YOLO input
        NSDictionary* attrs = @{
            (NSString*)kCVPixelBufferIOSurfacePropertiesKey: @{},
        };
        CVPixelBufferRef pb = nullptr;
        CVPixelBufferCreate(kCFAllocatorDefault, 640, 640,
                            kCVPixelFormatType_32BGRA,
                            (__bridge CFDictionaryRef)attrs, &pb);
        CVPixelBufferLockBaseAddress(pb, 0);
        memset(CVPixelBufferGetBaseAddress(pb), 128,
               CVPixelBufferGetBytesPerRow(pb) * 640);
        CVPixelBufferUnlockBaseAddress(pb, 0);

        MLFeatureValue* fv = [MLFeatureValue featureValueWithPixelBuffer:pb];
        NSDictionary* dict = @{input_name: fv};
        NSError* error = nil;
        MLDictionaryFeatureProvider* provider =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:dict
                                                             error:&error];

        auto t0 = std::chrono::steady_clock::now();
        [model predictionFromFeatures:provider error:&error];
        auto t1 = std::chrono::steady_clock::now();

        CVPixelBufferRelease(pb);
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
}

}  // anonymous namespace

TEST(ConcurrencyTest, GPUAndNERunConcurrently) {
    @autoreleasepool {
        // --- Skip if YOLO model not available ---
        const char* model_path = "models/yolo11s.mlpackage";
        NSString* path = [NSString stringWithUTF8String:model_path];
        BOOL is_dir = NO;
        if (![[NSFileManager defaultManager] fileExistsAtPath:path
                                                  isDirectory:&is_dir] || !is_dir) {
            GTEST_SKIP() << "Model not found at " << model_path;
        }

        // --- Load YOLO with CPUAndNeuralEngine (exclude GPU) ---
        NSError* error = nil;
        NSURL* url = [NSURL fileURLWithPath:path];
        NSURL* compiled = [MLModel compileModelAtURL:url error:&error];
        ASSERT_EQ(error, nil) << [[error localizedDescription] UTF8String];

        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;

        MLModel* yolo = [MLModel modelWithContentsOfURL:compiled
                                          configuration:config
                                                  error:&error];
        ASSERT_EQ(error, nil) << [[error localizedDescription] UTF8String];

        NSString* yolo_input_name =
            yolo.modelDescription.inputDescriptionsByName.allKeys.firstObject;
        ASSERT_NE(yolo_input_name, nil);

        // --- Create Metal pipeline ---
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        ASSERT_NE(device, nil);
        id<MTLCommandQueue> queue = [device newCommandQueue];

        NSString* src = [NSString stringWithUTF8String:kTestShaderSource];
        id<MTLLibrary> lib = [device newLibraryWithSource:src
                                                  options:nil
                                                    error:&error];
        ASSERT_NE(lib, nil) << [[error localizedDescription] UTF8String];
        id<MTLFunction> fn = [lib newFunctionWithName:@"tone_map_bgr"];
        ASSERT_NE(fn, nil);
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:fn error:&error];
        ASSERT_NE(pipeline, nil);

        // --- Prepare Metal buffers (4K frame) ---
        // Use a large frame so GPU work is comparable to YOLO on NE,
        // making overlap clearly measurable.
        const int w = 3840, h = 2160;
        const auto byte_count = static_cast<NSUInteger>(w) * h * 3;
        std::vector<uint8_t> input(byte_count);
        for (size_t i = 0; i < byte_count; i++) {
            input[i] = static_cast<uint8_t>((i * 7) % 256);
        }

        id<MTLBuffer> input_buf =
            [device newBufferWithBytes:input.data()
                                length:byte_count
                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> output_buf =
            [device newBufferWithLength:byte_count
                                options:MTLResourceStorageModeShared];

        GPUToneMapParams gpu_params{};
        gpu_params.exposure = 1.5f;
        gpu_params.white_point = 2.0f;
        gpu_params.gamma = 2.2f;
        gpu_params.gain_b = 1.0f;
        gpu_params.gain_g = 1.0f;
        gpu_params.gain_r = 1.0f;
        gpu_params.width = w;
        gpu_params.height = h;

        // --- Warmup (3 iterations each) ---
        for (int i = 0; i < 3; i++) {
            auto cmd = dispatch_metal_nonblocking(
                queue, pipeline, input_buf, output_buf, gpu_params, w, h);
            [cmd waitUntilCompleted];
        }
        for (int i = 0; i < 3; i++) {
            run_yolo_prediction(yolo, yolo_input_name);
        }

        // --- Sequential: Metal (blocking) then YOLO (blocking) ---
        constexpr int kTrials = 5;
        double best_seq = 1e9;
        double best_metal_alone = 1e9;
        double best_yolo_alone = 1e9;

        for (int t = 0; t < kTrials; t++) {
            auto t0 = std::chrono::steady_clock::now();
            auto cmd = dispatch_metal_nonblocking(
                queue, pipeline, input_buf, output_buf, gpu_params, w, h);
            [cmd waitUntilCompleted];
            auto t1 = std::chrono::steady_clock::now();
            double yolo_ms = run_yolo_prediction(yolo, yolo_input_name);
            auto t2 = std::chrono::steady_clock::now();

            double metal_ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            double total_ms =
                std::chrono::duration<double, std::milli>(t2 - t0).count();

            best_metal_alone = std::min(best_metal_alone, metal_ms);
            best_yolo_alone = std::min(best_yolo_alone, yolo_ms);
            best_seq = std::min(best_seq, total_ms);
        }

        // --- Concurrent: Metal commit (non-blocking) + YOLO (NE) ---
        double best_conc = 1e9;

        for (int t = 0; t < kTrials; t++) {
            auto t0 = std::chrono::steady_clock::now();

            // GPU: commit tone map work, returns immediately
            auto cmd = dispatch_metal_nonblocking(
                queue, pipeline, input_buf, output_buf, gpu_params, w, h);

            // NE: YOLO inference runs while GPU is working
            run_yolo_prediction(yolo, yolo_input_name);

            // GPU: wait for completion (should already be done if overlap worked)
            [cmd waitUntilCompleted];

            auto t1 = std::chrono::steady_clock::now();
            double conc_ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            best_conc = std::min(best_conc, conc_ms);
        }

        // --- Report ---
        double overlap_pct = (1.0 - best_conc / best_seq) * 100.0;
        printf("\n=== GPU/NE Concurrency Test ===\n");
        printf("Metal alone:  %.2f ms\n", best_metal_alone);
        printf("YOLO alone:   %.2f ms  (CPUAndNeuralEngine)\n", best_yolo_alone);
        printf("Sequential:   %.2f ms  (Metal + YOLO)\n", best_seq);
        printf("Concurrent:   %.2f ms  (Metal || YOLO)\n", best_conc);
        printf("Overlap:      %.1f%%\n", overlap_pct);
        printf("Expected max: %.2f ms  (max of individual)\n",
               std::max(best_metal_alone, best_yolo_alone));

        // Concurrent should be closer to max(metal, yolo) than to sum.
        // At least 10% faster than sequential proves real overlap.
        EXPECT_LT(best_conc, best_seq * 0.9)
            << "Expected at least 10% speedup from GPU/NE overlap";
    }
}
