#include <streetscope/inference/metal_tone_mapper.h>

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>

#include <stdexcept>
#include <string>

namespace {

// GPU-side params struct — must match Metal shader layout
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

constexpr const char* kToneMapShaderSource = R"(
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

// Shared tone mapping math
static float3 apply_reinhard(float3 rgb, constant ToneMapParams& p) {
    // AWB gains
    rgb *= float3(p.gain_r, p.gain_g, p.gain_b);

    // Exposure
    rgb *= p.exposure;

    // Luminance (Rec. 709)
    float lum = dot(rgb, float3(0.2126f, 0.7152f, 0.0722f));

    // Extended Reinhard
    if (lum > 1e-6f) {
        float ws = p.white_point * p.white_point;
        float lm = lum * (1.0f + lum / ws) / (1.0f + lum);
        rgb *= (lm / lum);
    }

    // Clamp and gamma
    rgb = clamp(rgb, 0.0f, 1.0f);
    float inv_gamma = 1.0f / p.gamma;
    rgb = pow(rgb, float3(inv_gamma));

    return rgb;
}

// BGR buffer kernel: reads/writes packed 3-byte BGR
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
        float(input[base + 2]) / 255.0f,   // R
        float(input[base + 1]) / 255.0f,   // G
        float(input[base])     / 255.0f    // B
    );

    rgb = apply_reinhard(rgb, params);

    output[base]     = uchar(rgb.z * 255.0f + 0.5f);  // B
    output[base + 1] = uchar(rgb.y * 255.0f + 0.5f);  // G
    output[base + 2] = uchar(rgb.x * 255.0f + 0.5f);  // R
}

// BGRA texture kernel: reads/writes BGRA textures (CVPixelBuffer zero-copy)
kernel void tone_map_bgra(
    texture2d<float, access::read>  input  [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    constant ToneMapParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= uint(params.width) || gid.y >= uint(params.height)) return;

    float4 color = input.read(gid);
    // BGRA8Unorm: texture.read() auto-swizzles to (R, G, B, A)
    float3 rgb = float3(color.x, color.y, color.z);

    rgb = apply_reinhard(rgb, params);

    output.write(float4(rgb.x, rgb.y, rgb.z, 1.0f), gid);
}
)";

} // anonymous namespace

namespace streetscope {

struct MetalToneMapper::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> command_queue = nil;
    id<MTLComputePipelineState> bgr_pipeline = nil;
    id<MTLComputePipelineState> bgra_pipeline = nil;
    CVMetalTextureCacheRef texture_cache = nullptr;

    Impl() {
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            throw std::runtime_error("MetalToneMapper: no Metal device available");
        }

        command_queue = [device newCommandQueue];

        // Compile shader from source
        NSError* error = nil;
        NSString* source = [NSString stringWithUTF8String:kToneMapShaderSource];
        id<MTLLibrary> library = [device newLibraryWithSource:source
                                                      options:nil
                                                        error:&error];
        if (!library) {
            std::string msg = "MetalToneMapper: shader compilation failed";
            if (error) {
                msg += ": ";
                msg += [[error localizedDescription] UTF8String];
            }
            throw std::runtime_error(msg);
        }

        // Create pipeline states for both kernels
        id<MTLFunction> bgr_fn = [library newFunctionWithName:@"tone_map_bgr"];
        if (!bgr_fn) {
            throw std::runtime_error("MetalToneMapper: tone_map_bgr function not found");
        }
        bgr_pipeline = [device newComputePipelineStateWithFunction:bgr_fn error:&error];
        if (!bgr_pipeline) {
            throw std::runtime_error("MetalToneMapper: BGR pipeline creation failed");
        }

        id<MTLFunction> bgra_fn = [library newFunctionWithName:@"tone_map_bgra"];
        if (!bgra_fn) {
            throw std::runtime_error("MetalToneMapper: tone_map_bgra function not found");
        }
        bgra_pipeline = [device newComputePipelineStateWithFunction:bgra_fn error:&error];
        if (!bgra_pipeline) {
            throw std::runtime_error("MetalToneMapper: BGRA pipeline creation failed");
        }

        // Create texture cache for CVPixelBuffer -> MTLTexture zero-copy
        CVReturn status = CVMetalTextureCacheCreate(
            kCFAllocatorDefault, nil,
            device, nil,
            &texture_cache);
        if (status != kCVReturnSuccess) {
            throw std::runtime_error("MetalToneMapper: texture cache creation failed");
        }
    }

    ~Impl() {
        if (texture_cache) {
            CFRelease(texture_cache);
        }
    }

    std::vector<uint8_t> tone_map_bgr(const uint8_t* bgr, int w, int h,
                                       const Params& params) {
        const auto byte_count = static_cast<NSUInteger>(w) * h * 3;

        // Upload input to GPU
        id<MTLBuffer> input_buf = [device newBufferWithBytes:bgr
                                                      length:byte_count
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> output_buf = [device newBufferWithLength:byte_count
                                                       options:MTLResourceStorageModeShared];

        // Params uniform
        GPUToneMapParams gpu_params{};
        gpu_params.exposure = params.exposure;
        gpu_params.white_point = params.white_point;
        gpu_params.gamma = params.gamma;
        gpu_params.gain_b = params.gain_b;
        gpu_params.gain_g = params.gain_g;
        gpu_params.gain_r = params.gain_r;
        gpu_params.width = w;
        gpu_params.height = h;

        // Encode and dispatch
        id<MTLCommandBuffer> cmd = [command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:bgr_pipeline];
        [encoder setBuffer:input_buf offset:0 atIndex:0];
        [encoder setBuffer:output_buf offset:0 atIndex:1];
        [encoder setBytes:&gpu_params length:sizeof(gpu_params) atIndex:2];

        NSUInteger total_threads = static_cast<NSUInteger>(w) * h;
        NSUInteger thread_width = bgr_pipeline.maxTotalThreadsPerThreadgroup;
        if (thread_width > total_threads) thread_width = total_threads;
        MTLSize grid = MTLSizeMake(total_threads, 1, 1);
        MTLSize group = MTLSizeMake(thread_width, 1, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:group];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        // Read back
        std::vector<uint8_t> result(byte_count);
        memcpy(result.data(), [output_buf contents], byte_count);
        return result;
    }

    void* tone_map_pixelbuffer(void* cv_pixel_buffer, const Params& params) {
        auto* src_pb = static_cast<CVPixelBufferRef>(cv_pixel_buffer);
        auto src_w = static_cast<int>(CVPixelBufferGetWidth(src_pb));
        auto src_h = static_cast<int>(CVPixelBufferGetHeight(src_pb));

        // Wrap source CVPixelBuffer as MTLTexture (zero-copy via IOSurface)
        CVMetalTextureRef src_tex_ref = nullptr;
        CVReturn status = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, texture_cache, src_pb, nil,
            MTLPixelFormatBGRA8Unorm,
            static_cast<size_t>(src_w), static_cast<size_t>(src_h),
            0, &src_tex_ref);
        if (status != kCVReturnSuccess || !src_tex_ref) {
            throw std::runtime_error("MetalToneMapper: failed to create source texture");
        }
        id<MTLTexture> src_tex = CVMetalTextureGetTexture(src_tex_ref);

        // Create output CVPixelBuffer (IOSurface-backed)
        NSDictionary* attrs = @{
            (NSString*)kCVPixelBufferIOSurfacePropertiesKey: @{},
            (NSString*)kCVPixelBufferMetalCompatibilityKey: @YES,
        };
        CVPixelBufferRef dst_pb = nullptr;
        status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            static_cast<size_t>(src_w), static_cast<size_t>(src_h),
            kCVPixelFormatType_32BGRA, (__bridge CFDictionaryRef)attrs, &dst_pb);
        if (status != kCVReturnSuccess || !dst_pb) {
            CFRelease(src_tex_ref);
            throw std::runtime_error("MetalToneMapper: failed to create output pixel buffer");
        }

        // Wrap output as MTLTexture
        CVMetalTextureRef dst_tex_ref = nullptr;
        status = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, texture_cache, dst_pb, nil,
            MTLPixelFormatBGRA8Unorm,
            static_cast<size_t>(src_w), static_cast<size_t>(src_h),
            0, &dst_tex_ref);
        if (status != kCVReturnSuccess || !dst_tex_ref) {
            CFRelease(src_tex_ref);
            CVPixelBufferRelease(dst_pb);
            throw std::runtime_error("MetalToneMapper: failed to create destination texture");
        }
        id<MTLTexture> dst_tex = CVMetalTextureGetTexture(dst_tex_ref);

        // Params uniform
        GPUToneMapParams gpu_params{};
        gpu_params.exposure = params.exposure;
        gpu_params.white_point = params.white_point;
        gpu_params.gamma = params.gamma;
        gpu_params.gain_b = params.gain_b;
        gpu_params.gain_g = params.gain_g;
        gpu_params.gain_r = params.gain_r;
        gpu_params.width = src_w;
        gpu_params.height = src_h;

        // Encode and dispatch
        id<MTLCommandBuffer> cmd = [command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:bgra_pipeline];
        [encoder setTexture:src_tex atIndex:0];
        [encoder setTexture:dst_tex atIndex:1];
        [encoder setBytes:&gpu_params length:sizeof(gpu_params) atIndex:0];

        MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(src_w),
                                   static_cast<NSUInteger>(src_h), 1);
        NSUInteger tw = bgra_pipeline.threadExecutionWidth;
        NSUInteger th = bgra_pipeline.maxTotalThreadsPerThreadgroup / tw;
        MTLSize group = MTLSizeMake(tw, th, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:group];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        // Cleanup texture refs (doesn't release underlying CVPixelBuffers)
        CFRelease(src_tex_ref);
        CFRelease(dst_tex_ref);

        // Caller owns dst_pb — must CVPixelBufferRelease when done
        return dst_pb;
    }

    std::vector<uint8_t> upscale_and_tone_map(const uint8_t* bgr, int w, int h,
                                               int scale, const Params& params) {
        if (scale < 1) {
            throw std::invalid_argument("MetalToneMapper: scale must be >= 1");
        }
        if (scale == 1) {
            return tone_map_bgr(bgr, w, h, params);
        }

        const int out_w = w * scale;
        const int out_h = h * scale;
        const auto pixel_count = static_cast<size_t>(w) * h;

        // CPU: BGR -> BGRA at source resolution (~0.1ms for 512x288)
        std::vector<uint8_t> bgra_in(pixel_count * 4);
        for (size_t i = 0; i < pixel_count; i++) {
            bgra_in[i * 4]     = bgr[i * 3];       // B
            bgra_in[i * 4 + 1] = bgr[i * 3 + 1];   // G
            bgra_in[i * 4 + 2] = bgr[i * 3 + 2];   // R
            bgra_in[i * 4 + 3] = 255;               // A
        }

        // Source texture: Shared (CPU uploads via replaceRegion)
        MTLTextureDescriptor* src_desc = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                         width:static_cast<NSUInteger>(w)
                                        height:static_cast<NSUInteger>(h)
                                     mipmapped:NO];
        src_desc.usage = MTLTextureUsageShaderRead;
        src_desc.storageMode = MTLStorageModeShared;
        id<MTLTexture> src_tex = [device newTextureWithDescriptor:src_desc];

        [src_tex replaceRegion:MTLRegionMake2D(0, 0,
                                                static_cast<NSUInteger>(w),
                                                static_cast<NSUInteger>(h))
                   mipmapLevel:0
                     withBytes:bgra_in.data()
                   bytesPerRow:static_cast<NSUInteger>(w) * 4];

        // Intermediate texture: Private (GPU-only, never touches CPU)
        MTLTextureDescriptor* mid_desc = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                         width:static_cast<NSUInteger>(out_w)
                                        height:static_cast<NSUInteger>(out_h)
                                     mipmapped:NO];
        mid_desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        mid_desc.storageMode = MTLStorageModePrivate;
        id<MTLTexture> mid_tex = [device newTextureWithDescriptor:mid_desc];

        // Output texture: Shared (GPU writes, CPU reads back)
        MTLTextureDescriptor* dst_desc = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                         width:static_cast<NSUInteger>(out_w)
                                        height:static_cast<NSUInteger>(out_h)
                                     mipmapped:NO];
        dst_desc.usage = MTLTextureUsageShaderWrite;
        dst_desc.storageMode = MTLStorageModeShared;
        id<MTLTexture> dst_tex = [device newTextureWithDescriptor:dst_desc];

        // === Single command buffer: MPS Lanczos → Reinhard compute ===
        id<MTLCommandBuffer> cmd = [command_queue commandBuffer];

        // MPS Lanczos upscale (src_tex → mid_tex)
        MPSImageLanczosScale* lanczos =
            [[MPSImageLanczosScale alloc] initWithDevice:device];
        MPSScaleTransform xform;
        xform.scaleX = static_cast<double>(scale);
        xform.scaleY = static_cast<double>(scale);
        xform.translateX = 0.0;
        xform.translateY = 0.0;
        [lanczos setScaleTransform:&xform];
        [lanczos encodeToCommandBuffer:cmd
                         sourceTexture:src_tex
                    destinationTexture:mid_tex];

        // Reinhard tone map (mid_tex → dst_tex)
        GPUToneMapParams gpu_params{};
        gpu_params.exposure = params.exposure;
        gpu_params.white_point = params.white_point;
        gpu_params.gamma = params.gamma;
        gpu_params.gain_b = params.gain_b;
        gpu_params.gain_g = params.gain_g;
        gpu_params.gain_r = params.gain_r;
        gpu_params.width = out_w;
        gpu_params.height = out_h;

        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
        [encoder setComputePipelineState:bgra_pipeline];
        [encoder setTexture:mid_tex atIndex:0];
        [encoder setTexture:dst_tex atIndex:1];
        [encoder setBytes:&gpu_params length:sizeof(gpu_params) atIndex:0];

        MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(out_w),
                                   static_cast<NSUInteger>(out_h), 1);
        NSUInteger tw = bgra_pipeline.threadExecutionWidth;
        NSUInteger th = bgra_pipeline.maxTotalThreadsPerThreadgroup / tw;
        MTLSize group = MTLSizeMake(tw, th, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:group];
        [encoder endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];

        // Readback: BGRA texture → BGR vector
        const auto out_pixels = static_cast<size_t>(out_w) * out_h;
        std::vector<uint8_t> bgra_out(out_pixels * 4);
        [dst_tex getBytes:bgra_out.data()
              bytesPerRow:static_cast<NSUInteger>(out_w) * 4
               fromRegion:MTLRegionMake2D(0, 0,
                                           static_cast<NSUInteger>(out_w),
                                           static_cast<NSUInteger>(out_h))
              mipmapLevel:0];

        std::vector<uint8_t> result(out_pixels * 3);
        for (size_t i = 0; i < out_pixels; i++) {
            result[i * 3]     = bgra_out[i * 4];       // B
            result[i * 3 + 1] = bgra_out[i * 4 + 1];   // G
            result[i * 3 + 2] = bgra_out[i * 4 + 2];   // R
        }
        return result;
    }
};

MetalToneMapper::MetalToneMapper()
    : impl_(std::make_unique<Impl>()) {}

MetalToneMapper::~MetalToneMapper() = default;

std::vector<uint8_t> MetalToneMapper::tone_map(const uint8_t* bgr, int w, int h,
                                                const Params& params) {
    return impl_->tone_map_bgr(bgr, w, h, params);
}

void* MetalToneMapper::tone_map_pixelbuffer(void* cv_pixel_buffer,
                                             const Params& params) {
    return impl_->tone_map_pixelbuffer(cv_pixel_buffer, params);
}

std::vector<uint8_t> MetalToneMapper::upscale_and_tone_map(
    const uint8_t* bgr, int w, int h, int scale, const Params& params) {
    return impl_->upscale_and_tone_map(bgr, w, h, scale, params);
}

}  // namespace streetscope
