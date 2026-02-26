#pragma once
#include <cstdint>

namespace streetscope {

static constexpr uint32_t kPlanMagic = 0x53535031;  // "SSP1"
static constexpr uint16_t kPlanVersion = 1;

enum class StageOp : uint8_t {
    kU8ToF32            = 0,
    kEmaAccumulate      = 1,
    kF32ToU8            = 2,
    kBackgroundSubtract = 3,
    kAeAwb              = 4,
    kBoxBlur            = 5,
    kAfBlend            = 6,
    kMemcpy             = 7,
};

enum class DType : uint8_t {
    kUint8   = 0,
    kFloat32 = 1,
};

static constexpr uint8_t kBufferNone = 0xFF;

// Well-known external buffer indices
static constexpr uint8_t kBufFrameU8       = 0;
static constexpr uint8_t kBufBackgroundF32 = 1;
static constexpr uint8_t kBufMaskOut       = 2;
static constexpr uint8_t kBufDisplayOut    = 3;
static constexpr uint8_t kBufAfAlphaMap    = 4;
static constexpr uint8_t kBufArenaStart    = 5;

struct StageDesc {
    StageOp op;
    uint8_t inputs[4];   // buffer indices, kBufferNone = unused
    uint8_t outputs[2];  // buffer indices, kBufferNone = unused
    uint8_t padding;
};
static_assert(sizeof(StageDesc) == 8, "StageDesc must be 8 bytes");

struct BufferDesc {
    uint32_t offset;      // byte offset from arena base (0 for external)
    uint32_t size_bytes;  // total size (0 for external)
    DType dtype;
    uint8_t padding[3];
};
static_assert(sizeof(BufferDesc) == 12, "BufferDesc must be 12 bytes");

struct PlanHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t num_stages;
    uint16_t num_buffers;
    uint16_t reserved;
    uint32_t arena_size_bytes;
    uint32_t width;
    uint32_t height;
};
static_assert(sizeof(PlanHeader) == 24, "PlanHeader must be 24 bytes");

}  // namespace streetscope
