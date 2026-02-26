"""Plan builder and binary serializer for the StreetScope execution plan."""

import struct

from .allocator import BufferDef, align_up, pack_buffers

MAGIC = 0x53535031  # "SSP1"
VERSION = 1

# Stage ops (must match C++ StageOp enum)
OP_U8_TO_F32 = 0
OP_EMA_ACCUMULATE = 1
OP_F32_TO_U8 = 2
OP_BACKGROUND_SUBTRACT = 3
OP_AE_AWB = 4
OP_BOX_BLUR = 5
OP_AF_BLEND = 6
OP_MEMCPY = 7

# DType (must match C++ DType enum)
DTYPE_UINT8 = 0
DTYPE_FLOAT32 = 1

# Well-known buffer indices
BUF_FRAME_U8 = 0
BUF_BACKGROUND_F32 = 1
BUF_MASK_OUT = 2
BUF_DISPLAY_OUT = 3
BUF_AF_ALPHA_MAP = 4
# Arena buffers start at 5
BUF_FRAME_F32 = 5
BUF_BG_U8 = 6
BUF_CORRECTED = 7
BUF_BLURRED = 8
BUF_BLUR_TEMP = 9

BUFFER_NONE = 0xFF

DTYPE_CODE = {"uint8": DTYPE_UINT8, "float32": DTYPE_FLOAT32}


def stage_bytes(op: int, inputs: list[int], outputs: list[int]) -> bytes:
    """Pack a single StageDesc into 8 bytes."""
    ins = (inputs + [BUFFER_NONE] * 4)[:4]
    outs = (outputs + [BUFFER_NONE] * 2)[:2]
    return struct.pack("<B4B2BB", op, *ins, *outs, 0)


def buffer_bytes(offset: int, size_bytes: int, dtype: str) -> bytes:
    """Pack a single BufferDesc into 12 bytes."""
    return struct.pack("<IIB3x", offset, size_bytes, DTYPE_CODE[dtype])


def build_plan(width: int, height: int) -> bytes:
    """Build a serialized execution plan for the given frame dimensions.

    ISP (AE + AWB + AF) always runs. The plan is fixed at 7 stages, 10 buffers.
    """
    pixels = width * height
    bytes_u8 = pixels * 3
    bytes_f32 = pixels * 3 * 4

    # ── Define buffers ────────────────────────────────────
    # External buffers (indices 0-4): offset/size 0, resolved from run_frame() args.
    buffers: list[BufferDef] = [
        BufferDef("frame_u8", bytes_u8, "uint8", True, -1, -1),
        BufferDef("background_f32", bytes_f32, "float32", True, -1, -1),
        BufferDef("mask_out", pixels, "uint8", True, -1, -1),
        BufferDef("display_out", bytes_u8, "uint8", True, -1, -1),
        BufferDef("af_alpha_map", pixels * 4, "float32", True, -1, -1),
    ]

    # Arena temporaries with stage lifetimes for packing.
    #
    # Stage ordering (fixed, 7 stages):
    #   0: kU8ToF32            frame_u8 -> frame_f32
    #   1: kEmaAccumulate      frame_f32, background_f32 -> background_f32
    #   2: kF32ToU8            background_f32 -> bg_u8
    #   3: kBackgroundSubtract frame_u8, bg_u8 -> mask_out
    #   4: kAeAwb              frame_u8 -> corrected
    #   5: kBoxBlur            corrected -> blurred (blur_temp)
    #   6: kAfBlend            corrected, blurred, af_alpha_map -> display_out

    # frame_f32: produced stage 0, consumed stage 1
    buffers.append(BufferDef("frame_f32", bytes_f32, "float32", False, 0, 1))
    # bg_u8: produced stage 2, consumed stage 3
    buffers.append(BufferDef("bg_u8", bytes_u8, "uint8", False, 2, 3))
    # corrected: produced stage 4, consumed stages 5-6
    buffers.append(BufferDef("corrected", bytes_u8, "uint8", False, 4, 6))
    # blurred: produced stage 5, consumed stage 6
    buffers.append(BufferDef("blurred", bytes_u8, "uint8", False, 5, 6))
    # blur_temp: scratch for stage 5 only
    buffers.append(BufferDef("blur_temp", bytes_u8, "uint8", False, 5, 5))

    # ── Define stages (always 7) ─────────────────────────
    stages: list[bytes] = [
        # Stage 0: kU8ToF32  in=frame_u8  out=frame_f32
        stage_bytes(OP_U8_TO_F32, [BUF_FRAME_U8], [BUF_FRAME_F32]),
        # Stage 1: kEmaAccumulate  in=frame_f32, background_f32  out=background_f32
        stage_bytes(OP_EMA_ACCUMULATE, [BUF_FRAME_F32, BUF_BACKGROUND_F32], [BUF_BACKGROUND_F32]),
        # Stage 2: kF32ToU8  in=background_f32  out=bg_u8
        stage_bytes(OP_F32_TO_U8, [BUF_BACKGROUND_F32], [BUF_BG_U8]),
        # Stage 3: kBackgroundSubtract  in=frame_u8, bg_u8  out=mask_out
        stage_bytes(OP_BACKGROUND_SUBTRACT, [BUF_FRAME_U8, BUF_BG_U8], [BUF_MASK_OUT]),
        # Stage 4: kAeAwb  in=frame_u8  out=corrected
        stage_bytes(OP_AE_AWB, [BUF_FRAME_U8], [BUF_CORRECTED]),
        # Stage 5: kBoxBlur  in=corrected, blur_temp  out=blurred
        stage_bytes(OP_BOX_BLUR, [BUF_CORRECTED, BUF_BLUR_TEMP], [BUF_BLURRED]),
        # Stage 6: kAfBlend  in=corrected, af_alpha_map, blurred  out=display_out
        stage_bytes(OP_AF_BLEND, [BUF_CORRECTED, BUF_AF_ALPHA_MAP, BUF_BLURRED], [BUF_DISPLAY_OUT]),
    ]

    # ── Pack arena ────────────────────────────────────────
    arena_offsets, arena_size = pack_buffers(buffers)

    # ── Serialize ─────────────────────────────────────────
    return serialize(stages, buffers, arena_offsets, arena_size, width, height)


def serialize(
    stages: list[bytes],
    buffers: list[BufferDef],
    arena_offsets: dict[str, int],
    arena_size: int,
    width: int,
    height: int,
) -> bytes:
    """Serialize the plan to a binary blob."""
    num_stages = len(stages)
    num_buffers = len(buffers)

    # Header: 24 bytes
    header = struct.pack(
        "<IHHHHIII",
        MAGIC,
        VERSION,
        num_stages,
        num_buffers,
        0,  # reserved
        arena_size,
        width,
        height,
    )

    # Stage descriptors
    stage_data = b"".join(stages)

    # Buffer descriptors
    buf_data = b""
    for buf in buffers:
        if buf.external:
            buf_data += buffer_bytes(0, 0, buf.dtype)
        else:
            offset = arena_offsets[buf.name]
            buf_data += buffer_bytes(offset, align_up(buf.size_bytes), buf.dtype)

    return header + stage_data + buf_data


def parse_plan(data: bytes) -> dict:
    """Parse a plan binary back into a dict (for testing/debugging)."""
    # Header: 24 bytes
    (magic, version, num_stages, num_buffers, _reserved, arena_size, width, height) = struct.unpack(
        "<IHHHHIII", data[:24]
    )

    offset = 24

    # Stages: 8 bytes each
    stages = []
    for _ in range(num_stages):
        chunk = data[offset : offset + 8]
        op, in0, in1, in2, in3, out0, out1, _pad = struct.unpack("<B4B2BB", chunk)
        inputs = [x for x in (in0, in1, in2, in3) if x != BUFFER_NONE]
        outputs = [x for x in (out0, out1) if x != BUFFER_NONE]
        stages.append({"op": op, "inputs": inputs, "outputs": outputs})
        offset += 8

    # Buffers: 12 bytes each
    buf_descs = []
    for _ in range(num_buffers):
        chunk = data[offset : offset + 12]
        buf_offset, size_bytes, dtype_code = struct.unpack("<IIB3x", chunk)
        dtype = "uint8" if dtype_code == DTYPE_UINT8 else "float32"
        buf_descs.append({"offset": buf_offset, "size_bytes": size_bytes, "dtype": dtype})
        offset += 12

    return {
        "magic": magic,
        "version": version,
        "num_stages": num_stages,
        "num_buffers": num_buffers,
        "arena_size": arena_size,
        "width": width,
        "height": height,
        "stages": stages,
        "buffers": buf_descs,
    }
