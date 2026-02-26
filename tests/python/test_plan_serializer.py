"""Tests for the static execution plan builder and serializer."""

import struct
import sys
from pathlib import Path

# Ensure src/python is importable
_src = Path(__file__).resolve().parent.parent.parent / "src" / "python"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from plan.allocator import ALIGNMENT, align_up, pack_buffers  # noqa: E402
from plan.serializer import (  # noqa: E402
    BUF_BG_U8,
    BUF_FRAME_F32,
    MAGIC,
    OP_AE_AWB,
    OP_AF_BLEND,
    OP_BACKGROUND_SUBTRACT,
    OP_BOX_BLUR,
    OP_EMA_ACCUMULATE,
    OP_F32_TO_U8,
    OP_U8_TO_F32,
    VERSION,
    build_plan,
    parse_plan,
)

W, H = 512, 288


class TestRoundTrip:
    """Round-trip: build_plan -> parse_plan -> verify fields."""

    def test_round_trip_full_isp(self):
        """Full ISP (AE+AWB+AF): always 7 stages, 10 buffers, positive arena."""
        data = build_plan(W, H)
        plan = parse_plan(data)

        assert plan["num_stages"] == 7
        assert plan["num_buffers"] == 10
        assert plan["arena_size"] > 0
        assert plan["width"] == W
        assert plan["height"] == H

        ops = [s["op"] for s in plan["stages"]]
        assert ops == [
            OP_U8_TO_F32,
            OP_EMA_ACCUMULATE,
            OP_F32_TO_U8,
            OP_BACKGROUND_SUBTRACT,
            OP_AE_AWB,
            OP_BOX_BLUR,
            OP_AF_BLEND,
        ]

    def test_different_sizes(self):
        """Plan builds correctly for various frame sizes."""
        for w, h in [(16, 8), (100, 100), (1920, 1080)]:
            data = build_plan(w, h)
            plan = parse_plan(data)
            assert plan["width"] == w
            assert plan["height"] == h
            assert plan["num_stages"] == 7
            assert plan["num_buffers"] == 10


class TestPacking:
    """Verify arena packing correctness."""

    def test_packing_saves_memory(self):
        """Arena size should be less than the naive sum of all arena buffers."""
        data = build_plan(W, H)
        plan = parse_plan(data)

        arena_buffers = [b for b in plan["buffers"] if b["size_bytes"] > 0]
        naive_sum = sum(b["size_bytes"] for b in arena_buffers)

        assert plan["arena_size"] < naive_sum, (
            f"Arena {plan['arena_size']} should be < naive sum {naive_sum}"
        )

    def test_alignment(self):
        """All arena buffer offsets must be multiples of 16."""
        data = build_plan(W, H)
        plan = parse_plan(data)

        for buf in plan["buffers"]:
            if buf["size_bytes"] > 0:
                assert buf["offset"] % ALIGNMENT == 0, (
                    f"Offset {buf['offset']} not aligned to {ALIGNMENT}"
                )

    def test_arena_size_aligned(self):
        """Total arena size must be a multiple of 16."""
        data = build_plan(W, H)
        plan = parse_plan(data)
        assert plan["arena_size"] % ALIGNMENT == 0, f"Arena size {plan['arena_size']} not aligned"


class TestBinaryFormat:
    """Verify the wire format."""

    def test_magic_and_version(self):
        """First 6 bytes: 4-byte magic + 2-byte version."""
        data = build_plan(W, H)
        magic, version = struct.unpack("<IH", data[:6])
        assert magic == MAGIC
        assert version == VERSION


class TestBufferLifetime:
    """Verify lifetime-aware buffer reuse."""

    def test_buffer_lifetime_overlap(self):
        """frame_f32 (stages 0-1) and bg_u8 (stages 2-3) should share offset 0."""
        data = build_plan(W, H)
        plan = parse_plan(data)

        # Buffer indices: frame_f32=5, bg_u8=6
        frame_f32_buf = plan["buffers"][BUF_FRAME_F32]
        bg_u8_buf = plan["buffers"][BUF_BG_U8]

        assert frame_f32_buf["offset"] == 0, (
            f"frame_f32 offset should be 0, got {frame_f32_buf['offset']}"
        )
        assert bg_u8_buf["offset"] == 0, f"bg_u8 offset should be 0, got {bg_u8_buf['offset']}"


class TestAllocatorHelpers:
    """Unit tests for allocator utility functions."""

    def test_align_up_exact(self):
        assert align_up(16) == 16
        assert align_up(32) == 32

    def test_align_up_rounds(self):
        assert align_up(1) == 16
        assert align_up(17) == 32
        assert align_up(15) == 16

    def test_pack_empty(self):
        offsets, size = pack_buffers([])
        assert offsets == {}
        assert size == 0
