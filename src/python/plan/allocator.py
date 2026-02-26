"""Buffer lifetime analysis and arena packing for the execution plan."""

from dataclasses import dataclass

ALIGNMENT = 16  # NEON requires 16-byte aligned buffers


@dataclass
class BufferDef:
    """A buffer in the execution plan."""

    name: str
    size_bytes: int
    dtype: str  # "uint8" or "float32"
    external: bool  # True = not arena-allocated (frame_u8, background, mask, display, alpha)
    first_stage: int  # First stage that uses this buffer (-1 for external)
    last_stage: int  # Last stage that uses this buffer (-1 for external)


def align_up(size: int, alignment: int = ALIGNMENT) -> int:
    """Round up to next multiple of alignment."""
    return (size + alignment - 1) & ~(alignment - 1)


def pack_buffers(buffers: list[BufferDef]) -> tuple[dict[str, int], int]:
    """
    Assign arena offsets to non-external buffers using greedy interval packing.

    Buffers with non-overlapping lifetimes can share the same memory.
    All offsets are 16-byte aligned.

    Returns (name -> offset, total_arena_bytes).
    """
    # Filter to non-external buffers only
    arena_bufs = [b for b in buffers if not b.external]

    # Sort by first_stage ascending, then by size_bytes descending (better packing)
    arena_bufs.sort(key=lambda b: (b.first_stage, -b.size_bytes))

    # Placed buffers: (offset, aligned_size, first_stage, last_stage)
    placed: list[tuple[int, int, int, int]] = []
    offsets: dict[str, int] = {}
    arena_size = 0

    for buf in arena_bufs:
        aligned_size = align_up(buf.size_bytes)
        # Find the lowest offset where this buffer doesn't conflict
        # with any already-placed buffer that has an overlapping lifetime
        candidate = 0

        while True:
            conflict = False
            for p_offset, p_size, p_first, p_last in placed:
                # Check lifetime overlap
                lifetimes_overlap = buf.first_stage <= p_last and buf.last_stage >= p_first
                if not lifetimes_overlap:
                    continue
                # Check memory overlap
                mem_overlap = candidate < p_offset + p_size and candidate + aligned_size > p_offset
                if mem_overlap:
                    conflict = True
                    # Move candidate past this placed buffer (aligned)
                    candidate = align_up(p_offset + p_size)
                    break
            if not conflict:
                break

        offsets[buf.name] = candidate
        placed.append((candidate, aligned_size, buf.first_stage, buf.last_stage))
        end = candidate + aligned_size
        if end > arena_size:
            arena_size = end

    arena_size = align_up(arena_size)
    return offsets, arena_size
