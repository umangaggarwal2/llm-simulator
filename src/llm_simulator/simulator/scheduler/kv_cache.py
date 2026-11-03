"""The paged KV cache: fixed-size blocks of token slots, held per sequence."""

from typing import Dict


def kv_capacity_blocks(
    hbm_bytes: float,
    gpu_memory_utilization: float,
    weight_bytes: float,
    kv_bytes_per_token: int,
    block_size: int,
) -> int:
    """Blocks that fit in the memory left after the weights.

    `gpu_memory_utilization` is the fraction of HBM the engine may use, as
    in vLLM; the rest is left for activations and the CUDA context.
    """
    if not 0 < gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    if kv_bytes_per_token < 1 or block_size < 1:
        raise ValueError("kv_bytes_per_token and block_size must be at least 1")
    free = hbm_bytes * gpu_memory_utilization - weight_bytes
    blocks = int(free // (kv_bytes_per_token * block_size))
    if blocks < 1:
        raise ValueError(
            f"no room for the KV cache: {hbm_bytes * gpu_memory_utilization:.3g} "
            f"usable bytes, {weight_bytes:.3g} taken by the weights"
        )
    return blocks


class BlockAllocator:
    """Counts the blocks each sequence holds.

    Blocks are interchangeable, so only counts are kept, not block ids; that
    changes with prefix caching, where a block's content matters.

    The watermark is a share of all blocks that admission must leave free.
    Running sequences may grow into it, so a sequence admitted into a nearly
    full cache does not force a preemption a few tokens later (vLLM's block
    manager, with its default of 1%).
    """

    def __init__(self, num_blocks: int, block_size: int, watermark: float = 0.01):
        if num_blocks < 1 or block_size < 1:
            raise ValueError("num_blocks and block_size must be at least 1")
        if not 0 <= watermark < 1:
            raise ValueError("watermark must be in [0, 1)")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.watermark_blocks = int(num_blocks * watermark)
        self._free = num_blocks
        self._held: Dict[int, int] = {}

    @property
    def free_blocks(self) -> int:
        return self._free

    @property
    def used_blocks(self) -> int:
        return self.num_blocks - self._free

    @property
    def usage(self) -> float:
        """The share of blocks in use, from 0 to 1."""
        return self.used_blocks / self.num_blocks

    @property
    def admissible_tokens(self) -> int:
        """The largest prompt that an empty cache can admit.

        A model whose context length exceeds it could accept prompts that
        never fit, so the engine refuses to start, as vLLM does.
        """
        return (self.num_blocks - self.watermark_blocks) * self.block_size

    def blocks_for(self, tokens: int) -> int:
        return -(-tokens // self.block_size)

    def held(self, seq: int) -> int:
        return self._held.get(seq, 0)

    def can_admit(self, tokens: int) -> bool:
        """Whether a new sequence of `tokens` fits and leaves the watermark free."""
        return self._free - self.blocks_for(tokens) >= self.watermark_blocks

    def grow(self, seq: int, tokens: int) -> bool:
        """Make `seq` hold enough blocks for `tokens`; False if they are not free.

        Growth may use the watermark. On False nothing changes.
        """
        held = self.held(seq)
        extra = self.blocks_for(tokens) - held
        if extra > self._free:
            return False
        if extra > 0:
            self._free -= extra
            self._held[seq] = held + extra
        return True

    def free(self, seq: int) -> None:
        """Release every block `seq` holds; a no-op if it holds none."""
        self._free += self._held.pop(seq, 0)
