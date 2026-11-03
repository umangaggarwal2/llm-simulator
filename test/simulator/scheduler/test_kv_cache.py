import pytest

from llm_simulator.simulator.scheduler.kv_cache import (
    BlockAllocator,
    kv_capacity_blocks,
)

GB = 1e9
KV_PER_TOKEN = 524288  # 512 KiB: a 7B model with 32 layers and 4096 hidden, fp16


# --- Capacity ---


def test_capacity_of_the_documented_example():
    # 80 GB x 0.9 - 13.5 GB of weights = 58.5 GB, at 8 MiB per 16-token block.
    blocks = kv_capacity_blocks(80 * GB, 0.9, 13.5 * GB, KV_PER_TOKEN, 16)
    assert blocks == 6973
    assert blocks * 16 == 111_568  # about 111,600 tokens


def test_capacity_rounds_down_to_whole_blocks():
    assert kv_capacity_blocks(100, 1.0, 0, 1, 16) == 6


def test_weights_larger_than_memory_leave_no_cache():
    with pytest.raises(ValueError, match="no room"):
        kv_capacity_blocks(80 * GB, 0.9, 75 * GB, KV_PER_TOKEN, 16)


@pytest.mark.parametrize(
    "utilization, kv_bytes, block_size",
    [(0, 1, 16), (1.1, 1, 16), (0.9, 0, 16), (0.9, 1, 0)],
)
def test_capacity_rejects_invalid_parameters(utilization, kv_bytes, block_size):
    with pytest.raises(ValueError):
        kv_capacity_blocks(80 * GB, utilization, 0, kv_bytes, block_size)


# --- Allocation ---


def make_allocator(num_blocks=100, block_size=16, watermark=0.0):
    return BlockAllocator(num_blocks, block_size, watermark)


@pytest.mark.parametrize("tokens, blocks", [(1, 1), (16, 1), (17, 2), (32, 2), (33, 3)])
def test_blocks_round_up(tokens, blocks):
    assert make_allocator().blocks_for(tokens) == blocks


def test_grow_takes_only_the_extra_blocks():
    kv = make_allocator()
    assert kv.grow(1, 20)  # 2 blocks
    assert kv.grow(1, 32)  # still 2
    assert kv.grow(1, 33)  # 3
    assert (kv.held(1), kv.free_blocks, kv.used_blocks) == (3, 97, 3)


def test_grow_never_shrinks():
    kv = make_allocator()
    kv.grow(1, 100)
    assert kv.grow(1, 10)
    assert kv.held(1) == 7


def test_grow_fails_without_changing_anything():
    kv = make_allocator(num_blocks=4)
    kv.grow(1, 48)  # 3 blocks
    assert not kv.grow(2, 32)  # needs 2, 1 free
    assert (kv.held(2), kv.free_blocks) == (0, 1)


def test_free_returns_every_block():
    kv = make_allocator()
    kv.grow(1, 100)
    kv.grow(2, 50)
    kv.free(1)
    assert (kv.held(1), kv.free_blocks) == (0, 96)
    kv.free(1)  # freeing twice, or a sequence that holds nothing, is harmless
    assert kv.free_blocks == 96


def test_usage():
    kv = make_allocator(num_blocks=8)
    kv.grow(1, 32)
    assert kv.usage == 0.25


# --- Watermark ---


def test_watermark_is_a_share_of_all_blocks_rounded_down():
    assert BlockAllocator(6973, 16, 0.01).watermark_blocks == 69
    assert BlockAllocator(99, 16, 0.01).watermark_blocks == 0


def test_admission_must_leave_the_watermark_free():
    kv = make_allocator(num_blocks=100, watermark=0.05)  # 5 blocks reserved
    assert kv.can_admit(95 * 16)
    assert not kv.can_admit(95 * 16 + 1)


def test_running_sequences_may_grow_into_the_watermark():
    kv = make_allocator(num_blocks=100, watermark=0.05)
    kv.grow(1, 95 * 16)
    assert not kv.can_admit(1)
    assert kv.grow(1, 100 * 16)  # all 100 blocks
    assert kv.free_blocks == 0


def test_admissible_tokens_excludes_the_watermark():
    assert make_allocator(num_blocks=100, watermark=0.05).admissible_tokens == 95 * 16
    assert BlockAllocator(6973, 16, 0.01).admissible_tokens == 110_464


@pytest.mark.parametrize(
    "num_blocks, block_size, watermark",
    [(0, 16, 0), (1, 0, 0), (1, 16, -0.1), (1, 16, 1)],
)
def test_allocator_rejects_invalid_parameters(num_blocks, block_size, watermark):
    with pytest.raises(ValueError):
        BlockAllocator(num_blocks, block_size, watermark)
