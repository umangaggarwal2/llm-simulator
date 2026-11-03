import random

import pytest

from llm_simulator.simulator.scheduler.kv_cache import BlockAllocator
from llm_simulator.simulator.scheduler.policy import Scheduler


def make_scheduler(
    num_blocks=1000, block_size=4, watermark=0.0, budget=32, max_sequences=8
):
    kv = BlockAllocator(num_blocks, block_size, watermark)
    return Scheduler(kv, budget, max_sequences), kv


def step(scheduler, now=0.0):
    plan = scheduler.schedule(now)
    return plan, scheduler.complete(plan)


def work_of(plan):
    """{sequence id: tokens} for the plan."""
    return {w.seq.id: w.tokens for w in plan.work}


def ids(seqs):
    return [seq.id for seq in seqs]


# --- Construction and add ---


@pytest.mark.parametrize("budget, max_sequences", [(8, 0), (7, 8)])
def test_rejects_invalid_limits(budget, max_sequences):
    with pytest.raises(ValueError):
        Scheduler(BlockAllocator(10, 4), budget, max_sequences)


def test_add_rejects_a_duplicate_id():
    scheduler, _ = make_scheduler()
    scheduler.add(1, 0.0, 10, 5)
    with pytest.raises(ValueError, match="already exists"):
        scheduler.add(1, 0.0, 10, 5)


def test_add_rejects_a_sequence_that_could_never_fit():
    scheduler, kv = make_scheduler(num_blocks=10, block_size=4)
    scheduler.add(1, 0.0, 30, 10)  # exactly 40 tokens
    with pytest.raises(ValueError, match="admits at most 40"):
        scheduler.add(2, 0.0, 30, 11)


def test_an_empty_prompt_counts_as_one_token():
    scheduler, _ = make_scheduler()
    assert scheduler.add(1, 0.0, 0, 3).prompt_tokens == 1


def test_has_work():
    scheduler, _ = make_scheduler()
    assert not scheduler.has_work()
    scheduler.add(1, 0.0, 4, 1)
    assert scheduler.has_work()
    step(scheduler)
    assert not scheduler.has_work()


# --- Admission ---


def test_admits_in_arrival_order_and_prefills_in_the_same_step():
    scheduler, kv = make_scheduler()
    for id in (1, 2, 3):
        scheduler.add(id, 0.0, 8, 5)
    plan, result = step(scheduler, now=1.5)
    assert ids(plan.admitted) == [1, 2, 3]
    assert work_of(plan) == {1: 8, 2: 8, 3: 8}
    assert all(w.prefill and w.cached == 0 for w in plan.work)
    assert ids(result.emitted) == [1, 2, 3]  # each prefill produced its first token
    assert all(seq.admitted_at == 1.5 for seq in plan.admitted)
    assert (plan.waiting, plan.running, plan.blocked) == (0, 3, None)
    assert kv.used_blocks == 6


def test_admission_stops_at_max_sequences():
    scheduler, _ = make_scheduler(max_sequences=2)
    for id in (1, 2, 3):
        scheduler.add(id, 0.0, 4, 5)
    plan = scheduler.schedule(0.0)
    assert ids(plan.admitted) == [1, 2]
    assert (plan.blocked, plan.waiting) == ("max_sequences", 1)


def test_admission_stops_when_the_budget_is_spent():
    scheduler, _ = make_scheduler(budget=16)
    for id in (1, 2):
        scheduler.add(id, 0.0, 16, 5)
    plan = scheduler.schedule(0.0)
    assert ids(plan.admitted) == [1]
    assert plan.blocked == "budget"


def test_a_head_that_does_not_fit_blocks_those_behind_it():
    scheduler, kv = make_scheduler(num_blocks=10, block_size=4)
    scheduler.add(1, 0.0, 24, 4)  # 6 blocks
    scheduler.add(2, 0.0, 20, 4)  # 5 blocks: does not fit beside 1
    scheduler.add(3, 0.0, 4, 4)  # 1 block: would fit, but waits behind 2
    plan = scheduler.schedule(0.0)
    assert ids(plan.admitted) == [1]
    assert (plan.blocked, plan.waiting) == ("kv", 2)


def test_admission_leaves_the_watermark_free():
    # 100 blocks with a 10% watermark: a new sequence may take only 90.
    scheduler, kv = make_scheduler(
        num_blocks=100, block_size=4, watermark=0.1, budget=400
    )
    scheduler.add(1, 0.0, 320, 1)  # 80 blocks
    scheduler.add(2, 0.0, 44, 1)  # 11 blocks: would leave 9, below the watermark
    plan = scheduler.schedule(0.0)
    assert ids(plan.admitted) == [1]
    assert plan.blocked == "kv"


def test_admission_reserves_the_whole_prompt_at_once():
    scheduler, kv = make_scheduler(budget=8, block_size=4)
    scheduler.add(1, 0.0, 40, 1)
    plan = scheduler.schedule(0.0)
    assert work_of(plan) == {1: 8}  # only the first chunk is computed
    assert kv.held(1) == 10  # but every block of the prompt is held


def test_queue_time_counts_the_first_admission_only():
    scheduler, _ = make_scheduler(num_blocks=6, block_size=4)
    for id in (1, 2, 3):
        scheduler.add(id, 0.0, 8, 10)
    step(scheduler, now=1.0)
    plan, _ = step(scheduler, now=2.0)
    [preemption] = plan.preempted
    victim = preemption.seq
    now = 2.0
    while not victim.computed:  # until it is readmitted and recomputed
        now += 1.0
        step(scheduler, now)
    assert victim.admitted_at == 1.0


# --- Chunked prefill ---


def test_a_prompt_longer_than_the_budget_is_split_and_only_the_last_chunk_emits():
    scheduler, _ = make_scheduler(budget=8)
    scheduler.add(1, 0.0, 20, 3)
    chunks, emitted = [], []
    for _ in range(3):
        plan, result = step(scheduler)
        chunks.append([(w.tokens, w.cached) for w in plan.work])
        emitted.append(ids(result.emitted))
    assert chunks == [[(8, 0)], [(8, 8)], [(4, 16)]]
    assert emitted == [[], [], [1]]


def test_decodes_go_first_and_prefill_takes_what_is_left():
    scheduler, _ = make_scheduler(budget=16)
    for id in (1, 2, 3):
        scheduler.add(id, 0.0, 4, 50)
    step(scheduler)  # 1-3 prefill and are now decoding
    scheduler.add(4, 0.0, 100, 1)
    for expected_chunk in (13, 13, 13):
        plan, _ = step(scheduler)
        assert work_of(plan) == {1: 1, 2: 1, 3: 1, 4: expected_chunk}
        assert (plan.decode_tokens, plan.prefill_tokens) == (3, expected_chunk)


def test_a_new_request_takes_what_a_finishing_prefill_leaves():
    scheduler, _ = make_scheduler(budget=16)
    scheduler.add(1, 0.0, 20, 1)
    step(scheduler)  # 16 of 20
    scheduler.add(2, 0.0, 30, 1)
    plan, _ = step(scheduler)
    assert work_of(plan) == {1: 4, 2: 12}


def test_shape_is_what_the_cost_model_takes():
    scheduler, _ = make_scheduler(budget=8)
    scheduler.add(1, 0.0, 12, 2)
    first, _ = step(scheduler)
    second, _ = step(scheduler)
    assert first.shape == ((8, 0),)
    assert second.shape == ((4, 8),)


# --- Growth and preemption ---


def test_a_decode_takes_a_new_block_at_the_boundary():
    scheduler, kv = make_scheduler(block_size=4)
    scheduler.add(1, 0.0, 4, 10)
    step(scheduler)  # 4 computed, 1 generated
    assert kv.held(1) == 1
    step(scheduler)  # decode computes token 5
    assert kv.held(1) == 2


def preemption_setup():
    # 6 blocks of 4 tokens, all taken by three 8-token prompts.
    scheduler, kv = make_scheduler(num_blocks=6, block_size=4, budget=32)
    for id in (1, 2, 3):
        scheduler.add(id, 0.0, 8, 10)
    step(scheduler)
    assert kv.free_blocks == 0
    return scheduler, kv


def test_the_newest_sequence_is_preempted_when_a_decode_needs_a_block():
    scheduler, kv = preemption_setup()
    plan, result = step(scheduler)
    [preemption] = plan.preempted
    assert preemption.seq.id == 3
    assert preemption.discarded == 8
    assert preemption.seq.computed == 0 and preemption.seq.preemptions == 1
    assert work_of(plan) == {1: 1, 2: 1}
    assert ids(result.emitted) == [1, 2]
    assert (plan.waiting, plan.running) == (1, 2)
    assert kv.held(3) == 0


def test_no_admission_in_a_step_that_preempted():
    scheduler, _ = preemption_setup()
    scheduler.add(4, 0.0, 1, 1)
    plan = scheduler.schedule(0.0)
    assert plan.preempted and not plan.admitted
    assert plan.blocked == "preemption"


def test_preempted_sequences_go_to_the_front_of_the_queue():
    scheduler, _ = preemption_setup()
    scheduler.add(4, 0.0, 1, 1)
    scheduler.complete(scheduler.schedule(0.0))
    assert ids(scheduler._waiting) == [3, 4]


def test_a_preempted_sequence_recomputes_what_it_knew_and_continues():
    scheduler, _ = preemption_setup()
    step(scheduler)  # 3 preempted after generating 1 token
    victim = None
    for _ in range(100):
        plan, result = step(scheduler)
        victim = next((w for w in plan.work if w.seq.id == 3), None)
        if victim:
            break
    # Its 8-token prompt and 1 generated token, from scratch, as one prefill.
    assert (victim.tokens, victim.cached, victim.prefill) == (9, 0, True)
    assert victim.seq.generated == 2  # the next token, not a repeat


def test_a_decode_that_is_itself_the_newest_preempts_itself():
    scheduler, _ = make_scheduler(num_blocks=2, block_size=4, budget=16)
    scheduler.add(1, 0.0, 2, 2)  # its next token still fits its block
    scheduler.add(2, 0.0, 4, 3)  # its next token needs a second block
    step(scheduler)  # the cache is full
    plan, _ = step(scheduler)
    assert [p.seq.id for p in plan.preempted] == [2]
    assert work_of(plan) == {1: 1}
    assert ids(scheduler._waiting) == [2]


def test_preemptions_free_enough_for_every_older_decode():
    # Four sequences fill the cache exactly; when all need a block, the two
    # newest are preempted so the two oldest can grow.
    scheduler, kv = make_scheduler(num_blocks=4, block_size=4, budget=16)
    for id in (1, 2, 3, 4):
        scheduler.add(id, 0.0, 4, 10)
    step(scheduler)
    plan, _ = step(scheduler)
    assert [p.seq.id for p in plan.preempted] == [4, 3]
    assert work_of(plan) == {1: 1, 2: 1}
    assert ids(scheduler._waiting) == [3, 4]  # the older victim first


# --- Completion ---


def test_finished_sequences_free_their_blocks():
    scheduler, kv = make_scheduler()
    scheduler.add(1, 0.0, 8, 2)
    scheduler.add(2, 0.0, 8, 5)
    step(scheduler)
    _, result = step(scheduler)
    assert ids(result.finished) == [1]
    assert kv.held(1) == 0
    assert scheduler.running == 1


def test_an_empty_output_finishes_after_prefill_without_emitting():
    scheduler, kv = make_scheduler()
    scheduler.add(1, 0.0, 8, 0)
    _, result = step(scheduler)
    assert (ids(result.emitted), ids(result.finished)) == ([], [1])
    assert kv.used_blocks == 0


def test_every_token_is_emitted_once_and_the_cache_ends_empty():
    scheduler, kv = make_scheduler(budget=8)
    outputs = {1: 5, 2: 1, 3: 12}
    for id, output in outputs.items():
        scheduler.add(id, 0.0, 10, output)
    emitted = {id: 0 for id in outputs}
    while scheduler.has_work():
        _, result = step(scheduler)
        for seq in result.emitted:
            emitted[seq.id] += 1
    assert emitted == outputs
    assert kv.used_blocks == 0


# --- Abort ---


def test_abort_while_waiting():
    scheduler, _ = make_scheduler(max_sequences=1)
    scheduler.add(1, 0.0, 4, 5)
    scheduler.add(2, 0.0, 4, 5)
    scheduler.abort(2)
    plan, _ = step(scheduler)
    assert (plan.waiting, plan.blocked) == (0, None)


def test_abort_while_running_frees_its_blocks():
    scheduler, kv = make_scheduler()
    scheduler.add(1, 0.0, 8, 5)
    step(scheduler)
    scheduler.abort(1)
    assert kv.used_blocks == 0
    assert not scheduler.has_work()


def test_abort_during_a_step_ignores_its_work():
    scheduler, kv = make_scheduler()
    scheduler.add(1, 0.0, 8, 5)
    scheduler.add(2, 0.0, 8, 5)
    plan = scheduler.schedule(0.0)
    scheduler.abort(1)
    result = scheduler.complete(plan)
    assert ids(result.emitted) == [2]
    assert kv.held(1) == 0


def test_abort_of_an_unknown_or_finished_sequence_is_a_no_op():
    scheduler, _ = make_scheduler()
    scheduler.add(1, 0.0, 4, 1)
    step(scheduler)
    scheduler.abort(1)
    scheduler.abort(99)


# --- Invariants under random load ---


@pytest.mark.parametrize("seed", range(5))
def test_random_load_keeps_every_invariant(seed):
    rng = random.Random(seed)
    budget, max_sequences = 24, 6
    scheduler, kv = make_scheduler(
        num_blocks=40,
        block_size=4,
        watermark=0.05,
        budget=budget,
        max_sequences=max_sequences,
    )
    outputs, emitted, aborted = {}, {}, set()
    next_id = 0
    preemptions = 0

    for _ in range(5000):
        if next_id < 300 and rng.random() < 0.4:
            prompt = rng.randint(0, 60)
            output = rng.randint(0, min(40, kv.admissible_tokens - max(1, prompt)))
            scheduler.add(next_id, 0.0, prompt, output)
            outputs[next_id], emitted[next_id] = output, 0
            next_id += 1
        alive = [
            id for id in outputs if id not in aborted and id in scheduler._sequences
        ]
        if alive and rng.random() < 0.03:
            victim = rng.choice(alive)
            scheduler.abort(victim)
            aborted.add(victim)
        if not scheduler.has_work():
            if next_id >= 300:
                break
            continue

        plan = scheduler.schedule(0.0)
        if alive and rng.random() < 0.02:  # a client leaves mid-step
            victim = rng.choice([w.seq.id for w in plan.work] or alive)
            scheduler.abort(victim)
            aborted.add(victim)
        result = scheduler.complete(plan)
        preemptions += len(plan.preempted)

        assert sum(w.tokens for w in plan.work) <= budget
        assert len({w.seq.id for w in plan.work}) == len(plan.work)
        assert scheduler.running <= max_sequences
        running = {seq.id for seq in scheduler._running}
        waiting = {seq.id for seq in scheduler._waiting}
        assert not running & waiting
        assert running | waiting == set(scheduler._sequences)
        assert all(kv.held(id) == 0 for id in waiting)
        assert sum(kv.held(id) for id in running) == kv.used_blocks
        for seq in result.emitted:
            emitted[seq.id] += 1
            assert emitted[seq.id] == seq.generated <= outputs[seq.id]

    assert not scheduler.has_work()
    assert kv.used_blocks == 0
    assert preemptions > 0  # the cache is small enough to exercise preemption
    for id, output in outputs.items():
        if id not in aborted:
            assert emitted[id] == output
