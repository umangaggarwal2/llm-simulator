import asyncio
from contextlib import aclosing

import pytest

from llm_simulator.simulator import EngineFailed, SchedulerEngine
from llm_simulator.simulator.scheduler.cost_model import RooflineCostModel
from llm_simulator.simulator.scheduler.kv_cache import BlockAllocator
from llm_simulator.simulator.scheduler.policy import Scheduler
from llm_simulator.types import GenerationPlan

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FakeClock:
    """A clock that only moves when the engine sleeps.

    Unlike FixedRateEngine's tests, sleeping also yields to the event loop:
    the step loop and its consumers are separate tasks. It yields before
    advancing, as a real sleep lets other tasks run before time has passed.
    """

    def __init__(self, now=100.0):
        self.now = now
        self.sleeps = 0

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.sleeps += 1
        await asyncio.sleep(0)
        self.now += seconds


# Round numbers: 1 ms per step for the weights, 1 us per KV token, and
# 10 us of compute per token, so prefill of 1,000 tokens takes 10 ms.
COST = RooflineCostModel(
    parameters=5e6,
    weight_bytes=1e6,
    kv_bytes_per_token=1000,
    peak_flops=1e12,
    memory_bandwidth=1e9,
)


def make_engine(clock, num_blocks=1000, budget=4096, max_sequences=8, cost=COST):
    policy = Scheduler(BlockAllocator(num_blocks, 16, 0.0), budget, max_sequences)
    return SchedulerEngine(policy, cost, clock=clock, sleep=clock.sleep)


def make_plan(n, prompt_tokens=100):
    return GenerationPlan(
        prompt_tokens=prompt_tokens,
        tokens=tuple(f"t{i}" for i in range(n)),
        finish_reason="stop",
        entry_id="e0",
        variant=0,
    )


async def collect(engine, plan, arrival):
    return [event async for event in engine.run(plan, arrival)]


def step(*work):
    return COST.step_cost(work).duration


@pytest.fixture
async def started():
    engines = []

    async def start(engine):
        await engine.start()
        engines.append(engine)
        return engine

    yield start
    for engine in engines:
        await engine.stop()


# --- Timeline ---


async def test_one_request_follows_the_cost_model_exactly(started):
    clock = FakeClock()
    engine = await started(make_engine(clock))
    events = await collect(engine, make_plan(3, prompt_tokens=100), clock.now)

    prefill = step((100, 0))
    first, second = step((1, 100)), step((1, 101))
    assert [e.text for e in events] == ["t0", "t1", "t2"]
    assert [e.index for e in events] == [0, 1, 2]
    assert [e.sim_time for e in events] == pytest.approx(
        [prefill, prefill + first, prefill + first + second]
    )
    assert all(e.lag == pytest.approx(0, abs=1e-12) for e in events)


async def test_queue_time_is_on_the_first_token_only(started):
    clock = FakeClock()
    engine = await started(make_engine(clock))
    events = await collect(engine, make_plan(3), clock.now)
    assert events[0].queue_time == 0
    assert [e.queue_time for e in events[1:]] == [None, None]


async def test_an_idle_engine_starts_at_the_arrival_not_when_run_is_called(started):
    # Time spent before run() (matching, tokenizing) is absorbed, as in
    # FixedRateEngine: the first token is still due one prefill after arrival.
    clock = FakeClock()
    engine = await started(make_engine(clock))
    arrival = clock.now
    clock.now += 0.002
    events = await collect(engine, make_plan(1), arrival)
    assert events[0].sim_time == pytest.approx(step((100, 0)))
    assert events[0].queue_time == 0


async def test_concurrent_requests_share_steps(started):
    clock = FakeClock()
    engine = await started(make_engine(clock))
    a, b = await asyncio.gather(
        collect(engine, make_plan(2), clock.now),
        collect(engine, make_plan(2), clock.now),
    )
    shared_prefill = step((100, 0), (100, 0))
    assert a[0].sim_time == b[0].sim_time == pytest.approx(shared_prefill)
    assert shared_prefill > step((100, 0))
    assert a[1].sim_time == b[1].sim_time


async def test_a_full_engine_queues_the_next_request(started):
    clock = FakeClock()
    engine = await started(make_engine(clock, max_sequences=1))
    first, second = await asyncio.gather(
        collect(engine, make_plan(3), clock.now),
        collect(engine, make_plan(2), clock.now),
    )
    # The second waits for the first to finish, then has the GPU to itself.
    assert second[0].queue_time == pytest.approx(first[-1].sim_time)
    assert second[0].sim_time == pytest.approx(first[-1].sim_time + step((100, 0)))


async def test_a_long_prompt_slows_every_decoding_stream(started):
    clock = FakeClock()
    engine = await started(make_engine(clock, budget=1000))

    async def late_long_prompt():
        await asyncio.sleep(0)
        while not engine._policy.running:
            await asyncio.sleep(0)
        return await collect(engine, make_plan(1, prompt_tokens=3000), clock.now)

    decoding, _ = await asyncio.gather(
        collect(engine, make_plan(8, prompt_tokens=10), clock.now),
        late_long_prompt(),
    )
    gaps = [b.sim_time - a.sim_time for a, b in zip(decoding, decoding[1:])]
    # A step that also carries a 999-token prefill chunk takes about 10 ms;
    # a decode alone about 1 ms.
    assert max(gaps) > 9 * min(gaps)


async def test_an_empty_plan_finishes_after_prefill(started):
    clock = FakeClock()
    engine = await started(make_engine(clock))
    start = clock.now
    assert await collect(engine, make_plan(0), start) == []
    assert clock.now - start == pytest.approx(step((100, 0)))
    assert not engine._policy.has_work()


# --- Disconnects ---


async def test_a_disconnect_frees_the_sequence_and_others_continue(started):
    clock = FakeClock()
    engine = await started(make_engine(clock))
    kv = engine._policy._kv

    async def leave_after_one_token():
        async with aclosing(engine.run(make_plan(50), clock.now)) as events:
            async for _ in events:
                break

    _, other = await asyncio.gather(
        leave_after_one_token(), collect(engine, make_plan(5), clock.now)
    )
    assert len(other) == 5
    assert kv.used_blocks == 0
    assert not engine._streams


async def test_a_disconnect_before_admission_is_never_scheduled(started):
    clock = FakeClock()
    engine = await started(make_engine(clock))
    events = engine.run(make_plan(5), clock.now)
    task = asyncio.ensure_future(events.__anext__())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await events.aclose()
    await collect(engine, make_plan(1), clock.now)
    assert not engine._policy.has_work() and not engine._arrivals


# --- A loop that falls behind real time ---


async def test_a_late_loop_keeps_the_simulated_timeline_and_reports_lag(started):
    clock = FakeClock()
    engine = await started(make_engine(clock))
    events = []
    async with aclosing(engine.run(make_plan(10), clock.now)) as stream:
        async for event in stream:
            events.append(event)
            if event.index == 2:
                clock.now += 0.5  # the process stalls for half a second

    decode = [b.sim_time - a.sim_time for a, b in zip(events, events[1:])]
    expected = [step((1, 100 + i)) for i in range(9)]
    assert decode == pytest.approx(expected)  # not shifted by the stall
    assert events[3].lag > 0.49
    assert events[-1].lag < events[3].lag  # catching up


async def test_a_loop_catching_up_yields_after_every_step(started):
    clock = FakeClock()
    engine = await started(make_engine(clock))
    still_running = []
    async with aclosing(engine.run(make_plan(50), clock.now)) as stream:
        async for event in stream:
            if event.index == 0:
                clock.now += 10.0  # every remaining step is now overdue
            elif event.index == 1:
                # Without the yield, the loop would have run all 49 overdue
                # steps before this consumer got to run again.
                still_running.append(engine._policy.has_work())
    assert still_running == [True]


async def test_a_request_never_joins_a_step_that_started_before_it_arrived(started):
    clock = FakeClock()
    engine = await started(make_engine(clock))
    late = {}

    async def first():
        # 200 decode steps of about 1.2 ms: still running when the second
        # request arrives, 50 ms after the stall began.
        async with aclosing(engine.run(make_plan(200), clock.now)) as stream:
            async for event in stream:
                if event.index == 2:
                    clock.now += 0.05
                    late["second"] = asyncio.ensure_future(
                        collect(engine, make_plan(2), clock.now)
                    )

    await first()
    second = await late["second"]
    # Handed to the policy at once, it would join the step the loop was about
    # to schedule, 50 ms before it arrived: a queue time of about -0.05.
    assert 0 <= second[0].queue_time < 0.002


# --- Lifecycle and failure ---


async def test_run_before_start_is_an_error():
    engine = make_engine(FakeClock())
    with pytest.raises(RuntimeError, match="start"):
        await collect(engine, make_plan(1), 0.0)


async def test_a_failing_step_loop_fails_open_and_later_requests():
    clock = FakeClock()
    engine = make_engine(clock)

    def broken(now):
        raise ValueError("bug")

    engine._policy.schedule = broken
    await engine.start()
    try:
        with pytest.raises(EngineFailed) as error:
            await collect(engine, make_plan(3), clock.now)
        assert isinstance(error.value.__cause__, ValueError)
        with pytest.raises(EngineFailed):
            await collect(engine, make_plan(3), clock.now)
    finally:
        await engine.stop()


async def test_stop_fails_open_streams():
    clock = FakeClock()
    engine = make_engine(clock)
    await engine.start()
    events = engine.run(make_plan(100), clock.now)
    await events.__anext__()
    await engine.stop()
    with pytest.raises(EngineFailed):
        async for _ in events:
            pass


async def test_stop_without_start_is_a_no_op():
    await make_engine(FakeClock()).stop()
