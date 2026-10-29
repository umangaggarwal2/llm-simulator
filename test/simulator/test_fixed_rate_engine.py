import time

import pytest

from llm_simulator.simulator import FixedRateEngine
from llm_simulator.types import GenerationPlan

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FakeClock:
    """A clock that only moves when the engine sleeps, so timing tests are exact."""

    def __init__(self, now=100.0):
        self.now = now
        self.sleeps = []

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def make_plan(n, prompt_tokens=1000):
    return GenerationPlan(
        prompt_tokens=prompt_tokens,
        tokens=tuple(f"t{i}" for i in range(n)),
        finish_reason="stop",
        entry_id="e0",
        variant=0,
    )


def make_engine(clock=None, prefill=100_000, decode=250, overhead=0.002):
    clock = clock or FakeClock()
    return FixedRateEngine(prefill, decode, overhead, clock=clock, sleep=clock.sleep)


# --- Timeline arithmetic ---


def test_ttft_is_overhead_plus_prefill():
    # 2 ms overhead + 1000 tokens at 100k tok/s = 12 ms; no extra decode step.
    assert make_engine().ttft(1000) == pytest.approx(0.012)


def test_timeline_has_constant_itl_after_ttft():
    timeline = make_engine().timeline(make_plan(4))
    assert timeline == pytest.approx((0.012, 0.016, 0.020, 0.024))


def test_architecture_doc_example():
    # 1000 prompt tokens, 200 output tokens: TTFT 12 ms, E2E 12 + 199 x 4 = 808 ms.
    timeline = make_engine().timeline(make_plan(200))
    assert timeline[0] == pytest.approx(0.012)
    assert timeline[-1] == pytest.approx(0.808)


def test_longer_prompt_means_longer_ttft():
    engine = make_engine()
    assert engine.ttft(10_000) - engine.ttft(1_000) == pytest.approx(0.09)


def test_no_overhead_by_default():
    engine = FixedRateEngine(1000, 100)
    assert engine.ttft(50) == pytest.approx(0.05)


def test_empty_plan_has_empty_timeline():
    assert make_engine().timeline(make_plan(0)) == ()


@pytest.mark.parametrize(
    "args",
    [(0, 250), (100, 0), (-1, 250), (100, 250, -0.001)],
)
def test_invalid_parameters(args):
    with pytest.raises(ValueError):
        FixedRateEngine(*args)


# --- Pacing ---


async def collect_with_release_times(engine, clock, plan, arrival):
    released = []
    async for event in engine.run(plan, arrival):
        released.append((event, clock.now))
    return released


async def test_each_token_is_released_at_its_deadline():
    clock = FakeClock(now=100.0)
    engine = make_engine(clock)
    released = await collect_with_release_times(engine, clock, make_plan(5), 100.0)

    for event, release_time in released:
        assert release_time == pytest.approx(100.0 + event.sim_time)
        assert event.lag == 0.0
    assert [e.index for e, _ in released] == [0, 1, 2, 3, 4]
    assert [e.text for e, _ in released] == ["t0", "t1", "t2", "t3", "t4"]


async def test_overhead_before_the_engine_is_absorbed_by_ttft():
    # The request arrived 5 ms before run() is called (parsing, matching...).
    clock = FakeClock(now=100.005)
    engine = make_engine(clock)
    released = await collect_with_release_times(engine, clock, make_plan(2), 100.0)

    first, release_time = released[0]
    assert release_time == pytest.approx(100.012)  # still arrival + TTFT
    assert clock.sleeps[0] == pytest.approx(0.007)  # only the remainder was slept
    assert first.lag == 0.0


async def test_overhead_longer_than_ttft_releases_immediately_with_lag():
    clock = FakeClock(now=100.030)  # 30 ms of overhead against a 12 ms TTFT
    engine = make_engine(clock)
    released = await collect_with_release_times(engine, clock, make_plan(6), 100.0)

    lags = [event.lag for event, _ in released]
    assert lags[0] == pytest.approx(0.018)
    # Tokens due before 30 ms are released at once; later ones are back on time.
    assert lags[4] == pytest.approx(0.002)
    assert lags[5] == 0.0
    assert released[-1][1] == pytest.approx(100.0 + released[-1][0].sim_time)


async def test_late_wake_does_not_delay_later_tokens():
    clock = FakeClock(now=100.0)

    async def oversleeping(seconds):
        clock.now += seconds + 0.001  # every wake-up is 1 ms late

    engine = FixedRateEngine(100_000, 250, 0.002, clock=clock, sleep=oversleeping)
    events = [event async for event in engine.run(make_plan(100), 100.0)]

    assert all(event.lag == pytest.approx(0.001) for event in events)
    # E2E is off by one wake-up, not by 100 of them.
    assert clock.now == pytest.approx(100.0 + events[-1].sim_time + 0.001)


async def test_slow_consumer_gets_overdue_tokens_at_once():
    clock = FakeClock(now=100.0)
    engine = make_engine(clock)
    stream = engine.run(make_plan(5), 100.0)

    first = await stream.__anext__()
    clock.now += 0.010  # the client stalls for 10 ms (2.5 ITLs)
    second = await stream.__anext__()
    third = await stream.__anext__()
    fourth = await stream.__anext__()
    await stream.aclose()

    assert first.lag == 0.0
    assert second.lag == pytest.approx(0.006)  # was due 4 ms after the first
    assert third.lag == pytest.approx(0.002)
    assert fourth.lag == 0.0  # back on schedule


async def test_empty_plan_still_waits_for_ttft():
    clock = FakeClock(now=100.0)
    engine = make_engine(clock)
    events = [event async for event in engine.run(make_plan(0), 100.0)]
    assert events == []
    assert clock.now == pytest.approx(100.012)


async def test_closing_early_stops_the_stream():
    clock = FakeClock(now=100.0)
    engine = make_engine(clock)
    stream = engine.run(make_plan(10), 100.0)
    await stream.__anext__()
    await stream.__anext__()
    await stream.aclose()
    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()
    assert clock.now == pytest.approx(100.016)  # never slept past token 1


async def test_concurrent_runs_keep_their_own_deadlines():
    clock = FakeClock(now=100.0)
    engine = make_engine(clock)
    small = engine.timeline(make_plan(3, prompt_tokens=100))
    large = engine.timeline(make_plan(3, prompt_tokens=5000))
    assert small[0] < large[0]  # independent: no shared state between requests


# --- Real clock ---


@pytest.mark.integration
async def test_real_time_pacing():
    # 50 tokens at 5 ms ITL after a 10 ms TTFT: E2E should be 255 ms.
    engine = FixedRateEngine(
        prefill_tokens_per_second=1000, decode_tokens_per_second=200
    )
    plan = make_plan(50, prompt_tokens=10)
    arrival = time.monotonic()
    events = [event async for event in engine.run(plan, arrival)]
    elapsed = time.monotonic() - arrival

    expected = engine.timeline(plan)[-1]
    assert expected == pytest.approx(0.255)
    # Each wake-up can be a millisecond or two late, but lateness never adds up.
    assert expected <= elapsed < expected + 0.02
    assert max(event.lag for event in events) < 0.02
