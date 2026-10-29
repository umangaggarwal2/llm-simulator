import time

import anyio
import pytest

from llm_simulator.simulator import InstantEngine
from llm_simulator.types import GenerationPlan, TokenEvent

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    # The app runs on asyncio under uvicorn, so test only that backend.
    return "asyncio"


def make_plan(*tokens):
    return GenerationPlan(
        prompt_tokens=8,
        tokens=tuple(tokens),
        finish_reason="stop",
        entry_id="e0",
        variant=0,
    )


async def collect(engine, plan):
    return [event async for event in engine.run(plan, time.monotonic())]


async def test_emits_every_token_in_order():
    plan = make_plan("Hello", ",", " world", "!")
    engine = InstantEngine(clock=lambda: 50.0)
    events = [event async for event in engine.run(plan, 50.0)]
    assert events == [
        TokenEvent(0, "Hello", 0.0),
        TokenEvent(1, ",", 0.0),
        TokenEvent(2, " world", 0.0),
        TokenEvent(3, "!", 0.0),
    ]


async def test_time_since_arrival_is_lag():
    # Every token is due at arrival, so a token released 0.15 s later is 0.15 s late.
    engine = InstantEngine(clock=lambda: 100.15)
    events = [event async for event in engine.run(make_plan("a", "b"), 100.0)]
    assert [event.sim_time for event in events] == [0.0, 0.0]
    assert [event.lag for event in events] == [pytest.approx(0.15)] * 2


async def test_lag_is_never_negative():
    engine = InstantEngine(clock=lambda: 99.0)
    events = [event async for event in engine.run(make_plan("a"), 100.0)]
    assert events[0].lag == 0.0


async def test_joined_text_matches_plan():
    plan = make_plan("To", " be", ",", " or", " not", " to", " be", ".")
    events = await collect(InstantEngine(), plan)
    assert "".join(event.text for event in events) == plan.text


async def test_empty_plan_emits_nothing():
    assert await collect(InstantEngine(), make_plan()) == []


async def test_closing_early_stops_the_stream():
    engine = InstantEngine(clock=lambda: 50.0)
    events = engine.run(make_plan("a", "b", "c"), 50.0)
    first = await events.__anext__()
    await events.aclose()
    assert first == TokenEvent(0, "a", 0.0)
    with pytest.raises(StopAsyncIteration):
        await events.__anext__()


async def test_concurrent_runs_are_independent():
    engine = InstantEngine()
    plans = [make_plan(*(f"{n}-{i}" for i in range(50))) for n in range(5)]
    results = [None] * len(plans)

    async def consume(slot, plan):
        results[slot] = await collect(engine, plan)

    async with anyio.create_task_group() as tasks:
        for slot, plan in enumerate(plans):
            tasks.start_soon(consume, slot, plan)

    for plan, events in zip(plans, results):
        assert tuple(event.text for event in events) == plan.tokens
        assert [event.index for event in events] == list(range(50))


async def test_lifecycle_hooks_are_no_ops():
    engine = InstantEngine()
    await engine.start()
    await engine.stop()
