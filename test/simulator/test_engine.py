import time

import anyio
import pytest

from llm_simulator.simulator.engine import Engine, InstantEngine, get_engine
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
    events = await collect(InstantEngine(), plan)
    assert events == [
        TokenEvent(0, "Hello", 0.0),
        TokenEvent(1, ",", 0.0),
        TokenEvent(2, " world", 0.0),
        TokenEvent(3, "!", 0.0),
    ]


async def test_joined_text_matches_plan():
    plan = make_plan("To", " be", ",", " or", " not", " to", " be", ".")
    events = await collect(InstantEngine(), plan)
    assert "".join(event.text for event in events) == plan.text


async def test_empty_plan_emits_nothing():
    assert await collect(InstantEngine(), make_plan()) == []


async def test_closing_early_stops_the_stream():
    events = InstantEngine().run(make_plan("a", "b", "c"), time.monotonic())
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


def test_get_engine_returns_instant_engine():
    engine = get_engine("instant")
    assert isinstance(engine, InstantEngine)
    assert isinstance(engine, Engine)


def test_get_engine_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown engine 'turbo'; available: instant"):
        get_engine("turbo")


def test_engine_interface_requires_run():
    class Incomplete(Engine):
        name = "incomplete"

    with pytest.raises(TypeError):
        Incomplete()
