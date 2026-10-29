import asyncio

import pytest

from llm_simulator.metrics import MetricsRecorder, RequestTimings, RequestTrace, traced
from llm_simulator.types import TokenEvent

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


class Clock:
    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now


class CapturingRecorder(MetricsRecorder):
    def __init__(self):
        self.calls = []

    def request_started(self, model, prompt_tokens):
        self.calls.append(("started", model, prompt_tokens))

    def first_token(self, model, ttft, queue_time):
        self.calls.append(("first_token", model, ttft, queue_time))

    def token(self, model, itl, lag):
        self.calls.append(("token", model, itl, lag))

    def request_finished(
        self, model, finish_reason, prompt_tokens, generation_tokens, timings
    ):
        self.calls.append(
            (
                "finished",
                model,
                finish_reason,
                prompt_tokens,
                generation_tokens,
                timings,
            )
        )

    def of(self, kind):
        return [call for call in self.calls if call[0] == kind]


def event(index, lag=0.0):
    return TokenEvent(index, f"t{index}", 0.0, lag)


def make_trace(recorder, clock, arrival=100.0, prompt_tokens=8):
    return RequestTrace(recorder, "m", arrival, prompt_tokens, clock=clock)


# --- RequestTrace ---


def test_records_ttft_itl_lag_and_timings():
    recorder, clock = CapturingRecorder(), Clock(100.0)
    trace = make_trace(recorder, clock)
    trace.start()
    for index, (at, lag) in enumerate(
        [(100.012, 0.001), (100.016, 0.0), (100.021, 0.002)]
    ):
        clock.now = at
        trace.token(event(index, lag))
    trace.finish("stop")

    assert recorder.of("started") == [("started", "m", 8)]
    assert recorder.of("first_token") == [
        ("first_token", "m", pytest.approx(0.012), 0.0)
    ]
    tokens = recorder.of("token")
    assert [call[2] for call in tokens] == [
        None,
        pytest.approx(0.004),
        pytest.approx(0.005),
    ]
    assert [call[3] for call in tokens] == [0.001, 0.0, 0.002]

    ((_, _, reason, prompt, generated, timings),) = recorder.of("finished")
    assert (reason, prompt, generated) == ("stop", 8, 3)
    assert timings == RequestTimings(
        e2e=pytest.approx(0.021),
        queue=0.0,
        prefill=pytest.approx(0.012),
        decode=pytest.approx(0.009),
        tpot=pytest.approx(0.0045),
    )


def test_length_counts_as_completed():
    recorder, clock = CapturingRecorder(), Clock()
    trace = make_trace(recorder, clock)
    trace.start()
    trace.token(event(0))
    trace.finish("length")
    assert recorder.of("finished")[0][5] is not None


def test_single_token_has_no_tpot():
    recorder, clock = CapturingRecorder(), Clock(100.0)
    trace = make_trace(recorder, clock)
    trace.start()
    clock.now = 100.010
    trace.token(event(0))
    trace.finish("stop")
    timings = recorder.of("finished")[0][5]
    assert timings.tpot is None
    assert timings.decode == 0.0


def test_cancelled_request_has_no_timings_but_keeps_its_tokens():
    recorder, clock = CapturingRecorder(), Clock(100.0)
    trace = make_trace(recorder, clock)
    trace.start()
    clock.now = 100.010
    trace.token(event(0))
    clock.now = 100.014
    trace.token(event(1))
    trace.finish("cancelled")

    assert len(recorder.of("first_token")) == 1
    assert len(recorder.of("token")) == 2
    assert recorder.of("finished") == [("finished", "m", "cancelled", 8, 2, None)]


def test_empty_completion_ends_when_finished():
    recorder, clock = CapturingRecorder(), Clock(100.0)
    trace = make_trace(recorder, clock)
    trace.start()
    clock.now = 100.012  # the engine waited out the prefill
    trace.finish("stop")
    assert recorder.of("first_token") == []
    timings = recorder.of("finished")[0][5]
    assert timings.e2e == pytest.approx(0.012)
    assert timings.prefill == pytest.approx(0.012)
    assert timings.decode == 0.0


def test_finish_is_reported_once():
    recorder, clock = CapturingRecorder(), Clock()
    trace = make_trace(recorder, clock)
    trace.start()
    trace.finish("stop")
    trace.finish("cancelled")
    assert len(recorder.of("finished")) == 1


def test_finish_before_start_records_nothing():
    recorder = CapturingRecorder()
    make_trace(recorder, Clock()).finish("stop")
    assert recorder.calls == []


# --- traced() ---


async def engine_events(n, closed=None):
    try:
        for index in range(n):
            yield event(index)
    finally:
        if closed is not None:
            closed.append(True)


async def test_traced_passes_events_through_and_finishes_with_the_plans_reason():
    recorder, clock = CapturingRecorder(), Clock()
    trace = make_trace(recorder, clock)
    events = [e async for e in traced(engine_events(3), trace, "length")]
    assert [e.index for e in events] == [0, 1, 2]
    assert len(recorder.of("token")) == 3
    assert recorder.of("finished")[0][2] == "length"


async def test_traced_closed_early_is_cancelled_and_closes_the_engine():
    recorder, clock, closed = CapturingRecorder(), Clock(), []
    stream = traced(engine_events(5, closed), make_trace(recorder, clock), "stop")
    await stream.__anext__()
    await stream.aclose()
    assert recorder.of("finished")[0][2] == "cancelled"
    assert recorder.of("finished")[0][5] is None
    assert closed == [True]


async def test_traced_task_cancellation_is_cancelled():
    recorder, clock = CapturingRecorder(), Clock()

    async def slow_engine():
        yield event(0)
        await asyncio.sleep(10)
        yield event(1)

    async def consume():
        async for _ in traced(slow_engine(), make_trace(recorder, clock), "stop"):
            pass

    task = asyncio.ensure_future(consume())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert recorder.of("finished")[0][2] == "cancelled"


async def test_traced_engine_failure_is_an_error():
    recorder, clock = CapturingRecorder(), Clock()

    async def failing_engine():
        yield event(0)
        raise RuntimeError("engine broke")

    with pytest.raises(RuntimeError):
        async for _ in traced(failing_engine(), make_trace(recorder, clock), "stop"):
            pass
    assert recorder.of("finished")[0][2] == "error"


async def test_stream_never_iterated_records_nothing():
    recorder, clock = CapturingRecorder(), Clock()
    stream = traced(engine_events(3), make_trace(recorder, clock), "stop")
    await stream.aclose()
    assert recorder.calls == []
