import asyncio
import itertools
import statistics
import time

import pytest

from llm_simulator.bench.health import LoopMonitor
from llm_simulator.bench.records import RequestRecord, ServerSummary
from llm_simulator.bench.runner import (
    WarmupFailed,
    closed_loop,
    open_loop,
    poisson_offsets,
    run_level,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FakeServer:
    """A send function that tracks what was sent and how much was in flight."""

    def __init__(self, delays=None, fail=(), delay=0.001):
        self.delays = delays or {}
        self.fail = fail
        self.delay = delay
        self.in_flight = 0
        self.peak = 0
        self.sent = []
        self.events = []

    async def send(self, index):
        self.sent.append(index)
        self.events.append(("start", index))
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        await asyncio.sleep(self.delays.get(index, self.delay))
        self.in_flight -= 1
        self.events.append(("end", index))
        failed = self.fail is True or index in self.fail
        return RequestRecord(
            index=index, sent=0.0, end=1.0, status=500 if failed else 200
        )


# --- closed_loop ---


async def test_every_index_is_sent_once_and_returned_in_order():
    server = FakeServer(delays={0: 0.03, 1: 0.001, 2: 0.02})
    records = await closed_loop(server.send, 3, range(10))
    assert sorted(server.sent) == list(range(10))
    assert [record.index for record in records] == list(range(10))


async def test_keeps_exactly_concurrency_requests_in_flight():
    server = FakeServer()
    await closed_loop(server.send, 4, range(40))
    assert server.peak == 4


async def test_more_workers_than_requests():
    server = FakeServer()
    records = await closed_loop(server.send, 50, range(3))
    assert len(records) == 3
    assert server.peak == 3


async def test_a_failed_request_does_not_stop_the_run():
    server = FakeServer(fail={2})
    records = await closed_loop(server.send, 2, range(6))
    assert [record.ok for record in records] == [True, True, False, True, True, True]


async def test_deadline_stops_new_sends_and_waits_for_those_in_flight():
    server = FakeServer(delay=0.02)
    started = time.perf_counter()
    records = await closed_loop(
        server.send, 2, iter(range(10**6)), until=started + 0.05
    )
    elapsed = time.perf_counter() - started
    # Sends at about 0, 20 and 40 ms on each of 2 workers; the last pair is
    # waited for, so the run ends at about 60 ms.
    assert 4 <= len(records) <= 8
    assert len(records) == len(server.sent)
    assert server.events[-1][0] == "end" and server.in_flight == 0
    assert elapsed < 0.2


# --- run_level ---


async def test_warmup_uses_negative_indexes_and_is_not_returned():
    server = FakeServer()
    run = await run_level(server.send, concurrency=2, requests=5, warmup=3)
    assert server.sent[:3] == [-1, -2, -3]
    assert [record.index for record in run.records] == [0, 1, 2, 3, 4]
    assert run.warmup == 3


async def test_measuring_starts_after_every_warmup_request_finished():
    server = FakeServer(delays={-1: 0.03})
    await run_level(server.send, concurrency=2, requests=4, warmup=2)
    last_warmup_end = server.events.index(("end", -1))
    first_measured = server.events.index(("start", 0))
    assert last_warmup_end < first_measured


async def test_window_starts_after_warmup():
    class Clock:
        now = 0.0

        def __call__(self):
            return self.now

    clock = Clock()

    async def send(index):
        clock.now += 1.0
        return RequestRecord(index=index, sent=clock.now, end=clock.now, status=200)

    run = await run_level(send, concurrency=1, requests=2, warmup=3, clock=clock)
    # 3 warm-up seconds pass before the window opens; 2 measured ones inside.
    assert (run.started, run.ended, run.duration) == (3.0, 5.0, 2.0)


async def test_no_warmup():
    server = FakeServer()
    run = await run_level(server.send, concurrency=2, requests=3, warmup=0)
    assert server.sent[0] >= 0
    assert run.warmup == 0


async def test_failed_warmup_stops_the_run():
    server = FakeServer(fail=True)
    with pytest.raises(WarmupFailed) as error:
        await run_level(server.send, concurrency=2, requests=10, warmup=4)
    assert len(error.value.records) == 4
    assert all(index < 0 for index in server.sent)


async def test_one_working_warmup_request_is_enough():
    server = FakeServer(fail={-1, -2, -3})
    run = await run_level(server.send, concurrency=2, requests=2, warmup=4)
    assert len(run.records) == 2


async def test_duration_mode():
    server = FakeServer(delay=0.01)
    run = await run_level(server.send, concurrency=2, duration=0.05, warmup=0)
    assert len(run.records) >= 4
    assert [record.index for record in run.records] == list(range(len(run.records)))
    assert run.duration >= 0.05


async def test_monitor_covers_the_measure_phase():
    run = await run_level(
        FakeServer(delay=0.03).send,
        concurrency=1,
        requests=2,
        warmup=1,
        monitor=LoopMonitor(interval=0.005),
    )
    assert run.loop is not None
    assert len(run.loop.lags) >= 5
    assert run.loop.wall_seconds == pytest.approx(run.duration, abs=0.02)


async def test_requests_or_duration_is_required():
    with pytest.raises(ValueError):
        await run_level(FakeServer().send, concurrency=1)
    with pytest.raises(ValueError):
        await run_level(FakeServer().send, concurrency=1, requests=1, duration=1.0)


class FakeProbe:
    def __init__(self, server, available=True):
        self.server = server
        self.available = available

    async def reset(self):
        self.server.events.append(("reset", None))
        return self.available

    async def fetch(self):
        self.server.events.append(("fetch", None))
        return ServerSummary(url="u", available=self.available)


async def test_server_is_reset_at_the_boundary_and_read_after_the_run():
    server = FakeServer(delays={-1: 0.02})
    run = await run_level(
        server.send, concurrency=2, requests=4, warmup=2, server=FakeProbe(server)
    )
    events = server.events
    reset, fetch = events.index(("reset", None)), events.index(("fetch", None))
    assert reset > events.index(("end", -1)) and reset > events.index(("end", -2))
    assert reset < events.index(("start", 0))
    assert fetch == len(events) - 1  # after every measured request ended
    assert run.server.available


async def test_no_server_probe():
    run = await run_level(FakeServer().send, concurrency=1, requests=2, warmup=0)
    assert run.server is None


# --- open loop ---


def test_poisson_offsets_are_seeded_and_have_the_right_mean():
    first = list(itertools.islice(poisson_offsets(10.0, seed=3), 20000))
    again = list(itertools.islice(poisson_offsets(10.0, seed=3), 20000))
    other = list(itertools.islice(poisson_offsets(10.0, seed=4), 5))
    assert first == again
    assert first[:5] != other
    gaps = [b - a for a, b in zip([0.0] + first, first)]
    assert statistics.fmean(gaps) == pytest.approx(0.1, rel=0.03)
    # Exponential gaps: the standard deviation equals the mean (CV = 1).
    assert statistics.pstdev(gaps) / statistics.fmean(gaps) == pytest.approx(
        1, abs=0.05
    )
    assert all(b > a for a, b in zip(first, first[1:]))


class FakeTime:
    """A clock that only moves when the loop sleeps on it."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        # Like a real sleep, let tasks already started run first.
        self.sleeps.append(seconds)
        await asyncio.sleep(0)
        self.now += seconds


async def test_requests_are_sent_at_their_scheduled_times():
    fake = FakeTime()
    sent_at = {}

    async def send(index):
        sent_at[index] = fake.now
        return RequestRecord(index=index, sent=fake.now, end=fake.now, status=200)

    records = await open_loop(
        send, 10.0, range(5), seed=1, clock=fake, sleep=fake.sleep
    )
    expected = list(itertools.islice(poisson_offsets(10.0, seed=1), 5))
    assert [sent_at[i] for i in range(5)] == pytest.approx(expected)
    assert [record.scheduled for record in records] == pytest.approx(expected)
    assert not any(record.waited for record in records)


async def test_a_late_send_does_not_shift_later_arrivals():
    fake = FakeTime()

    async def send(index):
        if index == 1:
            fake.now += 0.5  # the client was busy for half a second
        return RequestRecord(index=index, sent=fake.now, end=fake.now, status=200)

    records = await open_loop(
        send, 10.0, range(6), seed=2, clock=fake, sleep=fake.sleep
    )
    expected = list(itertools.islice(poisson_offsets(10.0, seed=2), 6))
    # Scheduled times are absolute: arrivals after the stall keep their slots.
    assert [record.scheduled for record in records] == pytest.approx(expected)


async def test_arrivals_continue_while_requests_are_running():
    server = FakeServer(delay=0.2)
    started = time.perf_counter()
    records = await open_loop(server.send, 200.0, range(20), seed=0)
    # A closed loop at concurrency 1 would need 20 x 0.2 s; the open loop
    # keeps sending, so nearly every request overlaps the others.
    assert time.perf_counter() - started < 1.0
    assert server.peak >= 15
    assert len(records) == 20


async def test_max_concurrency_caps_in_flight_requests_and_marks_waits():
    server = FakeServer(delay=0.05)
    records = await open_loop(server.send, 500.0, range(12), max_concurrency=3)
    assert server.peak == 3
    assert any(record.waited for record in records)
    waited = [record for record in records if record.waited]
    assert all(record.scheduled is not None for record in waited)


async def test_open_loop_deadline():
    server = FakeServer(delay=0.01)
    started = time.perf_counter()
    records = await open_loop(
        server.send, 100.0, itertools.count(), until=started + 0.2, seed=0
    )
    assert 5 <= len(records) <= 50
    assert all(record.scheduled < started + 0.2 for record in records)


async def test_run_level_open_loop_with_warmup_and_server():
    server = FakeServer(delay=0.005)
    run = await run_level(
        server.send,
        rate=100.0,
        max_concurrency=4,
        requests=10,
        warmup=4,
        warmup_concurrency=4,
        server=FakeProbe(server),
    )
    assert server.sent[:4] == [-1, -2, -3, -4]
    assert [record.index for record in run.records] == list(range(10))
    assert all(record.scheduled is not None for record in run.records)
    assert run.server.available


async def test_run_level_needs_one_load_shape():
    with pytest.raises(ValueError):
        await run_level(FakeServer().send, concurrency=1, rate=5.0, requests=1)
