"""Generating load: deciding when each request is sent.

The runner knows nothing about HTTP. It is given a send function that turns a
request index into a RequestRecord, so it can be tested with fakes. It
collects records and leaves interpreting them to results.py.
"""

import asyncio
import itertools
import random
import time
from datetime import datetime, timezone
from typing import (
    Awaitable,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Protocol,
)

from llm_simulator.bench.health import LoopMonitor
from llm_simulator.bench.records import RequestRecord, RunRecords, ServerSummary

Send = Callable[[int], Awaitable[RequestRecord]]
Clock = Callable[[], float]
Sleep = Callable[[float], Awaitable[None]]


class ServerProbe(Protocol):
    """The target's own metrics, as the runner needs them (see server.py)."""

    async def reset(self) -> bool: ...

    async def fetch(self) -> ServerSummary: ...


class WarmupFailed(Exception):
    """Every warm-up request failed, so there is nothing worth measuring."""

    def __init__(self, records: List[RequestRecord]):
        super().__init__(f"all {len(records)} warm-up requests failed")
        self.records = records


async def closed_loop(
    send: Send,
    concurrency: int,
    indexes: Iterable[int],
    *,
    until: Optional[float] = None,
    clock: Callable[[], float] = time.perf_counter,
) -> List[RequestRecord]:
    """Keep `concurrency` requests in flight, one per index, until the indexes
    run out or the clock passes `until`.

    Each worker sends its next request as soon as its previous one finishes,
    so a slower server is offered less load: right for measuring capacity at
    a fixed number of streams, but it hides queueing (the open loop shows it).
    A request sent before `until` is waited for. Records are returned in the
    order their indexes were taken, whatever order they finished in.
    """
    # Shared by every worker; taking the next index needs no lock because
    # everything runs on one event loop.
    remaining = iter(indexes)
    order: List[int] = []
    records: Dict[int, RequestRecord] = {}

    async def worker():
        while until is None or clock() < until:
            index = next(remaining, None)
            if index is None:
                return
            order.append(index)
            records[index] = await send(index)

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return [records[index] for index in order]


def poisson_offsets(rate: float, seed: int = 0) -> Iterator[float]:
    """Arrival times of a Poisson process, as seconds from the start.

    The gaps between arrivals are exponential with mean 1/rate:
    expovariate draws one by inverse-transform sampling, -ln(1 - U) / rate
    for a uniform U. The generator has its own seed stream, separate from
    the prompts', so the same seed gives the same arrival times.
    """
    rng = random.Random(f"arrivals:{seed}")
    offset = 0.0
    while True:
        offset += rng.expovariate(rate)
        yield offset


async def open_loop(
    send: Send,
    rate: float,
    indexes: Iterable[int],
    *,
    seed: int = 0,
    until: Optional[float] = None,
    max_concurrency: Optional[int] = None,
    clock: Clock = time.perf_counter,
    sleep: Sleep = asyncio.sleep,
) -> List[RequestRecord]:
    """Send a request at each Poisson arrival time, without waiting for replies.

    Unlike the closed loop, a slow server is not offered less load: requests
    keep arriving and pile up, as real users' would. Each arrival is slept
    for until its absolute time (start + offset), so late wake-ups do not
    accumulate into drift. With max_concurrency, an arrival that finds every
    slot busy waits for one; records note their scheduled time and whether
    they waited, so latency can be read from the send and from the schedule.
    Arrivals stop when the indexes run out or at `until`; every request
    sent is waited for.
    """
    slots = asyncio.Semaphore(max_concurrency) if max_concurrency else None
    start = clock()
    offsets = poisson_offsets(rate, seed)
    tasks = []

    async def arrive(index: int, scheduled: float) -> RequestRecord:
        waited = slots is not None and slots.locked()
        if slots is None:
            record = await send(index)
        else:
            async with slots:
                record = await send(index)
        record.scheduled = scheduled
        record.waited = waited
        return record

    for index in indexes:
        scheduled = start + next(offsets)
        if until is not None and scheduled >= until:
            break
        # Always yield, even when the arrival is already due: the requests
        # started so far only begin running when this loop gives way, and
        # overdue arrivals must not pile up and start together.
        await sleep(max(0.0, scheduled - clock()))
        tasks.append(asyncio.ensure_future(arrive(index, scheduled)))
    return list(await asyncio.gather(*tasks))


async def run_level(
    send: Send,
    *,
    concurrency: Optional[int] = None,
    rate: Optional[float] = None,
    max_concurrency: Optional[int] = None,
    seed: int = 0,
    requests: Optional[int] = None,
    duration: Optional[float] = None,
    warmup: int = 0,
    warmup_concurrency: Optional[int] = None,
    monitor: Optional[LoopMonitor] = None,
    server: Optional[ServerProbe] = None,
    clock: Callable[[], float] = time.perf_counter,
) -> RunRecords:
    """Warm up, then measure `requests` requests or `duration` seconds.

    Load is a closed loop at `concurrency`, or an open loop at `rate`
    (optionally capped at `max_concurrency`). Warm-up is always a short
    closed-loop burst, at `warmup_concurrency` (default: `concurrency`).

    Warm-up requests use indexes -1, -2, ..., so measured requests always
    start at index 0 and send the same prompts whatever the warm-up. Every
    warm-up request finishes before measuring starts, leaving the server
    idle at that boundary. The window runs from the first measured send to
    the last measured completion. The server's summary, if given, is reset
    at that idle boundary and read once every measured request has
    finished, so it covers the same requests. Raises WarmupFailed if no
    warm-up request succeeded.
    """
    if (requests is None) == (duration is None):
        raise ValueError("give requests or duration")
    if (concurrency is None) == (rate is None):
        raise ValueError("give concurrency (closed loop) or rate (open loop)")
    if warmup:
        burst = warmup_concurrency or concurrency or warmup
        warm = await closed_loop(send, burst, range(-1, -warmup - 1, -1))
        if not any(record.ok for record in warm):
            raise WarmupFailed(warm)

    # Before the monitor starts, so the reset's own request is not counted
    # as client loop lag.
    if server is not None:
        await server.reset()
    if monitor is not None:
        monitor.start()
    started_at = datetime.now(timezone.utc)
    started = clock()
    indexes = itertools.count() if duration is not None else range(requests)
    until = started + duration if duration is not None else None
    if rate is not None:
        records = await open_loop(
            send,
            rate,
            indexes,
            seed=seed,
            until=until,
            max_concurrency=max_concurrency,
            clock=clock,
        )
    else:
        records = await closed_loop(
            send, concurrency, indexes, until=until, clock=clock
        )
    ended = clock()
    loop = await monitor.stop() if monitor is not None else None
    summary = await server.fetch() if server is not None else None
    return RunRecords(records, started, ended, started_at, warmup, loop, summary)
