"""Watching the client itself, so its own limits are not blamed on the server.

The client takes every timestamp on its event loop. If the loop is busy, a
chunk that arrived on time is read late, and the latency it records is too
high. The monitor asks to wake at a fixed interval and records how late each
wake-up was: a late wake-up means timestamps taken then were late by about
as much.
"""

import asyncio
import time
from array import array
from contextlib import suppress
from typing import Awaitable, Callable, Optional

from llm_simulator.bench.records import LoopSamples

Clock = Callable[[], float]


class LoopMonitor:
    """Samples event-loop lateness and process CPU between start() and stop()."""

    def __init__(
        self,
        interval: float = 0.010,
        *,
        clock: Clock = time.perf_counter,
        cpu_clock: Clock = time.process_time,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self._interval = interval
        self._clock = clock
        self._cpu_clock = cpu_clock
        self._sleep = sleep
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        """Begin sampling; call from inside the running event loop."""
        self._lags = array("d")
        self._cpu_started = self._cpu_clock()
        self._wall_started = self._clock()
        self._task = asyncio.ensure_future(self._watch())

    async def stop(self) -> LoopSamples:
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task
        return LoopSamples(
            lags=self._lags,
            cpu_seconds=self._cpu_clock() - self._cpu_started,
            wall_seconds=self._clock() - self._wall_started,
        )

    async def _watch(self) -> None:
        while True:
            before = self._clock()
            await self._sleep(self._interval)
            self._lags.append(max(0.0, self._clock() - before - self._interval))
