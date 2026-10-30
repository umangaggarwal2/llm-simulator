"""Following one request from arrival to its last token."""

import asyncio
import time
from contextlib import aclosing
from typing import AsyncIterator, Callable, Optional

from llm_simulator.metrics.recorder import MetricsRecorder, RequestTimings
from llm_simulator.types import TokenEvent

# Requests ending with these produced their whole completion; only they
# contribute request-level timings (E2E, TPOT, prefill, decode, sizes).
COMPLETED_REASONS = frozenset({"stop", "length"})


class RequestTrace:
    """Measures one request and reports to a recorder.

    Times are server-observed: the moment the engine released each token.
    The trace keeps only running state, so its memory is constant however
    long the completion, and it belongs to a single request, so it needs no
    locking.
    """

    def __init__(
        self,
        recorder: MetricsRecorder,
        model: str,
        arrival: float,
        prompt_tokens: int,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._recorder = recorder
        self._model = model
        self._arrival = arrival
        self._prompt_tokens = prompt_tokens
        self._clock = clock
        # Always 0 until an engine with a waiting queue reports admission.
        self._queue_time = 0.0
        self._first: Optional[float] = None
        self._last: Optional[float] = None
        # The engine's scheduled times (TokenEvent.sim_time) of the first and
        # last token.
        self._first_sim: Optional[float] = None
        self._last_sim: Optional[float] = None
        self._count = 0
        self._started = False
        self._finished = False

    def start(self) -> None:
        self._started = True
        self._recorder.request_started(self._model, self._prompt_tokens)

    def token(self, event: TokenEvent) -> None:
        now = self._clock()
        if self._first is None:
            self._first = now
            self._first_sim = event.sim_time
            self._recorder.first_token(
                self._model, now - self._arrival, self._queue_time
            )
            itl = None
        else:
            itl = now - self._last
        self._recorder.token(self._model, itl, event.lag)
        self._last = now
        self._last_sim = event.sim_time
        self._count += 1

    def finish(self, finish_reason: str) -> None:
        """Report the outcome once; later calls, or calls before start(), do nothing."""
        if self._finished or not self._started:
            return
        self._finished = True

        timings = None
        if finish_reason in COMPLETED_REASONS:
            # An empty completion has no tokens, but the engine still waited
            # out the prefill, so it ends now.
            end = self._last if self._last is not None else self._clock()
            first = self._first if self._first is not None else end
            decode = end - first
            timings = RequestTimings(
                e2e=end - self._arrival,
                queue=self._queue_time,
                prefill=first - self._arrival - self._queue_time,
                decode=decode,
                tpot=decode / (self._count - 1) if self._count > 1 else None,
                simulated_ttft=self._first_sim,
                simulated_e2e=self._last_sim,
            )
        self._recorder.request_finished(
            self._model, finish_reason, self._prompt_tokens, self._count, timings
        )


async def traced(
    events: AsyncIterator[TokenEvent], trace: RequestTrace, finish_reason: str
) -> AsyncIterator[TokenEvent]:
    """Pass an engine's events through unchanged, recording each one.

    The request finishes with finish_reason when the engine runs to the end,
    "cancelled" when the stream is closed early (a client disconnect) and
    "error" when the engine fails. A stream that is never iterated records
    nothing, so started and finished requests always balance.
    """
    trace.start()
    outcome = "error"
    try:
        async with aclosing(events):
            async for event in events:
                trace.token(event)
                yield event
        outcome = finish_reason
    except (GeneratorExit, asyncio.CancelledError):
        outcome = "cancelled"
        raise
    finally:
        trace.finish(outcome)
