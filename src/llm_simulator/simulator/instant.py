"""An engine that emits every token immediately."""

import time
from typing import AsyncIterator

from llm_simulator.simulator._timing import Clock
from llm_simulator.simulator.engine import Engine
from llm_simulator.types import GenerationPlan, TokenEvent


class InstantEngine(Engine):
    """Emits every token immediately, with no simulated latency.

    Used to serve responses before any timing model exists, and for tests that
    check content rather than timing.
    """

    name = "instant"

    def __init__(self, *, clock: Clock = time.monotonic):
        self._clock = clock

    async def run(
        self, plan: GenerationPlan, arrival: float
    ) -> AsyncIterator[TokenEvent]:
        # No sleep between tokens: when streaming, each SSE write already
        # yields to the event loop.
        for index, text in enumerate(plan.tokens):
            # Every token is due at arrival (sim_time 0), so anything later is
            # lag, as for any engine: observed time = sim_time + lag.
            lag = max(0.0, self._clock() - arrival)
            yield TokenEvent(index, text, sim_time=0.0, lag=lag)
