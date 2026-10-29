"""An engine that streams every request on the same fixed timeline."""

import asyncio
import time
from typing import AsyncIterator, Tuple

from llm_simulator.simulator._timing import Clock, Sleep, sleep_until
from llm_simulator.simulator.engine import Engine
from llm_simulator.types import GenerationPlan, TokenEvent


class FixedRateEngine(Engine):
    """Streams every request on the same fixed timeline.

    The first token is due once the scheduler overhead and the prefill of the
    prompt have elapsed; each later token follows one decode step after the
    previous one. Requests do not affect each other, so there is no queueing
    and no load effect; that is SchedulerEngine's job.

    Tokens are released at absolute deadlines measured from the request's
    arrival, so time spent before the engine runs is absorbed by the simulated
    TTFT, and a late wake-up never delays the tokens after it.
    """

    name = "fixed_rate"
    required_sections = ("performance",)

    def __init__(
        self,
        prefill_tokens_per_second: float,
        decode_tokens_per_second: float,
        scheduler_overhead_s: float = 0.0,
        *,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ):
        if prefill_tokens_per_second <= 0 or decode_tokens_per_second <= 0:
            raise ValueError("token rates must be greater than 0")
        if scheduler_overhead_s < 0:
            raise ValueError("scheduler_overhead_s must not be negative")
        self._prefill_rate = prefill_tokens_per_second
        self._itl = 1 / decode_tokens_per_second
        # Paid once per request, before prefill; the decode rate is the
        # per-stream rate a client measures.
        self._overhead = scheduler_overhead_s
        self._clock = clock
        self._sleep = sleep

    def ttft(self, prompt_tokens: int) -> float:
        """Simulated seconds from arrival to the first token.

        The prefill pass itself produces the first token, so no decode step
        is added.
        """
        return self._overhead + prompt_tokens / self._prefill_rate

    def timeline(self, plan: GenerationPlan) -> Tuple[float, ...]:
        """Simulated seconds after arrival at which each token is due."""
        ttft = self.ttft(plan.prompt_tokens)
        return tuple(ttft + index * self._itl for index in range(len(plan.tokens)))

    async def run(
        self, plan: GenerationPlan, arrival: float
    ) -> AsyncIterator[TokenEvent]:
        if not plan.tokens:
            # Nothing to send, but the prompt was still processed.
            await sleep_until(
                arrival + self.ttft(plan.prompt_tokens), self._clock, self._sleep
            )
            return

        for index, (text, sim_time) in enumerate(zip(plan.tokens, self.timeline(plan))):
            deadline = arrival + sim_time
            lag = await sleep_until(deadline, self._clock, self._sleep)
            yield TokenEvent(index, text, sim_time, lag)
