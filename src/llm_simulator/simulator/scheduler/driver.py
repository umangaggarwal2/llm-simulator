"""SchedulerEngine: the asyncio driver, the only tier that deals with time."""

import asyncio
import heapq
import itertools
import logging
import time
from contextlib import suppress
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Optional, Tuple

from llm_simulator.simulator._timing import Clock, Sleep, sleep_until
from llm_simulator.simulator.engine import Engine
from llm_simulator.simulator.scheduler.cost_model import RooflineCostModel, StepCost
from llm_simulator.simulator.scheduler.policy import Scheduler, StepPlan
from llm_simulator.types import GenerationPlan, StepListener, StepStats, TokenEvent

logger = logging.getLogger(__name__)

# Put in a stream's queue after its last token.
_DONE = object()


class EngineFailed(RuntimeError):
    """The step loop has stopped; open and later requests fail with this."""


@dataclass(eq=False)
class _Stream:
    id: int
    arrival: float
    plan: GenerationPlan
    # (index, sim_time, queue_time) per token, then _DONE or an exception.
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    closed: bool = False


class SchedulerEngine(Engine):
    """Continuous batching on one simulated GPU, driven by a shared step loop.

    Each step, the policy decides what to compute, the cost model says how
    long that takes, and the loop sleeps until the step's end before handing
    each produced token to its request. Simulated time is anchored: a step
    ends at the previous step's end plus its duration, so a late wake-up
    shows as lag instead of shifting the simulated timeline.

    Anchoring means the loop can fall behind real time, which has two
    consequences handled here:

    - Overdue steps run back to back without sleeping, so the loop yields
      after every step; otherwise it would starve every other coroutine.
    - The next step can start, in simulated time, before a request that has
      already arrived in real time. Requests are therefore held in an
      arrival queue and handed to the policy only once simulated time has
      reached their arrival, so a request never joins a step that started
      before it arrived.

    The loop never waits for clients: each request has its own queue, as a
    GPU does not slow down for a slow reader.

    After each step, on_step receives its StepStats, for metrics. A listener
    that raises stops the engine like any other failure of the loop.
    """

    name = "scheduler"
    required_sections = ("hardware", "weights", "kv_cache", "scheduler")

    def __init__(
        self,
        policy: Scheduler,
        cost: RooflineCostModel,
        *,
        on_step: Optional[StepListener] = None,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ):
        self._policy = policy
        self._cost = cost
        self._on_step = on_step
        self._clock = clock
        self._sleep = sleep
        self._ids = itertools.count()
        self._streams: Dict[int, _Stream] = {}
        # (arrival, id, stream): requests not yet handed to the policy. A
        # heap, because requests can reach run() out of arrival order.
        self._arrivals: List[Tuple[float, int, _Stream]] = []
        self._wake = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._failure: Optional[BaseException] = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="scheduler-step-loop")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task
        self._task = None
        if self._failure is None:
            self._fail(EngineFailed("the engine has stopped"))

    async def run(
        self, plan: GenerationPlan, arrival: float
    ) -> AsyncIterator[TokenEvent]:
        if self._failure is not None:
            raise EngineFailed("the step loop has stopped") from self._failure
        if self._task is None:
            raise RuntimeError("SchedulerEngine.start() has not been called")

        stream = _Stream(next(self._ids), arrival, plan)
        self._streams[stream.id] = stream
        heapq.heappush(self._arrivals, (arrival, stream.id, stream))
        self._wake.set()
        try:
            while (item := await stream.queue.get()) is not _DONE:
                if isinstance(item, BaseException):
                    raise EngineFailed("the step loop has stopped") from item
                index, sim_time, queue_time = item
                # Measured when the consumer takes the token, so lag covers a
                # late step loop and a late consumer alike.
                lag = max(0.0, self._clock() - (arrival + sim_time))
                yield TokenEvent(index, plan.tokens[index], sim_time, lag, queue_time)
        finally:
            stream.closed = True
            del self._streams[stream.id]
            # Frees its KV blocks if the client left early; a no-op once it
            # has finished or before it reached the policy.
            self._policy.abort(stream.id)

    async def _loop(self) -> None:
        try:
            await self._steps()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("scheduler step loop failed")
            self._fail(exc)

    async def _steps(self) -> None:
        # The simulated time at which the last step ended.
        end = float("-inf")
        while True:
            if not self._policy.has_work():
                self._drop_closed_arrivals()
                if not self._arrivals:
                    self._wake.clear()
                    await self._wake.wait()
                    continue
                # The simulated GPU is idle until the next request arrives.
                # Starting at its arrival, not now, absorbs the time spent
                # before run() was called.
                end = max(end, self._arrivals[0][0])

            while self._arrivals and self._arrivals[0][0] <= end:
                _, _, stream = heapq.heappop(self._arrivals)
                if not stream.closed:
                    self._policy.add(
                        stream.id,
                        stream.arrival,
                        stream.plan.prompt_tokens,
                        len(stream.plan.tokens),
                    )
            if not self._policy.has_work():
                continue

            plan = self._policy.schedule(end)
            if not plan.work:
                # The policy's invariants make this impossible; fail loudly
                # rather than spin.
                raise RuntimeError("the scheduler planned an empty step")
            cost = self._cost.step_cost(plan.shape)
            end += cost.duration
            await sleep_until(end, self._clock, self._sleep)
            # Overdue steps do not sleep at all; yield so a loop catching up
            # never starves the rest of the server.
            await asyncio.sleep(0)

            result = self._policy.complete(plan)
            for seq in result.emitted:
                queue_time = (
                    seq.admitted_at - seq.arrival if seq.generated == 1 else None
                )
                self._streams[seq.id].queue.put_nowait(
                    (seq.generated - 1, end - seq.arrival, queue_time)
                )
            for seq in result.finished:
                self._streams[seq.id].queue.put_nowait(_DONE)
            if self._on_step is not None:
                self._on_step(self._stats(plan, cost))

    def _stats(self, plan: StepPlan, cost: StepCost) -> StepStats:
        kv = self._policy.kv
        return StepStats(
            duration=cost.duration,
            compute=cost.compute,
            memory=cost.memory,
            overhead=cost.overhead,
            sequences=len(plan.work),
            prefill_tokens=plan.prefill_tokens,
            decode_tokens=plan.decode_tokens,
            admitted=len(plan.admitted),
            preempted=len(plan.preempted),
            preempted_tokens=sum(p.discarded for p in plan.preempted),
            blocked=plan.blocked,
            # After complete(): the state the next step starts from.
            waiting=self._policy.waiting,
            running=self._policy.running,
            kv_blocks_used=kv.used_blocks,
            kv_blocks_total=kv.num_blocks,
        )

    def _drop_closed_arrivals(self) -> None:
        while self._arrivals and self._arrivals[0][2].closed:
            heapq.heappop(self._arrivals)

    def _fail(self, exc: BaseException) -> None:
        self._failure = exc
        for stream in self._streams.values():
            stream.queue.put_nowait(exc)
