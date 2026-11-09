"""Exact percentiles over recent traffic, for GET /metrics/summary.

Prometheus histograms estimate percentiles only to within a bucket's width,
too coarse to check millisecond timing. This store keeps the raw values of
recent requests and tokens in bounded windows instead, and computes exact
percentiles from them when the summary is read.

Reading is split in two. snapshot() copies the windows, which is cheap and
runs on the event loop alongside recording, so no locking is needed.
summarize() sorts the copies; it is pure and can run in a worker thread so
that sorting does not delay token releases.
"""

import time
from array import array
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from llm_simulator.metrics.recorder import (
    BLOCK_REASONS,
    FINISH_REASONS,
    MetricsRecorder,
    ModelDescription,
    RequestTimings,
)
from llm_simulator.metrics.window import RateWindow, SampleWindow, describe
from llm_simulator.types import StepStats

# Order of the latency_seconds section of the summary.
LATENCIES = (
    "ttft",
    "itl",
    "tpot",
    "e2e",
    "queue",
    "prefill",
    "decode",
    "lag",
    # The engine's timeline, for completed requests: what TTFT and E2E would
    # be with no lag. Compared with ttft and e2e, they separate the
    # simulator running late from delivery being slow.
    "ttft_simulated",
    "e2e_simulated",
)
SIZES = ("prompt", "generation")


@dataclass(frozen=True)
class WindowSnapshot:
    values: array
    count: int


@dataclass(frozen=True)
class SchedulerSnapshot:
    steps: int
    memory_bound: int
    gpu_seconds: float
    overhead_seconds: float
    prefill_tokens: int
    decode_tokens: int
    preemptions: int
    preempted_tokens: int
    blocked: Dict[str, int]
    waiting: int
    waiting_max: int
    kv_blocks_used: int
    kv_blocks_used_max: int
    kv_blocks_total: int
    durations: WindowSnapshot
    batch_tokens: WindowSnapshot
    batch_sequences: WindowSnapshot


@dataclass(frozen=True)
class ModelSnapshot:
    engine: Optional[str]
    running: int
    finished: Dict[str, int]
    prompt_tokens: int
    generation_tokens: int
    # (per second, seconds measured over)
    request_rate: Tuple[float, float]
    token_rate: Tuple[float, float]
    latencies: Dict[str, WindowSnapshot]
    sizes: Dict[str, WindowSnapshot]
    # None for engines without steps.
    scheduler: Optional[SchedulerSnapshot] = None


@dataclass(frozen=True)
class SummarySnapshot:
    since_reset: float
    models: Dict[str, ModelSnapshot]


def _copy(window: SampleWindow) -> WindowSnapshot:
    return WindowSnapshot(window.snapshot(), window.count)


class _SchedulerState:
    """A scheduler model's step measurements since the last reset."""

    def __init__(self, samples: int, first: StepStats):
        self.durations = SampleWindow(samples)
        self.batch_tokens = SampleWindow(samples)
        self.batch_sequences = SampleWindow(samples)
        self._current(first)
        self.clear()

    def _current(self, stats: StepStats) -> None:
        self.waiting = stats.waiting
        self.kv_blocks_used = stats.kv_blocks_used
        self.kv_blocks_total = stats.kv_blocks_total

    def clear(self) -> None:
        """Clear the period's totals and windows.

        The current queue and cache state survive, like requests running,
        and the maxima start again from them: a reset while the cache is
        full must not report that it was empty.
        """
        self.steps = 0
        self.memory_bound = 0
        self.gpu_seconds = 0.0
        self.overhead_seconds = 0.0
        self.prefill_tokens = 0
        self.decode_tokens = 0
        self.preemptions = 0
        self.preempted_tokens = 0
        self.blocked = dict.fromkeys(BLOCK_REASONS, 0)
        self.waiting_max = self.waiting
        self.kv_blocks_used_max = self.kv_blocks_used
        for window in (self.durations, self.batch_tokens, self.batch_sequences):
            window.clear()

    def add(self, stats: StepStats) -> None:
        self.steps += 1
        self.memory_bound += stats.bound == "memory"
        self.gpu_seconds += max(stats.compute, stats.memory)
        self.overhead_seconds += stats.overhead
        self.prefill_tokens += stats.prefill_tokens
        self.decode_tokens += stats.decode_tokens
        self.preemptions += stats.preempted
        self.preempted_tokens += stats.preempted_tokens
        if stats.blocked is not None:
            self.blocked[stats.blocked] += 1
        self._current(stats)
        self.waiting_max = max(self.waiting_max, stats.waiting)
        self.kv_blocks_used_max = max(self.kv_blocks_used_max, stats.kv_blocks_used)
        self.durations.add(stats.duration)
        self.batch_tokens.add(stats.prefill_tokens + stats.decode_tokens)
        self.batch_sequences.add(stats.sequences)

    def snapshot(self) -> SchedulerSnapshot:
        return SchedulerSnapshot(
            steps=self.steps,
            memory_bound=self.memory_bound,
            gpu_seconds=self.gpu_seconds,
            overhead_seconds=self.overhead_seconds,
            prefill_tokens=self.prefill_tokens,
            decode_tokens=self.decode_tokens,
            preemptions=self.preemptions,
            preempted_tokens=self.preempted_tokens,
            blocked=dict(self.blocked),
            waiting=self.waiting,
            waiting_max=self.waiting_max,
            kv_blocks_used=self.kv_blocks_used,
            kv_blocks_used_max=self.kv_blocks_used_max,
            kv_blocks_total=self.kv_blocks_total,
            durations=_copy(self.durations),
            batch_tokens=_copy(self.batch_tokens),
            batch_sequences=_copy(self.batch_sequences),
        )


class _ModelState:
    """Everything the summary keeps for one model."""

    def __init__(
        self,
        engine: Optional[str],
        request_samples: int,
        token_samples: int,
        rate_seconds: int,
        clock: Callable[[], float],
    ):
        self.engine = engine
        # In flight, so it survives a reset; see SummaryRecorder.reset().
        self.running = 0
        self.ttft = SampleWindow(request_samples)
        self.itl = SampleWindow(token_samples)
        self.lag = SampleWindow(token_samples)
        self.tpot = SampleWindow(request_samples)
        self.e2e = SampleWindow(request_samples)
        self.queue = SampleWindow(request_samples)
        self.prefill = SampleWindow(request_samples)
        self.decode = SampleWindow(request_samples)
        self.ttft_simulated = SampleWindow(request_samples)
        self.e2e_simulated = SampleWindow(request_samples)
        self.prompt_size = SampleWindow(request_samples)
        self.generation_size = SampleWindow(request_samples)
        self.request_rate = RateWindow(rate_seconds, clock)
        self.token_rate = RateWindow(rate_seconds, clock)
        # Created by the first step, so engines without steps have none.
        self.scheduler: Optional[_SchedulerState] = None
        self._step_samples = token_samples
        self.clear()

    def latencies(self) -> Dict[str, SampleWindow]:
        return {name: getattr(self, name) for name in LATENCIES}

    def sizes(self) -> Dict[str, SampleWindow]:
        return {"prompt": self.prompt_size, "generation": self.generation_size}

    def clear(self) -> None:
        self.finished = dict.fromkeys(FINISH_REASONS, 0)
        self.prompt_tokens = 0
        self.generation_tokens = 0
        for window in (*self.latencies().values(), *self.sizes().values()):
            window.clear()
        self.request_rate.clear()
        self.token_rate.clear()
        if self.scheduler is not None:
            self.scheduler.clear()

    def step(self, stats: StepStats) -> None:
        if self.scheduler is None:
            # Steps come about as often as tokens, so they share the
            # token window's capacity.
            self.scheduler = _SchedulerState(self._step_samples, stats)
        self.scheduler.add(stats)

    def snapshot(self, now: float) -> ModelSnapshot:
        def copy(windows):
            return {name: _copy(window) for name, window in windows.items()}

        return ModelSnapshot(
            engine=self.engine,
            running=self.running,
            finished=dict(self.finished),
            prompt_tokens=self.prompt_tokens,
            generation_tokens=self.generation_tokens,
            request_rate=self.request_rate.rate(now),
            token_rate=self.token_rate.rate(now),
            latencies=copy(self.latencies()),
            sizes=copy(self.sizes()),
            scheduler=None if self.scheduler is None else self.scheduler.snapshot(),
        )


class SummaryRecorder(MetricsRecorder):
    """Keeps recent raw measurements per model in bounded windows.

    Per-request values (TTFT, E2E, TPOT, queue, prefill, decode, sizes) keep
    the last request_samples requests; per-token values (ITL, lag) the last
    token_samples tokens across all requests. Inclusion follows the
    Prometheus store: cancelled requests count towards TTFT, ITL, lag and
    throughput, but only completed ones towards request-level timings.
    """

    def __init__(
        self,
        models: Mapping[str, ModelDescription],
        *,
        request_samples: int,
        token_samples: int,
        rate_seconds: int,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._request_samples = request_samples
        self._token_samples = token_samples
        self._rate_seconds = rate_seconds
        self._clock = clock
        self._since = clock()
        self._models: Dict[str, _ModelState] = {}
        for name, description in models.items():
            self._state(name, description.engine)

    def _state(self, model: str, engine: Optional[str] = None) -> _ModelState:
        """A model's state, created on first use for a model not configured."""
        state = self._models.get(model)
        if state is None:
            state = _ModelState(
                engine,
                self._request_samples,
                self._token_samples,
                self._rate_seconds,
                self._clock,
            )
            self._models[model] = state
        return state

    def request_started(self, model: str, prompt_tokens: int) -> None:
        state = self._state(model)
        state.running += 1
        state.prompt_tokens += prompt_tokens

    def first_token(self, model: str, ttft: float, queue_time: float) -> None:
        self._state(model).ttft.add(ttft)

    def token(self, model: str, itl: Optional[float], lag: float) -> None:
        state = self._state(model)
        if itl is not None:
            state.itl.add(itl)
        state.lag.add(lag)
        state.token_rate.add()

    def request_finished(
        self,
        model: str,
        finish_reason: str,
        prompt_tokens: int,
        generation_tokens: int,
        timings: Optional[RequestTimings],
    ) -> None:
        state = self._state(model)
        state.running -= 1
        state.finished[finish_reason] += 1
        state.generation_tokens += generation_tokens
        state.request_rate.add()
        if timings is None:
            return
        state.e2e.add(timings.e2e)
        state.queue.add(timings.queue)
        state.prefill.add(timings.prefill)
        state.decode.add(timings.decode)
        if timings.simulated_ttft is not None:
            state.ttft_simulated.add(timings.simulated_ttft)
        if timings.simulated_e2e is not None:
            state.e2e_simulated.add(timings.simulated_e2e)
        if timings.tpot is not None:
            state.tpot.add(timings.tpot)
        state.prompt_size.add(prompt_tokens)
        state.generation_size.add(generation_tokens)

    def step(self, model: str, stats: StepStats) -> None:
        self._state(model).step(stats)

    def reset(self) -> None:
        """Start a fresh measurement period.

        Clears every window, rate and count except requests running: those
        are still in flight and will finish, so zeroing them would drive the
        count negative. A request in flight across a reset is recorded in the
        new period when it finishes, so reset while the server is idle for a
        clean measurement.
        """
        self._since = self._clock()
        for state in self._models.values():
            state.clear()

    def snapshot(self) -> SummarySnapshot:
        """Copy the current state; cheap enough for the event loop."""
        now = self._clock()
        return SummarySnapshot(
            since_reset=now - self._since,
            models={name: state.snapshot(now) for name, state in self._models.items()},
        )


def summarize(snapshot: SummarySnapshot) -> Dict[str, Any]:
    """The JSON body of GET /metrics/summary; sorts every window."""

    def stats(windows: Dict[str, WindowSnapshot]):
        return {
            name: describe(window.values, window.count)
            for name, window in windows.items()
        }

    models = {}
    for name, model in snapshot.models.items():
        requests_per_second, window_seconds = model.request_rate
        tokens_per_second, _ = model.token_rate
        models[name] = {
            "engine": model.engine,
            "requests": {"running": model.running, "finished": model.finished},
            "tokens": {
                "prompt": model.prompt_tokens,
                "generation": model.generation_tokens,
            },
            "throughput": {
                "window_seconds": window_seconds,
                "requests_per_second": requests_per_second,
                "output_tokens_per_second": tokens_per_second,
            },
            "latency_seconds": stats(model.latencies),
            "tokens_per_request": stats(model.sizes),
        }
        if model.scheduler is not None:
            models[name]["scheduler"] = _scheduler(
                model.scheduler, snapshot.since_reset
            )
    return {"since_reset_seconds": snapshot.since_reset, "models": models}


def _scheduler(s: SchedulerSnapshot, since_reset: float) -> Dict[str, Any]:
    """The scheduler section: what the steps did since the reset."""
    busy = s.gpu_seconds + s.overhead_seconds

    def ratio(part, whole):
        return part / whole if whole else None

    def window(snapshot):
        return describe(snapshot.values, snapshot.count)

    return {
        "steps": s.steps,
        # Simulated GPU time over the period's wall time; near 1, the GPU
        # had no idle time left.
        "busy_fraction": ratio(busy, since_reset),
        "overhead_fraction": ratio(s.overhead_seconds, busy),
        "memory_bound_fraction": ratio(s.memory_bound, s.steps),
        "step_seconds": window(s.durations),
        "batch_tokens": window(s.batch_tokens),
        "batch_sequences": window(s.batch_sequences),
        "tokens": {"prefill": s.prefill_tokens, "decode": s.decode_tokens},
        "waiting": {"now": s.waiting, "max": s.waiting_max},
        "kv_cache": {
            "blocks": s.kv_blocks_total,
            "usage": s.kv_blocks_used / s.kv_blocks_total,
            "usage_max": s.kv_blocks_used_max / s.kv_blocks_total,
        },
        "preemptions": {"count": s.preemptions, "tokens": s.preempted_tokens},
        "blocked_steps": s.blocked,
    }
