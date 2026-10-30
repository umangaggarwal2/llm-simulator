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
    FINISH_REASONS,
    MetricsRecorder,
    ModelDescription,
    RequestTimings,
)
from llm_simulator.metrics.window import RateWindow, SampleWindow, describe

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


@dataclass(frozen=True)
class SummarySnapshot:
    since_reset: float
    models: Dict[str, ModelSnapshot]


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

    def snapshot(self, now: float) -> ModelSnapshot:
        def copy(windows):
            return {
                name: WindowSnapshot(window.snapshot(), window.count)
                for name, window in windows.items()
            }

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
    return {"since_reset_seconds": snapshot.since_reset, "models": models}
