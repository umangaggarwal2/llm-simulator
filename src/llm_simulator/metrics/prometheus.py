"""Metrics kept in a Prometheus registry and exposed in its text format.

The registry holds only current values: counters, gauges, and bucket counts
plus a sum and count per histogram, for each model. Memory is constant
regardless of traffic. History is kept by whatever scrapes /metrics, and
throughput comes from rate() over the counters.
"""

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import prometheus_client
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

from llm_simulator.metrics.recorder import (
    BLOCK_REASONS,
    FINISH_REASONS,
    MetricsRecorder,
    ModelDescription,
    RequestTimings,
)
from llm_simulator.types import StepStats

# Without this, every counter and histogram also exports a *_created series,
# roughly doubling the output; rate() and dashboards do not use them. The
# setting is process-wide in prometheus_client.
prometheus_client.disable_created_metrics()

NAMESPACE = "llm_sim"


def log_buckets(start: float, stop: float, factor: float) -> Tuple[float, ...]:
    """Bucket bounds from start, each factor times the last, until one reaches stop."""
    bounds = [start]
    while bounds[-1] < stop:
        bounds.append(bounds[-1] * factor)
    return tuple(round(bound, 9) for bound in bounds)


# 1 ms to 60 s, 25% apart.
LATENCY_BUCKETS = log_buckets(0.001, 60.0, 1.25)
# 0.1 ms to 5 s: decode steps are short.
ITL_BUCKETS = log_buckets(0.0001, 5.0, 1.25)
# 10 µs to 10 s: fine enough to show millisecond timer jitter.
LAG_BUCKETS = log_buckets(0.00001, 10.0, 1.5)
# Tokens computed in one step, and sequences in one batch, doubling.
STEP_TOKEN_BUCKETS = log_buckets(1, 32768, 2)
BATCH_BUCKETS = log_buckets(1, 1024, 2)

# Label values of the scheduler's metrics, all exported from a model's first
# step so that rates start from 0.
BOUNDS = ("compute", "memory")
BUSY_PARTS = ("gpu", "overhead")
PHASES = ("prefill", "decode")


@dataclass(frozen=True)
class _ModelMetrics:
    """The label children for one model, looked up once instead of per call."""

    ttft: Histogram
    itl: Histogram
    tpot: Histogram
    e2e: Histogram
    queue: Histogram
    prefill: Histogram
    decode: Histogram
    prompt_size: Histogram
    generation_size: Histogram
    lag: Histogram
    requests: Dict[str, Counter]
    prompt_tokens: Counter
    generation_tokens: Counter
    running: Gauge


@dataclass(frozen=True)
class _SchedulerMetrics:
    """The label children for one scheduler model, created on its first step."""

    waiting: Gauge
    running: Gauge
    kv_usage: Gauge
    kv_blocks: Gauge
    steps: Dict[str, Counter]
    busy: Dict[str, Counter]
    step_tokens: Dict[str, Counter]
    preemptions: Counter
    preempted_tokens: Counter
    blocked: Dict[str, Counter]
    duration: Histogram
    tokens_per_step: Histogram
    batch: Histogram


class PrometheusRecorder(MetricsRecorder):
    """Records into a registry of its own, so apps and tests never share metrics."""

    content_type = CONTENT_TYPE_LATEST

    def __init__(self, models: Mapping[str, ModelDescription]):
        self.registry = CollectorRegistry()
        largest_context = max((m.context_length for m in models.values()), default=1)
        token_buckets = log_buckets(1, largest_context, 2)

        def gauge(name, doc, labels=("model",)):
            return Gauge(name, doc, labels, namespace=NAMESPACE, registry=self.registry)

        def histogram(name, doc, buckets):
            return Histogram(
                name,
                doc,
                ["model"],
                namespace=NAMESPACE,
                buckets=buckets,
                registry=self.registry,
            )

        def counter(name, doc, labels=("model",)):
            return Counter(
                name, doc, labels, namespace=NAMESPACE, registry=self.registry
            )

        ttft = histogram(
            "time_to_first_token_seconds",
            "Time from arrival until the engine released the first token.",
            LATENCY_BUCKETS,
        )
        itl = histogram(
            "inter_token_latency_seconds",
            "Time between consecutive tokens of a request.",
            ITL_BUCKETS,
        )
        tpot = histogram(
            "request_time_per_output_token_seconds",
            "Mean time per output token after the first, for completed requests.",
            ITL_BUCKETS,
        )
        e2e = histogram(
            "e2e_request_latency_seconds",
            "Time from arrival until the last token, for completed requests.",
            LATENCY_BUCKETS,
        )
        queue = histogram(
            "request_queue_time_seconds",
            "Time a completed request waited before processing started.",
            LATENCY_BUCKETS,
        )
        prefill = histogram(
            "request_prefill_time_seconds",
            "Time from the end of queueing until the first token, for completed requests.",
            LATENCY_BUCKETS,
        )
        decode = histogram(
            "request_decode_time_seconds",
            "Time from the first token until the last, for completed requests.",
            LATENCY_BUCKETS,
        )
        prompt_size = histogram(
            "request_prompt_tokens",
            "Prompt tokens of completed requests.",
            token_buckets,
        )
        generation_size = histogram(
            "request_generation_tokens",
            "Generated tokens of completed requests.",
            token_buckets,
        )
        lag = histogram(
            "token_lag_seconds",
            "How late the engine released each token against its own timeline; "
            "the simulator's fidelity error.",
            LAG_BUCKETS,
        )
        requests = counter(
            "requests",
            "Requests that finished, by finish reason.",
            ("model", "finish_reason"),
        )
        prompt_tokens = counter("prompt_tokens", "Prompt tokens processed.")
        generation_tokens = counter("generation_tokens", "Tokens generated.")
        running = Gauge(
            "requests_running",
            "Requests in flight: from the start of their completion until it "
            "ends, including any time spent waiting in a scheduler's queue.",
            ["model"],
            namespace=NAMESPACE,
            registry=self.registry,
        )
        info = Info(
            "model",
            "The engine and context length each model is served with.",
            ["model"],
            namespace=NAMESPACE,
            registry=self.registry,
        )

        self._families = dict(
            ttft=ttft,
            itl=itl,
            tpot=tpot,
            e2e=e2e,
            queue=queue,
            prefill=prefill,
            decode=decode,
            prompt_size=prompt_size,
            generation_size=generation_size,
            lag=lag,
            prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens,
            running=running,
        )
        self._requests = requests
        self._models: Dict[str, _ModelMetrics] = {}

        # Scheduler engines only: one sample per step, from the step loop.
        self._scheduler_families = dict(
            sequences=gauge(
                "scheduler_sequences",
                "Sequences in the scheduler by state: waiting for admission, or "
                "running in the batch. requests_running counts requests in "
                "flight instead, waiting ones included.",
                ("model", "state"),
            ),
            kv_usage=gauge(
                "kv_cache_usage_ratio", "Share of KV-cache blocks in use, from 0 to 1."
            ),
            kv_blocks=gauge("kv_cache_blocks", "KV-cache capacity in blocks."),
            steps=counter(
                "scheduler_steps",
                "Scheduler steps, by what set their GPU time: compute or memory.",
                ("model", "bound"),
            ),
            busy=counter(
                "scheduler_busy_seconds",
                "Simulated time the GPU spent on steps: on the GPU work, and on "
                "the per-step CPU overhead. rate() is the GPU's utilisation.",
                ("model", "part"),
            ),
            step_tokens=counter(
                "scheduler_step_tokens",
                "Tokens computed by scheduler steps, by phase. Prefill includes "
                "recomputation after preemption.",
                ("model", "phase"),
            ),
            preemptions=counter(
                "preemptions", "Sequences preempted because the KV cache was full."
            ),
            preempted_tokens=counter(
                "preempted_tokens",
                "Tokens whose KV preemption threw away, to be computed again.",
            ),
            blocked=counter(
                "scheduler_blocked_steps",
                "Steps that left requests waiting, by the reason admission stopped.",
                ("model", "reason"),
            ),
            duration=histogram(
                "scheduler_step_duration_seconds",
                "Simulated duration of each scheduler step.",
                ITL_BUCKETS,
            ),
            tokens_per_step=histogram(
                "scheduler_batch_tokens",
                "Tokens computed in each scheduler step, prefill and decode.",
                STEP_TOKEN_BUCKETS,
            ),
            batch=histogram(
                "scheduler_batch_sequences",
                "Sequences in each scheduler step.",
                BATCH_BUCKETS,
            ),
        )
        self._schedulers: Dict[str, _SchedulerMetrics] = {}
        for name, description in models.items():
            info.labels(name).info(
                {
                    "engine": description.engine,
                    "context_length": str(description.context_length),
                }
            )
            self._children(name)

    def exposition(self) -> bytes:
        """The registry in Prometheus text format, for GET /metrics."""
        return generate_latest(self.registry)

    def _children(self, model: str) -> _ModelMetrics:
        """The label children for a model, created on first use.

        Configured models get theirs at startup, so the hot path is a dict
        lookup; a model the recorder was not told about (for example a
        runtime swapped in by a test) still gets recorded rather than failing
        the request.
        """
        metrics = self._models.get(model)
        if metrics is None:
            metrics = _ModelMetrics(
                **{
                    field: family.labels(model)
                    for field, family in self._families.items()
                },
                # Created up front so every reason is exported, at 0 until used.
                requests={
                    reason: self._requests.labels(model, reason)
                    for reason in FINISH_REASONS
                },
            )
            self._models[model] = metrics
        return metrics

    def request_started(self, model: str, prompt_tokens: int) -> None:
        metrics = self._children(model)
        metrics.running.inc()
        metrics.prompt_tokens.inc(prompt_tokens)

    def first_token(self, model: str, ttft: float, queue_time: float) -> None:
        self._children(model).ttft.observe(ttft)

    def token(self, model: str, itl: Optional[float], lag: float) -> None:
        metrics = self._children(model)
        if itl is not None:
            metrics.itl.observe(itl)
        metrics.lag.observe(lag)

    def request_finished(
        self,
        model: str,
        finish_reason: str,
        prompt_tokens: int,
        generation_tokens: int,
        timings: Optional[RequestTimings],
    ) -> None:
        metrics = self._children(model)
        metrics.running.dec()
        metrics.requests[finish_reason].inc()
        # Counted for every request: these tokens were produced either way.
        metrics.generation_tokens.inc(generation_tokens)
        if timings is None:
            return
        metrics.e2e.observe(timings.e2e)
        metrics.queue.observe(timings.queue)
        metrics.prefill.observe(timings.prefill)
        metrics.decode.observe(timings.decode)
        if timings.tpot is not None:
            metrics.tpot.observe(timings.tpot)
        metrics.prompt_size.observe(prompt_tokens)
        metrics.generation_size.observe(generation_tokens)

    def step(self, model: str, stats: StepStats) -> None:
        metrics = self._scheduler_children(model)
        metrics.waiting.set(stats.waiting)
        metrics.running.set(stats.running)
        metrics.kv_usage.set(stats.kv_blocks_used / stats.kv_blocks_total)
        metrics.kv_blocks.set(stats.kv_blocks_total)
        metrics.steps[stats.bound].inc()
        metrics.busy["gpu"].inc(max(stats.compute, stats.memory))
        metrics.busy["overhead"].inc(stats.overhead)
        metrics.step_tokens["prefill"].inc(stats.prefill_tokens)
        metrics.step_tokens["decode"].inc(stats.decode_tokens)
        if stats.preempted:
            metrics.preemptions.inc(stats.preempted)
            metrics.preempted_tokens.inc(stats.preempted_tokens)
        if stats.blocked is not None:
            metrics.blocked[stats.blocked].inc()
        metrics.duration.observe(stats.duration)
        metrics.tokens_per_step.observe(stats.prefill_tokens + stats.decode_tokens)
        metrics.batch.observe(stats.sequences)

    def _scheduler_children(self, model: str) -> _SchedulerMetrics:
        """A model's scheduler children, created on its first step.

        Only engines with steps report them, so other models export none of
        these series without the recorder knowing engine names.
        """
        metrics = self._schedulers.get(model)
        if metrics is None:
            f = self._scheduler_families

            def by(family, values):
                return {value: family.labels(model, value) for value in values}

            metrics = _SchedulerMetrics(
                waiting=f["sequences"].labels(model, "waiting"),
                running=f["sequences"].labels(model, "running"),
                kv_usage=f["kv_usage"].labels(model),
                kv_blocks=f["kv_blocks"].labels(model),
                steps=by(f["steps"], BOUNDS),
                busy=by(f["busy"], BUSY_PARTS),
                step_tokens=by(f["step_tokens"], PHASES),
                preemptions=f["preemptions"].labels(model),
                preempted_tokens=f["preempted_tokens"].labels(model),
                blocked=by(f["blocked"], BLOCK_REASONS),
                duration=f["duration"].labels(model),
                tokens_per_step=f["tokens_per_step"].labels(model),
                batch=f["batch"].labels(model),
            )
            self._schedulers[model] = metrics
        return metrics
