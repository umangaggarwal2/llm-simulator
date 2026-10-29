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
    FINISH_REASONS,
    MetricsRecorder,
    ModelDescription,
    RequestTimings,
)

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


class PrometheusRecorder(MetricsRecorder):
    """Records into a registry of its own, so apps and tests never share metrics."""

    content_type = CONTENT_TYPE_LATEST

    def __init__(self, models: Mapping[str, ModelDescription]):
        self.registry = CollectorRegistry()
        largest_context = max((m.context_length for m in models.values()), default=1)
        token_buckets = log_buckets(1, largest_context, 2)

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
            "Requests currently producing their completion.",
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
