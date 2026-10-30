"""Turning a run's records into the numbers a benchmark reports.

Pure functions over records: no network, so a run can be re-analysed and the
analysis tested with hand-made records. Latency and throughput cover
successful requests only; failures are counted and grouped separately, since
the timing of a request that failed part-way means nothing.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from llm_simulator.bench.config import BenchConfig
from llm_simulator.bench.records import LoopSamples, RequestRecord, RunRecords
from llm_simulator.metrics.window import describe

Stats = Dict[str, Optional[float]]

# An idle asyncio loop wakes about 1 ms late; beyond this at p99, the client
# was busy enough to delay the timestamps it took.
SATURATION_LAG = 0.005

# Metrics measured by both the client and the server.
COMPARED = ("ttft", "itl", "tpot", "e2e")
# Statistics whose difference is meaningful; min and max are single values,
# too noisy to subtract.
DELIVERY_KEYS = ("mean", "p50", "p90", "p95", "p99")
# The server's summary names for the engine's own timeline.
SIMULATED = {"ttft": "ttft_simulated", "e2e": "e2e_simulated"}


@dataclass(frozen=True)
class Throughput:
    requests_per_second: float
    output_tokens_per_second: float
    # Prompt and output tokens together.
    total_tokens_per_second: float


@dataclass(frozen=True)
class ClientHealth:
    # How late the client's event loop woke, in seconds.
    loop_lag: Stats
    # Client CPU time over wall time; near 100, its one core was the limit.
    cpu_percent: float
    saturated: bool


@dataclass(frozen=True)
class Arrivals:
    """How the open loop's schedule was kept."""

    target_rate: float
    # Arrivals per second as scheduled, and as actually sent: the schedule
    # is random, so a short run's offered rate differs from the target, and
    # sends fall behind it when requests wait for a slot or the client lags.
    offered_rate: float
    achieved_rate: float
    # Sent minus scheduled, for every request: near zero unless the request
    # waited for a slot (max_concurrency) or the client was late.
    queue_wait: Stats
    waited: int

    @property
    def keeping_up(self) -> bool:
        return self.achieved_rate >= 0.95 * self.offered_rate


@dataclass(frozen=True)
class ServerComparison:
    url: str
    # The server's statistics for COMPARED and "lag", in seconds.
    server: Dict[str, Stats]
    # The engine's timeline for "ttft" and "e2e"; empty if the server does
    # not report it.
    simulated: Dict[str, Stats]
    # Client minus server, per metric and DELIVERY_KEYS statistic: a
    # difference of percentiles, since the summary has no per-request data
    # to pair with the client's.
    delivery: Dict[str, Dict[str, Optional[float]]]
    # Requests the server finished, for any reason, and client requests it
    # accepted (HTTP 200); rejected requests are never traced by the server.
    server_requests: int
    accepted_requests: int
    # The server's generated tokens over the client's window.
    server_output_tokens_per_second: float

    @property
    def counts_match(self) -> bool:
        return self.server_requests == self.accepted_requests


@dataclass(frozen=True)
class BenchResult:
    config: BenchConfig
    workload: str
    run: RunRecords
    requests: int
    ok: int
    failed: int
    finish_reasons: Dict[str, int]
    # Failure description -> count, most frequent first.
    errors: Dict[str, int]
    # Totals from usage, over successful requests.
    prompt_tokens: int
    output_tokens: int
    throughput: Throughput
    # ttft, itl, tpot, e2e: describe() statistics in seconds. ITL pools every
    # gap of every request; the others have one value per request.
    latency: Dict[str, Stats]
    # prompt, output: describe() statistics in tokens.
    tokens_per_request: Dict[str, Stats]
    # Successful requests that returned fewer tokens than they asked for.
    shortfall: int
    # Successful requests whose content chunks differ from usage's token
    # count, so ITL is per chunk rather than per token.
    chunk_mismatch: int
    # None when the run was not monitored.
    client: Optional[ClientHealth] = None
    # None when the server was not compared; server_note then says why, if
    # a comparison was attempted.
    server: Optional[ServerComparison] = None
    server_note: Optional[str] = None
    # Open loop only: the schedule, and TTFT and E2E measured from each
    # request's scheduled time (what an arriving user waits), next to
    # `latency`, which is measured from the actual send like the server's.
    arrivals: Optional[Arrivals] = None
    latency_from_schedule: Optional[Dict[str, Stats]] = None

    @property
    def warmup(self) -> int:
        return self.run.warmup

    @property
    def duration(self) -> float:
        return self.run.duration


def summarize(run: RunRecords, config: BenchConfig, workload: str) -> BenchResult:
    ok = [record for record in run.records if record.ok]
    failed = [record for record in run.records if not record.ok]

    ttfts = [record.ttft for record in ok if record.ttft is not None]
    itls = [gap for record in ok for gap in record.itls]
    tpots = [
        (record.last_token - record.first_token) / (record.chunks - 1)
        for record in ok
        if record.chunks > 1
    ]
    e2es = [record.e2e for record in ok]
    prompts = [
        record.prompt_tokens for record in ok if record.prompt_tokens is not None
    ]
    outputs = [
        record.completion_tokens
        for record in ok
        if record.completion_tokens is not None
    ]

    prompt_tokens, output_tokens = sum(prompts), sum(outputs)
    # A run that sent nothing has no duration to divide by.
    duration = run.duration if run.duration > 0 else float("inf")
    latency = {
        "ttft": _describe(ttfts),
        "itl": _describe(itls),
        "tpot": _describe(tpots),
        "e2e": _describe(e2es),
    }
    server, server_note = compare(latency, run)
    arrivals, from_schedule = _schedule(run, ok, config)
    return BenchResult(
        config=config,
        workload=workload,
        run=run,
        requests=len(run.records),
        ok=len(ok),
        failed=len(failed),
        finish_reasons=dict(Counter(record.finish_reason for record in ok)),
        errors=dict(Counter(failure(record) for record in failed).most_common()),
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        throughput=Throughput(
            requests_per_second=len(ok) / duration,
            output_tokens_per_second=output_tokens / duration,
            total_tokens_per_second=(prompt_tokens + output_tokens) / duration,
        ),
        latency=latency,
        tokens_per_request={"prompt": _describe(prompts), "output": _describe(outputs)},
        shortfall=sum(
            1
            for record in ok
            if record.completion_tokens is not None
            and record.requested_tokens is not None
            and record.completion_tokens < record.requested_tokens
        ),
        chunk_mismatch=sum(
            1
            for record in ok
            if record.completion_tokens is not None
            and record.chunks != record.completion_tokens
        ),
        client=None if run.loop is None else client_health(run.loop),
        server=server,
        server_note=server_note,
        arrivals=arrivals,
        latency_from_schedule=from_schedule,
    )


def _schedule(
    run: RunRecords, ok: List[RequestRecord], config: BenchConfig
) -> Tuple[Optional[Arrivals], Optional[Dict[str, Stats]]]:
    scheduled = [record for record in run.records if record.scheduled is not None]
    if config.rate is None or not scheduled:
        return None, None
    last_scheduled = max(record.scheduled for record in scheduled) - run.started
    last_sent = max(record.sent for record in scheduled) - run.started
    arrivals = Arrivals(
        target_rate=config.rate,
        offered_rate=len(scheduled) / last_scheduled if last_scheduled > 0 else 0.0,
        achieved_rate=len(scheduled) / last_sent if last_sent > 0 else 0.0,
        queue_wait=_describe([record.queue_wait for record in scheduled]),
        waited=sum(1 for record in scheduled if record.waited),
    )
    from_schedule = {
        "ttft": _describe(
            [r.ttft_from_schedule for r in ok if r.ttft_from_schedule is not None]
        ),
        "e2e": _describe(
            [r.e2e_from_schedule for r in ok if r.e2e_from_schedule is not None]
        ),
    }
    return arrivals, from_schedule


def compare(
    latency: Dict[str, Stats], run: RunRecords
) -> Tuple[Optional[ServerComparison], Optional[str]]:
    """Set the client's statistics against the server's summary, if there is one.

    Reads the summary defensively: a missing field becomes a missing value,
    not an error, so a different summary shape degrades the comparison
    instead of the report.
    """
    summary = run.server
    if summary is None:
        return None, None
    if not summary.available or summary.model is None:
        return None, summary.reason
    model = summary.model
    server_latency = _dict(model.get("latency_seconds"))
    server = {name: _dict(server_latency.get(name)) for name in (*COMPARED, "lag")}
    simulated = {
        name: _dict(server_latency[key])
        for name, key in SIMULATED.items()
        if isinstance(server_latency.get(key), dict)
    }
    delivery = {
        name: {
            key: _difference(latency[name].get(key), server[name].get(key))
            for key in DELIVERY_KEYS
        }
        for name in COMPARED
    }
    finished = _dict(_dict(model.get("requests")).get("finished"))
    generated = _dict(model.get("tokens")).get("generation") or 0
    duration = run.duration if run.duration > 0 else float("inf")
    return (
        ServerComparison(
            url=summary.url,
            server=server,
            simulated=simulated,
            delivery=delivery,
            server_requests=sum(v for v in finished.values() if isinstance(v, int)),
            accepted_requests=sum(1 for record in run.records if record.status == 200),
            server_output_tokens_per_second=generated / duration,
        ),
        None,
    )


def _dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _difference(client: Optional[float], server: Optional[float]) -> Optional[float]:
    if client is None or server is None:
        return None
    return client - server


def client_health(loop: LoopSamples) -> ClientHealth:
    lag = _describe(list(loop.lags))
    return ClientHealth(
        loop_lag=lag,
        cpu_percent=(
            100 * loop.cpu_seconds / loop.wall_seconds if loop.wall_seconds > 0 else 0.0
        ),
        saturated=lag["p99"] is not None and lag["p99"] > SATURATION_LAG,
    )


def failure(record: RequestRecord) -> str:
    """How a failed request is described and grouped: status and message."""
    status = f"HTTP {record.status}" if record.status else "no response"
    return f"{status}: {record.error}"


def _describe(values: List[float]) -> Stats:
    return describe(values, len(values))
