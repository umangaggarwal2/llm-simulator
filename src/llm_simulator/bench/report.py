"""Presenting a BenchResult: a terminal table and a JSON results file."""

import dataclasses
import json
import platform
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_simulator.bench.results import BenchResult
from llm_simulator.bench.sweep import SweepLevel

# Bumped when a field changes meaning, so results files stay comparable.
FORMAT_VERSION = 1
LATENCY_ROWS = (("TTFT", "ttft"), ("ITL", "itl"), ("TPOT", "tpot"), ("E2E", "e2e"))
STAT_COLUMNS = ("mean", "min", "p50", "p90", "p95", "p99", "max")
TOKEN_COLUMNS = ("mean", "min", "p50", "max")
TOP_ERRORS = 5
# Rows of the client-against-server table.
COMPARISON_ROWS = (
    ("TTFT", "ttft", "p50"),
    ("TTFT", "ttft", "p99"),
    ("ITL", "itl", "p50"),
    ("ITL", "itl", "p99"),
    ("TPOT", "tpot", "p50"),
    ("E2E", "e2e", "p50"),
    ("E2E", "e2e", "p99"),
)


def format_result(result: BenchResult) -> str:
    config = result.config
    throughput = result.throughput
    finishes = "".join(
        f" · finish {reason} {count}"
        for reason, count in sorted(result.finish_reasons.items(), key=str)
    )
    duration = f"duration    {result.duration:.2f} s"
    if config.duration is not None:
        duration += f" (target {config.duration:g} s)"
    lines = [
        f"{config.model} @ {config.base_url} · {describe_load(config)}"
        f" · warm-up {result.warmup} · {result.requests} requests · {result.workload}",
        f"requests    {result.ok} ok · {result.failed} failed{finishes}",
        duration,
        f"throughput  {throughput.requests_per_second:.2f} req/s"
        f" · {throughput.output_tokens_per_second:,.0f} output tok/s"
        f" · {throughput.total_tokens_per_second:,.0f} total tok/s",
    ]
    if result.client is not None:
        lag = result.client.loop_lag
        lines.append(
            f"client      loop lag p50 {_seconds(lag['p50'])} · p99 {_seconds(lag['p99'])}"
            f" · max {_seconds(lag['max'])} · CPU {result.client.cpu_percent:.0f}%"
            f" · {'saturated' if result.client.saturated else 'healthy'}"
        )
    arrivals = result.arrivals
    if arrivals is not None:
        wait = arrivals.queue_wait
        lines.append(
            f"arrivals    target {arrivals.target_rate:g}/s · offered"
            f" {arrivals.offered_rate:.2f}/s · achieved {arrivals.achieved_rate:.2f}/s"
            f" · queue wait p50 {_seconds(wait['p50'])} · p99 {_seconds(wait['p99'])}"
            f" · {arrivals.waited} waited"
        )
    server = result.server
    if server is not None:
        mark = "✓" if server.counts_match else "≠"
        lines.append(
            f"server      {server.url} · {server.server_requests} requests"
            f" (client {server.accepted_requests} {mark})"
        )
    elif result.server_note is not None:
        lines.append(f"server      not available ({result.server_note})")
    lines += [
        "",
        _row("latency", STAT_COLUMNS),
    ]
    for label, key in LATENCY_ROWS:
        stats = result.latency[key]
        lines.append(_row(f"  {label}", [_seconds(stats[c]) for c in STAT_COLUMNS]))
    if result.latency_from_schedule is not None:
        # Open loop: what an arriving user waits, including any queue wait.
        for label, key in (("TTFT sched", "ttft"), ("E2E sched", "e2e")):
            stats = result.latency_from_schedule[key]
            lines.append(_row(f"  {label}", [_seconds(stats[c]) for c in STAT_COLUMNS]))
        wait = result.arrivals.queue_wait
        lines.append(_row("  queue wait", [_seconds(wait[c]) for c in STAT_COLUMNS]))
    lines.append(_row("tokens/req", TOKEN_COLUMNS))
    for key in ("prompt", "output"):
        stats = result.tokens_per_request[key]
        lines.append(_row(f"  {key}", [_tokens(stats[c]) for c in TOKEN_COLUMNS]))
    if server is not None:
        lines += [""] + _comparison(result)

    warnings = _warnings(result)
    if warnings:
        lines += ["", "warnings"] + [f"  {warning}" for warning in warnings]
    if result.errors:
        lines += ["", "errors"]
        for message, count in list(result.errors.items())[:TOP_ERRORS]:
            lines.append(f"  {count} × {message}")
        hidden = len(result.errors) - TOP_ERRORS
        if hidden > 0:
            lines.append(f"  … and {hidden} other kinds")
    return "\n".join(lines)


def to_json(result: BenchResult) -> Dict[str, Any]:
    """The results file: everything needed to compare this run with another."""
    config = dataclasses.asdict(result.config)
    if config["api_key"] is not None:
        config["api_key"] = "<redacted>"
    if config["dataset"] is not None:
        config["dataset"] = str(config["dataset"])
    started_at = result.run.started_at
    return {
        "version": FORMAT_VERSION,
        "config": config,
        "workload": result.workload,
        # No hostname: results files get shared.
        "environment": {
            "started_at": started_at.isoformat() if started_at else None,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "llm_simulator": _version("llm-simulator"),
            "aiohttp": _version("aiohttp"),
        },
        "result": {
            "duration_seconds": result.duration,
            "warmup_requests": result.warmup,
            "client": (
                None if result.client is None else dataclasses.asdict(result.client)
            ),
            "requests": {
                "total": result.requests,
                "ok": result.ok,
                "failed": result.failed,
                "finish_reasons": result.finish_reasons,
            },
            "tokens": {"prompt": result.prompt_tokens, "output": result.output_tokens},
            "throughput": dataclasses.asdict(result.throughput),
            "latency_seconds": result.latency,
            "tokens_per_request": result.tokens_per_request,
            "warnings": _warnings(result),
            "errors": result.errors,
            "server": _server_json(result),
            "arrivals": (
                None if result.arrivals is None else dataclasses.asdict(result.arrivals)
            ),
            "latency_from_schedule_seconds": result.latency_from_schedule,
        },
    }


def describe_load(config) -> str:
    if config.mode == "open":
        text = f"open loop · rate {config.rate:g}/s (Poisson, seed {config.seed})"
        if config.max_concurrency is not None:
            text += f" · max concurrency {config.max_concurrency}"
        return text
    return f"closed loop · concurrency {config.closed_concurrency}"


def _comparison(result: BenchResult) -> List[str]:
    server = result.server
    simulated = bool(server.simulated)
    header = ["client", "server"] + (["simulated"] if simulated else []) + ["delivery*"]
    lines = [_row("vs server", header)]
    for label, name, key in COMPARISON_ROWS:
        cells = [
            _seconds(result.latency[name].get(key)),
            _seconds(server.server[name].get(key)),
        ]
        if simulated:
            cells.append(_seconds(server.simulated.get(name, {}).get(key)))
        cells.append(_seconds(server.delivery[name].get(key)))
        lines.append(_row(f"  {label} {key}", cells))
    lines.append(
        _row(
            "  out tok/s",
            [
                f"{result.throughput.output_tokens_per_second:,.0f}",
                f"{server.server_output_tokens_per_second:,.0f}",
            ],
        )
    )
    lag = server.server["lag"]
    lines.append(
        f"fidelity    server lag p50 {_seconds(lag.get('p50'))}"
        f" · p99 {_seconds(lag.get('p99'))} · max {_seconds(lag.get('max'))}"
    )
    lines.append("* client percentile − server percentile")
    return lines


def _server_json(result: BenchResult) -> Optional[Dict[str, Any]]:
    summary = result.run.server
    if summary is None:
        return None
    data: Dict[str, Any] = {
        "url": summary.url,
        "available": summary.available,
        "reason": summary.reason,
        "since_reset_seconds": summary.since_reset,
        "summary": summary.model,
    }
    if result.server is not None:
        data["requests"] = {
            "server": result.server.server_requests,
            "client_accepted": result.server.accepted_requests,
        }
        data["delivery_seconds"] = result.server.delivery
    return data


def write_json(result: BenchResult, path: Path) -> None:
    Path(path).write_text(
        json.dumps(to_json(result), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _warnings(result: BenchResult) -> List[str]:
    warnings = []
    if result.shortfall:
        warnings.append(
            f"{result.shortfall} successful requests returned fewer tokens than"
            " requested: the server's response ended before max_tokens"
        )
    if result.client is not None and result.client.saturated:
        warnings.append(
            "client event loop was late by"
            f" {_seconds(result.client.loop_lag['p99'])} at p99"
            f" (CPU {result.client.cpu_percent:.0f}%): client-side latencies are"
            " inflated by up to that much; lower the load"
        )
    arrivals = result.arrivals
    if arrivals is not None and arrivals.waited:
        limit = result.config.max_concurrency
        warnings.append(
            f"{arrivals.waited} requests waited for a free slot (max concurrency"
            f" {limit}): the server cannot keep up at this rate; see TTFT sched"
        )
    if arrivals is not None and not arrivals.keeping_up:
        warnings.append(
            f"offered {arrivals.offered_rate:.2f}/s, achieved"
            f" {arrivals.achieved_rate:.2f}/s: sends fell behind the schedule"
        )
    if result.server is not None and not result.server.counts_match:
        warnings.append(
            f"server finished {result.server.server_requests} requests, client had"
            f" {result.server.accepted_requests} accepted: other traffic reached"
            " the server during the run, so its figures are mixed"
        )
    if result.chunk_mismatch:
        warnings.append(
            f"content chunks differ from usage tokens in {result.chunk_mismatch}"
            " requests; ITL is measured per chunk"
        )
    return warnings


def _row(label: str, cells) -> str:
    return f"{label:<12}" + "".join(f"{cell:>10}" for cell in cells)


def _seconds(value: Optional[float]) -> str:
    if value is None:
        return "—"
    if value < 1:
        return f"{value * 1000:.2f} ms"
    return f"{value:.3f} s"


def _tokens(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value:,.0f}" if value == int(value) else f"{value:,.1f}"


def _version(package: str) -> Optional[str]:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


# Sweep table columns: header, and the cell for one level's result.
SWEEP_COLUMNS = (
    ("req/s", lambda r: f"{r.throughput.requests_per_second:.2f}"),
    ("out tok/s", lambda r: f"{r.throughput.output_tokens_per_second:,.0f}"),
    ("TTFT p50", lambda r: _seconds(r.latency["ttft"]["p50"])),
    ("TTFT p99", lambda r: _seconds(r.latency["ttft"]["p99"])),
    ("ITL p50", lambda r: _seconds(r.latency["itl"]["p50"])),
    ("E2E p50", lambda r: _seconds(r.latency["e2e"]["p50"])),
    (
        "lag p99",
        lambda r: _seconds(r.server.server["lag"].get("p99")) if r.server else "—",
    ),
    (
        "dlv TTFT",
        lambda r: _seconds(r.server.delivery["ttft"].get("p50")) if r.server else "—",
    ),
    (
        "client",
        lambda r: "—" if r.client is None else ("SAT" if r.client.saturated else "ok"),
    ),
    ("errors", lambda r: str(r.failed)),
)
# Extra open-loop columns: how the schedule was kept, and what users waited.
OPEN_SWEEP_COLUMNS = (
    ("achieved/s", lambda r: f"{r.arrivals.achieved_rate:.2f}" if r.arrivals else "—"),
    (
        "sched p99",
        lambda r: (
            _seconds(r.latency_from_schedule["ttft"]["p99"])
            if r.latency_from_schedule
            else "—"
        ),
    ),
    ("waited", lambda r: str(r.arrivals.waited) if r.arrivals else "—"),
)


def format_progress(number: int, total: int, level: SweepLevel) -> str:
    head = f"level {number}/{total} · {level.parameter} {level.value:g}"
    if level.result is None:
        return f"{head} · failed: {level.error}"
    result = level.result
    return f"{head} · {result.duration:.1f} s · {result.ok} ok · {result.failed} failed"


def format_sweep(levels: List[SweepLevel], workload: str) -> str:
    """One row per level, for comparing how the server behaves as load grows."""
    first = levels[0].config
    parameter = levels[0].parameter
    length = (
        f"{first.duration:g} s per level"
        if first.duration is not None
        else f"{first.measured_requests} requests per level"
    )
    lines = [f"sweep {parameter} · {first.model} · {workload} · {length}", ""]
    columns = SWEEP_COLUMNS + (OPEN_SWEEP_COLUMNS if parameter == "rate" else ())
    lines.append(_sweep_row(parameter, [header for header, _ in columns]))
    for level in levels:
        if level.result is None:
            lines.append(f"{level.value:<12g}  failed: {level.error}")
            continue
        lines.append(
            _sweep_row(f"{level.value:g}", [cell(level.result) for _, cell in columns])
        )
    lines.append("")
    lines.append(
        "lag p99: server's lag · dlv TTFT: client − server TTFT p50"
        " · client: SAT = client saturated"
        + (" · sched p99: TTFT from the scheduled time" if parameter == "rate" else "")
    )
    return "\n".join(lines)


def _sweep_row(label: str, cells) -> str:
    # Wider than the single-run table: sweep headers fill ten characters.
    return f"{label:<12}" + "".join(f"{cell:>11}" for cell in cells)


def sweep_to_json(levels: List[SweepLevel]) -> Dict[str, Any]:
    return {
        "version": FORMAT_VERSION,
        "sweep": {
            "parameter": levels[0].parameter,
            "values": [level.value for level in levels],
        },
        "levels": [
            {"value": level.value, "error": level.error}
            | ({} if level.result is None else to_json(level.result))
            for level in levels
        ],
    }


def write_sweep_json(levels: List[SweepLevel], path: Path) -> None:
    Path(path).write_text(
        json.dumps(sweep_to_json(levels), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
