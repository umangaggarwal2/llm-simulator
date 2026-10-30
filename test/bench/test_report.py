import json

import pytest
from array import array
from datetime import datetime, timezone

from llm_simulator.bench.config import BenchConfig
from llm_simulator.bench.records import (
    LoopSamples,
    RequestRecord,
    RunRecords,
    ServerSummary,
)
from llm_simulator.bench.report import format_result, to_json, write_json
from llm_simulator.bench.results import summarize


def record(index, ttft=0.01, gaps=(0.004, 0.004, 0.004), status=200, error=None):
    if status != 200 or error:
        return RequestRecord(index=index, sent=0.0, end=0.1, status=status, error=error)
    return RequestRecord(
        index=index,
        sent=0.0,
        end=ttft + sum(gaps) + 0.001,
        first_token=ttft,
        last_token=ttft + sum(gaps),
        itls=array("d", gaps),
        chunks=len(gaps) + 1,
        prompt_tokens=8,
        completion_tokens=len(gaps) + 1,
        finish_reason="length",
        requested_tokens=len(gaps) + 1,
        status=200,
    )


def result(records, loop=None, server=None, **config):
    config = BenchConfig(**{"model": "qwen-mock", "concurrency": 2, **config})
    started_at = datetime(2026, 10, 30, 10, 0, tzinfo=timezone.utc)
    run = RunRecords(records, 0.0, 2.0, started_at, warmup=2, loop=loop, server=server)
    return summarize(run, config, "fixed prompt")


HEALTHY = LoopSamples(array("d", [0.001] * 100), 0.6, 2.0)
SATURATED = LoopSamples(array("d", [0.001] * 90 + [0.02] * 10), 1.9, 2.0)


def test_table():
    text = format_result(result([record(0), record(1, ttft=0.02)]))
    lines = text.splitlines()
    assert lines[0].startswith("qwen-mock @ http://localhost:8080/v1 · closed loop")
    assert "fixed prompt" in lines[0]
    assert "2 ok · 0 failed · finish length 2" in text
    assert "duration    2.00 s" in text
    assert "1.00 req/s · 4 output tok/s · 12 total tok/s" in text
    ttft_row = next(line for line in lines if line.strip().startswith("TTFT"))
    assert "15.00 ms" in ttft_row  # the mean
    assert "warnings" not in text and "errors" not in text


def test_seconds_above_one_are_shown_in_seconds():
    text = format_result(result([record(0, ttft=1.5)]))
    ttft_row = next(line for line in text.splitlines() if "TTFT" in line)
    assert "1.500 s" in ttft_row


def test_warnings_and_errors():
    short = record(0)
    short.requested_tokens = 100
    failures = [record(i, status=404, error=f"model {i % 7}") for i in range(1, 15)]
    text = format_result(result([short] + failures))
    assert "1 successful requests returned fewer tokens than requested" in text
    assert "2 × HTTP 404: model 1" in text
    assert "… and 2 other kinds" in text


def test_json(tmp_path):
    data = to_json(result([record(0)], api_key="sk-secret"))
    assert data["version"] == 1
    assert data["config"]["api_key"] == "<redacted>"
    assert "sk-secret" not in json.dumps(data)
    assert data["workload"] == "fixed prompt"
    assert data["environment"]["started_at"] == "2026-10-30T10:00:00+00:00"
    assert set(data["environment"]) == {
        "started_at",
        "python",
        "platform",
        "llm_simulator",
        "aiohttp",
    }
    assert data["result"]["requests"]["ok"] == 1
    assert data["result"]["latency_seconds"]["ttft"]["p50"] == 0.01
    assert data["result"]["throughput"]["requests_per_second"] == 0.5


def test_json_without_a_key_keeps_null_and_paths_become_strings(tmp_path):
    data = to_json(result([record(0)], dataset=tmp_path / "d.jsonl"))
    assert data["config"]["api_key"] is None
    assert data["config"]["dataset"].endswith("d.jsonl")


def test_write_json_round_trips_as_readable_utf8(tmp_path):
    path = tmp_path / "run.json"
    run = result([record(0)])
    write_json(run, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == json.loads(json.dumps(to_json(run)))
    assert data["result"]["tokens"] == {"prompt": 8, "output": 4}


def test_non_ascii_is_written_as_is(tmp_path):
    path = tmp_path / "run.json"
    config = BenchConfig(model="m")
    run = RunRecords([record(0)], 0.0, 1.0)
    write_json(summarize(run, config, "synthetic 8 → 4 tokens"), path)
    assert '"synthetic 8 → 4 tokens"' in path.read_text(encoding="utf-8")


def test_warmup_in_the_header():
    assert "· warm-up 2 ·" in format_result(result([record(0)])).splitlines()[0]


def test_duration_target():
    text = format_result(result([record(0)], duration=2.0))
    assert "duration    2.00 s (target 2 s)" in text


def test_client_health_line():
    text = format_result(result([record(0)], loop=HEALTHY))
    line = next(line for line in text.splitlines() if line.startswith("client"))
    assert "loop lag p50 1.00 ms" in line
    assert "CPU 30%" in line and line.endswith("healthy")
    assert "warnings" not in text


def test_saturated_client_warns():
    text = format_result(result([record(0)], loop=SATURATED))
    assert "· saturated" in text
    assert "client event loop was late by 20.00 ms at p99 (CPU 95%)" in text


def test_no_client_line_without_a_monitor():
    assert "client " not in format_result(result([record(0)]))


def test_json_phases_and_client():
    data = to_json(result([record(0)], loop=HEALTHY))
    assert data["result"]["warmup_requests"] == 2
    assert data["result"]["client"]["cpu_percent"] == 30.0
    assert data["result"]["client"]["saturated"] is False
    assert data["result"]["client"]["loop_lag"]["p50"] == 0.001
    assert to_json(result([record(0)]))["result"]["client"] is None


def stat(p50, p99=None):
    return {"count": 1, "samples": 1, "mean": p50, "p50": p50, "p99": p99 or p50}


def section(simulated=True, finished=1):
    latency = {
        "ttft": stat(0.008),
        "itl": stat(0.004),
        "tpot": stat(0.004),
        "e2e": stat(0.020),
        "lag": {"p50": 0.0005, "p99": 0.001, "max": 0.002},
    }
    if simulated:
        latency["ttft_simulated"] = stat(0.007)
        latency["e2e_simulated"] = stat(0.019)
    return {
        "latency_seconds": latency,
        "requests": {"finished": {"length": finished}},
        "tokens": {"generation": 4},
    }


URL = "http://localhost:8080/metrics/summary"


def test_comparison_table():
    text = format_result(
        result([record(0)], server=ServerSummary(URL, True, model=section()))
    )
    assert f"server      {URL} · 1 requests (client 1 ✓)" in text
    header = next(line for line in text.splitlines() if line.startswith("vs server"))
    assert "simulated" in header and "delivery*" in header
    ttft = next(line for line in text.splitlines() if "TTFT p50" in line)
    # client 10 ms, server 8 ms, simulated 7 ms, delivery 2 ms
    assert ttft.split()[2:] == ["10.00", "ms", "8.00", "ms", "7.00", "ms", "2.00", "ms"]
    assert "fidelity    server lag p50 0.50 ms · p99 1.00 ms · max 2.00 ms" in text
    assert "* client percentile − server percentile" in text
    assert "warnings" not in text


def test_no_simulated_column_from_an_older_server():
    summary = ServerSummary(URL, True, model=section(simulated=False))
    header = next(
        line
        for line in format_result(result([record(0)], server=summary)).splitlines()
        if line.startswith("vs server")
    )
    assert "simulated" not in header


def test_unavailable_server_line():
    text = format_result(
        result([record(0)], server=ServerSummary(URL, False, reason="HTTP 404"))
    )
    assert "server      not available (HTTP 404)" in text
    assert "vs server" not in text


def test_mismatched_counts_warn():
    summary = ServerSummary(URL, True, model=section(finished=3))
    text = format_result(result([record(0)], server=summary))
    assert "(client 1 ≠)" in text
    assert "server finished 3 requests, client had 1 accepted" in text


def test_server_json():
    summary = ServerSummary(URL, True, since_reset=2.1, model=section())
    data = to_json(result([record(0)], server=summary))["result"]["server"]
    assert data["url"] == URL and data["available"] is True
    assert data["since_reset_seconds"] == 2.1
    assert data["summary"] == section()
    assert data["requests"] == {"server": 1, "client_accepted": 1}
    assert data["delivery_seconds"]["ttft"]["p50"] == pytest.approx(0.002)
    assert to_json(result([record(0)]))["result"]["server"] is None


# --- open loop and sweeps ---

from llm_simulator.bench.report import (  # noqa: E402
    format_progress,
    format_sweep,
    sweep_to_json,
)
from llm_simulator.bench.sweep import SweepLevel  # noqa: E402


def open_result(waited=False):
    first = record(0)
    first.scheduled = -0.2 if waited else 0.0
    first.waited = waited
    config = BenchConfig(model="qwen-mock", rate=5.0, max_concurrency=2)
    return summarize(RunRecords([first], 0.0, 2.0, warmup=2), config, "fixed prompt")


def test_open_loop_header_and_rows():
    text = format_result(open_result())
    first = text.splitlines()[0]
    assert "open loop · rate 5/s (Poisson, seed 0) · max concurrency 2" in first
    assert "arrivals    target 5/s" in text
    assert "  TTFT sched" in text and "  E2E sched" in text and "  queue wait" in text
    assert "waited for a free slot" not in text


def test_waiting_warns():
    text = format_result(open_result(waited=True))
    assert "1 requests waited for a free slot (max concurrency 2)" in text


def test_open_loop_json():
    data = to_json(open_result())["result"]
    assert data["arrivals"]["target_rate"] == 5.0
    assert set(data["latency_from_schedule_seconds"]) == {"ttft", "e2e"}


def level(value, error=None):
    config = BenchConfig(model="qwen-mock", concurrency=value)
    if error:
        return SweepLevel("concurrency", value, config, error=error)
    run = RunRecords([record(0)], 0.0, 2.0, warmup=value)
    return SweepLevel(
        "concurrency", value, config, result=summarize(run, config, "fixed prompt")
    )


def test_sweep_table():
    text = format_sweep(
        [level(1), level(8), level(32, "all 32 warm-up requests failed")], "w"
    )
    lines = text.splitlines()
    assert lines[0] == "sweep concurrency · qwen-mock · w · 100 requests per level"
    header = lines[2]
    assert header.startswith("concurrency") and "TTFT p99" in header
    assert lines[3].strip().startswith("1") and "10.00 ms" in lines[3]
    assert "failed: all 32 warm-up requests failed" in text
    # Every header is separated from the next.
    assert "  errors" in header and "TTFT p50  " in header


def test_progress_lines():
    assert format_progress(2, 3, level(8)) == (
        "level 2/3 · concurrency 8 · 2.0 s · 1 ok · 0 failed"
    )
    assert format_progress(3, 3, level(32, "boom")) == (
        "level 3/3 · concurrency 32 · failed: boom"
    )


def test_sweep_json():
    data = sweep_to_json([level(1), level(8, "boom")])
    assert data["sweep"] == {"parameter": "concurrency", "values": [1, 8]}
    assert data["levels"][0]["value"] == 1
    assert data["levels"][0]["result"]["requests"]["ok"] == 1
    assert data["levels"][1] == {"value": 8, "error": "boom"}
