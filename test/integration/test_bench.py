"""The load generator against a real simulator, over HTTP."""

import asyncio
import json
from pathlib import Path

import pytest

from llm_simulator.bench import cli
from llm_simulator.bench.config import BenchConfig
from llm_simulator.bench.results import summarize

pytestmark = pytest.mark.integration

BENCH_RESPONSES = Path(__file__).parents[2] / "config" / "bench-responses.yml"
MODELS = {
    "version": 1,
    "models": [
        {"name": "fast-mock", "context_length": 8192},
        {
            # 20 ms overhead, prefill at 1000 tok/s, then one token every 10 ms.
            "name": "paced-mock",
            "context_length": 8192,
            "engine": "fixed_rate",
            "performance": {
                "prefill_tokens_per_second": 1000,
                "decode_tokens_per_second": 100,
            },
            "latency": {"scheduler_overhead_ms": 20},
        },
    ],
}


@pytest.fixture(scope="module")
def base_url(tmp_path_factory, serve_process):
    # A separate process: in-process, the server thread would hold the GIL
    # and delay the client's timestamps (see conftest.py).
    with serve_process(
        tmp_path_factory.mktemp("config"), MODELS, BENCH_RESPONSES
    ) as root:
        yield f"{root}/v1"


def run(**config):
    config = BenchConfig(**config)
    records = asyncio.run(cli.run(config, cli.build_workload(config)))
    return records.records, records.duration


def test_output_length_is_exactly_max_tokens(base_url):
    records, duration = run(
        model="fast-mock", base_url=base_url, concurrency=4, requests=12, max_tokens=50
    )
    assert duration > 0
    assert all(record.ok for record in records)
    for record in records:
        assert record.completion_tokens == record.chunks == 50
        assert record.finish_reason == "length"
        assert len(record.itls) == 49


def test_client_timing_is_never_earlier_than_the_simulated_timeline(base_url):
    (record,), _ = run(model="paced-mock", base_url=base_url, requests=1, max_tokens=8)
    assert record.ok
    ttft = 0.020 + record.prompt_tokens / 1000
    # The client sees the server's release plus delivery: never earlier.
    assert ttft <= record.ttft < ttft + 0.05
    assert record.e2e >= ttft + 7 * 0.010
    assert min(record.itls) > 0.005


def test_command_end_to_end(base_url, capsys):
    argv = ["--model", "fast-mock", "--base-url", base_url, "--requests", "3"]
    assert cli.main(argv + ["--max-tokens", "10"]) == 0
    out = capsys.readouterr().out
    assert "3 ok · 0 failed · finish length 3" in out
    assert "warnings" not in out


def test_unknown_model_fails_the_command(base_url, capsys):
    argv = ["--model", "nope", "--base-url", base_url, "--requests", "2"]
    assert cli.main(argv) == 1
    # Every request fails, so the run stops in warm-up.
    assert "1 × HTTP 404: The model `nope` does not exist" in capsys.readouterr().err


def test_input_tokens_set_the_prompt_length(base_url):
    records, _ = run(
        model="fast-mock",
        base_url=base_url,
        concurrency=2,
        requests=6,
        input_tokens=300,
        max_tokens=5,
    )
    assert [record.prompt_tokens for record in records] == [300] * 6


def test_prefix_selects_a_tagged_entry(base_url):
    # [short] is 32 to 128 tokens in bench-responses.yml, so a large
    # max_tokens is not reached and the response ends with stop.
    records, _ = run(
        model="fast-mock",
        base_url=base_url,
        requests=5,
        input_tokens=50,
        prompt_prefix="[short] ",
        max_tokens=1000,
    )
    for record in records:
        assert record.finish_reason == "stop"
        assert 32 <= record.completion_tokens <= 128


def test_dataset_replay(base_url, tmp_path):
    dataset = tmp_path / "prompts.jsonl"
    dataset.write_text('{"prompt": "one"}\n{"prompt": "two", "max_tokens": 3}\n')
    records, _ = run(
        model="fast-mock", base_url=base_url, requests=4, dataset=dataset, max_tokens=9
    )
    assert [record.completion_tokens for record in records] == [9, 3, 9, 3]


def test_results_file(base_url, tmp_path, capsys):
    output = tmp_path / "run.json"
    argv = ["--model", "fast-mock", "--base-url", base_url, "--requests", "4"]
    assert cli.main(argv + ["--input-tokens", "64", "--output", str(output)]) == 0
    assert "p99" in capsys.readouterr().out
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["result"]["requests"]["ok"] == 4
    assert data["result"]["tokens_per_request"]["prompt"]["min"] == 64
    assert data["result"]["warnings"] == []


def test_warmup_brings_ttft_close_to_the_simulated_timeline(base_url):
    records, _ = run(
        model="paced-mock",
        base_url=base_url,
        concurrency=2,
        requests=4,
        warmup=2,
        input_tokens=100,
        max_tokens=3,
    )
    simulated = 0.020 + 100 / 1000
    for record in records:
        assert simulated <= record.ttft < simulated + 0.015


def test_duration_mode_measures_for_the_given_time(base_url):
    config = BenchConfig(
        model="paced-mock", base_url=base_url, concurrency=2, duration=0.5, max_tokens=3
    )
    records = asyncio.run(cli.run(config, cli.build_workload(config)))
    # Each request takes about 20 + 8 + 20 ms, so 2 streams send about 20.
    assert 8 <= len(records.records) <= 30
    assert all(record.sent < records.started + 0.5 for record in records.records)
    assert records.warmup == 2
    assert records.loop is not None


def test_unreachable_server_fails_in_warmup(capsys):
    argv = ["--model", "m", "--base-url", "http://127.0.0.1:1/v1", "--concurrency", "3"]
    assert cli.main(argv) == 1
    err = capsys.readouterr().err
    assert "all 3 warm-up requests failed; nothing was measured" in err
    assert "3 × no response" in err


def test_report_shows_phases_and_client_health(base_url, capsys):
    argv = ["--model", "fast-mock", "--base-url", base_url, "--requests", "5"]
    assert cli.main(argv + ["--concurrency", "2", "--max-tokens", "4"]) == 0
    out = capsys.readouterr().out
    assert "· warm-up 2 ·" in out
    assert "client      loop lag" in out


def test_server_comparison_shows_all_three_layers(base_url):
    config = BenchConfig(
        model="paced-mock",
        base_url=base_url,
        concurrency=2,
        requests=4,
        input_tokens=100,
        max_tokens=3,
    )
    run_records = asyncio.run(cli.run(config, cli.build_workload(config)))
    result = summarize(run_records, config, "w")
    comparison = result.server
    assert comparison is not None and comparison.counts_match
    simulated = comparison.simulated["ttft"]["p50"]
    # fixed_rate: 20 ms overhead + 100 prompt tokens at 1000 tok/s, exactly.
    assert simulated == pytest.approx(0.020 + 100 / 1000)
    server = comparison.server["ttft"]["p50"]
    client = result.latency["ttft"]["p50"]
    assert simulated <= server <= client
    assert comparison.server["lag"]["count"] == 4 * 3


def test_comparison_can_be_turned_off(base_url, capsys):
    argv = ["--model", "fast-mock", "--base-url", base_url, "--requests", "2"]
    assert cli.main(argv + ["--no-server-metrics"]) == 0
    assert "server " not in capsys.readouterr().out


def test_target_without_a_summary_is_noted(base_url, capsys):
    root = base_url.removesuffix("/v1")
    argv = ["--model", "fast-mock", "--base-url", base_url, "--requests", "2"]
    assert cli.main(argv + ["--server-metrics-url", f"{root}/nope"]) == 0
    out = capsys.readouterr().out
    assert "server      not available (HTTP 404)" in out
    assert "TTFT" in out


def test_open_loop_keeps_its_rate(base_url):
    config = BenchConfig(
        model="fast-mock", base_url=base_url, rate=50.0, requests=60, max_tokens=4
    )
    result = summarize(
        asyncio.run(cli.run(config, cli.build_workload(config))), config, "w"
    )
    assert result.ok == 60
    arrivals = result.arrivals
    assert arrivals.keeping_up and arrivals.waited == 0
    # The schedule is random; the sends follow it closely.
    assert arrivals.achieved_rate == pytest.approx(arrivals.offered_rate, rel=0.05)
    assert result.latency_from_schedule["ttft"]["p50"] == pytest.approx(
        result.latency["ttft"]["p50"], abs=0.003
    )


def test_overload_queues_on_the_client_and_shows_in_ttft_from_schedule(base_url):
    # paced-mock takes about 20 + 8 + 20 ms per request with 3 tokens, so 2
    # slots serve about 40 per second; 200 per second must queue.
    config = BenchConfig(
        model="paced-mock",
        base_url=base_url,
        rate=200.0,
        max_concurrency=2,
        requests=30,
        max_tokens=3,
    )
    result = summarize(
        asyncio.run(cli.run(config, cli.build_workload(config))), config, "w"
    )
    assert result.arrivals.waited > 10
    from_send = result.latency["ttft"]["p99"]
    from_schedule = result.latency_from_schedule["ttft"]["p99"]
    assert from_schedule > from_send + 0.1


def test_sweep_command(base_url, tmp_path, capsys):
    output = tmp_path / "sweep.json"
    argv = ["--model", "fast-mock", "--base-url", base_url, "--requests", "6"]
    argv += ["--concurrency", "1,3", "--max-tokens", "4", "--output", str(output)]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "level 1/2 · concurrency 1" in out and "level 2/2 · concurrency 3" in out
    assert "sweep concurrency · fast-mock" in out
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["sweep"] == {"parameter": "concurrency", "values": [1, 3]}
    assert [level["result"]["requests"]["ok"] for level in data["levels"]] == [6, 6]
