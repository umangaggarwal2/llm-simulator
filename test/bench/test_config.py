import pytest

from llm_simulator.bench.config import BenchConfig


def test_defaults():
    config = BenchConfig(model="m")
    assert config.base_url == "http://localhost:8080/v1"
    assert (config.mode, config.closed_concurrency, config.max_tokens) == (
        "closed",
        1,
        256,
    )
    assert (config.measured_requests, config.duration) == (100, None)
    assert config.warmup_requests == 1
    assert config.api_key is None
    assert config.timeout == 60.0


def test_chat_url_tolerates_a_trailing_slash():
    assert BenchConfig(model="m", base_url="http://h:1/v1/").chat_url == (
        "http://h:1/v1/chat/completions"
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"model": ""},
        {"base_url": "localhost:8080/v1"},
        {"concurrency": 0},
        {"requests": 0},
        {"max_tokens": -1},
        {"timeout": 0},
        {"input_tokens": 0},
        {"input_tokens": 100, "dataset": "d.jsonl"},
        {"requests": 10, "duration": 5.0},
        {"duration": 0},
        {"warmup": -1},
        {"server_metrics_url": "localhost/metrics/summary"},
        {"concurrency": 0},
        {"concurrency": 2, "rate": 5.0},
        {"rate": 0},
        {"max_concurrency": 4},
        {"rate": 5.0, "max_concurrency": 0},
    ],
)
def test_invalid_values_are_rejected(changes):
    with pytest.raises(ValueError):
        BenchConfig(**{"model": "m", **changes})


def test_warmup_defaults_to_one_per_stream():
    assert BenchConfig(model="m", concurrency=8).warmup_requests == 8
    assert BenchConfig(model="m", concurrency=8, warmup=0).warmup_requests == 0


def test_duration_replaces_the_request_count():
    config = BenchConfig(model="m", duration=30.0)
    assert config.measured_requests is None
    assert BenchConfig(model="m", requests=7).measured_requests == 7


def test_open_loop_sizing():
    uncapped = BenchConfig(model="m", rate=10.0)
    assert (uncapped.mode, uncapped.streams) == ("open", 0)
    assert (uncapped.warmup_requests, uncapped.warmup_concurrency) == (8, 8)
    capped = BenchConfig(model="m", rate=10.0, max_concurrency=4)
    assert (capped.streams, capped.warmup_requests, capped.warmup_concurrency) == (
        4,
        4,
        4,
    )
    bigger_warmup = BenchConfig(model="m", rate=10.0, max_concurrency=4, warmup=20)
    assert bigger_warmup.warmup_concurrency == 4


def test_closed_loop_sizing():
    config = BenchConfig(model="m", concurrency=8)
    assert (config.streams, config.warmup_requests, config.warmup_concurrency) == (
        8,
        8,
        8,
    )
