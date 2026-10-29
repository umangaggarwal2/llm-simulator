import pytest

from llm_simulator.metrics import (
    ModelDescription,
    RequestTimings,
    SummaryRecorder,
    summarize,
)

MODELS = {
    "qwen-mock": ModelDescription("fixed_rate", 32768),
    "tiny-mock": ModelDescription("instant", 8192),
}
TIMINGS = RequestTimings(e2e=0.808, queue=0.0, prefill=0.012, decode=0.796, tpot=0.004)


class Clock:
    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now


def make_recorder(clock=None, **sizes):
    sizes = {
        "request_samples": 100,
        "token_samples": 1000,
        "rate_seconds": 60,
        **sizes,
    }
    return SummaryRecorder(MODELS, clock=clock or Clock(), **sizes)


def complete(recorder, model="qwen-mock", tokens=3, timings=TIMINGS):
    recorder.request_started(model, 1000)
    recorder.first_token(model, ttft=0.012, queue_time=0.0)
    for index in range(tokens):
        recorder.token(model, itl=None if index == 0 else 0.004, lag=0.001)
    recorder.request_finished(model, "stop", 1000, tokens, timings)


def summary(recorder):
    return summarize(recorder.snapshot())


def test_shape_for_every_configured_model():
    body = summary(make_recorder())
    assert body["since_reset_seconds"] == 0.0
    assert set(body["models"]) == {"qwen-mock", "tiny-mock"}
    model = body["models"]["qwen-mock"]
    assert model["engine"] == "fixed_rate"
    assert model["requests"] == {
        "running": 0,
        "finished": {"stop": 0, "length": 0, "cancelled": 0, "error": 0},
    }
    assert model["tokens"] == {"prompt": 0, "generation": 0}
    assert set(model["throughput"]) == {
        "window_seconds",
        "requests_per_second",
        "output_tokens_per_second",
    }
    assert list(model["latency_seconds"]) == [
        "ttft",
        "itl",
        "tpot",
        "e2e",
        "queue",
        "prefill",
        "decode",
        "lag",
    ]
    assert list(model["tokens_per_request"]) == ["prompt", "generation"]
    assert model["latency_seconds"]["ttft"]["p50"] is None


def test_completed_request():
    recorder = make_recorder()
    recorder.request_started("qwen-mock", 1000)
    assert summary(recorder)["models"]["qwen-mock"]["requests"]["running"] == 1
    recorder.first_token("qwen-mock", ttft=0.012, queue_time=0.0)
    recorder.token("qwen-mock", itl=None, lag=0.001)
    recorder.token("qwen-mock", itl=0.004, lag=0.0)
    recorder.request_finished("qwen-mock", "stop", 1000, 2, TIMINGS)

    model = summary(recorder)["models"]["qwen-mock"]
    assert model["requests"]["running"] == 0
    assert model["requests"]["finished"]["stop"] == 1
    assert model["tokens"] == {"prompt": 1000, "generation": 2}
    latency = model["latency_seconds"]
    assert latency["ttft"]["p50"] == 0.012
    assert latency["itl"]["count"] == 1
    assert latency["lag"]["count"] == 2
    assert latency["e2e"]["max"] == 0.808
    assert latency["tpot"]["mean"] == 0.004
    assert (latency["prefill"]["p99"], latency["decode"]["min"]) == (0.012, 0.796)
    assert model["tokens_per_request"]["prompt"]["p50"] == 1000
    assert model["tokens_per_request"]["generation"]["p50"] == 2
    # The other model is untouched.
    assert (
        summary(recorder)["models"]["tiny-mock"]["latency_seconds"]["ttft"]["count"]
        == 0
    )


def test_cancelled_request_is_counted_but_not_timed():
    recorder = make_recorder()
    recorder.request_started("qwen-mock", 50)
    recorder.first_token("qwen-mock", ttft=0.012, queue_time=0.0)
    recorder.token("qwen-mock", itl=None, lag=0.0)
    recorder.request_finished("qwen-mock", "cancelled", 50, 1, None)

    model = summary(recorder)["models"]["qwen-mock"]
    assert model["requests"]["finished"]["cancelled"] == 1
    assert model["requests"]["running"] == 0
    assert model["tokens"]["generation"] == 1
    assert model["latency_seconds"]["ttft"]["count"] == 1
    assert model["latency_seconds"]["lag"]["count"] == 1
    assert model["latency_seconds"]["e2e"]["count"] == 0
    assert model["tokens_per_request"]["generation"]["count"] == 0


def test_single_token_request_skips_tpot():
    recorder = make_recorder()
    timings = RequestTimings(e2e=0.01, queue=0.0, prefill=0.01, decode=0.0, tpot=None)
    complete(recorder, tokens=1, timings=timings)
    latency = summary(recorder)["models"]["qwen-mock"]["latency_seconds"]
    assert latency["tpot"]["count"] == 0
    assert latency["e2e"]["count"] == 1


def test_windows_keep_the_most_recent_values():
    recorder = make_recorder(request_samples=2, token_samples=5)
    for _ in range(4):
        complete(recorder, tokens=3)
    latency = summary(recorder)["models"]["qwen-mock"]["latency_seconds"]
    assert (latency["ttft"]["count"], latency["ttft"]["samples"]) == (4, 2)
    assert (latency["lag"]["count"], latency["lag"]["samples"]) == (12, 5)
    assert (latency["itl"]["count"], latency["itl"]["samples"]) == (8, 5)


def test_throughput():
    clock = Clock(100.0)
    recorder = make_recorder(clock)
    clock.now = 102.0
    complete(recorder, tokens=10)
    complete(recorder, tokens=10)
    clock.now = 104.0
    throughput = summary(recorder)["models"]["qwen-mock"]["throughput"]
    assert throughput == {
        "window_seconds": 4.0,
        "requests_per_second": 0.5,
        "output_tokens_per_second": 5.0,
    }


def test_unknown_model_is_recorded_rather_than_failing():
    recorder = make_recorder()
    complete(recorder, model="swapped-in")
    model = summary(recorder)["models"]["swapped-in"]
    assert model["engine"] is None
    assert model["requests"]["finished"]["stop"] == 1


def test_reset_clears_everything_but_requests_in_flight():
    clock = Clock(100.0)
    recorder = make_recorder(clock)
    complete(recorder)
    recorder.request_started("qwen-mock", 7)  # still running at the reset
    clock.now = 110.0
    recorder.reset()
    clock.now = 112.5

    body = summary(recorder)
    assert body["since_reset_seconds"] == 2.5
    model = body["models"]["qwen-mock"]
    assert model["requests"] == {
        "running": 1,
        "finished": {"stop": 0, "length": 0, "cancelled": 0, "error": 0},
    }
    assert model["tokens"] == {"prompt": 0, "generation": 0}
    assert model["throughput"]["output_tokens_per_second"] == 0.0
    assert all(stats["count"] == 0 for stats in model["latency_seconds"].values())

    # The request in flight finishes into the new period, and running
    # returns to 0 rather than going negative.
    recorder.request_finished("qwen-mock", "cancelled", 7, 0, None)
    model = summary(recorder)["models"]["qwen-mock"]
    assert model["requests"]["running"] == 0
    assert model["requests"]["finished"]["cancelled"] == 1


def test_snapshot_is_unaffected_by_later_recording():
    recorder = make_recorder()
    complete(recorder)
    snapshot = recorder.snapshot()
    complete(recorder)
    recorder.reset()
    model = summarize(snapshot)["models"]["qwen-mock"]
    assert model["requests"]["finished"]["stop"] == 1
    assert model["latency_seconds"]["ttft"]["count"] == 1
    assert model["latency_seconds"]["ttft"]["samples"] == 1


def test_percentiles_over_many_requests():
    recorder = make_recorder()
    for ttft in range(1, 101):
        recorder.request_started("qwen-mock", 1)
        recorder.first_token("qwen-mock", ttft=ttft / 1000, queue_time=0.0)
        recorder.request_finished("qwen-mock", "cancelled", 1, 1, None)
    ttft = summary(recorder)["models"]["qwen-mock"]["latency_seconds"]["ttft"]
    assert ttft["p50"] == pytest.approx(0.0505)
    assert ttft["p99"] == pytest.approx(0.09901)
    assert (ttft["min"], ttft["max"]) == (0.001, 0.1)
