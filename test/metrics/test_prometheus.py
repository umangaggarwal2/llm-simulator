import pytest
from prometheus_client.parser import text_string_to_metric_families

from llm_simulator.metrics import ModelDescription, PrometheusRecorder, RequestTimings
from llm_simulator.metrics.prometheus import (
    FINISH_REASONS,
    LAG_BUCKETS,
    LATENCY_BUCKETS,
    log_buckets,
)

MODELS = {
    "qwen-mock": ModelDescription("fixed_rate", 32768),
    "tiny-mock": ModelDescription("instant", 8192),
}
TIMINGS = RequestTimings(e2e=0.808, queue=0.0, prefill=0.012, decode=0.796, tpot=0.004)


def value(recorder, name, **labels):
    return recorder.registry.get_sample_value(name, labels)


def test_every_catalogued_metric_is_exported():
    families = {
        family.name
        for family in text_string_to_metric_families(
            PrometheusRecorder(MODELS).exposition().decode()
        )
    }
    assert families == {
        "llm_sim_time_to_first_token_seconds",
        "llm_sim_inter_token_latency_seconds",
        "llm_sim_request_time_per_output_token_seconds",
        "llm_sim_e2e_request_latency_seconds",
        "llm_sim_request_queue_time_seconds",
        "llm_sim_request_prefill_time_seconds",
        "llm_sim_request_decode_time_seconds",
        "llm_sim_request_prompt_tokens",
        "llm_sim_request_generation_tokens",
        "llm_sim_token_lag_seconds",
        "llm_sim_requests",
        "llm_sim_prompt_tokens",
        "llm_sim_generation_tokens",
        "llm_sim_requests_running",
        "llm_sim_model_info",
    }


def test_no_created_series():
    assert b"_created" not in PrometheusRecorder(MODELS).exposition()


def test_model_info():
    recorder = PrometheusRecorder(MODELS)
    assert (
        value(
            recorder,
            "llm_sim_model_info",
            model="qwen-mock",
            engine="fixed_rate",
            context_length="32768",
        )
        == 1.0
    )


def test_finish_reasons_start_at_zero_for_every_model():
    recorder = PrometheusRecorder(MODELS)
    for model in MODELS:
        for reason in FINISH_REASONS:
            assert (
                value(
                    recorder,
                    "llm_sim_requests_total",
                    model=model,
                    finish_reason=reason,
                )
                == 0.0
            )


def test_completed_request():
    recorder = PrometheusRecorder(MODELS)
    recorder.request_started("qwen-mock", 1000)
    assert value(recorder, "llm_sim_requests_running", model="qwen-mock") == 1.0

    recorder.first_token("qwen-mock", ttft=0.012, queue_time=0.0)
    recorder.token("qwen-mock", itl=None, lag=0.001)
    recorder.token("qwen-mock", itl=0.004, lag=0.0)
    recorder.request_finished("qwen-mock", "stop", 1000, 2, TIMINGS)

    labels = {"model": "qwen-mock"}
    assert value(recorder, "llm_sim_requests_running", **labels) == 0.0
    assert (
        value(recorder, "llm_sim_requests_total", finish_reason="stop", **labels) == 1
    )
    assert value(recorder, "llm_sim_prompt_tokens_total", **labels) == 1000
    assert value(recorder, "llm_sim_generation_tokens_total", **labels) == 2
    assert value(recorder, "llm_sim_time_to_first_token_seconds_sum", **labels) == 0.012
    assert value(recorder, "llm_sim_inter_token_latency_seconds_count", **labels) == 1
    assert value(recorder, "llm_sim_token_lag_seconds_count", **labels) == 2
    assert value(recorder, "llm_sim_e2e_request_latency_seconds_sum", **labels) == 0.808
    assert (
        value(recorder, "llm_sim_request_time_per_output_token_seconds_count", **labels)
        == 1
    )
    assert (
        value(recorder, "llm_sim_request_prefill_time_seconds_sum", **labels) == 0.012
    )
    assert value(recorder, "llm_sim_request_decode_time_seconds_sum", **labels) == 0.796
    assert value(recorder, "llm_sim_request_prompt_tokens_sum", **labels) == 1000
    assert value(recorder, "llm_sim_request_generation_tokens_sum", **labels) == 2
    # The other model is untouched.
    assert (
        value(recorder, "llm_sim_time_to_first_token_seconds_count", model="tiny-mock")
        == 0
    )


def test_cancelled_request_is_counted_but_not_timed():
    recorder = PrometheusRecorder(MODELS)
    recorder.request_started("qwen-mock", 50)
    recorder.first_token("qwen-mock", ttft=0.012, queue_time=0.0)
    recorder.token("qwen-mock", itl=None, lag=0.0)
    recorder.request_finished("qwen-mock", "cancelled", 50, 1, None)

    labels = {"model": "qwen-mock"}
    assert (
        value(recorder, "llm_sim_requests_total", finish_reason="cancelled", **labels)
        == 1
    )
    assert value(recorder, "llm_sim_requests_running", **labels) == 0.0
    assert value(recorder, "llm_sim_generation_tokens_total", **labels) == 1
    assert value(recorder, "llm_sim_time_to_first_token_seconds_count", **labels) == 1
    assert value(recorder, "llm_sim_e2e_request_latency_seconds_count", **labels) == 0
    assert value(recorder, "llm_sim_request_generation_tokens_count", **labels) == 0


def test_single_token_request_skips_tpot():
    recorder = PrometheusRecorder(MODELS)
    recorder.request_started("qwen-mock", 5)
    timings = RequestTimings(e2e=0.01, queue=0.0, prefill=0.01, decode=0.0, tpot=None)
    recorder.request_finished("qwen-mock", "stop", 5, 1, timings)
    assert (
        value(
            recorder,
            "llm_sim_request_time_per_output_token_seconds_count",
            model="qwen-mock",
        )
        == 0
    )
    assert (
        value(recorder, "llm_sim_e2e_request_latency_seconds_count", model="qwen-mock")
        == 1
    )


def test_unknown_model_is_recorded_rather_than_failing():
    recorder = PrometheusRecorder(MODELS)
    recorder.request_started("swapped-in", 3)
    recorder.request_finished("swapped-in", "stop", 3, 0, None)
    assert (
        value(
            recorder, "llm_sim_requests_total", model="swapped-in", finish_reason="stop"
        )
        == 1
    )


def test_each_recorder_has_its_own_registry():
    first, second = PrometheusRecorder(MODELS), PrometheusRecorder(MODELS)
    first.request_started("qwen-mock", 10)
    assert value(first, "llm_sim_prompt_tokens_total", model="qwen-mock") == 10
    assert value(second, "llm_sim_prompt_tokens_total", model="qwen-mock") == 0


def test_buckets():
    assert log_buckets(1, 8, 2) == (1, 2, 4, 8)
    assert LATENCY_BUCKETS[0] == 0.001 and LATENCY_BUCKETS[-1] >= 60
    # Fine enough at the bottom to see millisecond timer jitter.
    assert LAG_BUCKETS[0] == 0.00001
    assert sum(1 for bound in LAG_BUCKETS if bound <= 0.002) >= 10


def test_token_buckets_cover_the_largest_context():
    recorder = PrometheusRecorder(MODELS)
    recorder.request_started("qwen-mock", 30000)
    recorder.request_finished("qwen-mock", "stop", 30000, 0, TIMINGS)
    assert (
        value(
            recorder,
            "llm_sim_request_prompt_tokens_bucket",
            model="qwen-mock",
            le="16384.0",
        )
        == 0
    )
    assert (
        value(
            recorder,
            "llm_sim_request_prompt_tokens_bucket",
            model="qwen-mock",
            le="32768.0",
        )
        == 1
    )
