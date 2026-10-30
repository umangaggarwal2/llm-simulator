import dataclasses
import json
import time

import pytest
import yaml
from fastapi.testclient import TestClient

from llm_simulator.client.api import create_app
from llm_simulator.client.dependencies import get_runtimes
from llm_simulator.llm.models import ModelsConfigError
from llm_simulator.llm.responses import ResponsesConfigError
from llm_simulator.llm.tokenizer import RegexTokenizer
from llm_simulator import runtime as runtime_module
from llm_simulator.runtime import EngineBuilder, ModelRuntime
from llm_simulator.settings import Settings
from llm_simulator.simulator import InstantEngine
from llm_simulator.types import Message

MODELS = {
    "version": 1,
    "models": [
        {"name": "big-mock", "context_length": 1000},
        {"name": "small-mock", "context_length": 40},
    ],
}
RESPONSES = {
    "version": 1,
    "responses": [
        {
            "id": "greeting",
            "match": {"exact": "hello"},
            "content": "Hello from {model}!",
        },
        {
            "id": "colours",
            "match": {"starts_with": "colour"},
            "variants": ["Red.", "Green.", "Blue.", "Yellow.", "Purple."],
        },
    ],
    "default": {"content": "one two three four five six seven eight"},
}


def write_configs(tmp_path, models=MODELS, responses=RESPONSES):
    models_path = tmp_path / "models.yml"
    responses_path = tmp_path / "responses.yml"
    models_path.write_text(yaml.safe_dump(models))
    responses_path.write_text(yaml.safe_dump(responses))
    return Settings(models_path=models_path, responses_path=responses_path)


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(write_configs(tmp_path))) as client:
        yield client


def chat(client, text="hello", model="big-mock", **extra):
    body = {"model": model, "messages": [{"role": "user", "content": text}], **extra}
    return client.post("/v1/chat/completions", json=body)


def stream_chunks(client, **body):
    body = {"model": "big-mock", "stream": True, **body}
    with client.stream("POST", "/v1/chat/completions", json=body) as response:
        assert response.status_code == 200
        lines = [line for line in response.iter_lines() if line]
        headers = response.headers
    assert lines[-1] == "data: [DONE]"
    return [json.loads(line.removeprefix("data: ")) for line in lines[:-1]], headers


def prompt_tokens(text):
    return RegexTokenizer().count_messages([Message("user", text)])


# --- Models ---


def test_lists_every_configured_model(client):
    body = client.get("/v1/models").json()
    assert body["object"] == "list"
    assert [m["id"] for m in body["data"]] == ["big-mock", "small-mock"]
    assert all(
        m["object"] == "model" and isinstance(m["created"], int) for m in body["data"]
    )


# --- Non-streaming ---


def test_serves_the_configured_response(client):
    body = chat(client, "hello", model="small-mock").json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "small-mock"
    assert body["id"].startswith("chatcmpl-")
    choice = body["choices"][0]
    assert choice["message"] == {
        "role": "assistant",
        "content": "Hello from small-mock!",
    }
    assert choice["finish_reason"] == "stop"


def test_usage_comes_from_the_plan(client):
    usage = chat(client, "hello").json()["usage"]
    assert usage["prompt_tokens"] == prompt_tokens("hello")
    assert usage["completion_tokens"] == len(
        RegexTokenizer().encode("Hello from big-mock!")
    )
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_max_completion_tokens_truncates(client):
    body = chat(client, "anything", max_completion_tokens=3).json()
    assert body["choices"][0]["message"]["content"] == "one two three"
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] == 3


def test_max_tokens_still_works(client):
    body = chat(client, "anything", max_tokens=2).json()
    assert body["choices"][0]["message"]["content"] == "one two"


def test_max_completion_tokens_wins_over_max_tokens(client):
    body = chat(client, "anything", max_tokens=2, max_completion_tokens=4).json()
    assert body["usage"]["completion_tokens"] == 4


def test_seed_selects_reproducible_variants(client):
    def answer(seed):
        return chat(client, "colours please", seed=seed).json()["choices"][0]["message"]

    assert answer(1) == answer(1)
    assert len({answer(seed)["content"] for seed in range(20)}) > 1


# --- Streaming ---


def test_stream_chunks_match_openai(client):
    chunks, headers = stream_chunks(
        client, messages=[{"role": "user", "content": "hello"}]
    )
    assert headers["content-type"].startswith("text/event-stream")
    assert headers["cache-control"] == "no-cache"
    assert headers["x-accel-buffering"] == "no"

    deltas = [c["choices"][0]["delta"] for c in chunks]
    assert deltas[0] == {"role": "assistant", "content": ""}
    assert "".join(d.get("content", "") for d in deltas) == "Hello from big-mock!"
    assert deltas[-1] == {}
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert all("usage" not in c for c in chunks)
    assert len({c["id"] for c in chunks}) == 1


def test_stream_with_include_usage(client):
    chunks, _ = stream_chunks(
        client,
        messages=[{"role": "user", "content": "anything"}],
        stream_options={"include_usage": True},
        max_completion_tokens=3,
    )
    assert all(c["usage"] is None for c in chunks[:-1])
    assert chunks[-2]["choices"][0]["finish_reason"] == "length"
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"] == {
        "prompt_tokens": prompt_tokens("anything"),
        "completion_tokens": 3,
        "total_tokens": prompt_tokens("anything") + 3,
    }


def test_stream_options_are_ignored_without_stream(client):
    body = chat(client, "hello", stream_options={"include_usage": True}).json()
    assert body["object"] == "chat.completion"


# --- Errors ---


def test_unknown_model(client):
    response = chat(client, model="gpt-9")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"
    assert response.json()["error"]["param"] == "model"


def test_model_is_required(client):
    response = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]}
    )
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "Missing required parameter: 'model'."


def test_context_length_is_per_model(client):
    assert chat(client, "hello", model="big-mock", max_tokens=100).status_code == 200
    response = chat(client, "hello", model="small-mock", max_tokens=100)
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "context_length_exceeded"
    assert error["param"] == "messages"
    assert "maximum context length is 40 tokens" in error["message"]


def test_context_length_error_is_not_streamed(client):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "small-mock",
            "stream": True,
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")


# --- Startup and wiring ---


def test_bad_models_config_fails_at_startup(tmp_path):
    models = {"version": 1, "models": [{"name": "m", "context_length": 0}]}
    with pytest.raises(ModelsConfigError):
        create_app(write_configs(tmp_path, models=models))


def test_bad_responses_config_fails_at_startup(tmp_path):
    responses = {"version": 1, "default": {"content": "{nope}"}}
    with pytest.raises(ResponsesConfigError):
        create_app(write_configs(tmp_path, responses=responses))


class RecordingEngine(InstantEngine):
    name = "recording"
    calls = []

    async def start(self):
        RecordingEngine.calls.append("start")

    async def stop(self):
        RecordingEngine.calls.append("stop")

    async def run(self, plan, arrival):
        RecordingEngine.calls.append(("run", arrival))
        async for event in super().run(plan, arrival):
            yield event


RECORDING = EngineBuilder(RecordingEngine, lambda deployment: RecordingEngine())


def test_engines_start_and_stop_with_the_app(tmp_path, monkeypatch):
    monkeypatch.setitem(runtime_module.ENGINES, "recording", RECORDING)
    RecordingEngine.calls = []
    models = {
        "version": 1,
        "models": [{"name": "m", "context_length": 100, "engine": "recording"}],
    }
    app = create_app(write_configs(tmp_path, models=models))
    with TestClient(app):
        assert RecordingEngine.calls == ["start"]
    assert RecordingEngine.calls == ["start", "stop"]


def test_engine_receives_arrival_from_the_middleware(tmp_path, monkeypatch):
    monkeypatch.setitem(runtime_module.ENGINES, "recording", RECORDING)
    RecordingEngine.calls = []
    models = {
        "version": 1,
        "models": [{"name": "m", "context_length": 100, "engine": "recording"}],
    }
    with TestClient(create_app(write_configs(tmp_path, models=models))) as client:
        before = time.monotonic()
        chat(client, model="m")
        after = time.monotonic()
    ((_, arrival),) = [call for call in RecordingEngine.calls if call[0] == "run"]
    assert before <= arrival <= after


def test_runtimes_can_be_overridden(tmp_path):
    app = create_app(write_configs(tmp_path))
    real = app.state.runtimes["big-mock"]
    app.dependency_overrides[get_runtimes] = lambda: {
        "only-mock": ModelRuntime(
            real.deployment, real.tokenizer, real.generator, real.engine
        )
    }
    with TestClient(app) as client:
        assert [m["id"] for m in client.get("/v1/models").json()["data"]] == [
            "only-mock"
        ]
        assert chat(client, model="big-mock").status_code == 404
        assert chat(client, model="only-mock").status_code == 200


# --- Metrics ---


def scrape(client):
    from prometheus_client.parser import text_string_to_metric_families

    response = client.get("/metrics")
    assert response.status_code == 200
    samples = {}
    for family in text_string_to_metric_families(response.text):
        for sample in family.samples:
            samples[(sample.name, tuple(sorted(sample.labels.items())))] = sample.value
    return response, samples


def sample(samples, name, **labels):
    return samples[(name, tuple(sorted(labels.items())))]


def test_metrics_endpoint_uses_prometheus_format(client):
    response, samples = scrape(client)
    assert response.headers["content-type"].startswith("text/plain")
    assert "version=" in response.headers["content-type"]
    assert (
        sample(
            samples,
            "llm_sim_model_info",
            model="big-mock",
            engine="instant",
            context_length="1000",
        )
        == 1.0
    )


def test_metrics_record_completed_requests(client):
    chat(client, "hello", model="big-mock")
    stream_chunks(
        client,
        messages=[{"role": "user", "content": "anything"}],
        max_completion_tokens=3,
    )
    _, samples = scrape(client)

    labels = {"model": "big-mock"}
    assert (
        sample(samples, "llm_sim_requests_total", finish_reason="stop", **labels) == 1
    )
    assert (
        sample(samples, "llm_sim_requests_total", finish_reason="length", **labels) == 1
    )
    assert sample(samples, "llm_sim_requests_running", **labels) == 0
    assert sample(samples, "llm_sim_prompt_tokens_total", **labels) == (
        prompt_tokens("hello") + prompt_tokens("anything")
    )
    generated = len(RegexTokenizer().encode("Hello from big-mock!")) + 3
    assert sample(samples, "llm_sim_generation_tokens_total", **labels) == generated
    assert sample(samples, "llm_sim_time_to_first_token_seconds_count", **labels) == 2
    assert sample(samples, "llm_sim_e2e_request_latency_seconds_count", **labels) == 2
    assert sample(samples, "llm_sim_token_lag_seconds_count", **labels) == generated
    assert sample(samples, "llm_sim_inter_token_latency_seconds_count", **labels) == (
        generated - 2
    )
    # The other model saw nothing.
    assert (
        sample(samples, "llm_sim_time_to_first_token_seconds_count", model="small-mock")
        == 0
    )


def test_rejected_requests_are_not_traced(client):
    chat(client, model="gpt-9")
    chat(client, "hello", model="small-mock", max_tokens=100)
    _, samples = scrape(client)
    for model in ("big-mock", "small-mock"):
        assert (
            sum(
                sample(
                    samples, "llm_sim_requests_total", model=model, finish_reason=reason
                )
                for reason in ("stop", "length", "cancelled", "error")
            )
            == 0
        )


def test_each_app_has_its_own_metrics(tmp_path):
    settings = write_configs(tmp_path)
    with TestClient(create_app(settings)) as first:
        chat(first, "hello")
    with TestClient(create_app(settings)) as second:
        _, samples = scrape(second)
    assert (
        sample(
            samples, "llm_sim_requests_total", model="big-mock", finish_reason="stop"
        )
        == 0
    )


# --- Metrics summary ---


def test_summary_records_completed_requests(client):
    chat(client, "hello", model="big-mock")
    stream_chunks(
        client,
        messages=[{"role": "user", "content": "anything"}],
        max_completion_tokens=3,
    )
    response = client.get("/metrics/summary")
    assert response.status_code == 200
    model = response.json()["models"]["big-mock"]

    assert model["engine"] == "instant"
    assert model["requests"]["finished"]["stop"] == 1
    assert model["requests"]["finished"]["length"] == 1
    assert model["requests"]["running"] == 0
    generated = len(RegexTokenizer().encode("Hello from big-mock!")) + 3
    assert model["tokens"] == {
        "prompt": prompt_tokens("hello") + prompt_tokens("anything"),
        "generation": generated,
    }
    latency = model["latency_seconds"]
    assert latency["ttft"]["count"] == 2
    assert latency["e2e"]["count"] == 2
    assert latency["lag"]["count"] == generated
    assert latency["itl"]["count"] == generated - 2
    assert latency["ttft"]["p50"] >= 0
    assert model["throughput"]["output_tokens_per_second"] > 0
    assert (
        response.json()["models"]["small-mock"]["latency_seconds"]["ttft"]["count"] == 0
    )


def test_summary_reset(client):
    chat(client, "hello")
    response = client.post("/metrics/summary/reset")
    assert response.status_code == 204
    assert response.content == b""

    body = client.get("/metrics/summary").json()
    model = body["models"]["big-mock"]
    assert body["since_reset_seconds"] < 5
    assert sum(model["requests"]["finished"].values()) == 0
    assert model["latency_seconds"]["ttft"]["count"] == 0
    # Prometheus counters only go up; the reset leaves them alone.
    _, samples = scrape(client)
    assert (
        sample(
            samples, "llm_sim_requests_total", model="big-mock", finish_reason="stop"
        )
        == 1
    )


def test_summary_windows_come_from_settings(tmp_path):
    settings = dataclasses.replace(
        write_configs(tmp_path), summary_request_samples=2, summary_token_samples=3
    )
    with TestClient(create_app(settings)) as client:
        for _ in range(3):
            chat(client, "hello")
        latency = client.get("/metrics/summary").json()["models"]["big-mock"][
            "latency_seconds"
        ]
    assert (latency["ttft"]["count"], latency["ttft"]["samples"]) == (3, 2)
    assert latency["lag"]["samples"] == 3


def test_metrics_routes_are_not_in_the_openapi_schema(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert not any(path.startswith("/metrics") for path in paths)


# --- Content parts ---


def parts(*texts):
    return [{"type": "text", "text": t} for t in texts]


def chat_with(client, content, **extra):
    body = {
        "model": "big-mock",
        "messages": [{"role": "user", "content": content}],
        **extra,
    }
    return client.post("/v1/chat/completions", json=body)


def test_text_parts_get_the_same_answer_as_a_string(client):
    as_string = chat_with(client, "hello").json()
    as_parts = chat_with(client, parts("hello")).json()
    assert as_parts["choices"][0]["message"]["content"] == "Hello from big-mock!"
    assert as_parts["choices"] == as_string["choices"]
    assert as_parts["usage"] == as_string["usage"]


def test_text_parts_are_joined_before_matching(client):
    # "col" + "ours?" is "colours?", which the colours entry matches.
    response = chat_with(client, parts("col", "ours?"), seed=1)
    assert response.json()["choices"][0]["message"]["content"] in {
        "Red.",
        "Green.",
        "Blue.",
        "Yellow.",
        "Purple.",
    }
    assert response.json()["usage"]["prompt_tokens"] == prompt_tokens("colours?")


def test_system_message_as_parts(client):
    body = {
        "model": "big-mock",
        "messages": [
            {"role": "system", "content": parts("Be ", "brief.")},
            {"role": "user", "content": "hello"},
        ],
    }
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == (
        "Hello from big-mock!"
    )


def test_streamed_request_with_parts(client):
    chunks, _ = stream_chunks(
        client, messages=[{"role": "user", "content": parts("hel", "lo")}]
    )
    text = "".join(
        chunk["choices"][0]["delta"].get("content") or ""
        for chunk in chunks
        if chunk["choices"]
    )
    assert text == "Hello from big-mock!"


@pytest.mark.parametrize("stream", [False, True])
def test_image_part_is_rejected(client, stream):
    content = [
        {"type": "text", "text": "What is this?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
    ]
    response = chat_with(client, content, stream=stream)
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "messages.[0].content.[1].type"
    assert "image_url is not supported" in error["message"]


@pytest.mark.parametrize(
    "content, param, message",
    [
        (
            [{"type": "video", "text": "x"}],
            "messages.[0].content.[0].type",
            "Invalid value for 'messages.[0].content.[0].type'",
        ),
        (
            [{"type": "text"}],
            "messages.[0].content.[0].text",
            "Missing required parameter: 'messages.[0].content.[0].text'.",
        ),
        ([], "messages.[0].content", "Invalid value for 'messages.[0].content'"),
        (
            5,
            "messages.[0].content",
            "Invalid value for 'messages.[0].content': Input should be a string "
            "or an array of content parts.",
        ),
    ],
)
def test_invalid_content_is_rejected(client, content, param, message):
    response = chat_with(client, content)
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["param"] == param
    assert error["message"].startswith(message)


def test_rejected_content_is_not_traced(client):
    chat_with(client, [{"type": "image_url", "image_url": {"url": "x"}}])
    model = client.get("/metrics/summary").json()["models"]["big-mock"]
    assert sum(model["requests"]["finished"].values()) == 0
    assert model["requests"]["running"] == 0


def test_summary_reports_the_simulated_timeline(client):
    # big-mock uses the instant engine: every token is due at arrival.
    chat(client, "hello")
    latency = client.get("/metrics/summary").json()["models"]["big-mock"][
        "latency_seconds"
    ]
    assert latency["ttft_simulated"]["count"] == 1
    assert latency["ttft_simulated"]["p50"] == 0.0
    assert latency["e2e_simulated"]["p50"] == 0.0
    # Observed = simulated + lag, so observed is never earlier.
    assert latency["ttft"]["p50"] >= latency["ttft_simulated"]["p50"]
