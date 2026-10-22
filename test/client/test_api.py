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
from llm_simulator.runtime import ModelRuntime
from llm_simulator.settings import Settings
from llm_simulator.simulator import engine as engine_module
from llm_simulator.simulator.engine import InstantEngine
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


def test_engines_start_and_stop_with_the_app(tmp_path, monkeypatch):
    monkeypatch.setitem(engine_module._ENGINES, "recording", RecordingEngine)
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
    monkeypatch.setitem(engine_module._ENGINES, "recording", RecordingEngine)
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
