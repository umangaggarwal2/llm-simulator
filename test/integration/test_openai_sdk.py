"""End-to-end checks with OpenAI's official Python SDK.

A real uvicorn server runs in a background thread, and the SDK talks to it
over HTTP exactly as it would talk to api.openai.com. If the SDK can parse
every response and error, clients written for OpenAI work unchanged.
"""

import socket
import threading
import time

import openai
import pytest
import uvicorn
import yaml
from openai import AsyncOpenAI, OpenAI

from llm_simulator.client.api import create_app
from llm_simulator.llm.tokenizer import RegexTokenizer
from llm_simulator.settings import Settings
from llm_simulator.types import Message

pytestmark = pytest.mark.integration

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


@pytest.fixture(scope="module")
def base_url(tmp_path_factory):
    config_dir = tmp_path_factory.mktemp("config")
    (config_dir / "models.yml").write_text(yaml.safe_dump(MODELS))
    (config_dir / "responses.yml").write_text(yaml.safe_dump(RESPONSES))
    app = create_app(
        Settings(
            models_path=config_dir / "models.yml",
            responses_path=config_dir / "responses.yml",
        )
    )

    # Bind the socket first, so the port is known and cannot be taken by
    # anything else before the server starts.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]

    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]})
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise RuntimeError("simulator did not start")
        time.sleep(0.01)

    yield f"http://127.0.0.1:{port}/v1"

    server.should_exit = True
    thread.join(timeout=10)
    sock.close()


@pytest.fixture
def client(base_url):
    # max_retries=0 so error responses surface at once instead of being retried.
    with OpenAI(base_url=base_url, api_key="not-needed", max_retries=0) as client:
        yield client


def user(text):
    return [{"role": "user", "content": text}]


def prompt_tokens(*messages):
    return RegexTokenizer().count_messages([Message(**m) for m in messages])


# --- Models ---


def test_list_models(client):
    models = client.models.list()
    assert [model.id for model in models] == ["big-mock", "small-mock"]
    assert all(model.object == "model" for model in models)


# --- Non-streaming ---


def test_chat_completion(client):
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "hello"},
    ]
    response = client.chat.completions.create(model="small-mock", messages=messages)

    assert response.object == "chat.completion"
    assert response.model == "small-mock"
    choice = response.choices[0]
    assert choice.message.role == "assistant"
    assert choice.message.content == "Hello from small-mock!"
    assert choice.finish_reason == "stop"
    assert response.usage.prompt_tokens == prompt_tokens(*messages)
    assert response.usage.total_tokens == (
        response.usage.prompt_tokens + response.usage.completion_tokens
    )


def test_max_completion_tokens_truncates(client):
    response = client.chat.completions.create(
        model="big-mock", messages=user("anything"), max_completion_tokens=3
    )
    assert response.choices[0].message.content == "one two three"
    assert response.choices[0].finish_reason == "length"
    assert response.usage.completion_tokens == 3


def test_seed_is_reproducible(client):
    def answer(seed):
        response = client.chat.completions.create(
            model="big-mock", messages=user("colours please"), seed=seed
        )
        return response.choices[0].message.content

    assert answer(7) == answer(7)
    assert len({answer(seed) for seed in range(20)}) > 1


# --- Streaming ---


def test_streaming(client):
    stream = client.chat.completions.create(
        model="big-mock", messages=user("hello"), stream=True
    )
    chunks = list(stream)

    assert chunks[0].choices[0].delta.role == "assistant"
    text = "".join(chunk.choices[0].delta.content or "" for chunk in chunks)
    assert text == "Hello from big-mock!"
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert all(chunk.usage is None for chunk in chunks)
    assert len({chunk.id for chunk in chunks}) == 1


def test_streaming_with_usage(client):
    stream = client.chat.completions.create(
        model="big-mock",
        messages=user("anything"),
        stream=True,
        stream_options={"include_usage": True},
        max_completion_tokens=4,
    )
    chunks = list(stream)

    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert text == "one two three four"
    finish = [c.choices[0].finish_reason for c in chunks if c.choices]
    assert finish[-1] == "length"
    assert chunks[-1].choices == []
    assert chunks[-1].usage.completion_tokens == 4
    assert chunks[-1].usage.prompt_tokens == prompt_tokens(*user("anything"))


@pytest.mark.anyio
async def test_async_client_streaming(base_url):
    async with AsyncOpenAI(
        base_url=base_url, api_key="not-needed", max_retries=0
    ) as client:
        stream = await client.chat.completions.create(
            model="big-mock", messages=user("hello"), stream=True
        )
        text = "".join([chunk.choices[0].delta.content or "" async for chunk in stream])
    assert text == "Hello from big-mock!"


# --- Errors ---


def test_unknown_model(client):
    with pytest.raises(openai.NotFoundError) as error:
        client.chat.completions.create(model="gpt-9", messages=user("hi"))
    assert error.value.status_code == 404
    assert error.value.code == "model_not_found"
    assert error.value.param == "model"


def test_context_length_exceeded(client):
    with pytest.raises(openai.BadRequestError) as error:
        client.chat.completions.create(
            model="small-mock", messages=user("hello"), max_tokens=100
        )
    assert error.value.status_code == 400
    assert error.value.code == "context_length_exceeded"
    assert error.value.param == "messages"
    assert "maximum context length is 40 tokens" in error.value.message


def test_context_length_exceeded_when_streaming(client):
    # The error must arrive as a normal 400 before any stream starts.
    with pytest.raises(openai.BadRequestError) as error:
        client.chat.completions.create(
            model="small-mock", messages=user("hello"), max_tokens=100, stream=True
        )
    assert error.value.code == "context_length_exceeded"
