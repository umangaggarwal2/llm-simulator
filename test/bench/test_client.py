import asyncio
import json
import socket

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from llm_simulator.bench.client import Event, build_body, open_session, parse_line
from llm_simulator.bench.client import stream_chat
from llm_simulator.bench.config import BenchConfig
from llm_simulator.bench.workload import RequestSpec

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def chunk(delta=None, finish_reason=None, usage=None):
    choices = (
        []
        if delta is None
        else [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
    )
    return {"object": "chat.completion.chunk", "choices": choices, "usage": usage}


def data(payload):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return f"data: {text}\n\n".encode()


# --- parse_line ---


def test_role_chunk_carries_no_content():
    event = parse_line(data(chunk({"role": "assistant", "content": ""})))
    assert event == Event()


def test_content_chunk():
    assert parse_line(data(chunk({"content": "Hi"}))).content == "Hi"


def test_finish_chunk():
    assert parse_line(data(chunk({}, finish_reason="length"))).finish_reason == (
        "length"
    )


def test_usage_chunk():
    usage = {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11}
    assert parse_line(data(chunk(usage=usage))).usage == usage


def test_done():
    assert parse_line(b"data: [DONE]\n").done


@pytest.mark.parametrize("line", [b"\n", b": keep-alive\n", b"event: message\n"])
def test_lines_without_data(line):
    assert parse_line(line) is None


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"error": {"message": "boom", "type": "server_error"}}, "boom"),
        ({"object": "error", "message": "vllm boom"}, "vllm boom"),
    ],
)
def test_in_stream_errors(payload, message):
    assert parse_line(data(payload)).error == message


def test_invalid_json_raises_value_error():
    with pytest.raises(ValueError):
        parse_line(b"data: {not json\n")


def test_body():
    body = json.loads(build_body(BenchConfig(model="m"), RequestSpec("hello", 7)))
    assert body == {
        "model": "m",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_completion_tokens": 7,
    }


# --- stream_chat against a scripted server ---

USAGE = {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11}
NORMAL = [
    chunk({"role": "assistant", "content": ""}),
    chunk({"content": "a"}),
    chunk({"content": "b"}),
    chunk({"content": "c"}),
    chunk({}, finish_reason="length"),
    chunk(usage=USAGE),
    "[DONE]",
]


class StepClock:
    """Each reading is one later than the last, so times count clock reads."""

    def __init__(self):
        self.now = -1.0

    def __call__(self):
        self.now += 1
        return self.now


class Scripted:
    """An aiohttp app that streams a scripted list of SSE events."""

    def __init__(self, events=NORMAL, status=200, body=None, pause=0.0):
        self.events = events
        self.status = status
        self.body = body
        self.pause = pause
        self.requests = []

    async def handle(self, request):
        self.requests.append((dict(request.headers), await request.json()))
        if self.status != 200:
            return web.Response(status=self.status, text=self.body)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for event in self.events:
            if event == "pause":
                await asyncio.sleep(self.pause)
                continue
            await response.write(data(event))
        return response


async def run(scripted, clock=None, **config):
    app = web.Application()
    app.router.add_post("/v1/chat/completions", scripted.handle)
    server = TestServer(app)
    await server.start_server()
    try:
        config = BenchConfig(model="m", base_url=str(server.make_url("/v1")), **config)
        async with open_session(config) as session:
            kwargs = {} if clock is None else {"clock": clock}
            return await stream_chat(
                session, config, 5, RequestSpec("hello", 3), **kwargs
            )
    finally:
        await server.close()


async def test_normal_stream():
    scripted = Scripted()
    record = await run(scripted, clock=StepClock())
    assert record.ok
    assert (record.index, record.status, record.error) == (5, 200, None)
    # Clock reads: sent=0, then one per line, each event followed by a blank
    # line: role 1-2, a 3-4, b 5-6, c 7-8, finish 9-10, usage 11-12, [DONE] 13.
    assert record.sent == 0
    assert record.ttft == 3
    assert list(record.itls) == [2, 2]
    assert record.last_token == 7
    assert record.end == record.e2e == 13
    assert record.chunks == 3
    assert (record.prompt_tokens, record.completion_tokens) == (8, 3)
    assert record.finish_reason == "length"

    assert record.requested_tokens == 3
    headers, body = scripted.requests[0]
    assert body["model"] == "m" and body["stream"] is True
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert body["max_completion_tokens"] == 3
    assert "Authorization" not in headers


async def test_api_key_is_sent_as_a_bearer_token():
    scripted = Scripted()
    await run(scripted, api_key="sk-test")
    assert scripted.requests[0][0]["Authorization"] == "Bearer sk-test"


async def test_http_error_keeps_the_openai_message():
    body = json.dumps({"error": {"message": "too long", "type": "x"}})
    record = await run(Scripted(status=400, body=body))
    assert not record.ok
    assert (record.status, record.error) == (400, "too long")
    assert record.chunks == 0


async def test_http_error_without_a_json_body():
    record = await run(Scripted(status=502, body="bad gateway"))
    assert (record.status, record.error) == (502, "HTTP 502: bad gateway")


async def test_stream_cut_before_done():
    record = await run(Scripted(events=NORMAL[:3]))
    assert record.status == 200
    assert record.error == "stream ended before [DONE]"
    assert record.chunks == 2
    assert not record.ok


async def test_in_stream_error_fails_the_request():
    events = [chunk({"content": "a"}), {"error": {"message": "engine died"}}, "[DONE]"]
    record = await run(Scripted(events=events))
    assert record.error == "engine died"
    assert not record.ok


async def test_invalid_stream_data():
    record = await run(Scripted(events=["{not json"]))
    assert record.error.startswith("invalid stream data")


async def test_stalled_stream_times_out():
    events = [chunk({"content": "a"}), "pause", "[DONE]"]
    record = await run(Scripted(events=events, pause=1.0), timeout=0.1)
    assert record.status == 200
    assert "Timeout" in record.error
    assert record.chunks == 1


async def test_connection_refused_is_recorded_not_raised():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()  # nothing listens on this port now
    config = BenchConfig(model="m", base_url=f"http://127.0.0.1:{port}/v1")
    async with open_session(config) as session:
        record = await stream_chat(session, config, 0, RequestSpec("hi", 1))
    assert record.status == 0
    assert "ClientConnectorError" in record.error
    assert record.end >= record.sent
