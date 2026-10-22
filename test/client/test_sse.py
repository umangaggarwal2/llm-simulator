import json

import pytest

from llm_simulator.client.sse import sse_stream
from llm_simulator.types import GenerationPlan, TokenEvent

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


PLAN = GenerationPlan(
    prompt_tokens=8,
    tokens=("Hello", " world"),
    finish_reason="stop",
    entry_id="e0",
    variant=0,
)


class TrackingEvents:
    """An engine-like event stream that records whether it was closed."""

    def __init__(self, tokens):
        self.tokens = tokens
        self.closed = False

    def __aiter__(self):
        return self._run()

    async def _run(self):
        try:
            for index, text in enumerate(self.tokens):
                yield TokenEvent(index, text, 0.0)
        finally:
            self.closed = True


async def events_for(tokens):
    for index, text in enumerate(tokens):
        yield TokenEvent(index, text, 0.0)


def parse(line):
    assert line.startswith("data: ") and line.endswith("\n\n")
    return json.loads(line[6:])


async def test_chunk_sequence():
    lines = [
        line
        async for line in sse_stream(
            events_for(PLAN.tokens), PLAN, "chatcmpl-1", 123, "m", include_usage=False
        )
    ]
    assert lines[-1] == "data: [DONE]\n\n"
    chunks = [parse(line) for line in lines[:-1]]
    assert [c["choices"][0]["delta"] for c in chunks] == [
        {"role": "assistant", "content": ""},
        {"content": "Hello"},
        {"content": " world"},
        {},
    ]
    assert [c["choices"][0]["finish_reason"] for c in chunks] == [
        None,
        None,
        None,
        "stop",
    ]
    assert all("usage" not in c for c in chunks)
    assert {c["id"] for c in chunks} == {"chatcmpl-1"}
    assert {c["object"] for c in chunks} == {"chat.completion.chunk"}


async def test_include_usage_adds_final_usage_chunk():
    lines = [
        line
        async for line in sse_stream(
            events_for(PLAN.tokens), PLAN, "chatcmpl-1", 123, "m", include_usage=True
        )
    ]
    chunks = [parse(line) for line in lines[:-1]]
    assert all(c["usage"] is None for c in chunks[:-1])
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"] == {
        "prompt_tokens": 8,
        "completion_tokens": 2,
        "total_tokens": 10,
    }


async def test_finish_reason_comes_from_the_plan():
    plan = GenerationPlan(8, ("a",), "length", "e0", 0)
    lines = [
        line
        async for line in sse_stream(events_for(plan.tokens), plan, "c", 1, "m", False)
    ]
    assert parse(lines[-2])["choices"][0]["finish_reason"] == "length"


async def test_engine_stream_is_closed_when_the_client_disconnects():
    events = TrackingEvents(["a", "b", "c", "d"])
    stream = sse_stream(aiter(events), PLAN, "c", 1, "m", False)
    await stream.__anext__()  # role chunk
    await stream.__anext__()  # first token
    await stream.aclose()  # what Starlette does when the client goes away
    assert events.closed
