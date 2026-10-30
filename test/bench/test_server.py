import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from llm_simulator.bench.client import open_session
from llm_simulator.bench.config import BenchConfig
from llm_simulator.bench.server import ServerMetrics, summary_url

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


SECTION = {"engine": "fixed_rate", "latency_seconds": {"ttft": {"p50": 0.008}}}


@pytest.mark.parametrize(
    "base_url, expected",
    [
        ("http://h:8080/v1", "http://h:8080/metrics/summary"),
        ("http://h:8080/v1/", "http://h:8080/metrics/summary"),
        ("http://h:8080/api", "http://h:8080/api/metrics/summary"),
    ],
)
def test_summary_url(base_url, expected):
    assert summary_url(base_url) == expected


async def probe(reset_status=204, summary=None, summary_status=200, model="m"):
    """Reset and fetch against a scripted metrics endpoint."""
    calls = []

    async def reset(request):
        calls.append("reset")
        return web.Response(status=reset_status)

    async def get(request):
        calls.append("fetch")
        if summary_status != 200:
            return web.Response(status=summary_status)
        body = summary if summary is not None else {"models": {"m": SECTION}}
        text = body if isinstance(body, str) else json.dumps(body)
        return web.Response(text=text, content_type="application/json")

    app = web.Application()
    app.router.add_post("/metrics/summary/reset", reset)
    app.router.add_get("/metrics/summary", get)
    server = TestServer(app)
    await server.start_server()
    try:
        url = str(server.make_url("/metrics/summary"))
        async with open_session(BenchConfig(model=model)) as session:
            metrics = ServerMetrics(session, url, model)
            available = await metrics.reset()
            return available, await metrics.fetch(), calls
    finally:
        await server.close()


async def test_reset_then_fetch():
    available, summary, calls = await probe(
        summary={"since_reset_seconds": 2.5, "models": {"m": SECTION}}
    )
    assert available and calls == ["reset", "fetch"]
    assert summary.available and summary.reason is None
    assert summary.model == SECTION
    assert summary.since_reset == 2.5


@pytest.mark.parametrize("status", [404, 405, 500])
async def test_no_summary_endpoint(status):
    available, summary, calls = await probe(reset_status=status)
    assert not available
    assert (summary.available, summary.reason) == (False, f"HTTP {status}")
    # Nothing to read after a failed reset.
    assert calls == ["reset"]


async def test_missing_model():
    _, summary, _ = await probe(model="other", summary={"models": {"m": SECTION}})
    assert summary.reason == "no section for model 'other'"


@pytest.mark.parametrize("body", ["{not json", '["a list"]', '{"models": 5}'])
async def test_unexpected_bodies_are_unavailable(body):
    _, summary, _ = await probe(summary=body)
    assert not summary.available


async def test_fetch_error_status():
    _, summary, _ = await probe(summary_status=503)
    assert summary.reason == "HTTP 503"


async def test_unreachable_server_does_not_raise():
    async with open_session(BenchConfig(model="m")) as session:
        metrics = ServerMetrics(session, "http://127.0.0.1:1/metrics/summary", "m")
        assert not await metrics.reset()
        summary = await metrics.fetch()
    assert summary.reason.startswith("no response")
