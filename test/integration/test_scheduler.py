"""SchedulerEngine end to end: a real server, the OpenAI SDK, concurrent streams."""

import asyncio
import time

import httpx2
import pytest
from openai import AsyncOpenAI
from prometheus_client.parser import text_string_to_metric_families

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

# A slow GPU, so steps are long enough to measure over HTTP: 1 GB of weights
# at 100 GB/s makes every step about 10 ms, whatever its batch.
HARDWARE = {
    "slow-gpu": {"hbm_gib": 2, "memory_bandwidth_gbps": 100, "peak_tflops": 100}
}
MODEL = {
    "context_length": 1000,
    "engine": "scheduler",
    "hardware": "slow-gpu",
    "weights": {"parameters_billions": 0.5},
    "kv_cache": {"bytes_per_token": 1024},
}
MODELS = {
    "version": 1,
    "hardware": HARDWARE,
    "models": [
        {
            "name": "batched",
            **MODEL,
            "scheduler": {"max_batch_tokens": 512, "max_num_sequences": 16},
        },
        {
            "name": "one-at-a-time",
            **MODEL,
            "scheduler": {"max_batch_tokens": 512, "max_num_sequences": 1},
        },
    ],
}
TEXT = "one two three four five six seven eight"
RESPONSES = {"version": 1, "default": {"content": TEXT}}
STEP = 0.010
TOKENS = 8  # what the regex tokenizer makes of TEXT


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module")
def base_url(tmp_path_factory, serve):
    with serve(tmp_path_factory.mktemp("config"), MODELS, RESPONSES) as root:
        yield f"{root}/v1"


async def stream_text(client, model):
    stream = await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": "hi"}], stream=True
    )
    return "".join([chunk.choices[0].delta.content or "" async for chunk in stream])


async def timed_batch(base_url, model, n):
    async with AsyncOpenAI(
        base_url=base_url, api_key="not-needed", max_retries=0
    ) as client:
        start = time.monotonic()
        texts = await asyncio.gather(*(stream_text(client, model) for _ in range(n)))
        return texts, time.monotonic() - start


def scrape(base_url):
    text = httpx2.get(base_url.removesuffix("/v1") + "/metrics").text
    return {
        (sample.name, tuple(sorted(sample.labels.items()))): sample.value
        for family in text_string_to_metric_families(text)
        for sample in family.samples
    }


def queue_time_sum(samples, model):
    return samples.get(
        ("llm_sim_request_queue_time_seconds_sum", (("model", model),)), 0.0
    )


async def test_streams_the_configured_content(base_url):
    [text], elapsed = await timed_batch(base_url, "batched", 1)
    assert text == TEXT
    # One prefill step, then a decode step per further token.
    assert elapsed >= TOKENS * STEP


async def test_concurrent_requests_are_batched_into_the_same_steps(base_url):
    texts, elapsed = await timed_batch(base_url, "batched", 8)
    assert texts == [TEXT] * 8
    # Eight requests in shared steps take about as long as one; served one
    # after another they would take eight times as long.
    assert elapsed < 3 * TOKENS * STEP


async def test_a_full_engine_queues_requests(base_url):
    before = queue_time_sum(scrape(base_url), "one-at-a-time")
    texts, elapsed = await timed_batch(base_url, "one-at-a-time", 2)
    after = queue_time_sum(scrape(base_url), "one-at-a-time")
    assert texts == [TEXT] * 2
    assert elapsed >= 2 * TOKENS * STEP
    # The second request waited for the whole of the first.
    assert after - before >= (TOKENS - 1) * STEP
