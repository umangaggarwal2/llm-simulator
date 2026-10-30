import asyncio
import time

import pytest

from llm_simulator.bench.health import LoopMonitor
from llm_simulator.bench.records import LoopSamples
from llm_simulator.bench.results import client_health

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def monitored(work):
    monitor = LoopMonitor(interval=0.005)
    monitor.start()
    await work()
    return await monitor.stop()


async def test_an_idle_loop_is_healthy():
    samples = await monitored(lambda: asyncio.sleep(0.2))
    assert len(samples.lags) >= 20
    health = client_health(samples)
    assert not health.saturated
    assert 0 <= health.cpu_percent < 50


async def test_a_blocked_loop_is_saturated():
    async def block():
        # Busy work that holds the event loop, as an overloaded client would
        # (time.sleep would block it too, but without using CPU).
        for _ in range(6):
            until = time.perf_counter() + 0.05
            while time.perf_counter() < until:
                pass
            await asyncio.sleep(0)

    health = client_health(await monitored(block))
    assert health.saturated
    assert health.loop_lag["max"] >= 0.04
    assert health.cpu_percent > 50


async def test_stop_ends_the_sampling_task():
    monitor = LoopMonitor(interval=0.001)
    monitor.start()
    await asyncio.sleep(0.01)
    samples = await monitor.stop()
    count = len(samples.lags)
    await asyncio.sleep(0.01)
    assert len(samples.lags) == count


def test_health_from_samples():
    from array import array

    samples = LoopSamples(array("d", [0.001] * 98 + [0.02, 0.03]), 0.5, 2.0)
    health = client_health(samples)
    assert health.cpu_percent == 25.0
    assert health.saturated  # p99 is above 5 ms
    assert client_health(LoopSamples(array("d"), 0.0, 0.0)).saturated is False
