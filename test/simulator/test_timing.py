import pytest

from llm_simulator.simulator._timing import sleep_until

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FakeClock:
    def __init__(self, now):
        self.now = now
        self.sleeps = []

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


async def test_sleeps_until_a_future_deadline():
    clock = FakeClock(10.0)
    lag = await sleep_until(10.25, clock, clock.sleep)
    assert clock.sleeps == [pytest.approx(0.25)]
    assert lag == 0.0


async def test_past_deadline_returns_at_once_with_its_lag():
    clock = FakeClock(10.0)
    lag = await sleep_until(9.9, clock, clock.sleep)
    assert clock.sleeps == []
    assert lag == pytest.approx(0.1)


async def test_late_wake_up_is_reported_as_lag():
    clock = FakeClock(10.0)

    async def oversleep(seconds):
        clock.now += seconds + 0.002

    assert await sleep_until(10.5, clock, oversleep) == pytest.approx(0.002)


async def test_early_wake_up_sleeps_again_for_the_rest():
    clock = FakeClock(10.0)
    sleeps = []

    async def undersleep(seconds):
        # Like uvloop: the first sleep wakes 1.2 ms before the deadline; the
        # sleep for the remainder then lets that much time pass.
        sleeps.append(seconds)
        clock.now += seconds - 0.0012 if len(sleeps) == 1 else seconds

    lag = await sleep_until(10.5, clock, undersleep)
    assert clock.now >= 10.5
    assert lag == 0.0
    assert sleeps == [pytest.approx(0.5), pytest.approx(0.0012)]


def test_never_releases_early_on_uvloop():
    uvloop = pytest.importorskip("uvloop")
    import asyncio
    import random
    import time

    async def probe():
        earliest = float("inf")
        for _ in range(500):
            deadline = time.monotonic() + random.uniform(0.0001, 0.005)
            await sleep_until(deadline, time.monotonic, asyncio.sleep)
            earliest = min(earliest, time.monotonic() - deadline)
        return earliest

    assert uvloop.run(probe()) >= 0.0
