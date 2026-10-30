import pytest

from llm_simulator.bench.config import BenchConfig
from llm_simulator.bench.runner import WarmupFailed
from llm_simulator.bench.sweep import levels, sweep

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_levels_vary_one_parameter():
    base = BenchConfig(model="m", concurrency=1, requests=50, input_tokens=100)
    configs = levels(base, "concurrency", [1, 8, 32])
    assert [config.concurrency for config in configs] == [1, 8, 32]
    assert all(config.input_tokens == 100 for config in configs)
    rates = levels(BenchConfig(model="m", rate=1.0), "rate", [1.0, 5.0])
    assert [config.rate for config in rates] == [1.0, 5.0]


def test_levels_are_validated():
    with pytest.raises(ValueError):
        levels(BenchConfig(model="m", concurrency=1), "concurrency", [0])
    with pytest.raises(ValueError):
        levels(BenchConfig(model="m"), "seed", [1])


async def test_runs_every_level_in_order_and_survives_a_failure():
    ran, reported = [], []

    async def run_one(config):
        ran.append(config.concurrency)
        if config.concurrency == 8:
            raise WarmupFailed([])
        return f"result {config.concurrency}"

    configs = levels(BenchConfig(model="m", concurrency=1), "concurrency", [1, 8, 32])
    done = await sweep(
        "concurrency",
        configs,
        run_one,
        lambda number, level: reported.append((number, level.value)),
    )
    assert ran == [1, 8, 32]
    assert reported == [(1, 1), (2, 8), (3, 32)]
    assert [level.result for level in done] == ["result 1", None, "result 32"]
    assert done[1].error == "all 0 warm-up requests failed"
