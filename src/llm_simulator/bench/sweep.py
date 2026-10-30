"""Running the same benchmark at several load levels, one after another.

Each level is a complete run (warm-up, reset, measure, drain, fetch) with
the same prompts in the same order, so differences between levels come
from the load alone. A level that fails is recorded and the sweep goes on.
"""

import dataclasses
from dataclasses import dataclass
from typing import Awaitable, Callable, List, Optional, Sequence

from llm_simulator.bench.config import BenchConfig
from llm_simulator.bench.results import BenchResult
from llm_simulator.bench.runner import WarmupFailed

# What a sweep can vary: closed-loop concurrency or open-loop rate.
PARAMETERS = ("concurrency", "rate")


@dataclass(frozen=True)
class SweepLevel:
    parameter: str
    value: float
    config: BenchConfig
    # Exactly one is set: the level's result, or why it could not be run.
    result: Optional[BenchResult] = None
    error: Optional[str] = None


def levels(
    config: BenchConfig, parameter: str, values: Sequence[float]
) -> List[BenchConfig]:
    """One config per value, otherwise the same as `config`."""
    if parameter not in PARAMETERS:
        raise ValueError(f"cannot sweep {parameter!r}")
    return [dataclasses.replace(config, **{parameter: value}) for value in values]


async def sweep(
    parameter: str,
    configs: Sequence[BenchConfig],
    run_one: Callable[[BenchConfig], Awaitable[BenchResult]],
    on_level: Callable[[int, SweepLevel], None] = lambda number, level: None,
) -> List[SweepLevel]:
    """Run every level in order; on_level is told as each one finishes."""
    done = []
    for number, config in enumerate(configs, start=1):
        value = getattr(config, parameter)
        try:
            level = SweepLevel(parameter, value, config, result=await run_one(config))
        except WarmupFailed as exc:
            level = SweepLevel(parameter, value, config, error=str(exc))
        done.append(level)
        on_level(number, level)
    return done
