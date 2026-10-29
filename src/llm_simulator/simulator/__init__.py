"""When tokens are emitted: the engine interface and its implementations."""

from llm_simulator.simulator.engine import Engine
from llm_simulator.simulator.fixed_rate import FixedRateEngine
from llm_simulator.simulator.instant import InstantEngine

__all__ = ["Engine", "FixedRateEngine", "InstantEngine"]
