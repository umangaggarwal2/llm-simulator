"""Engines decide when each token of a generation plan is emitted."""

from abc import ABC, abstractmethod
from typing import AsyncIterator, ClassVar, Tuple

from llm_simulator.types import GenerationPlan, TokenEvent


class Engine(ABC):
    """Turns a plan into a timed stream of token events.

    Engines never change what is said, only when. Callers may close the
    iterator early (for example when a client disconnects); engines that hold
    resources for a sequence release them in a try/finally around their loop.
    """

    name: ClassVar[str]
    # models.yml sections a model must define to use this engine, such as
    # "performance"; checked at startup.
    required_sections: ClassVar[Tuple[str, ...]] = ()

    @abstractmethod
    def run(self, plan: GenerationPlan, arrival: float) -> AsyncIterator[TokenEvent]:
        """Yield the plan's tokens in order.

        arrival is the time.monotonic() value recorded when the request came
        in; timed engines compute their deadlines from it.
        """

    async def start(self) -> None:
        """Called once when the app starts, e.g. to launch a background loop."""

    async def stop(self) -> None:
        """Called once when the app shuts down."""


class InstantEngine(Engine):
    """Emits every token immediately, with no simulated latency.

    Used to serve responses before any timing model exists, and for tests that
    check content rather than timing.
    """

    name = "instant"

    async def run(
        self, plan: GenerationPlan, arrival: float
    ) -> AsyncIterator[TokenEvent]:
        # No sleep between tokens: when streaming, each SSE write already
        # yields to the event loop.
        for index, text in enumerate(plan.tokens):
            yield TokenEvent(index, text, sim_time=0.0)


_ENGINES = {InstantEngine.name: InstantEngine}


def get_engine(name: str) -> Engine:
    """Build an engine by its config name, e.g. "instant"."""
    try:
        return _ENGINES[name]()
    except KeyError:
        available = ", ".join(sorted(_ENGINES))
        raise ValueError(f"unknown engine '{name}'; available: {available}") from None
