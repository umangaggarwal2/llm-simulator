"""The engine interface: engines decide when each token of a plan is emitted.

Implementations live in their own modules (instant.py, fixed_rate.py, ...).
This module never imports them; choosing one by name is runtime.py's job.
"""

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
