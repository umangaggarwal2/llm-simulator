"""Domain types shared by the API, generator and engine layers.

This module imports nothing from the project, so the llm and simulator
packages can both depend on it without depending on each other.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

FinishReason = Literal["stop", "length"]


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class GenerationRequest:
    model: str
    messages: Tuple[Message, ...]
    max_tokens: Optional[int] = None
    # Overrides the seed in responses.yml when set.
    seed: Optional[int] = None


@dataclass(frozen=True)
class GenerationPlan:
    """What to send for one request. Timing is decided later by the engine."""

    prompt_tokens: int
    tokens: Tuple[str, ...]
    finish_reason: FinishReason
    # The responses.yml entry that answered, and which of its choices was picked.
    entry_id: str
    variant: int

    @property
    def completion_tokens(self) -> int:
        return len(self.tokens)

    @property
    def text(self) -> str:
        return "".join(self.tokens)


@dataclass(frozen=True)
class TokenEvent:
    """One token of a plan, as emitted by an engine."""

    # Position in the completion: 0, 1, 2, ...
    index: int
    text: str
    # Simulated seconds after the request arrived at which this token is due,
    # so token 0's value is the simulated TTFT and the gaps between tokens are ITLs.
    sim_time: float
    # Seconds the engine released this token after its deadline: the
    # simulator's own fidelity error, not part of the simulated timeline.
    # Observed TTFT is sim_time + lag for token 0, and observed E2E is the last
    # token's sim_time + lag; lag never accumulates across tokens.
    lag: float = 0.0
