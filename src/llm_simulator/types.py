"""Domain types shared by the API, generator and engine layers.

This module imports nothing from the project, so the llm and simulator
packages can both depend on it without depending on each other.
"""

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple

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
    # Simulated seconds the request waited before an engine with a waiting
    # queue admitted it. Set on token 0 only, and only by such engines.
    queue_time: Optional[float] = None


@dataclass(frozen=True)
class StepStats:
    """One scheduler step, reported after its tokens were released.

    The batch fields describe the step as it ran. waiting, running and the
    KV fields are the state after it completed, so the step before the engine
    goes idle leaves them at 0 rather than at its batch's values.
    """

    # Simulated seconds: max(compute, memory) + overhead.
    duration: float
    compute: float
    memory: float
    overhead: float
    sequences: int
    prefill_tokens: int
    decode_tokens: int
    admitted: int
    preempted: int
    # Tokens whose KV preemption threw away, to be computed again.
    preempted_tokens: int
    # Why requests were left waiting: "kv", "max_sequences", "budget" or
    # "preemption"; None when none were.
    blocked: Optional[str]
    waiting: int
    running: int
    kv_blocks_used: int
    kv_blocks_total: int

    @property
    def bound(self) -> str:
        """Which limit set the step's GPU time: "compute" or "memory"."""
        return "compute" if self.compute > self.memory else "memory"


# Receives every step of one engine; runtime.py binds the model's name.
StepListener = Callable[[StepStats], None]
