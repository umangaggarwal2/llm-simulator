"""Token bookkeeping for one request inside the scheduler."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(eq=False)
class Sequence:
    """How far one request has got: which tokens exist, and which have KV.

    A single rule covers every phase. Each step computes some of the
    `uncomputed` tokens; when none are left afterwards, the step has produced
    a new token. Prefill computes the whole prompt, a chunked prefill part of
    it, a decode step the one token generated last, and a recomputation after
    preemption the prompt plus everything generated so far.

    `output_tokens` comes from the plan and is used only to stop, like an
    end-of-sequence token; admission must never look at it, because a real
    server cannot know it in advance.

    Sequences compare by identity: two requests with the same progress are
    still different requests.
    """

    id: int
    arrival: float
    prompt_tokens: int
    output_tokens: int
    generated: int = 0
    # Tokens whose KV is in the cache.
    computed: int = 0
    # When the sequence was first admitted; readmission after preemption
    # does not change it, so queue time counts the first wait only.
    admitted_at: Optional[float] = None
    preemptions: int = 0
    finished: bool = field(default=False, init=False)

    def __post_init__(self):
        if self.prompt_tokens < 1:
            raise ValueError("prompt_tokens must be at least 1")
        if self.output_tokens < 0:
            raise ValueError("output_tokens must not be negative")

    @property
    def known(self) -> int:
        """The prompt and every token generated so far."""
        return self.prompt_tokens + self.generated

    @property
    def uncomputed(self) -> int:
        return self.known - self.computed

    @property
    def prefilling(self) -> bool:
        """Whether the next step computes more than the last generated token."""
        return self.uncomputed > 1 or self.generated == 0

    def advance(self, tokens: int) -> bool:
        """Apply a step that computed `tokens` of this sequence.

        Returns whether the step produced a token for the client.
        """
        if self.finished:
            raise ValueError(f"sequence {self.id} has finished")
        if not 1 <= tokens <= self.uncomputed:
            raise ValueError(
                f"sequence {self.id} has {self.uncomputed} uncomputed tokens, "
                f"not {tokens}"
            )
        self.computed += tokens
        if self.uncomputed:
            return False  # a prefill chunk that is not the last
        if self.output_tokens == 0:
            # The prompt was processed, but there is nothing to send.
            self.finished = True
            return False
        self.generated += 1
        self.finished = self.generated == self.output_tokens
        return True

    def preempt(self) -> None:
        """Drop the sequence's KV; it must recompute everything it knows."""
        self.computed = 0
        self.preemptions += 1
