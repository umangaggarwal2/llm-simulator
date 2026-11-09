"""Scheduling decisions: what each step computes, in the style of vLLM.

Pure: no clock and no asyncio. The driver calls schedule(), waits for the
step's duration, then calls complete() with the same plan.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

from llm_simulator.simulator.scheduler.kv_cache import BlockAllocator
from llm_simulator.simulator.scheduler.sequence import Sequence


@dataclass(frozen=True)
class Work:
    """One sequence's share of a step."""

    seq: Sequence
    tokens: int
    # Tokens whose KV the sequence held before the step.
    cached: int
    prefill: bool


@dataclass(frozen=True)
class Preemption:
    seq: Sequence
    # Tokens whose KV was thrown away and must be computed again.
    discarded: int


@dataclass(frozen=True)
class StepPlan:
    work: Tuple[Work, ...]
    admitted: Tuple[Sequence, ...]
    preempted: Tuple[Preemption, ...]
    # Why requests were left waiting: "kv", "max_sequences", "budget" or
    # "preemption"; None when nothing is waiting.
    blocked: Optional[str]
    # After scheduling.
    waiting: int
    running: int
    kv_usage: float

    @property
    def shape(self) -> Tuple[Tuple[int, int], ...]:
        """(tokens, cached) per sequence, as the cost model takes them."""
        return tuple((w.tokens, w.cached) for w in self.work)

    @property
    def prefill_tokens(self) -> int:
        return sum(w.tokens for w in self.work if w.prefill)

    @property
    def decode_tokens(self) -> int:
        return sum(w.tokens for w in self.work if not w.prefill)


@dataclass(frozen=True)
class StepResult:
    # Sequences that produced a token, in batch order.
    emitted: Tuple[Sequence, ...]
    # Sequences that are done; their blocks are free.
    finished: Tuple[Sequence, ...]


class Scheduler:
    """Continuous batching over a paged KV cache, with chunked prefill.

    Each step:

    1. Decodes of running sequences, oldest first, one token each. A decode
       that needs a block when none is free preempts the newest running
       sequence, which drops its KV and goes to the front of the queue to
       recompute it.
    2. The next chunk of every running sequence still in prefill.
    3. If nothing was preempted: admissions in arrival order while the
       budget, max_num_sequences and the KV cache allow. Admission reserves
       blocks for everything the sequence knows (its prompt, plus what it
       generated before a preemption) and must leave the watermark free. A
       head of the queue that does not fit blocks those behind it.

    Prefills never need to grow, because admission reserved their blocks, so
    only decodes preempt.
    """

    def __init__(
        self, kv: BlockAllocator, max_batch_tokens: int, max_num_sequences: int
    ):
        if max_num_sequences < 1:
            raise ValueError("max_num_sequences must be at least 1")
        if max_batch_tokens < max_num_sequences:
            # As in vLLM: otherwise a full batch of decodes would not fit in
            # one step.
            raise ValueError("max_batch_tokens must be at least max_num_sequences")
        self._kv = kv
        self._max_batch_tokens = max_batch_tokens
        self._max_num_sequences = max_num_sequences
        self._waiting: Deque[Sequence] = deque()
        self._running: List[Sequence] = []
        # Every sequence waiting or running, by id.
        self._sequences: Dict[int, Sequence] = {}

    @property
    def waiting(self) -> int:
        return len(self._waiting)

    @property
    def running(self) -> int:
        return len(self._running)

    @property
    def kv(self) -> BlockAllocator:
        """The cache, for reading its state; only the scheduler changes it."""
        return self._kv

    def has_work(self) -> bool:
        return bool(self._sequences)

    def add(
        self, id: int, arrival: float, prompt_tokens: int, output_tokens: int
    ) -> Sequence:
        """Queue a new request.

        An empty prompt counts as one token: a real server always processes
        at least the chat template's.
        """
        if id in self._sequences:
            raise ValueError(f"sequence {id} already exists")
        seq = Sequence(id, arrival, max(1, prompt_tokens), output_tokens)
        # A safety check, not a scheduling input: the generator keeps prompt
        # and output within the context length, and the engine refuses to
        # start if that does not fit an empty cache. A sequence that could
        # never fit would block the queue forever.
        if seq.prompt_tokens + seq.output_tokens > self._kv.admissible_tokens:
            raise ValueError(
                f"sequence {id} needs {seq.prompt_tokens + seq.output_tokens} "
                f"tokens; the cache admits at most {self._kv.admissible_tokens}"
            )
        self._sequences[id] = seq
        self._waiting.append(seq)
        return seq

    def abort(self, id: int) -> None:
        """Drop a sequence and free its blocks; a no-op if it is gone.

        Safe during a step: complete() ignores work of aborted sequences, and
        the freed blocks are not reused before the next schedule().
        """
        seq = self._sequences.pop(id, None)
        if seq is None:
            return
        if seq in self._running:
            self._running.remove(seq)
        else:
            self._waiting.remove(seq)
        self._kv.free(id)

    def schedule(self, now: float) -> StepPlan:
        budget = self._max_batch_tokens
        work: List[Work] = []
        preempted: List[Preemption] = []

        i = 0
        while i < len(self._running):
            seq = self._running[i]
            if seq.prefilling:
                i += 1
            elif self._kv.grow(seq.id, seq.computed + 1):
                work.append(Work(seq, 1, seq.computed, prefill=False))
                budget -= 1
                i += 1
            else:
                # The victim is last, so it has no work yet; if it is seq
                # itself, the loop ends.
                preempted.append(self._preempt_newest())

        for seq in self._running:
            if not budget:
                break
            if seq.prefilling:
                tokens = min(seq.uncomputed, budget)
                work.append(Work(seq, tokens, seq.computed, prefill=True))
                budget -= tokens

        admitted: List[Sequence] = []
        blocked = None
        while self._waiting:
            seq = self._waiting[0]
            if preempted:
                blocked = "preemption"
            elif not budget:
                blocked = "budget"
            elif len(self._running) >= self._max_num_sequences:
                blocked = "max_sequences"
            elif not self._kv.can_admit(seq.known):
                blocked = "kv"
            if blocked:
                break
            self._waiting.popleft()
            self._kv.grow(seq.id, seq.known)
            if seq.admitted_at is None:
                seq.admitted_at = now
            self._running.append(seq)
            admitted.append(seq)
            tokens = min(seq.uncomputed, budget)
            work.append(Work(seq, tokens, 0, prefill=True))
            budget -= tokens

        return StepPlan(
            work=tuple(work),
            admitted=tuple(admitted),
            preempted=tuple(preempted),
            blocked=blocked,
            waiting=len(self._waiting),
            running=len(self._running),
            kv_usage=self._kv.usage,
        )

    def complete(self, plan: StepPlan) -> StepResult:
        """Apply a step whose time has passed."""
        emitted: List[Sequence] = []
        finished: List[Sequence] = []
        for w in plan.work:
            seq = w.seq
            if self._sequences.get(seq.id) is not seq:
                continue  # aborted during the step
            if seq.advance(w.tokens):
                emitted.append(seq)
            if seq.finished:
                del self._sequences[seq.id]
                self._kv.free(seq.id)
                finished.append(seq)
        if finished:
            self._running = [seq for seq in self._running if not seq.finished]
        return StepResult(tuple(emitted), tuple(finished))

    def _preempt_newest(self) -> Preemption:
        victim = self._running.pop()
        discarded = victim.computed
        self._kv.free(victim.id)
        victim.preempt()
        self._waiting.appendleft(victim)
        return Preemption(victim, discarded)
