"""How long one forward pass over a batch takes: a roofline model."""

from dataclasses import dataclass
from typing import Iterable, Tuple


@dataclass(frozen=True)
class StepCost:
    """The parts of one step's duration, in seconds."""

    compute: float
    memory: float
    overhead: float

    @property
    def duration(self) -> float:
        # Compute and memory traffic overlap, so the slower one sets the pace;
        # the CPU-side overhead does not overlap with either.
        return max(self.compute, self.memory) + self.overhead

    @property
    def bound(self) -> str:
        return "compute" if self.compute > self.memory else "memory"


@dataclass(frozen=True)
class RooflineCostModel:
    """A step takes as long as its arithmetic or its memory reads, whichever
    is slower, plus a fixed CPU-side overhead.

    - compute: about 2 FLOPs per parameter per token computed in the step.
    - memory: every step reads all the weights once, however many tokens it
      computes, plus the KV of every token each sequence attends to.

    The efficiencies scale the hardware's peak figures down to what real
    kernels reach. Calibrate them against a real server: batch-1 ITL measured
    at two context lengths fits ITL = overhead + bytes / (bandwidth x
    memory_efficiency), and a long prompt's TTFT gives compute_efficiency.

    Attention FLOPs, which grow with context, are left out; they matter only
    for very long prompts.
    """

    parameters: float
    weight_bytes: float
    kv_bytes_per_token: int
    peak_flops: float  # FLOP/s
    memory_bandwidth: float  # bytes/s
    compute_efficiency: float = 1.0
    memory_efficiency: float = 1.0
    step_overhead_s: float = 0.0

    def __post_init__(self):
        if min(self.parameters, self.weight_bytes, self.kv_bytes_per_token) <= 0:
            raise ValueError(
                "parameters, weight_bytes and kv_bytes_per_token must be > 0"
            )
        if self.peak_flops <= 0 or self.memory_bandwidth <= 0:
            raise ValueError("peak_flops and memory_bandwidth must be > 0")
        for name in ("compute_efficiency", "memory_efficiency"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.step_overhead_s < 0:
            raise ValueError("step_overhead_s must not be negative")

    def step_cost(self, work: Iterable[Tuple[int, int]]) -> StepCost:
        """The cost of a step.

        `work` holds one (computed, cached) pair per sequence in the batch:
        the tokens the step computes for it, and the tokens whose KV it
        already holds. A decode is (1, context); a prefill chunk is
        (chunk, tokens of the prompt computed before it).
        """
        tokens = 0
        kv_tokens = 0
        for computed, cached in work:
            if computed < 1 or cached < 0:
                raise ValueError(f"invalid work item ({computed}, {cached})")
            tokens += computed
            # The KV the sequence reads, and the KV this step writes.
            kv_tokens += cached + computed
        if not tokens:
            raise ValueError("a step must compute at least one token")
        flops = 2 * self.parameters * tokens
        bytes_read = self.weight_bytes + kv_tokens * self.kv_bytes_per_token
        return StepCost(
            compute=flops / (self.peak_flops * self.compute_efficiency),
            memory=bytes_read / (self.memory_bandwidth * self.memory_efficiency),
            overhead=self.step_overhead_s,
        )
