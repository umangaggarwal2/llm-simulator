"""What a benchmark run sends, and how much of it."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Open-loop warm-up without a concurrency cap to size it by.
DEFAULT_OPEN_WARMUP = 8


@dataclass(frozen=True)
class BenchConfig:
    """Settings for one benchmark run.

    Plain data: the CLI reads arguments and the environment, everything else
    takes a BenchConfig.
    """

    model: str
    base_url: str = "http://localhost:8080/v1"
    # How load is generated. Closed loop: `concurrency` requests in flight,
    # each worker sending its next as soon as its previous one finishes
    # (the default, with 1). Open loop: requests arrive at `rate` per second
    # on a Poisson schedule, whether or not earlier ones have finished;
    # `max_concurrency` optionally caps how many are in flight, and an
    # arrival that finds it full waits for a slot.
    concurrency: Optional[int] = None
    rate: Optional[float] = None
    max_concurrency: Optional[int] = None
    # How long the measure phase runs: a number of requests, or a number of
    # seconds (requests sent before the deadline are counted and waited for).
    # With neither, 100 requests.
    requests: Optional[int] = None
    duration: Optional[float] = None
    # Requests sent first and not measured, so connections are open and code
    # paths warm; None means one per concurrent stream.
    warmup: Optional[int] = None
    # max_completion_tokens of every request; a dataset line may set its own.
    max_tokens: int = 256
    # Where prompts come from: synthetic prompts of input_tokens prompt
    # tokens, the lines of a JSONL dataset, or else this fixed prompt.
    prompt: str = "Benchmark request."
    input_tokens: Optional[int] = None
    dataset: Optional[Path] = None
    # Prepended to every prompt, e.g. "[short] " for bench-responses.yml.
    prompt_prefix: str = ""
    # Seeds the synthetic prompts and, separately, the arrival schedule, so
    # runs send the same prompts at the same times.
    seed: int = 0
    # Sent as "Authorization: Bearer <api_key>" when set.
    api_key: Optional[str] = None
    # Seconds to wait for the next chunk, not for the whole request: long
    # generations are expected, a stalled stream is not.
    timeout: float = 60.0
    # Compare with the target's /metrics/summary, when it has one. The URL
    # defaults to the base URL with its /v1 replaced by /metrics/summary.
    server_metrics: bool = True
    server_metrics_url: Optional[str] = None

    def __post_init__(self):
        if not self.model:
            raise ValueError("model must not be empty")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError(
                f"base_url must start with http:// or https://, got {self.base_url!r}"
            )
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if self.concurrency is not None and self.rate is not None:
            raise ValueError(
                "use concurrency (closed loop) or rate (open loop), not both"
            )
        if self.concurrency is not None and self.concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        if self.rate is not None and self.rate <= 0:
            raise ValueError("rate must be positive")
        if self.max_concurrency is not None:
            if self.rate is None:
                raise ValueError("max_concurrency applies to the open loop: set rate")
            if self.max_concurrency < 1:
                raise ValueError("max_concurrency must be at least 1")
        if self.requests is not None and self.duration is not None:
            raise ValueError("use requests or duration, not both")
        if self.requests is not None and self.requests < 1:
            raise ValueError("requests must be at least 1")
        if self.duration is not None and self.duration <= 0:
            raise ValueError("duration must be positive")
        if self.warmup is not None and self.warmup < 0:
            raise ValueError("warmup must not be negative")
        if (
            self.server_metrics_url is not None
            and not self.server_metrics_url.startswith(("http://", "https://"))
        ):
            raise ValueError(
                "server_metrics_url must start with http:// or https://, got"
                f" {self.server_metrics_url!r}"
            )
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.input_tokens is not None and self.dataset is not None:
            raise ValueError("use input_tokens or dataset, not both")
        if self.input_tokens is not None and self.input_tokens < 1:
            raise ValueError("input_tokens must be at least 1")

    @property
    def measured_requests(self) -> Optional[int]:
        """Requests to measure; None when the run is bounded by duration."""
        if self.duration is not None:
            return None
        return 100 if self.requests is None else self.requests

    @property
    def mode(self) -> str:
        return "open" if self.rate is not None else "closed"

    @property
    def closed_concurrency(self) -> int:
        return 1 if self.concurrency is None else self.concurrency

    @property
    def streams(self) -> int:
        """Connections the client may open; 0 means no limit (open loop, no cap)."""
        if self.mode == "closed":
            return self.closed_concurrency
        return self.max_concurrency or 0

    @property
    def warmup_requests(self) -> int:
        if self.warmup is not None:
            return self.warmup
        if self.mode == "closed":
            return self.closed_concurrency
        return self.max_concurrency or DEFAULT_OPEN_WARMUP

    @property
    def warmup_concurrency(self) -> int:
        """Warm-up is always a closed-loop burst, at most one per stream."""
        streams = self.streams or self.warmup_requests
        return max(1, min(self.warmup_requests, streams))

    @property
    def chat_url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"
