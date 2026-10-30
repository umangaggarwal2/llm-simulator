"""What the client measured, with no dependencies, so analysis needs no network."""

from array import array
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class RequestRecord:
    """One request as the client saw it.

    Times come from the client's clock (time.perf_counter by default) and are
    only meaningful relative to each other. A content chunk is a chunk whose
    delta carries non-empty content; the opening role chunk, the finish chunk
    and the usage chunk are not counted.
    """

    index: int
    # Just before the request was written.
    sent: float
    # When the stream was fully read ([DONE]), or when the request failed.
    end: float
    first_token: Optional[float] = None
    last_token: Optional[float] = None
    # Gaps between consecutive content chunks.
    itls: array = field(default_factory=lambda: array("d"))
    # Content chunks received. Equal to tokens for the simulator; a server
    # that packs several tokens into a chunk is counted by usage instead.
    chunks: int = 0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    # The max_completion_tokens the request asked for.
    requested_tokens: Optional[int] = None
    # Open loop only: when the arrival schedule said to send the request,
    # and whether it had to wait for a free slot (max_concurrency) first.
    scheduled: Optional[float] = None
    waited: bool = False
    # HTTP status; 0 when no response arrived (connection error or timeout).
    status: int = 0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.error is None

    @property
    def ttft(self) -> Optional[float]:
        return None if self.first_token is None else self.first_token - self.sent

    @property
    def e2e(self) -> float:
        return self.end - self.sent

    @property
    def queue_wait(self) -> Optional[float]:
        """How long after its scheduled time the request was sent."""
        return None if self.scheduled is None else self.sent - self.scheduled

    @property
    def ttft_from_schedule(self) -> Optional[float]:
        """TTFT as the arriving user waits for it: from the scheduled time."""
        if self.scheduled is None or self.first_token is None:
            return None
        return self.first_token - self.scheduled

    @property
    def e2e_from_schedule(self) -> Optional[float]:
        return None if self.scheduled is None else self.end - self.scheduled


@dataclass
class LoopSamples:
    """What the client's loop monitor saw during the measure phase."""

    # How late each wake-up was, in seconds.
    lags: array
    # CPU time the client process used, and the wall time it was used over.
    cpu_seconds: float
    wall_seconds: float


@dataclass
class ServerSummary:
    """What the target's /metrics/summary reported for the measured window."""

    url: str
    available: bool
    # Why there is no summary: "HTTP 404", "no response: ...", a missing model.
    reason: Optional[str] = None
    # Seconds between the reset and the fetch, as the server measured them.
    since_reset: Optional[float] = None
    # The section for the benchmarked model, as the server sent it.
    model: Optional[Dict[str, Any]] = None


@dataclass
class RunRecords:
    """One run's records and the window they were measured over."""

    records: List[RequestRecord]
    # Client clock when the first measured request was sent and when the
    # last one finished.
    started: float
    ended: float
    # Wall-clock start, for the results file.
    started_at: Optional[datetime] = None
    # Requests sent before measuring, whose records were discarded.
    warmup: int = 0
    loop: Optional[LoopSamples] = None
    server: Optional[ServerSummary] = None

    @property
    def duration(self) -> float:
        return self.ended - self.started
