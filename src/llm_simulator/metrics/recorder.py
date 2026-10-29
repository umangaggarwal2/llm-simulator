"""The interface every metrics store implements, and one that records nothing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

# Every way a request can end: it ran to a stop or its token limit, the
# client went away, or the engine failed.
FINISH_REASONS = ("stop", "length", "cancelled", "error")


@dataclass(frozen=True)
class ModelDescription:
    """What a store needs to know about a served model, beyond its name."""

    engine: str
    context_length: int


@dataclass(frozen=True)
class RequestTimings:
    """Request-level timings in seconds, reported only for completed requests."""

    e2e: float
    queue: float
    prefill: float
    decode: float
    # Mean time per output token after the first; None with fewer than two tokens.
    tpot: Optional[float]


class MetricsRecorder(ABC):
    """Receives measurements from request traces.

    Calls arrive from the event loop, one request at a time per trace.
    Implementations decide how to store them; the trace never touches a
    metrics library directly.
    """

    @abstractmethod
    def request_started(self, model: str, prompt_tokens: int) -> None:
        """A request began producing its completion."""

    @abstractmethod
    def first_token(self, model: str, ttft: float, queue_time: float) -> None:
        """The first token was released, ttft seconds after arrival."""

    @abstractmethod
    def token(self, model: str, itl: Optional[float], lag: float) -> None:
        """A token was released; itl is None for the first token."""

    @abstractmethod
    def request_finished(
        self,
        model: str,
        finish_reason: str,
        prompt_tokens: int,
        generation_tokens: int,
        timings: Optional[RequestTimings],
    ) -> None:
        """A request ended. timings is None unless it completed (stop or length)."""


class NoopRecorder(MetricsRecorder):
    """Records nothing; for tests and for running with metrics turned off."""

    def request_started(self, model, prompt_tokens):
        pass

    def first_token(self, model, ttft, queue_time):
        pass

    def token(self, model, itl, lag):
        pass

    def request_finished(
        self, model, finish_reason, prompt_tokens, generation_tokens, timings
    ):
        pass


class FanOutRecorder(MetricsRecorder):
    """Forwards every measurement to several recorders, in order.

    Lets traces report once while more than one store keeps the results.
    A failing recorder raises rather than being skipped, so bugs surface.
    """

    def __init__(self, recorders: Sequence[MetricsRecorder]):
        self._recorders = tuple(recorders)

    def request_started(self, model, prompt_tokens):
        for recorder in self._recorders:
            recorder.request_started(model, prompt_tokens)

    def first_token(self, model, ttft, queue_time):
        for recorder in self._recorders:
            recorder.first_token(model, ttft, queue_time)

    def token(self, model, itl, lag):
        for recorder in self._recorders:
            recorder.token(model, itl, lag)

    def request_finished(
        self, model, finish_reason, prompt_tokens, generation_tokens, timings
    ):
        for recorder in self._recorders:
            recorder.request_finished(
                model, finish_reason, prompt_tokens, generation_tokens, timings
            )
