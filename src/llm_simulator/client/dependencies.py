"""FastAPI dependencies that give handlers the components built at startup.

Tests can replace any of them through app.dependency_overrides.
"""

from fastapi import Request

from llm_simulator.client.exception import OpenAIError
from llm_simulator.metrics import MetricsRecorder, PrometheusRecorder, SummaryRecorder
from llm_simulator.runtime import ModelRuntime, Runtimes


def get_runtimes(request: Request) -> Runtimes:
    return request.app.state.runtimes


def get_recorder(request: Request) -> MetricsRecorder:
    """Where request traces report their measurements."""
    return request.app.state.recorder


def get_prometheus(request: Request) -> PrometheusRecorder:
    """The store GET /metrics exposes."""
    return request.app.state.prometheus


def get_summary(request: Request) -> SummaryRecorder:
    """The store GET /metrics/summary reads."""
    return request.app.state.summary


def find_runtime(runtimes: Runtimes, model: str) -> ModelRuntime:
    try:
        return runtimes[model]
    except KeyError:
        raise OpenAIError(
            404,
            f"The model `{model}` does not exist or you do not have access to it.",
            param="model",
            code="model_not_found",
        ) from None
