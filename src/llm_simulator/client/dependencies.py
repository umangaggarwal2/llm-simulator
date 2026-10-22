"""FastAPI dependencies that give handlers access to the per-model runtimes."""

from fastapi import Request

from llm_simulator.client.exception import OpenAIError
from llm_simulator.runtime import ModelRuntime, Runtimes


def get_runtimes(request: Request) -> Runtimes:
    """The runtimes built at startup. Tests can replace them through
    app.dependency_overrides[get_runtimes]."""
    return request.app.state.runtimes


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
