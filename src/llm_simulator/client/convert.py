"""Conversions between the HTTP API's schemas and the domain types."""

from llm_simulator.client.schemas import ChatCompletionRequest, Usage
from llm_simulator.types import GenerationPlan, GenerationRequest, Message


def to_generation_request(body: ChatCompletionRequest) -> GenerationRequest:
    # max_tokens is deprecated in favour of max_completion_tokens, which wins
    # when both are set.
    max_tokens = (
        body.max_completion_tokens
        if body.max_completion_tokens is not None
        else body.max_tokens
    )
    return GenerationRequest(
        model=body.model,
        messages=tuple(Message(m.role, m.content) for m in body.messages),
        max_tokens=max_tokens,
        seed=body.seed,
    )


def to_usage(plan: GenerationPlan) -> Usage:
    return Usage(
        prompt_tokens=plan.prompt_tokens,
        completion_tokens=plan.completion_tokens,
        total_tokens=plan.prompt_tokens + plan.completion_tokens,
    )
