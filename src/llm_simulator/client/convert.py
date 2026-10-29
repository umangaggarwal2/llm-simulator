"""Conversions between the HTTP API's schemas and the domain types."""

from llm_simulator.client.exception import OpenAIError
from llm_simulator.client.schemas import ChatCompletionRequest, ChatMessageParam, Usage
from llm_simulator.types import GenerationPlan, GenerationRequest, Message

# Placed between the text parts of one message. OpenAI does not document how
# it joins them; no separator is assumed until measured. See docs/schema.md.
PART_SEPARATOR = ""


def to_generation_request(body: ChatCompletionRequest) -> GenerationRequest:
    """The domain request for an API request.

    Raises OpenAIError (400) for content the simulator cannot serve, before
    anything is generated or streamed.
    """
    # max_tokens is deprecated in favour of max_completion_tokens, which wins
    # when both are set.
    max_tokens = (
        body.max_completion_tokens
        if body.max_completion_tokens is not None
        else body.max_tokens
    )
    return GenerationRequest(
        model=body.model,
        messages=tuple(
            Message(m.role, _text(m, index)) for index, m in enumerate(body.messages)
        ),
        max_tokens=max_tokens,
        seed=body.seed,
    )


def _text(message: ChatMessageParam, index: int) -> str:
    """The message's text parts joined into one string.

    The simulated models are text-only, so any other part is rejected rather
    than dropped, which would silently undercount the prompt.
    """
    texts = []
    for position, part in enumerate(message.content):
        param = f"messages.[{index}].content.[{position}]"
        if part.type != "text":
            raise OpenAIError(
                400,
                f"Invalid content type. {part.type} is not supported by this "
                "simulator; only text content parts are accepted.",
                param=f"{param}.type",
            )
        if part.text is None:
            raise OpenAIError(
                400,
                f"Missing required parameter: '{param}.text'.",
                param=f"{param}.text",
            )
        texts.append(part.text)
    return PART_SEPARATOR.join(texts)


def to_usage(plan: GenerationPlan) -> Usage:
    return Usage(
        prompt_tokens=plan.prompt_tokens,
        completion_tokens=plan.completion_tokens,
        total_tokens=plan.prompt_tokens + plan.completion_tokens,
    )
