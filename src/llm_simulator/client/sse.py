"""Server-sent event stream for streaming chat completions, in OpenAI's format."""

from contextlib import aclosing
from typing import AsyncIterator, List, Optional

from llm_simulator.client.convert import to_usage
from llm_simulator.client.schemas import ChatCompletionChunk, ChunkChoice, Delta, Usage
from llm_simulator.types import GenerationPlan, TokenEvent

# Stop reverse proxies and caches from holding chunks back, which would undo
# the engine's pacing. Responses must also not go through compression
# middleware, which buffers for the same reason.
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


async def sse_stream(
    events: AsyncIterator[TokenEvent],
    plan: GenerationPlan,
    completion_id: str,
    created: int,
    model: str,
    include_usage: bool,
) -> AsyncIterator[str]:
    """Yield the SSE lines for one completion.

    Each content chunk is yielded only when the engine releases its token, so
    the engine decides when every chunk reaches the client.
    """

    def chunk(choices: List[ChunkChoice], usage: Optional[Usage] = None) -> str:
        payload = ChatCompletionChunk(
            id=completion_id, created=created, model=model, choices=choices, usage=usage
        )
        # With include_usage, OpenAI sends "usage": null on every chunk except
        # the last; without it, the field is left out entirely.
        exclude = None if include_usage else {"usage"}
        return f"data: {payload.model_dump_json(exclude=exclude)}\n\n"

    yield chunk([ChunkChoice(index=0, delta=Delta(role="assistant", content=""))])

    # aclosing makes sure the engine's iterator is closed, and its cleanup
    # runs, even when the client disconnects mid-stream.
    async with aclosing(events):
        async for event in events:
            yield chunk([ChunkChoice(index=0, delta=Delta(content=event.text))])

    yield chunk([ChunkChoice(index=0, delta=Delta(), finish_reason=plan.finish_reason)])
    if include_usage:
        yield chunk([], usage=to_usage(plan))
    yield "data: [DONE]\n\n"
