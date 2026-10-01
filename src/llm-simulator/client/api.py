import time
import uuid

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from exception import OpenAIError, register_exception_handlers
from models import (
    MODEL_NAME,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ChunkChoice,
    Delta,
    ErrorResponse,
    ModelCard,
    ModelList,
    Usage,
)


def create_app() -> FastAPI:
    app = FastAPI(
        title="LLM Simulator",
        version="0.1.0",
        responses={
            status: {"model": ErrorResponse} for status in (400, 404, 405, 500)
        },
    )
    register_exception_handlers(app)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models", response_model=ModelList)
    async def list_models():
        return ModelList(data=[ModelCard(id=MODEL_NAME)])

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(body: ChatCompletionRequest):
        if body.model != MODEL_NAME:
            raise OpenAIError(
                404,
                f"The model `{body.model}` does not exist or you do not have access to it.",
                param="model",
                code="model_not_found",
            )

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        # TODO: replace with the simulator's deterministic token generator.
        tokens = ["Hello", " from", " the", " LLM", " simulator", "."]

        if body.stream:
            return StreamingResponse(
                _stream_chunks(completion_id, created, body.model, tokens),
                media_type="text/event-stream",
            )

        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=body.model,
            choices=[
                Choice(
                    index=0,
                    message=ChatMessage(role="assistant", content="".join(tokens)),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=0,
                completion_tokens=len(tokens),
                total_tokens=len(tokens),
            ),
        )

    return app


async def _stream_chunks(completion_id, created, model, tokens):
    def chunk(delta: Delta, finish_reason=None) -> str:
        payload = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model,
            choices=[ChunkChoice(index=0, delta=delta, finish_reason=finish_reason)],
        )
        return f"data: {payload.model_dump_json()}\n\n"

    for token in tokens:
        yield chunk(Delta(content=token))

    yield chunk(Delta(), finish_reason="stop")
    yield "data: [DONE]\n\n"


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
