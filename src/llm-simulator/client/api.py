import json
import time
import uuid
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

MODEL_NAME = "qwen-mock"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_NAME
    messages: List[ChatMessage]
    stream: bool = False
    max_tokens: Optional[int] = None


def create_app() -> FastAPI:
    app = FastAPI(title="LLM Simulator", version="0.1.0")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "llm-simulator"}],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest):
        if not body.messages:
            raise HTTPException(status_code=400, detail="'messages' must not be empty")

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        # TODO: replace with the simulator's deterministic token generator.
        tokens = ["Hello", " from", " the", " LLM", " simulator", "."]

        if body.stream:
            return StreamingResponse(
                _stream_chunks(completion_id, created, body.model, tokens),
                media_type="text/event-stream",
            )

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": body.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(tokens)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": len(tokens),
                "total_tokens": len(tokens),
            },
        }

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": exc.detail, "code": exc.status_code}},
        )

    return app


async def _stream_chunks(completion_id, created, model, tokens):
    for token in tokens:
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    final = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
