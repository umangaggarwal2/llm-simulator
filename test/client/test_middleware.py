import time

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel

from llm_simulator.client.middleware import ArrivalMiddleware


class Body(BaseModel):
    text: str


def make_app():
    app = FastAPI()
    app.add_middleware(ArrivalMiddleware)
    seen = {}

    @app.post("/echo")
    async def echo(body: Body, request: Request):
        seen["arrival"] = request.state.arrival
        seen["handler"] = time.monotonic()
        return {"text": body.text}

    @app.get("/stream")
    async def stream():
        async def chunks():
            for i in range(3):
                yield f"chunk {i}\n"

        return StreamingResponse(chunks(), media_type="text/plain")

    return app, seen


def test_arrival_is_recorded_before_the_handler_runs():
    app, seen = make_app()
    before = time.monotonic()
    response = TestClient(app).post("/echo", json={"text": "hi"})
    assert response.json() == {"text": "hi"}
    assert before <= seen["arrival"] <= seen["handler"]


def test_each_request_gets_its_own_arrival():
    app, seen = make_app()
    client = TestClient(app)
    client.post("/echo", json={"text": "a"})
    first = seen["arrival"]
    client.post("/echo", json={"text": "b"})
    assert seen["arrival"] > first


def test_streamed_responses_pass_through_unchanged():
    app, _ = make_app()
    with TestClient(app).stream("GET", "/stream") as response:
        assert list(response.iter_lines()) == ["chunk 0", "chunk 1", "chunk 2"]
