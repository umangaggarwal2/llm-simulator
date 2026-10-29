"""The OpenAI-compatible HTTP API."""

import time
import uuid
from contextlib import aclosing, asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response, StreamingResponse

from llm_simulator.client.convert import to_generation_request, to_usage
from llm_simulator.client.dependencies import (
    find_runtime,
    get_prometheus,
    get_recorder,
    get_runtimes,
    get_summary,
)
from llm_simulator.client.exception import register_exception_handlers
from llm_simulator.client.middleware import ArrivalMiddleware
from llm_simulator.client.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ErrorResponse,
    ModelCard,
    ModelList,
)
from llm_simulator.client.sse import SSE_HEADERS, sse_stream
from llm_simulator.llm.models import load_models
from llm_simulator.llm.responses import load_responses
from llm_simulator.metrics import (
    FanOutRecorder,
    MetricsRecorder,
    ModelDescription,
    PrometheusRecorder,
    RequestTrace,
    SummaryRecorder,
    summarize,
    traced,
)
from llm_simulator.runtime import Runtimes, build_runtimes
from llm_simulator.settings import Settings


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """Build the app from configuration.

    Both config files are loaded and every model's runtime is built here, so a
    configuration error stops the server before it accepts any traffic.
    """
    settings = settings or Settings.from_env()
    runtimes = build_runtimes(
        load_models(settings.models_path), load_responses(settings.responses_path)
    )
    models_created = int(time.time())
    descriptions = {
        name: ModelDescription(runtime.engine.name, runtime.deployment.context_length)
        for name, runtime in runtimes.items()
    }
    prometheus = PrometheusRecorder(descriptions)
    summary = SummaryRecorder(
        descriptions,
        request_samples=settings.summary_request_samples,
        token_samples=settings.summary_token_samples,
        rate_seconds=settings.summary_rate_seconds,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        started = []
        try:
            for runtime in runtimes.values():
                await runtime.engine.start()
                started.append(runtime.engine)
            yield
        finally:
            for engine in reversed(started):
                await engine.stop()

    app = FastAPI(
        title="LLM Simulator",
        version="0.1.0",
        lifespan=lifespan,
        responses={status: {"model": ErrorResponse} for status in (400, 404, 405, 500)},
    )
    app.state.runtimes = runtimes
    # Request traces report once; both stores receive every measurement.
    app.state.recorder = FanOutRecorder([prometheus, summary])
    app.state.prometheus = prometheus
    app.state.summary = summary
    app.add_middleware(ArrivalMiddleware)
    register_exception_handlers(app)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/metrics", include_in_schema=False)
    async def metrics(prometheus: PrometheusRecorder = Depends(get_prometheus)):
        return Response(prometheus.exposition(), media_type=prometheus.content_type)

    @app.get("/metrics/summary", include_in_schema=False)
    async def metrics_summary(summary: SummaryRecorder = Depends(get_summary)):
        # Copied on the event loop, sorted in a worker thread: percentiles over
        # 100k values take milliseconds, which would otherwise delay tokens.
        return JSONResponse(await run_in_threadpool(summarize, summary.snapshot()))

    @app.post("/metrics/summary/reset", include_in_schema=False, status_code=204)
    async def reset_metrics_summary(summary: SummaryRecorder = Depends(get_summary)):
        summary.reset()
        return Response(status_code=204)

    @app.get("/v1/models", response_model=ModelList)
    async def list_models(runtimes: Runtimes = Depends(get_runtimes)):
        return ModelList(
            data=[ModelCard(id=name, created=models_created) for name in runtimes]
        )

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(
        body: ChatCompletionRequest,
        request: Request,
        runtimes: Runtimes = Depends(get_runtimes),
        recorder: MetricsRecorder = Depends(get_recorder),
    ):
        arrival = request.state.arrival
        runtime = find_runtime(runtimes, body.model)

        # Generated before any response is sent, so errors such as an over-long
        # prompt come back as a proper 400 rather than breaking a stream that
        # has already sent 200. It runs in a worker thread so that slow
        # matching on a long prompt does not stall every other stream.
        plan = await run_in_threadpool(
            runtime.generator.generate, to_generation_request(body)
        )
        trace = RequestTrace(recorder, body.model, arrival, plan.prompt_tokens)
        events = traced(runtime.engine.run(plan, arrival), trace, plan.finish_reason)
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        if body.stream:
            include_usage = (
                body.stream_options is not None and body.stream_options.include_usage
            )
            return StreamingResponse(
                sse_stream(
                    events, plan, completion_id, created, body.model, include_usage
                ),
                media_type="text/event-stream",
                headers=SSE_HEADERS,
            )

        # Waits out the engine's timing, so the response arrives when the whole
        # simulated generation would have finished.
        async with aclosing(events):
            text = "".join([event.text async for event in events])

        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=body.model,
            choices=[
                Choice(
                    index=0,
                    message=ChatMessage(role="assistant", content=text),
                    finish_reason=plan.finish_reason,
                )
            ],
            usage=to_usage(plan),
        )

    return app


def main():
    import uvicorn

    settings = Settings.from_env()
    # Factory mode: the app, and its configuration, are built when the server
    # starts rather than when this module is imported.
    uvicorn.run(
        "llm_simulator.client.api:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
    )


if __name__ == "__main__":
    main()
