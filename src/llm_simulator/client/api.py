"""The OpenAI-compatible HTTP API."""

import argparse
import os
import sys
import time
import uuid
from contextlib import aclosing, asynccontextmanager
from importlib import metadata
from pathlib import Path
from typing import Optional, Sequence

from fastapi import Depends, FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response, StreamingResponse

from llm_simulator import defaults
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


def package_version() -> str:
    """The installed package's version, so pyproject.toml is its one source."""
    try:
        return metadata.version("llm-simulator")
    except metadata.PackageNotFoundError:
        # Running from a source tree that was never installed.
        return "0+unknown"


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """Build the app from configuration.

    Both config files are loaded and every model's runtime is built here, so a
    configuration error stops the server before it accepts any traffic.
    """
    settings = settings or Settings.from_env()
    models = load_models(settings.models_path)
    # The recorders come first: engines with steps report to them directly.
    descriptions = {
        model.name: ModelDescription(model.engine, model.context_length)
        for model in models.models
    }
    prometheus = PrometheusRecorder(descriptions)
    summary = SummaryRecorder(
        descriptions,
        request_samples=settings.summary_request_samples,
        token_samples=settings.summary_token_samples,
        rate_seconds=settings.summary_rate_seconds,
    )
    # Request traces and engine steps report once; both stores receive every
    # measurement.
    recorder = FanOutRecorder([prometheus, summary])
    runtimes = build_runtimes(models, load_responses(settings.responses_path), recorder)
    models_created = int(time.time())

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
        version=package_version(),
        lifespan=lifespan,
        responses={status: {"model": ErrorResponse} for status in (400, 404, 405, 500)},
    )
    app.state.runtimes = runtimes
    app.state.recorder = recorder
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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="llmsimulator",
        description="Serve simulated LLMs through an OpenAI-compatible API.",
        epilog="Each option overrides its LLM_SIM_* environment variable; "
        "see the README for the others. A FILE is a path or builtin:NAME, one "
        "of the files shipped with the package; without one, the built-in "
        "models and responses are served.",
    )
    parser.add_argument("--host", help="interface to listen on (LLM_SIM_HOST)")
    parser.add_argument("--port", type=_port, help="port to listen on (LLM_SIM_PORT)")
    parser.add_argument(
        "--models",
        type=_config_file,
        metavar="FILE",
        help="models.yml (LLM_SIM_MODELS)",
    )
    parser.add_argument(
        "--responses",
        type=_config_file,
        metavar="FILE",
        help="responses.yml (LLM_SIM_RESPONSES)",
    )
    parser.add_argument(
        "--print-config",
        choices=defaults.NAMES,
        metavar="NAME",
        help="print a built-in config file, to copy and edit, and exit: "
        + ", ".join(defaults.NAMES),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {package_version()}"
    )
    return parser.parse_args(argv)


def _config_file(text: str) -> Path:
    # argparse shows an ArgumentTypeError's message but replaces a
    # ValueError's with a generic one, which would hide the available names.
    try:
        return defaults.resolve(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _port(text: str) -> int:
    try:
        port = int(text)
    except ValueError:
        port = 0
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"invalid port: {text!r}")
    return port


def _print(text: str) -> None:
    """Write text to stdout, tolerating a reader that stops early.

    `llmsimulator --print-config responses | head` closes the pipe after a
    few lines; that is not an error. Flushing here raises it while it can be
    caught, and pointing stdout at /dev/null stops Python from raising it
    again when it flushes at exit (the recipe in Python's signal docs).
    """
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except BrokenPipeError:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())


def main(argv: Optional[Sequence[str]] = None) -> int:
    import uvicorn

    args = parse_args(argv)
    if args.print_config:
        _print(defaults.path(args.print_config).read_text(encoding="utf-8"))
        return 0
    # The app is built here rather than by uvicorn, so the flags can reach it
    # and a broken configuration stops with its message instead of a
    # traceback from inside the server's startup. Config errors, unreadable
    # files included, are ValueErrors.
    try:
        settings = Settings.from_args(args)
        app = create_app(settings)
    except ValueError as exc:
        print(f"llmsimulator: error: {exc}", file=sys.stderr)
        return 1
    uvicorn.run(app, host=settings.host, port=settings.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
