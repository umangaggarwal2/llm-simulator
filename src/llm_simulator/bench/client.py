"""One streaming chat completion request, timed as the client sees it.

The only module, with server.py later, that touches the network. It speaks
the OpenAI streaming protocol directly instead of using the OpenAI SDK, which
builds a typed object for every chunk: the client's own overhead must stay
far below the server's, or the benchmark measures the client.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import aiohttp

from llm_simulator.bench.config import BenchConfig
from llm_simulator.bench.records import RequestRecord
from llm_simulator.bench.workload import RequestSpec

# Waiting for a connection is not the same as waiting for tokens.
CONNECT_TIMEOUT = 10.0


def build_body(config: BenchConfig, spec: RequestSpec) -> bytes:
    """The request body, serialised before the clock starts."""
    return json.dumps(
        {
            "model": config.model,
            "messages": [{"role": "user", "content": spec.prompt}],
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_completion_tokens": spec.max_tokens,
        }
    ).encode()


@dataclass
class Event:
    """What one SSE data line carried that the client cares about."""

    # Non-empty delta.content only: the opening role chunk carries "".
    content: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    done: bool = False


def parse_line(line: bytes) -> Optional[Event]:
    """Parse one line of the stream; None for lines without data.

    Assumes one JSON object per "data:" line, as OpenAI, vLLM and the
    simulator all send. Raises ValueError for data that is not valid JSON.
    """
    if not line.startswith(b"data:"):
        # Blank separators, ":" comments and other SSE fields.
        return None
    data = line[5:].strip()
    if data == b"[DONE]":
        return Event(done=True)
    chunk = json.loads(data)

    error = chunk.get("error")
    if error is not None:
        message = error.get("message") if isinstance(error, dict) else None
        return Event(error=message or str(error))
    if chunk.get("object") == "error":  # vLLM's in-stream error shape
        return Event(error=chunk.get("message") or str(chunk))

    event = Event(usage=chunk.get("usage"))
    choices = chunk.get("choices")
    if choices:
        choice = choices[0]
        event.content = (choice.get("delta") or {}).get("content") or None
        event.finish_reason = choice.get("finish_reason")
    return event


def open_session(config: BenchConfig) -> aiohttp.ClientSession:
    """A session sized for the run; call it inside the event loop."""
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    return aiohttp.ClientSession(
        # aiohttp's default limit of 100 connections would silently cap the
        # run at 100 concurrent streams; 0 is no limit, for an uncapped open
        # loop.
        connector=aiohttp.TCPConnector(limit=config.streams),
        # No total limit, since long generations are expected; a stream that
        # stalls between chunks fails.
        timeout=aiohttp.ClientTimeout(
            total=None, sock_connect=CONNECT_TIMEOUT, sock_read=config.timeout
        ),
        headers=headers,
    )


async def stream_chat(
    session: aiohttp.ClientSession,
    config: BenchConfig,
    index: int,
    spec: RequestSpec,
    clock: Callable[[], float] = time.perf_counter,
) -> RequestRecord:
    """Send one streaming request and record when each part arrived.

    Never raises for a failed request: HTTP errors, broken streams, timeouts
    and connection errors become a record with status and error set, so one
    failure does not stop a run. Cancellation still propagates.
    """
    body = build_body(config, spec)
    sent = clock()
    record = RequestRecord(
        index=index, sent=sent, end=sent, requested_tokens=spec.max_tokens
    )
    try:
        async with session.post(config.chat_url, data=body) as response:
            record.status = response.status
            if response.status != 200:
                record.error = await _error_message(response)
                record.end = clock()
                return record

            async for line in response.content:
                now = clock()
                event = parse_line(line)
                if event is None:
                    continue
                if event.done:
                    record.end = now
                    return record
                if event.error is not None:
                    record.error = event.error
                if event.content is not None:
                    if record.last_token is None:
                        record.first_token = now
                    else:
                        record.itls.append(now - record.last_token)
                    record.last_token = now
                    record.chunks += 1
                if event.finish_reason is not None:
                    record.finish_reason = event.finish_reason
                if event.usage is not None:
                    record.prompt_tokens = event.usage.get("prompt_tokens")
                    record.completion_tokens = event.usage.get("completion_tokens")
            record.error = record.error or "stream ended before [DONE]"
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        record.error = _describe(exc)
    except ValueError as exc:
        record.error = f"invalid stream data: {exc}"
    record.end = clock()
    return record


async def _error_message(response: aiohttp.ClientResponse) -> str:
    """The message of an OpenAI-style error body, or the start of the body."""
    text = await response.text()
    try:
        return json.loads(text)["error"]["message"]
    except (ValueError, KeyError, TypeError):
        return f"HTTP {response.status}: {text[:200]}"


def _describe(exc: BaseException) -> str:
    # Timeouts often have an empty message.
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__
