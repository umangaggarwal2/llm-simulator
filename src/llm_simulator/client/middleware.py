"""ASGI middleware for the HTTP API."""

import time

from starlette.types import ASGIApp, Receive, Scope, Send


class ArrivalMiddleware:
    """Records when each HTTP request arrived, as request.state.arrival.

    It runs as soon as the request's headers are parsed, before the body is
    read and validated, which is where a client's TTFT clock starts. It is a
    plain ASGI middleware rather than BaseHTTPMiddleware, which would route
    every streamed chunk through an extra buffer.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            # Starlette backs request.state with scope["state"].
            scope.setdefault("state", {})["arrival"] = time.monotonic()
        await self.app(scope, receive, send)
