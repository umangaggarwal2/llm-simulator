"""The target's own measurements, from the simulator's /metrics/summary.

Reset after warm-up and read after the run, the summary covers the same
requests as the client's window, so the report can put server-observed
figures next to client-observed ones. A target without the endpoint (a real
vLLM or OpenAI server) is simply not compared: nothing here raises.
"""

import asyncio
import json
from typing import Optional

import aiohttp

from llm_simulator.bench.records import ServerSummary

# A metrics endpoint that hangs must not hang the run.
TIMEOUT = aiohttp.ClientTimeout(total=10)


def summary_url(base_url: str) -> str:
    """The summary endpoint for an OpenAI base URL: .../v1 → .../metrics/summary."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return root + "/metrics/summary"


class ServerMetrics:
    def __init__(self, session: aiohttp.ClientSession, url: str, model: str):
        self.url = url
        self._session = session
        self._model = model
        # Set when the target has no usable summary.
        self.reason: Optional[str] = None

    async def reset(self) -> bool:
        """Start a fresh summary period; False if the target has no summary."""
        try:
            async with self._session.post(
                self.url + "/reset", timeout=TIMEOUT
            ) as response:
                if response.status == 204:
                    return True
                self.reason = f"HTTP {response.status}"
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            self.reason = f"no response: {type(exc).__name__}"
        return False

    async def fetch(self) -> ServerSummary:
        """This model's summary since the reset."""
        if self.reason is not None:
            return self._unavailable(self.reason)
        try:
            async with self._session.get(self.url, timeout=TIMEOUT) as response:
                if response.status != 200:
                    return self._unavailable(f"HTTP {response.status}")
                body = json.loads(await response.text())
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            return self._unavailable(f"no response: {type(exc).__name__}")
        except ValueError:
            return self._unavailable("the summary is not valid JSON")

        models = body.get("models") if isinstance(body, dict) else None
        model = models.get(self._model) if isinstance(models, dict) else None
        if not isinstance(model, dict):
            return self._unavailable(f"no section for model {self._model!r}")
        return ServerSummary(
            url=self.url,
            available=True,
            since_reset=body.get("since_reset_seconds"),
            model=model,
        )

    def _unavailable(self, reason: str) -> ServerSummary:
        self.reason = reason
        return ServerSummary(url=self.url, available=False, reason=reason)
