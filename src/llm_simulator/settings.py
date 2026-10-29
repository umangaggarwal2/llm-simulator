"""Process-level settings, read from environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional


@dataclass(frozen=True)
class Settings:
    models_path: Path = Path("config/models.yml")
    responses_path: Path = Path("config/responses.yml")
    host: str = "0.0.0.0"
    port: int = 8080
    # Values /metrics/summary keeps: the last N requests for per-request
    # metrics, the last N tokens for per-token ones, and the seconds that
    # throughput is averaged over.
    summary_request_samples: int = 10_000
    summary_token_samples: int = 100_000
    summary_rate_seconds: int = 60

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "Settings":
        """Read the LLM_SIM_* variables listed in the README.

        Unset variables keep their defaults; relative paths resolve against the
        working directory. A number that is not a positive integer raises
        ValueError naming the variable.
        """
        env = os.environ if env is None else env
        defaults = cls()
        return cls(
            models_path=Path(env.get("LLM_SIM_MODELS", defaults.models_path)),
            responses_path=Path(env.get("LLM_SIM_RESPONSES", defaults.responses_path)),
            host=env.get("LLM_SIM_HOST", defaults.host),
            port=_positive_int(env, "LLM_SIM_PORT", defaults.port),
            summary_request_samples=_positive_int(
                env, "LLM_SIM_SUMMARY_REQUEST_SAMPLES", defaults.summary_request_samples
            ),
            summary_token_samples=_positive_int(
                env, "LLM_SIM_SUMMARY_TOKEN_SAMPLES", defaults.summary_token_samples
            ),
            summary_rate_seconds=_positive_int(
                env, "LLM_SIM_SUMMARY_RATE_SECONDS", defaults.summary_rate_seconds
            ),
        )


def _positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from None
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return value
