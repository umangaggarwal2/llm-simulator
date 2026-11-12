"""Process-level settings, read from environment variables and command-line flags."""

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from llm_simulator import defaults


@dataclass(frozen=True)
class Settings:
    # The built-in files shipped with the package, unless a flag or variable
    # names others.
    models_path: Path = field(default_factory=lambda: defaults.path("models"))
    responses_path: Path = field(default_factory=lambda: defaults.path("responses"))
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

        Unset or empty variables keep their defaults. A path is a file,
        relative to the working directory, or `builtin:<name>` for a file
        shipped with the package. A number that is not a positive integer, or
        an unknown built-in name, raises ValueError naming the variable.
        """
        env = os.environ if env is None else env
        fallback = cls()
        return cls(
            models_path=_path(env, "LLM_SIM_MODELS", fallback.models_path),
            responses_path=_path(env, "LLM_SIM_RESPONSES", fallback.responses_path),
            host=_get(env, "LLM_SIM_HOST") or fallback.host,
            port=_positive_int(env, "LLM_SIM_PORT", fallback.port),
            summary_request_samples=_positive_int(
                env, "LLM_SIM_SUMMARY_REQUEST_SAMPLES", fallback.summary_request_samples
            ),
            summary_token_samples=_positive_int(
                env, "LLM_SIM_SUMMARY_TOKEN_SAMPLES", fallback.summary_token_samples
            ),
            summary_rate_seconds=_positive_int(
                env, "LLM_SIM_SUMMARY_RATE_SECONDS", fallback.summary_rate_seconds
            ),
        )

    @classmethod
    def from_args(
        cls, args: Any, env: Optional[Mapping[str, str]] = None
    ) -> "Settings":
        """The environment's settings, with each command-line flag that was
        given taking precedence over its variable.

        args has the attributes of the server's parsed flags (host, port,
        models, responses), each None when the flag was not given; models and
        responses are already resolved to paths, `builtin:` names included.
        """
        flags = {
            "host": args.host,
            "port": args.port,
            "models_path": args.models,
            "responses_path": args.responses,
        }
        return dataclasses.replace(
            cls.from_env(env),
            **{name: value for name, value in flags.items() if value is not None},
        )


def _get(env: Mapping[str, str], name: str) -> Optional[str]:
    """A variable's value, or None when it is unset or empty.

    Empty counts as unset so that tools which can only set a variable, such
    as Docker Compose substituting a default, can still leave it to the
    default.
    """
    return env.get(name) or None


def _path(env: Mapping[str, str], name: str, default: Path) -> Path:
    raw = _get(env, name)
    if raw is None:
        return default
    try:
        return defaults.resolve(raw)
    except ValueError as exc:
        raise ValueError(f"{name}: {exc}") from None


def _positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = _get(env, name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from None
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return value
