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

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "Settings":
        """Read LLM_SIM_MODELS, LLM_SIM_RESPONSES, LLM_SIM_HOST and LLM_SIM_PORT.

        Unset variables keep their defaults; relative paths resolve against the
        working directory.
        """
        env = os.environ if env is None else env
        defaults = cls()
        port = env.get("LLM_SIM_PORT")
        return cls(
            models_path=Path(env.get("LLM_SIM_MODELS", defaults.models_path)),
            responses_path=Path(env.get("LLM_SIM_RESPONSES", defaults.responses_path)),
            host=env.get("LLM_SIM_HOST", defaults.host),
            port=defaults.port if port is None else int(port),
        )
