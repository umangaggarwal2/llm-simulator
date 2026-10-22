"""Schema and loader for models.yml: the models the simulator serves.

Each entry is a model deployed on a hardware profile, with its serving
settings. Sections only some engines use are optional here; each engine
declares the sections it requires, and runtime.py checks them at startup.

Only runtime.py reads these classes. The generator and engines receive the
individual values they need, so the simulator package never imports this
module.
"""

from pathlib import Path
from typing import Annotated, Dict, Literal, Optional, Tuple, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    model_validator,
)

from llm_simulator.llm._loading import load_yaml_model

PositiveInt = Annotated[StrictInt, Field(ge=1)]
PositiveNumber = Annotated[Union[StrictInt, StrictFloat], Field(gt=0)]
NonNegativeNumber = Annotated[Union[StrictInt, StrictFloat], Field(ge=0)]

# What clients send as `model`: any non-empty name without whitespace,
# e.g. "qwen-mock" or "Qwen/Qwen2.5-7B-Instruct".
MODEL_NAME_PATTERN = r"^\S+$"


class ModelsConfigError(ValueError):
    """Raised when models.yml cannot be read or does not match the schema."""


class _Model(BaseModel):
    # Reject unknown keys so typos fail at startup instead of being ignored.
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- Sections ---


class HardwareProfile(_Model):
    """A GPU type. Each model that references it gets its own simulated GPU."""

    hbm_gb: PositiveNumber
    memory_bandwidth_gbps: PositiveNumber


class KvCache(_Model):
    bytes_per_token: PositiveInt


class Performance(_Model):
    prefill_tokens_per_second: PositiveNumber
    decode_tokens_per_second: PositiveNumber


class Scheduler(_Model):
    max_batch_tokens: PositiveInt
    max_num_sequences: PositiveInt


class Latency(_Model):
    scheduler_overhead_ms: NonNegativeNumber = 0


# Names of the optional sections, as engines refer to them in required_sections.
SECTIONS = ("hardware", "kv_cache", "performance", "scheduler", "latency")


# --- Models ---


class ModelDeployment(_Model):
    name: str = Field(pattern=MODEL_NAME_PATTERN)
    context_length: PositiveInt
    tokenizer: str = "regex"
    engine: str = "instant"
    # Name of a profile under the top-level `hardware` key.
    hardware: Optional[str] = None
    kv_cache: Optional[KvCache] = None
    performance: Optional[Performance] = None
    scheduler: Optional[Scheduler] = None
    latency: Optional[Latency] = None


class ModelsConfig(_Model):
    version: Literal[1]
    hardware: Dict[str, HardwareProfile] = {}
    models: Tuple[ModelDeployment, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_names_and_references(self):
        seen, duplicates = set(), []
        for model in self.models:
            if model.name in seen and model.name not in duplicates:
                duplicates.append(model.name)
            seen.add(model.name)
        if duplicates:
            raise ValueError(f"duplicate model names: {', '.join(duplicates)}")

        for model in self.models:
            if model.hardware is not None and model.hardware not in self.hardware:
                available = ", ".join(sorted(self.hardware)) or "none defined"
                raise ValueError(
                    f"model '{model.name}' references unknown hardware "
                    f"'{model.hardware}'; available: {available}"
                )
        return self

    def hardware_for(self, model: ModelDeployment) -> Optional[HardwareProfile]:
        return None if model.hardware is None else self.hardware[model.hardware]


def load_models(path: Union[str, Path]) -> ModelsConfig:
    """Read and validate a models file, raising ModelsConfigError on any problem."""
    return load_yaml_model(
        Path(path), ModelsConfig, ModelsConfigError, "models", "name"
    )
