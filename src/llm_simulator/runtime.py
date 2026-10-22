"""Builds what each configured model needs to serve requests.

This is the one module that assembles every layer: it reads the parsed
configuration and gives the generator and engine only the values they need.
"""

from dataclasses import dataclass
from typing import Mapping

from llm_simulator.llm.generator import Generator
from llm_simulator.llm.matcher import Matcher
from llm_simulator.llm.models import ModelDeployment, ModelsConfig, ModelsConfigError
from llm_simulator.llm.responses import ResponsesConfig
from llm_simulator.llm.tokenizer import Tokenizer, get_tokenizer
from llm_simulator.simulator.engine import Engine, get_engine


@dataclass(frozen=True)
class ModelRuntime:
    deployment: ModelDeployment
    tokenizer: Tokenizer
    generator: Generator
    engine: Engine


Runtimes = Mapping[str, ModelRuntime]


def build_runtimes(models: ModelsConfig, responses: ResponsesConfig) -> Runtimes:
    """Build one runtime per model, raising ModelsConfigError for unusable entries.

    All models share one Matcher, so the response rules are compiled once.
    """
    matcher = Matcher(responses)
    runtimes = {}
    for deployment in models.models:
        tokenizer = _build(deployment, "tokenizer", get_tokenizer)
        engine = _build(deployment, "engine", get_engine)
        _check_required_sections(deployment, engine)
        runtimes[deployment.name] = ModelRuntime(
            deployment=deployment,
            tokenizer=tokenizer,
            generator=Generator(
                matcher, tokenizer, deployment.context_length, responses.seed
            ),
            engine=engine,
        )
    return runtimes


def _build(deployment: ModelDeployment, field: str, factory):
    try:
        return factory(getattr(deployment, field))
    except ValueError as exc:
        raise ModelsConfigError(f"model '{deployment.name}': {field}: {exc}") from None


def _check_required_sections(deployment: ModelDeployment, engine: Engine) -> None:
    missing = [
        section
        for section in engine.required_sections
        if getattr(deployment, section) is None
    ]
    if missing:
        raise ModelsConfigError(
            f"model '{deployment.name}': engine '{engine.name}' requires "
            f"section(s) {', '.join(missing)}"
        )
