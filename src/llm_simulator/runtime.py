"""Builds what each configured model needs to serve requests.

This is the composition root: the one module that reads the parsed
configuration and constructs every component. Components themselves take
plain parameters, so they never depend on the configuration classes.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Type

from llm_simulator.llm.generator import Generator
from llm_simulator.llm.matcher import Matcher
from llm_simulator.llm.models import ModelDeployment, ModelsConfig, ModelsConfigError
from llm_simulator.llm.responses import ResponsesConfig
from llm_simulator.llm.tokenizer import Tokenizer, get_tokenizer
from llm_simulator.simulator import Engine, FixedRateEngine, InstantEngine


@dataclass(frozen=True)
class ModelRuntime:
    deployment: ModelDeployment
    tokenizer: Tokenizer
    generator: Generator
    engine: Engine


Runtimes = Mapping[str, ModelRuntime]


@dataclass(frozen=True)
class EngineBuilder:
    engine: Type[Engine]
    # Called only after the engine's required sections are known to be present.
    build: Callable[[ModelDeployment], Engine]


def _fixed_rate(deployment: ModelDeployment) -> Engine:
    latency = deployment.latency
    return FixedRateEngine(
        prefill_tokens_per_second=deployment.performance.prefill_tokens_per_second,
        decode_tokens_per_second=deployment.performance.decode_tokens_per_second,
        scheduler_overhead_s=latency.scheduler_overhead_ms / 1000 if latency else 0.0,
    )


ENGINES: Dict[str, EngineBuilder] = {
    InstantEngine.name: EngineBuilder(
        InstantEngine, lambda deployment: InstantEngine()
    ),
    FixedRateEngine.name: EngineBuilder(FixedRateEngine, _fixed_rate),
}


def build_runtimes(models: ModelsConfig, responses: ResponsesConfig) -> Runtimes:
    """Build one runtime per model, raising ModelsConfigError for unusable entries.

    All models share one Matcher, so the response rules are compiled once.
    """
    matcher = Matcher(responses)
    runtimes = {}
    for deployment in models.models:
        tokenizer = _build_tokenizer(deployment)
        engine = _build_engine(deployment)
        runtimes[deployment.name] = ModelRuntime(
            deployment=deployment,
            tokenizer=tokenizer,
            generator=Generator(
                matcher, tokenizer, deployment.context_length, responses.seed
            ),
            engine=engine,
        )
    return runtimes


def _build_tokenizer(deployment: ModelDeployment) -> Tokenizer:
    try:
        return get_tokenizer(deployment.tokenizer)
    except ValueError as exc:
        raise ModelsConfigError(
            f"model '{deployment.name}': tokenizer: {exc}"
        ) from None


def _build_engine(deployment: ModelDeployment) -> Engine:
    builder = ENGINES.get(deployment.engine)
    if builder is None:
        available = ", ".join(sorted(ENGINES))
        raise ModelsConfigError(
            f"model '{deployment.name}': engine: unknown engine "
            f"'{deployment.engine}'; available: {available}"
        )

    missing = [
        section
        for section in builder.engine.required_sections
        if getattr(deployment, section) is None
    ]
    if missing:
        raise ModelsConfigError(
            f"model '{deployment.name}': engine '{deployment.engine}' requires "
            f"section(s) {', '.join(missing)}"
        )

    try:
        return builder.build(deployment)
    except ValueError as exc:
        raise ModelsConfigError(f"model '{deployment.name}': engine: {exc}") from None
