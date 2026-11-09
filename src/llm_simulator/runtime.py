"""Builds what each configured model needs to serve requests.

This is the composition root: the one module that reads the parsed
configuration and constructs every component. Components themselves take
plain parameters, so they never depend on the configuration classes.
"""

import functools
import logging
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Type

from llm_simulator.llm.generator import Generator
from llm_simulator.llm.matcher import Matcher
from llm_simulator.llm.models import (
    Efficiency,
    HardwareProfile,
    Latency,
    ModelDeployment,
    ModelsConfig,
    ModelsConfigError,
)
from llm_simulator.llm.responses import ResponsesConfig
from llm_simulator.llm.tokenizer import Tokenizer, get_tokenizer
from llm_simulator.metrics import MetricsRecorder
from llm_simulator.simulator import (
    Engine,
    FixedRateEngine,
    InstantEngine,
    SchedulerEngine,
)
from llm_simulator.simulator.scheduler.cost_model import RooflineCostModel
from llm_simulator.simulator.scheduler.kv_cache import (
    BlockAllocator,
    kv_capacity_blocks,
)
from llm_simulator.simulator.scheduler.policy import Scheduler
from llm_simulator.types import StepListener

logger = logging.getLogger(__name__)

GIB = 2**30


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
    # Called only after the engine's required sections are known to be
    # present, with the model's hardware profile when it references one, and
    # the listener for its steps when metrics are recorded. Engines without
    # steps ignore the listener.
    build: Callable[
        [ModelDeployment, Optional[HardwareProfile], Optional[StepListener]], Engine
    ]


def _fixed_rate(
    deployment: ModelDeployment,
    hardware: Optional[HardwareProfile],
    on_step: Optional[StepListener],
) -> Engine:
    latency = deployment.latency
    return FixedRateEngine(
        prefill_tokens_per_second=deployment.performance.prefill_tokens_per_second,
        decode_tokens_per_second=deployment.performance.decode_tokens_per_second,
        scheduler_overhead_s=latency.scheduler_overhead_ms / 1000 if latency else 0.0,
    )


@dataclass(frozen=True)
class SchedulerParts:
    policy: Scheduler
    cost: RooflineCostModel


def _scheduler_parts(
    deployment: ModelDeployment, hardware: HardwareProfile
) -> SchedulerParts:
    """The scheduler's KV cache, policy and cost model, with startup checks.

    Raises ValueError for a configuration a real engine would refuse to
    start with.
    """
    if hardware.peak_tflops is None:
        raise ValueError(f"hardware '{deployment.hardware}' needs peak_tflops")
    weights, kv = deployment.weights, deployment.kv_cache
    weight_bytes = weights.parameters_billions * 1e9 * weights.bytes_per_parameter
    blocks = kv_capacity_blocks(
        hbm_bytes=hardware.hbm_gib * GIB,
        gpu_memory_utilization=kv.gpu_memory_utilization,
        weight_bytes=weight_bytes,
        kv_bytes_per_token=kv.bytes_per_token,
        block_size=kv.block_size,
    )
    allocator = BlockAllocator(blocks, kv.block_size, kv.watermark)
    if deployment.context_length > allocator.admissible_tokens:
        # Otherwise a request the API accepts could never be admitted.
        raise ValueError(
            f"context_length {deployment.context_length} does not fit the KV "
            f"cache, which admits at most {allocator.admissible_tokens} tokens"
        )
    policy = Scheduler(
        allocator,
        deployment.scheduler.max_batch_tokens,
        deployment.scheduler.max_num_sequences,
    )
    latency = deployment.latency or Latency()
    efficiency = deployment.efficiency or Efficiency()
    cost = RooflineCostModel(
        parameters=weights.parameters_billions * 1e9,
        weight_bytes=weight_bytes,
        kv_bytes_per_token=kv.bytes_per_token,
        peak_flops=hardware.peak_tflops * 1e12,
        memory_bandwidth=hardware.memory_bandwidth_gbps * 1e9,
        compute_efficiency=efficiency.compute,
        memory_efficiency=efficiency.memory,
        step_overhead_s=latency.step_overhead_ms / 1000,
    )
    logger.info(
        "model '%s': KV cache of %d blocks (%d tokens), %d reserved by the watermark",
        deployment.name,
        blocks,
        blocks * kv.block_size,
        allocator.watermark_blocks,
    )
    return SchedulerParts(policy, cost)


def _scheduler(
    deployment: ModelDeployment,
    hardware: HardwareProfile,
    on_step: Optional[StepListener],
) -> Engine:
    parts = _scheduler_parts(deployment, hardware)
    return SchedulerEngine(parts.policy, parts.cost, on_step=on_step)


ENGINES: Dict[str, EngineBuilder] = {
    InstantEngine.name: EngineBuilder(
        InstantEngine, lambda deployment, hardware, on_step: InstantEngine()
    ),
    FixedRateEngine.name: EngineBuilder(FixedRateEngine, _fixed_rate),
    SchedulerEngine.name: EngineBuilder(SchedulerEngine, _scheduler),
}


def build_runtimes(
    models: ModelsConfig,
    responses: ResponsesConfig,
    recorder: Optional[MetricsRecorder] = None,
) -> Runtimes:
    """Build one runtime per model, raising ModelsConfigError for unusable entries.

    All models share one Matcher, so the response rules are compiled once.
    Engines with steps report each one to recorder, labelled with the model's
    name; request metrics are recorded by the API, not here.
    """
    matcher = Matcher(responses)
    runtimes = {}
    for deployment in models.models:
        tokenizer = _build_tokenizer(deployment)
        on_step = (
            None
            if recorder is None
            else functools.partial(recorder.step, deployment.name)
        )
        engine = _build_engine(deployment, models.hardware_for(deployment), on_step)
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


def _build_engine(
    deployment: ModelDeployment,
    hardware: Optional[HardwareProfile],
    on_step: Optional[StepListener],
) -> Engine:
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
        return builder.build(deployment, hardware, on_step)
    except ValueError as exc:
        raise ModelsConfigError(f"model '{deployment.name}': engine: {exc}") from None
