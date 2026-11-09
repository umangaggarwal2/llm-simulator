import pytest

from llm_simulator.llm.models import ModelsConfig, ModelsConfigError
from llm_simulator.llm.responses import ResponsesConfig
from llm_simulator.llm.tokenizer import RegexTokenizer
from llm_simulator import runtime as runtime_module
from llm_simulator.runtime import EngineBuilder, _scheduler_parts, build_runtimes
from llm_simulator.simulator import FixedRateEngine, InstantEngine
from llm_simulator.metrics import NoopRecorder
from llm_simulator.types import GenerationRequest, Message

RESPONSES = ResponsesConfig.model_validate(
    {"version": 1, "default": {"content": "Hi from {model}"}}
)


def models(*entries):
    return ModelsConfig.model_validate({"version": 1, "models": list(entries)})


def test_builds_one_runtime_per_model():
    runtimes = build_runtimes(
        models(
            {"name": "big", "context_length": 1000},
            {"name": "small", "context_length": 20},
        ),
        RESPONSES,
    )
    assert list(runtimes) == ["big", "small"]
    assert isinstance(runtimes["big"].engine, InstantEngine)
    assert isinstance(runtimes["big"].tokenizer, RegexTokenizer)
    # Each model gets its own engine instance.
    assert runtimes["big"].engine is not runtimes["small"].engine


def test_each_generator_uses_its_models_context_length():
    runtimes = build_runtimes(
        models(
            {"name": "big", "context_length": 1000},
            {"name": "small", "context_length": 20},
        ),
        RESPONSES,
    )
    request = GenerationRequest("small", (Message("user", "hello"),), max_tokens=15)
    assert runtimes["big"].generator.generate(request).text == "Hi from small"
    with pytest.raises(Exception, match="maximum context length is 20"):
        runtimes["small"].generator.generate(request)


def test_unknown_tokenizer_names_the_model():
    with pytest.raises(
        ModelsConfigError, match="model 'm': tokenizer: unknown tokenizer 'bpe'"
    ):
        build_runtimes(
            models({"name": "m", "context_length": 10, "tokenizer": "bpe"}), RESPONSES
        )


def test_unknown_engine_names_the_model():
    with pytest.raises(ModelsConfigError, match="model 'm': engine: unknown engine"):
        build_runtimes(
            models({"name": "m", "context_length": 10, "engine": "warp"}), RESPONSES
        )


def test_engine_required_sections_are_enforced(monkeypatch):
    class NeedsPerformance(InstantEngine):
        name = "needs_performance"
        required_sections = ("performance", "latency")

    monkeypatch.setitem(
        runtime_module.ENGINES,
        NeedsPerformance.name,
        EngineBuilder(
            NeedsPerformance, lambda deployment, hardware, on_step: NeedsPerformance()
        ),
    )
    with pytest.raises(
        ModelsConfigError,
        match="engine 'needs_performance' requires section\\(s\\) performance, latency",
    ):
        build_runtimes(
            models({"name": "m", "context_length": 10, "engine": "needs_performance"}),
            RESPONSES,
        )


def test_unknown_engine_lists_available_engines():
    with pytest.raises(ModelsConfigError, match="available: fixed_rate, instant"):
        build_runtimes(
            models({"name": "m", "context_length": 10, "engine": "warp"}), RESPONSES
        )


def test_instant_is_the_default_engine():
    runtimes = build_runtimes(models({"name": "m", "context_length": 10}), RESPONSES)
    assert isinstance(runtimes["m"].engine, InstantEngine)


PERFORMANCE = {"prefill_tokens_per_second": 1000, "decode_tokens_per_second": 50}


def test_fixed_rate_engine_gets_its_parameters_from_the_config():
    runtimes = build_runtimes(
        models(
            {
                "name": "m",
                "context_length": 10,
                "engine": "fixed_rate",
                "performance": PERFORMANCE,
                "latency": {"scheduler_overhead_ms": 5},
            }
        ),
        RESPONSES,
    )
    engine = runtimes["m"].engine
    assert isinstance(engine, FixedRateEngine)
    # 5 ms overhead + 100 prompt tokens at 1000 tok/s.
    assert engine.ttft(100) == pytest.approx(0.105)


def test_fixed_rate_latency_section_is_optional():
    runtimes = build_runtimes(
        models(
            {
                "name": "m",
                "context_length": 10,
                "engine": "fixed_rate",
                "performance": PERFORMANCE,
            }
        ),
        RESPONSES,
    )
    assert runtimes["m"].engine.ttft(100) == pytest.approx(0.1)


def test_fixed_rate_requires_performance():
    with pytest.raises(
        ModelsConfigError,
        match="model 'm': engine 'fixed_rate' requires section\\(s\\) performance",
    ):
        build_runtimes(
            models({"name": "m", "context_length": 10, "engine": "fixed_rate"}),
            RESPONSES,
        )


# --- Scheduler parts ---

H100 = {"hbm_gib": 80, "memory_bandwidth_gbps": 3000, "peak_tflops": 989}
SCHEDULED = {
    "name": "m",
    "context_length": 32768,
    "hardware": "h100",
    "weights": {"parameters_billions": 6.74},
    "kv_cache": {"bytes_per_token": 524288},
    "scheduler": {"max_batch_tokens": 8192, "max_num_sequences": 64},
}


def scheduler_parts(hardware=None, **overrides):
    config = ModelsConfig.model_validate(
        {
            "version": 1,
            "hardware": {"h100": {**H100, **(hardware or {})}},
            "models": [{**SCHEDULED, **overrides}],
        }
    )
    deployment = config.models[0]
    return _scheduler_parts(deployment, config.hardware_for(deployment))


def test_scheduler_kv_capacity_uses_gib_of_hbm():
    # 80 GiB x 0.9 = 77.3e9 bytes; minus 13.48e9 of weights; 8 MiB per block.
    policy = scheduler_parts().policy
    kv = policy._kv
    assert kv.num_blocks == 7609
    assert kv.watermark_blocks == 76
    assert kv.block_size == 16


def test_scheduler_cost_model_converts_units():
    cost = scheduler_parts(
        latency={"step_overhead_ms": 1.5},
        efficiency={"compute": 0.5, "memory": 0.8},
    ).cost
    assert cost.parameters == pytest.approx(6.74e9)
    assert cost.weight_bytes == pytest.approx(13.48e9)
    assert cost.peak_flops == pytest.approx(989e12)
    assert cost.memory_bandwidth == pytest.approx(3000e9)
    assert (cost.compute_efficiency, cost.memory_efficiency) == (0.5, 0.8)
    assert cost.step_overhead_s == pytest.approx(0.0015)


def test_scheduler_optional_sections_have_defaults():
    cost = scheduler_parts().cost
    assert (cost.compute_efficiency, cost.memory_efficiency) == (1.0, 1.0)
    assert cost.step_overhead_s == 0


def test_scheduler_bytes_per_parameter_sets_weight_size():
    cost = scheduler_parts(
        weights={"parameters_billions": 6.74, "bytes_per_parameter": 1}
    ).cost
    assert cost.weight_bytes == pytest.approx(6.74e9)
    assert cost.parameters == pytest.approx(6.74e9)


def test_scheduler_needs_peak_tflops():
    with pytest.raises(ValueError, match="hardware 'h100' needs peak_tflops"):
        scheduler_parts(hardware={"peak_tflops": None})


def test_scheduler_weights_must_fit():
    with pytest.raises(ValueError, match="no room for the KV cache"):
        scheduler_parts(weights={"parameters_billions": 40})


def test_scheduler_context_length_must_fit_the_cache():
    # 7,609 blocks less a 76-block watermark admit 120,528 tokens.
    scheduler_parts(context_length=120_528)
    with pytest.raises(ValueError, match="admits at most 120528 tokens"):
        scheduler_parts(context_length=120_529)


def test_scheduler_batch_must_hold_a_decode_per_sequence():
    with pytest.raises(ValueError, match="max_batch_tokens"):
        scheduler_parts(scheduler={"max_batch_tokens": 32, "max_num_sequences": 64})


class StepLog(NoopRecorder):
    def __init__(self):
        self.steps = []

    def step(self, model, stats):
        self.steps.append((model, stats))


def test_scheduler_steps_reach_the_recorder_with_the_models_name():
    config = ModelsConfig.model_validate(
        {
            "version": 1,
            "hardware": {"h100": H100},
            "models": [{**SCHEDULED, "name": "batched", "engine": "scheduler"}],
        }
    )
    recorder = StepLog()
    engine = build_runtimes(config, RESPONSES, recorder)["batched"].engine
    engine._on_step("stats")
    assert recorder.steps == [("batched", "stats")]


def test_without_a_recorder_steps_are_not_reported():
    config = ModelsConfig.model_validate(
        {
            "version": 1,
            "hardware": {"h100": H100},
            "models": [{**SCHEDULED, "engine": "scheduler"}],
        }
    )
    assert build_runtimes(config, RESPONSES)["m"].engine._on_step is None
