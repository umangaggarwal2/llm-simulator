import pytest

from llm_simulator.llm.models import ModelsConfig, ModelsConfigError
from llm_simulator.llm.responses import ResponsesConfig
from llm_simulator.llm.tokenizer import RegexTokenizer
from llm_simulator.runtime import build_runtimes
from llm_simulator.simulator import engine as engine_module
from llm_simulator.simulator.engine import InstantEngine
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

    monkeypatch.setitem(engine_module._ENGINES, NeedsPerformance.name, NeedsPerformance)
    with pytest.raises(
        ModelsConfigError,
        match="engine 'needs_performance' requires section\\(s\\) performance, latency",
    ):
        build_runtimes(
            models({"name": "m", "context_length": 10, "engine": "needs_performance"}),
            RESPONSES,
        )
