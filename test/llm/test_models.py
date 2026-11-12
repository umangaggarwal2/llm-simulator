from pathlib import Path

import pytest
import yaml

from llm_simulator import defaults
from llm_simulator.llm.models import ModelsConfig, ModelsConfigError, load_models

REPO_ROOT = Path(__file__).resolve().parents[2]


def write(tmp_path, data):
    path = tmp_path / "models.yml"
    path.write_text(data if isinstance(data, str) else yaml.safe_dump(data))
    return path


def minimal(**model):
    return {"version": 1, "models": [{"name": "m", "context_length": 100, **model}]}


def test_repo_config_loads():
    config = load_models(defaults.path("models"))
    names = [model.name for model in config.models]
    assert names == ["qwen-mock", "qwen-mock-batched", "tiny-mock"]
    qwen = config.models[0]
    assert config.hardware_for(qwen).hbm_gib == 80
    assert qwen.performance.decode_tokens_per_second == 250
    assert config.hardware_for(config.models[2]) is None


def test_defaults():
    model = ModelsConfig.model_validate(minimal()).models[0]
    assert model.tokenizer == "regex"
    assert model.engine == "instant"
    assert model.hardware is None
    assert model.performance is None


def test_model_names_allow_slashes_and_dots():
    config = ModelsConfig.model_validate(minimal(name="Qwen/Qwen2.5-7B-Instruct"))
    assert config.models[0].name == "Qwen/Qwen2.5-7B-Instruct"


@pytest.mark.parametrize(
    "data, message",
    [
        ({"version": 2, "models": [{"name": "m", "context_length": 1}]}, "version"),
        ({"version": 1, "models": []}, "at least 1 item"),
        (minimal(context_length=0), "greater than or equal to 1"),
        (minimal(context_length=True), "valid integer"),
        (minimal(name="has space"), "should match pattern"),
        (minimal(contxt_length=5), "Extra inputs are not permitted"),
        (minimal(hardware="h100"), "unknown hardware 'h100'; available: none defined"),
        (
            minimal(
                performance={
                    "prefill_tokens_per_second": 0,
                    "decode_tokens_per_second": 1,
                }
            ),
            "greater than 0",
        ),
        (minimal(kv_cache={"bytes_per_token": 1.5}), "valid integer"),
        (minimal(latency={"scheduler_overhead_ms": -1}), "greater than or equal to 0"),
        (minimal(latency={"step_overhead_ms": -1}), "greater than or equal to 0"),
        (minimal(weights={"parameters_billions": 0}), "greater than 0"),
        (minimal(weights={"parameters_billions": "6.74e9"}), "valid"),
        (
            minimal(kv_cache={"bytes_per_token": 1, "gpu_memory_utilization": 0}),
            "greater than 0",
        ),
        (
            minimal(kv_cache={"bytes_per_token": 1, "gpu_memory_utilization": 1.5}),
            "less than or equal to 1",
        ),
        (minimal(kv_cache={"bytes_per_token": 1, "watermark": 1}), "less than 1"),
        (minimal(kv_cache={"bytes_per_token": 1, "block_size": 0}), "greater than"),
        (minimal(efficiency={"memory": 0}), "greater than 0"),
        (minimal(efficiency={"compute": 1.2}), "less than or equal to 1"),
    ],
)
def test_invalid_config(tmp_path, data, message):
    with pytest.raises(ModelsConfigError, match=message):
        load_models(write(tmp_path, data))


def test_duplicate_names(tmp_path):
    data = {
        "version": 1,
        "models": [
            {"name": "a", "context_length": 1},
            {"name": "a", "context_length": 2},
        ],
    }
    with pytest.raises(ModelsConfigError, match="duplicate model names: a"):
        load_models(write(tmp_path, data))


def test_hardware_reference_resolves(tmp_path):
    data = minimal(hardware="gpu")
    data["hardware"] = {"gpu": {"hbm_gib": 24, "memory_bandwidth_gbps": 600}}
    config = load_models(write(tmp_path, data))
    assert config.hardware_for(config.models[0]).memory_bandwidth_gbps == 600


def test_errors_name_the_model(tmp_path):
    data = {
        "version": 1,
        "models": [
            {"name": "good", "context_length": 1},
            {"name": "broken", "context_length": 0},
        ],
    }
    with pytest.raises(ModelsConfigError) as error:
        load_models(write(tmp_path, data))
    assert "models[1] (broken).context_length" in str(error.value)


def test_unreadable_and_malformed_files(tmp_path):
    with pytest.raises(ModelsConfigError, match="cannot read file"):
        load_models(tmp_path / "missing.yml")
    with pytest.raises(ModelsConfigError, match="invalid YAML"):
        load_models(write(tmp_path, "models: [\n"))
    with pytest.raises(ModelsConfigError, match="expected a mapping"):
        load_models(write(tmp_path, "- 1\n"))


def test_scheduler_defaults():
    model = ModelsConfig.model_validate(
        minimal(
            weights={"parameters_billions": 7},
            kv_cache={"bytes_per_token": 1024},
            latency={},
            efficiency={},
        )
    ).models[0]
    assert model.weights.bytes_per_parameter == 2
    assert (
        model.kv_cache.block_size,
        model.kv_cache.gpu_memory_utilization,
        model.kv_cache.watermark,
    ) == (16, 0.9, 0.01)
    assert model.latency.step_overhead_ms == 0
    assert (model.efficiency.compute, model.efficiency.memory) == (1.0, 1.0)


def test_peak_tflops_is_optional_on_a_profile(tmp_path):
    data = minimal(hardware="gpu")
    data["hardware"] = {"gpu": {"hbm_gib": 24, "memory_bandwidth_gbps": 600}}
    config = load_models(write(tmp_path, data))
    assert config.hardware_for(config.models[0]).peak_tflops is None


def test_hbm_gb_was_renamed(tmp_path):
    data = minimal()
    data["hardware"] = {"gpu": {"hbm_gb": 24, "memory_bandwidth_gbps": 600}}
    with pytest.raises(ModelsConfigError, match="hbm_gb"):
        load_models(write(tmp_path, data))


def test_validation_variants_differ_from_base_only_where_intended():
    config = load_models(REPO_ROOT / "config" / "validation" / "models.yml")
    models = {model.name: model.model_dump() for model in config.models}
    base = models.pop("base")
    batched = next(
        m
        for m in load_models(defaults.path("models")).models
        if m.name == "qwen-mock-batched"
    ).model_dump()
    assert {**base, "name": None} == {**batched, "name": None}

    def changes(variant):
        return {
            (section, key)
            for section, value in variant.items()
            if section != "name"
            for key in (value if isinstance(value, dict) else {None: value})
            if (value.get(key) if isinstance(value, dict) else value)
            != (base[section].get(key) if isinstance(value, dict) else base[section])
        }

    assert {name: changes(m) for name, m in models.items()} == {
        "budget-2048": {("scheduler", "max_batch_tokens")},
        "small-kv": {("context_length", None), ("kv_cache", "gpu_memory_utilization")},
        "small-kv-no-watermark": {
            ("context_length", None),
            ("kv_cache", "gpu_memory_utilization"),
            ("kv_cache", "watermark"),
        },
        "overhead-1ms": {("latency", "step_overhead_ms")},
        "memory-eff-80": {("efficiency", "memory")},
    }
