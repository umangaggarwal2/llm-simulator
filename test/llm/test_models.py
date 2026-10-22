from pathlib import Path

import pytest
import yaml

from llm_simulator.llm.models import ModelsConfig, ModelsConfigError, load_models

REPO_ROOT = Path(__file__).resolve().parents[2]


def write(tmp_path, data):
    path = tmp_path / "models.yml"
    path.write_text(data if isinstance(data, str) else yaml.safe_dump(data))
    return path


def minimal(**model):
    return {"version": 1, "models": [{"name": "m", "context_length": 100, **model}]}


def test_repo_config_loads():
    config = load_models(REPO_ROOT / "config" / "models.yml")
    names = [model.name for model in config.models]
    assert names == ["qwen-mock", "tiny-mock"]
    qwen = config.models[0]
    assert config.hardware_for(qwen).hbm_gb == 80
    assert qwen.performance.decode_tokens_per_second == 250
    assert config.hardware_for(config.models[1]) is None


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
    data["hardware"] = {"gpu": {"hbm_gb": 24, "memory_bandwidth_gbps": 600}}
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
