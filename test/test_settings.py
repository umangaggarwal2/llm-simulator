from pathlib import Path

import pytest

from llm_simulator.settings import Settings


def test_defaults_when_nothing_is_set():
    settings = Settings.from_env({})
    assert settings == Settings()
    assert settings.models_path == Path("config/models.yml")
    assert settings.responses_path == Path("config/responses.yml")
    assert settings.host == "0.0.0.0"
    assert settings.port == 8080
    assert settings.summary_request_samples == 10_000
    assert settings.summary_token_samples == 100_000
    assert settings.summary_rate_seconds == 60


def test_every_variable_is_read():
    settings = Settings.from_env(
        {
            "LLM_SIM_MODELS": "/etc/sim/models.yml",
            "LLM_SIM_RESPONSES": "/etc/sim/responses.yml",
            "LLM_SIM_HOST": "127.0.0.1",
            "LLM_SIM_PORT": "9000",
            "LLM_SIM_SUMMARY_REQUEST_SAMPLES": "500",
            "LLM_SIM_SUMMARY_TOKEN_SAMPLES": "20000",
            "LLM_SIM_SUMMARY_RATE_SECONDS": "10",
        }
    )
    assert settings == Settings(
        models_path=Path("/etc/sim/models.yml"),
        responses_path=Path("/etc/sim/responses.yml"),
        host="127.0.0.1",
        port=9000,
        summary_request_samples=500,
        summary_token_samples=20000,
        summary_rate_seconds=10,
    )


def test_unset_variables_keep_their_defaults():
    settings = Settings.from_env({"LLM_SIM_PORT": "9000"})
    assert settings.port == 9000
    assert settings.models_path == Settings().models_path


def test_reads_the_process_environment_by_default(monkeypatch):
    monkeypatch.setenv("LLM_SIM_HOST", "10.0.0.1")
    assert Settings.from_env().host == "10.0.0.1"


def test_non_numeric_port_is_rejected():
    with pytest.raises(ValueError):
        Settings.from_env({"LLM_SIM_PORT": "eighty"})


@pytest.mark.parametrize(
    "name",
    [
        "LLM_SIM_PORT",
        "LLM_SIM_SUMMARY_REQUEST_SAMPLES",
        "LLM_SIM_SUMMARY_TOKEN_SAMPLES",
        "LLM_SIM_SUMMARY_RATE_SECONDS",
    ],
)
@pytest.mark.parametrize("raw", ["eighty", "0", "-5", "1.5"])
def test_numbers_must_be_positive_integers(name, raw):
    with pytest.raises(ValueError, match=name):
        Settings.from_env({name: raw})
