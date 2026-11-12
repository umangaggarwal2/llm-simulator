from argparse import Namespace
from pathlib import Path

import pytest

from llm_simulator import defaults
from llm_simulator.settings import Settings


def test_defaults_when_nothing_is_set():
    settings = Settings.from_env({})
    assert settings == Settings()
    # The files shipped with the package, so it runs from any directory.
    assert settings.models_path == defaults.path("models")
    assert settings.responses_path == defaults.path("responses")
    assert settings.models_path.is_file() and settings.responses_path.is_file()
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


def flags(host=None, port=None, models=None, responses=None):
    return Namespace(host=host, port=port, models=models, responses=responses)


def test_flags_override_their_variables():
    env = {
        "LLM_SIM_HOST": "10.0.0.1",
        "LLM_SIM_PORT": "9000",
        "LLM_SIM_MODELS": "/env/models.yml",
        "LLM_SIM_RESPONSES": "/env/responses.yml",
    }
    settings = Settings.from_args(
        flags("127.0.0.1", 9100, Path("/flag/models.yml"), Path("/flag/responses.yml")),
        env,
    )
    assert (settings.host, settings.port) == ("127.0.0.1", 9100)
    assert settings.models_path == Path("/flag/models.yml")
    assert settings.responses_path == Path("/flag/responses.yml")


def test_flags_not_given_keep_the_environment_or_the_default():
    settings = Settings.from_args(
        flags(responses=Path("/flag/responses.yml")), {"LLM_SIM_PORT": "9000"}
    )
    assert settings.responses_path == Path("/flag/responses.yml")
    assert settings.port == 9000
    assert settings.models_path == Settings().models_path
    assert settings.host == Settings().host


def test_flags_leave_the_summary_settings_to_the_environment():
    settings = Settings.from_args(
        flags(port=9100), {"LLM_SIM_SUMMARY_RATE_SECONDS": "10"}
    )
    assert settings.summary_rate_seconds == 10


# --- Built-in files ---


def test_builtin_names_resolve_to_the_shipped_files():
    settings = Settings.from_env(
        {
            "LLM_SIM_MODELS": "builtin:models",
            "LLM_SIM_RESPONSES": "builtin:bench-responses",
        }
    )
    assert settings.models_path == defaults.path("models")
    assert settings.responses_path == defaults.path("bench-responses")
    assert settings.responses_path.name == "bench-responses.yml"


def test_an_unknown_builtin_name_names_the_variable_and_the_choices():
    with pytest.raises(ValueError) as error:
        Settings.from_env({"LLM_SIM_RESPONSES": "builtin:bench"})
    message = str(error.value)
    assert message.startswith("LLM_SIM_RESPONSES: unknown built-in config 'bench'")
    assert "models, responses, bench-responses" in message


def test_other_values_are_paths():
    settings = Settings.from_env({"LLM_SIM_MODELS": "my/models.yml"})
    assert settings.models_path == Path("my/models.yml")


@pytest.mark.parametrize(
    "name", ["LLM_SIM_MODELS", "LLM_SIM_RESPONSES", "LLM_SIM_HOST", "LLM_SIM_PORT"]
)
def test_an_empty_variable_counts_as_unset(name):
    assert Settings.from_env({name: ""}) == Settings()


def test_every_builtin_file_is_shipped():
    for name in defaults.NAMES:
        assert defaults.path(name).is_file(), name
