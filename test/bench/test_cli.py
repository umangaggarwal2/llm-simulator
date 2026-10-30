import json
import sys

import pytest

from llm_simulator.bench import cli
from llm_simulator.bench.config import BenchConfig


def test_parse_args():
    config, output, *_ = cli.parse_args(
        [
            "--model",
            "qwen-mock",
            "--base-url",
            "http://h:9/v1",
            "--concurrency",
            "8",
            "--requests",
            "50",
            "--max-tokens",
            "128",
            "--input-tokens",
            "512",
            "--prompt-prefix",
            "[short] ",
            "--seed",
            "3",
            "--timeout",
            "5",
            "--output",
            "run.json",
        ]
    )
    assert config == BenchConfig(
        model="qwen-mock",
        base_url="http://h:9/v1",
        concurrency=8,
        requests=50,
        max_tokens=128,
        input_tokens=512,
        prompt_prefix="[short] ",
        seed=3,
        timeout=5.0,
    )
    assert str(output) == "run.json"


def test_defaults_use_the_fixed_prompt_and_write_nothing():
    config, output, *_ = cli.parse_args(["--model", "m"])
    assert (config.input_tokens, config.dataset, output) == (None, None, None)
    assert cli.build_workload(config).request(0).prompt == config.prompt


def test_api_key_falls_back_to_the_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    assert cli.parse_args(["--model", "m"])[0].api_key == "sk-env"
    assert cli.parse_args(["--model", "m", "--api-key", "sk-arg"])[0].api_key == (
        "sk-arg"
    )


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--model", "m", "--concurrency", "0"],
        ["--model", "m", "--requests", "x"],
        ["--model", "m", "--input-tokens", "10", "--dataset", "d.jsonl"],
        ["--model", "m", "--requests", "10", "--duration", "5"],
        ["--model", "m", "--warmup", "-1"],
    ],
)
def test_bad_arguments_exit_with_usage(argv, capsys):
    with pytest.raises(SystemExit) as exit:
        cli.parse_args(argv)
    assert exit.value.code == 2


def test_build_workload_picks_the_source(tmp_path):
    dataset = tmp_path / "d.jsonl"
    dataset.write_text('{"prompt": "from file"}\n')
    by_tokens = cli.build_workload(BenchConfig(model="m", input_tokens=50))
    by_file = cli.build_workload(BenchConfig(model="m", dataset=dataset))
    assert by_tokens.description.startswith("synthetic 50 input tokens")
    assert by_file.request(0).prompt == "from file"


def test_workload_errors_stop_before_sending(tmp_path, capsys):
    missing = tmp_path / "missing.jsonl"
    assert cli.main(["--model", "m", "--dataset", str(missing)]) == 2
    assert "missing.jsonl" in capsys.readouterr().err
    assert cli.main(["--model", "m", "--input-tokens", "3"]) == 2
    assert "input_tokens 3 is less than" in capsys.readouterr().err


def test_missing_aiohttp_prints_the_install_hint(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "aiohttp", None)  # makes the import fail
    assert cli.main(["--model", "m"]) == 1
    assert cli.INSTALL_HINT in capsys.readouterr().err


def test_phase_arguments():
    config, *_ = cli.parse_args(
        ["--model", "m", "--duration", "30", "--warmup", "0", "--concurrency", "4"]
    )
    assert (config.duration, config.warmup, config.measured_requests) == (30.0, 0, None)


def test_server_metrics_arguments():
    config, *_ = cli.parse_args(["--model", "m"])
    assert (config.server_metrics, config.server_metrics_url) == (True, None)
    config, *_ = cli.parse_args(
        ["--model", "m", "--no-server-metrics", "--server-metrics-url", "http://h/s"]
    )
    assert (config.server_metrics, config.server_metrics_url) == (False, "http://h/s")


def test_one_value_is_a_single_run():
    args = cli.parse_args(["--model", "m", "--concurrency", "8"])
    assert (args.config.concurrency, args.sweep) == (8, None)


def test_a_list_is_a_sweep():
    args = cli.parse_args(["--model", "m", "--concurrency", "1,8,32"])
    assert (args.sweep, list(args.values)) == ("concurrency", [1, 8, 32])
    assert args.config.concurrency == 1


def test_open_loop_arguments():
    args = cli.parse_args(
        ["--model", "m", "--rate", "5,10.5", "--max-concurrency", "16", "--seed", "4"]
    )
    assert (args.sweep, list(args.values)) == ("rate", [5.0, 10.5])
    config = args.config
    assert (config.rate, config.max_concurrency, config.mode) == (5.0, 16, "open")


@pytest.mark.parametrize(
    "argv",
    [
        ["--model", "m", "--concurrency", "1,x"],
        ["--model", "m", "--concurrency", "4", "--rate", "5"],
        ["--model", "m", "--concurrency", "4", "--max-concurrency", "2"],
        ["--model", "m", "--rate", "5,0"],
        ["--model", "m", "--concurrency", ","],
    ],
)
def test_bad_load_arguments(argv):
    with pytest.raises(SystemExit) as exit:
        cli.parse_args(argv)
    assert exit.value.code == 2
