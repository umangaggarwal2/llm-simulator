"""The llmsimulator command, run as a real process with its flags."""

import json
import socket
import subprocess
import sys
import time
import urllib.request

import pytest
import yaml

pytestmark = pytest.mark.integration

COMMAND = [sys.executable, "-m", "llm_simulator.client.api"]


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_flags_configure_the_server(tmp_path):
    models = tmp_path / "models.yml"
    responses = tmp_path / "responses.yml"
    models.write_text(
        yaml.safe_dump({"version": 1, "models": [{"name": "m", "context_length": 100}]})
    )
    responses.write_text(
        yaml.safe_dump({"version": 1, "default": {"content": "From the flag file."}})
    )
    port = free_port()
    server = subprocess.Popen(
        COMMAND
        + ["--host", "127.0.0.1", "--port", str(port)]
        + ["--models", str(models), "--responses", str(responses)],
        # Run from elsewhere: nothing may depend on the working directory.
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 20
        while True:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                break
            except OSError:
                if server.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("the server did not start")
                time.sleep(0.05)
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(
                {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        body = json.load(urllib.request.urlopen(request, timeout=5))
        assert body["choices"][0]["message"]["content"] == "From the flag file."
    finally:
        server.terminate()
        server.wait(timeout=10)


def test_a_missing_config_exits_with_a_message(tmp_path):
    result = subprocess.run(
        COMMAND + ["--models", str(tmp_path / "missing.yml")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 1
    assert result.stderr.startswith("llmsimulator: error: ")
    assert "Traceback" not in result.stderr
