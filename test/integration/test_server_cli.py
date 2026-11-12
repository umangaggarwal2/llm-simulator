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


class Server:
    """The llmsimulator command running as a process, until stopped."""

    def __init__(self, args, cwd):
        self.port = free_port()
        self.process = subprocess.Popen(
            COMMAND + ["--host", "127.0.0.1", "--port", str(self.port)] + args,
            # Run from elsewhere: nothing may depend on the working directory.
            cwd=cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 20
        while True:
            try:
                urllib.request.urlopen(self.url("/health"), timeout=1)
                return
            except OSError:
                if self.process.poll() is not None or time.monotonic() > deadline:
                    self.stop()
                    raise RuntimeError("the server did not start")
                time.sleep(0.05)

    def url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def get(self, path):
        return json.load(urllib.request.urlopen(self.url(path), timeout=5))

    def chat(self, model, text, **extra):
        body = {"model": model, "messages": [{"role": "user", "content": text}]}
        request = urllib.request.Request(
            self.url("/v1/chat/completions"),
            data=json.dumps({**body, **extra}).encode(),
            headers={"Content-Type": "application/json"},
        )
        return json.load(urllib.request.urlopen(request, timeout=5))

    def stop(self):
        self.process.terminate()
        self.process.wait(timeout=10)


def test_flags_configure_the_server(tmp_path):
    models = tmp_path / "models.yml"
    responses = tmp_path / "responses.yml"
    models.write_text(
        yaml.safe_dump({"version": 1, "models": [{"name": "m", "context_length": 100}]})
    )
    responses.write_text(
        yaml.safe_dump({"version": 1, "default": {"content": "From the flag file."}})
    )
    server = Server(["--models", str(models), "--responses", str(responses)], tmp_path)
    try:
        body = server.chat("m", "hi")
        assert body["choices"][0]["message"]["content"] == "From the flag file."
    finally:
        server.stop()


def test_with_no_configuration_it_serves_the_builtin_files(tmp_path):
    # An empty directory and no files given, as after `pip install`.
    server = Server([], tmp_path)
    try:
        models = {model["id"] for model in server.get("/v1/models")["data"]}
        assert models == {"qwen-mock", "qwen-mock-batched", "tiny-mock"}
        body = server.chat("tiny-mock", "Tell me about Hamlet")
        assert "Hamlet" in body["choices"][0]["message"]["content"]
    finally:
        server.stop()


def test_builtin_names_select_shipped_files(tmp_path):
    server = Server(["--responses", "builtin:bench-responses"], tmp_path)
    try:
        # Benchmark responses are synthetic, so max_tokens sets the length.
        body = server.chat("tiny-mock", "anything", max_tokens=7)
        assert body["usage"]["completion_tokens"] == 7
        assert body["choices"][0]["finish_reason"] == "length"
    finally:
        server.stop()


def test_print_config_tolerates_a_reader_that_stops_early(tmp_path):
    # As in `llmsimulator --print-config responses | head`: the reader closes
    # the pipe long before the 265 KB file is written.
    process = subprocess.Popen(
        COMMAND + ["--print-config", "responses"],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout.read(100).startswith(b"# Sample responses")
    process.stdout.close()
    error = process.stderr.read().decode()
    assert process.wait(timeout=60) == 0
    assert "Traceback" not in error and "BrokenPipeError" not in error


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
