"""A real simulator for tests that talk HTTP to it.

serve runs it in a background thread of the test process: quick, and enough
for checking behaviour. serve_process runs it in a process of its own, for
tests that time the client precisely: in one process the server and the
client share the GIL, so a busy server thread delays the client's event loop
by several milliseconds.
"""

import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Union

import pytest
import uvicorn
import yaml

from llm_simulator.client.api import create_app
from llm_simulator.settings import Settings


def _write_configs(config_dir: Path, models: dict, responses: Union[dict, Path]):
    models_path = config_dir / "models.yml"
    models_path.write_text(yaml.safe_dump(models))
    if isinstance(responses, Path):
        return models_path, responses
    responses_path = config_dir / "responses.yml"
    responses_path.write_text(yaml.safe_dump(responses))
    return models_path, responses_path


@contextmanager
def _serve(
    config_dir: Path, models: dict, responses: Union[dict, Path]
) -> Iterator[str]:
    """Run the simulator until the block ends; yields its root URL.

    responses is a config as a dict, or the path of an existing file.
    """
    models_path, responses_path = _write_configs(config_dir, models, responses)
    app = create_app(Settings(models_path=models_path, responses_path=responses_path))

    # Bind the socket first, so the port is known and cannot be taken by
    # anything else before the server starts.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]

    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]})
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("simulator did not start")
            time.sleep(0.01)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()


@pytest.fixture(scope="session")
def serve():
    """serve(config_dir, models, responses) runs a simulator as a context manager."""
    return _serve


@contextmanager
def _serve_process(
    config_dir: Path, models: dict, responses: Union[dict, Path]
) -> Iterator[str]:
    """Run the simulator in its own process until the block ends; yields its root URL."""
    models_path, responses_path = _write_configs(config_dir, models, responses)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    # Freed for the server to bind; nothing else on the machine is expected
    # to take it in the moment between.
    sock.close()

    env = {
        **os.environ,
        "LLM_SIM_MODELS": str(models_path),
        "LLM_SIM_RESPONSES": str(responses_path),
    }
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "llm_simulator.client.api:create_app",
        "--factory",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    server = subprocess.Popen(command, env=env)
    root = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 20
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"simulator exited with {server.returncode}")
            try:
                urllib.request.urlopen(f"{root}/health", timeout=1).close()
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError("simulator did not start")
                time.sleep(0.05)
        yield root
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


@pytest.fixture(scope="session")
def serve_process():
    """serve_process(config_dir, models, responses) runs a simulator in its own process."""
    return _serve_process
