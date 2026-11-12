"""The built package carries everything the commands need at run time."""

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from llm_simulator import defaults

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_the_wheel_contains_the_builtin_config_files(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--quiet",
            "-w",
            str(tmp_path),
            str(REPO_ROOT),
        ],
        check=True,
        timeout=300,
    )
    [wheel] = tmp_path.glob("llm_simulator-*.whl")
    names = set(zipfile.ZipFile(wheel).namelist())
    assert "llm_simulator/defaults/__init__.py" in names
    for name in defaults.NAMES:
        assert f"llm_simulator/defaults/{name}.yml" in names
