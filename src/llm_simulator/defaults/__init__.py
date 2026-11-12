"""Configuration files shipped with the package, usable as `builtin:<name>`.

The server falls back to `models` and `responses` when no file is given, so
an installed package runs from any directory. To start from one, copy it with
`llmsimulator --print-config <name>`.
"""

from importlib import resources
from pathlib import Path

PREFIX = "builtin:"
NAMES = ("models", "responses", "bench-responses")


def path(name: str) -> Path:
    """The installed path of a built-in file, e.g. path("models")."""
    if name not in NAMES:
        raise ValueError(
            f"unknown built-in config {name!r}; available: {', '.join(NAMES)}"
        )
    return Path(str(resources.files(__name__) / f"{name}.yml"))


def resolve(value: str) -> Path:
    """A `builtin:<name>` value as its installed path; any other value as a path."""
    if value.startswith(PREFIX):
        return path(value[len(PREFIX) :])
    return Path(value)
