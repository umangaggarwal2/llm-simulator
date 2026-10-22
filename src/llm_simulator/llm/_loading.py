"""Loading YAML config files and reporting their validation errors readably."""

from pathlib import Path
from typing import Type, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

M = TypeVar("M", bound=BaseModel)


def load_yaml_model(
    path: Path,
    schema: Type[M],
    error: Type[Exception],
    list_key: str,
    label_key: str,
) -> M:
    """Read a YAML file and validate it against schema.

    Every problem is raised as error, with a message that lists all validation
    errors at once. Locations inside the list under list_key are labelled with
    each item's label_key value, e.g. "responses[3] (hamlet-themes).match".
    """
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise error(f"{path}: cannot read file: {exc.strerror}") from exc
    except yaml.YAMLError as exc:
        raise error(f"{path}: invalid YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise error(f"{path}: expected a mapping at the top level")

    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        # The formatted message lists every error; pydantic's own trace adds nothing.
        raise error(_format_errors(path, data, exc, list_key, label_key)) from None


def _format_errors(
    path: Path, data: dict, exc: ValidationError, list_key: str, label_key: str
) -> str:
    lines = [f"{path}: {exc.error_count()} error(s)"]
    for error in exc.errors():
        message = error["msg"].removeprefix("Value error, ")
        location = _format_loc(error["loc"], data, list_key, label_key)
        lines.append(f"  {location}: {message}")
    return "\n".join(lines)


def _format_loc(loc: tuple, data: dict, list_key: str, label_key: str) -> str:
    # ("responses", 3, "match", "regex", 0) -> "responses[3] (hamlet-themes).match.regex[0]"
    if not loc:
        return "(top level)"

    text = ""
    for part in loc:
        if isinstance(part, int):
            text += f"[{part}]"
        else:
            text += f".{part}" if text else str(part)

    if loc[0] == list_key and len(loc) > 1 and isinstance(loc[1], int):
        items = data.get(list_key)
        item = (
            items[loc[1]] if isinstance(items, list) and loc[1] < len(items) else None
        )
        if isinstance(item, dict) and isinstance(item.get(label_key), str):
            prefix = f"{list_key}[{loc[1]}]"
            text = f"{prefix} ({item[label_key]}){text[len(prefix):]}"
    return text
