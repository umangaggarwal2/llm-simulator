"""Schema, loader and validation for responses.yml. See docs/schema.md."""

import re
from pathlib import Path
from string import Formatter
from typing import Annotated, Literal, Optional, Tuple, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StrictInt,
    field_validator,
    model_validator,
)

from llm_simulator.llm._loading import load_yaml_model

# Variables every template can use; regex entries add their named groups.
BUILTIN_VARIABLES = frozenset({"last_user_message", "model"})

MATCH_OPERATORS = ("exact", "contains", "starts_with", "regex")
BODY_SOURCES = ("content", "variants", "synthetic")
VARIANT_SOURCES = ("content", "synthetic")

ID_PATTERN = r"^[a-z0-9][a-z0-9_-]*$"
RESERVED_IDS = frozenset({"default"})

TokenCount = Annotated[StrictInt, Field(ge=1)]

# A parsed template is a sequence of (literal text, placeholder name or None).
TemplatePart = Tuple[str, Optional[str]]


class ResponsesConfigError(ValueError):
    """Raised when responses.yml cannot be read or does not match the schema."""


class _Model(BaseModel):
    # Reject unknown keys so typos fail at startup instead of being ignored.
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- Bodies ---


class Synthetic(_Model):
    # A fixed count is stored as (n, n), so callers always see a range.
    tokens: Tuple[TokenCount, TokenCount]

    @field_validator("tokens", mode="before")
    @classmethod
    def _expand_fixed_count(cls, tokens):
        return (tokens, tokens) if isinstance(tokens, int) else tokens

    @field_validator("tokens")
    @classmethod
    def _check_range(cls, tokens):
        if tokens[0] > tokens[1]:
            raise ValueError(f"range min ({tokens[0]}) is greater than max ({tokens[1]})")
        return tokens

    @property
    def min_tokens(self) -> int:
        return self.tokens[0]

    @property
    def max_tokens(self) -> int:
        return self.tokens[1]


class Variant(_Model):
    content: Optional[str] = None
    synthetic: Optional[Synthetic] = None

    # The content template, parsed once at load for both validation and rendering.
    _parts: Tuple[TemplatePart, ...] = PrivateAttr(default=())

    @model_validator(mode="before")
    @classmethod
    def _expand_shorthand(cls, data):
        # A bare string is shorthand for {content: <string>}.
        return {"content": data} if isinstance(data, str) else data

    @model_validator(mode="after")
    def _check_source(self):
        _require_exactly_one(self, VARIANT_SOURCES)
        if self.content is not None:
            self._parts = parse_template(self.content)
        return self

    @property
    def parts(self) -> Tuple[TemplatePart, ...]:
        """The parsed content template; empty for synthetic variants."""
        return self._parts

    @property
    def fields(self) -> frozenset:
        """Placeholder names the content template uses."""
        return frozenset(name for _, name in self._parts if name is not None)


class Body(_Model):
    content: Optional[str] = None
    variants: Optional[Tuple[Variant, ...]] = None
    synthetic: Optional[Synthetic] = None

    _choices: Tuple[Variant, ...] = PrivateAttr(default=())

    @field_validator("variants")
    @classmethod
    def _check_not_empty(cls, variants):
        # Checked here rather than with min_length, which also fires when every
        # variant is invalid and buries the real errors.
        if variants is not None and not variants:
            raise ValueError("must not be an empty list")
        return variants

    @model_validator(mode="after")
    def _check_source(self):
        _require_exactly_one(self, BODY_SOURCES)
        if self.variants is not None:
            self._choices = self.variants
        else:
            self._choices = (self._single_choice(),)
        return self

    def _single_choice(self) -> Variant:
        # content and synthetic were already validated as Body fields, so build
        # the Variant directly instead of validating them a second time.
        variant = Variant.model_construct(content=self.content, synthetic=self.synthetic)
        if self.content is not None:
            try:
                variant._parts = parse_template(self.content)
            except ValueError as exc:
                raise ValueError(f"content: {exc}") from None
        return variant

    @property
    def choices(self) -> Tuple[Variant, ...]:
        """Candidate responses; a body without variants has exactly one."""
        return self._choices


# --- Matching ---


class Match(_Model):
    exact: Optional[Tuple[str, ...]] = None
    contains: Optional[Tuple[str, ...]] = None
    starts_with: Optional[Tuple[str, ...]] = None
    regex: Optional[Tuple[str, ...]] = None

    _patterns: Tuple[re.Pattern, ...] = PrivateAttr(default=())

    @field_validator(*MATCH_OPERATORS, mode="before")
    @classmethod
    def _expand_single_value(cls, value):
        return (value,) if isinstance(value, str) else value

    @field_validator(*MATCH_OPERATORS)
    @classmethod
    def _check_values(cls, values):
        if values is None:
            return values
        if not values:
            raise ValueError("must not be an empty list")
        if any(not value for value in values):
            raise ValueError("must not contain empty strings")
        return values

    @model_validator(mode="after")
    def _check_operator(self):
        _require_exactly_one(self, MATCH_OPERATORS)
        if self.regex is not None:
            self._patterns = tuple(_compile(pattern) for pattern in self.regex)
        return self

    @property
    def operator(self) -> str:
        return next(name for name in MATCH_OPERATORS if getattr(self, name) is not None)

    @property
    def values(self) -> Tuple[str, ...]:
        return getattr(self, self.operator)

    @property
    def patterns(self) -> Tuple[re.Pattern, ...]:
        """Compiled case-insensitive regexes; empty unless the operator is regex."""
        return self._patterns

    @property
    def group_names(self) -> frozenset:
        """Named groups defined by every pattern, so always safe to use in templates."""
        if not self._patterns:
            return frozenset()
        return frozenset.intersection(*(frozenset(p.groupindex) for p in self._patterns))


# --- Top level ---


class Entry(Body):
    id: str = Field(pattern=ID_PATTERN)
    match: Match

    @field_validator("id")
    @classmethod
    def _check_reserved(cls, id):
        if id in RESERVED_IDS:
            raise ValueError(f"'{id}' is reserved")
        return id

    @model_validator(mode="after")
    def _check_entry_templates(self):
        clashes = self.match.group_names & BUILTIN_VARIABLES
        if clashes:
            raise ValueError(
                f"regex group name(s) {', '.join(sorted(clashes))} clash with "
                "built-in template variables; rename the group(s)"
            )
        _check_templates(self, BUILTIN_VARIABLES | self.match.group_names)
        return self


class ResponsesConfig(_Model):
    version: Literal[1]
    seed: StrictInt = 0
    responses: Tuple[Entry, ...] = ()
    default: Body

    @field_validator("default")
    @classmethod
    def _check_default_templates(cls, default):
        _check_templates(default, BUILTIN_VARIABLES)
        return default

    @model_validator(mode="after")
    def _check_unique_ids(self):
        seen, duplicates = set(), []
        for entry in self.responses:
            if entry.id in seen and entry.id not in duplicates:
                duplicates.append(entry.id)
            seen.add(entry.id)
        if duplicates:
            raise ValueError(f"duplicate entry ids: {', '.join(duplicates)}")
        return self


# --- Loading ---


def load_responses(path: Union[str, Path]) -> ResponsesConfig:
    """Read and validate a responses file, raising ResponsesConfigError on any problem."""
    return load_yaml_model(
        Path(path), ResponsesConfig, ResponsesConfigError, "responses", "id"
    )


# --- Templates ---


def parse_template(template: str) -> Tuple[TemplatePart, ...]:
    """Split a template into (literal text, placeholder name or None) parts.

    "{{" and "}}" come back as literal braces. Raises ValueError for unbalanced
    braces, and for placeholders that are not plain names (attribute access,
    indexing, conversions or format specs). Rendering joins the parts, so
    templates are never passed to str.format.
    """
    try:
        parsed = list(Formatter().parse(template))
    except ValueError as exc:
        raise ValueError(
            f"malformed template ({exc}); write {{{{ or }}}} for literal braces"
        ) from None

    parts = []
    for literal, name, spec, conversion in parsed:
        if name is not None and (not name.isidentifier() or spec or conversion):
            raw = (
                name
                + (f"!{conversion}" if conversion else "")
                + (f":{spec}" if spec else "")
            )
            raise ValueError(
                f"invalid placeholder {{{raw}}}; use a plain name such as {{model}}"
            )
        parts.append((literal, name))
    return tuple(parts)


# --- Helpers ---


def _require_exactly_one(model: BaseModel, names: Tuple[str, ...]) -> None:
    present = [name for name in names if getattr(model, name) is not None]
    if len(present) != 1:
        expected = ", ".join(f"'{name}'" for name in names)
        got = ", ".join(f"'{name}'" for name in present) or "none"
        raise ValueError(f"expected exactly one of {expected}, got {got}")


def _compile(pattern: str) -> re.Pattern:
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        raise ValueError(f"invalid regex {pattern!r}: {exc}") from None


def _check_templates(body: Body, allowed: frozenset) -> None:
    for index, choice in enumerate(body.choices):
        if choice.content is None:
            continue
        where = f"variants[{index}]" if body.variants is not None else "content"
        unknown = choice.fields - allowed
        if unknown:
            raise ValueError(
                f"{where}: unknown placeholder(s) {', '.join(sorted(unknown))}; "
                f"available: {', '.join(sorted(allowed))}"
            )
