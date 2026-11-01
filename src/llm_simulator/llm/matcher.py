"""Request-time matching of user input against responses.yml entries."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, FrozenSet, Mapping, Optional, Tuple

from llm_simulator.llm.prefilter import (
    Requirements,
    filterable,
    requirements,
    satisfied,
)
from llm_simulator.llm.responses import Body, Match, ResponsesConfig

DEFAULT_ID = "default"

# Shared, read-only "matched with no captures" result for non-regex operators.
_NO_GROUPS: Mapping[str, str] = MappingProxyType({})

# A test receives the input and its casefolded form, and returns the captured
# template variables when it matches, or None when it doesn't.
Test = Callable[[str, str], Optional[Mapping[str, str]]]


@dataclass(frozen=True)
class MatchResult:
    entry_id: str
    body: Body
    groups: Mapping[str, str]


@dataclass(frozen=True)
class _Rule:
    entry_id: str
    body: Body
    test: Test
    # Each pattern's literal requirements (see prefilter); None when the rule
    # is always tested: not a regex, or a pattern with no requirements.
    requires: Optional[Tuple[Requirements, ...]] = None

    def possible(self, present: FrozenSet[str]) -> bool:
        return self.requires is None or any(
            satisfied(pattern, present) for pattern in self.requires
        )


class Matcher:
    """Finds the first responses.yml entry that matches the input.

    Entries are tried in file order and all matching is case-insensitive. Rules
    are prepared once here, so match() does no per-operator work.

    A regex rule is skipped without running when the input lacks a string
    every match must contain; those strings are searched for once per input.
    """

    def __init__(self, config: ResponsesConfig):
        self._rules = tuple(
            _Rule(entry.id, entry, _build_test(entry.match), _requires(entry.match))
            for entry in config.responses
        )
        self._needles = tuple(
            sorted(
                {
                    string
                    for rule in self._rules
                    if rule.requires
                    for pattern in rule.requires
                    for group in pattern
                    for string in group
                }
            )
        )
        self._default = MatchResult(DEFAULT_ID, config.default, _NO_GROUPS)

    def match(self, text: str) -> MatchResult:
        folded = text.casefold()
        present = self._present(text, folded)
        for rule in self._rules:
            if present is not None and not rule.possible(present):
                continue
            groups = rule.test(text, folded)
            if groups is not None:
                return MatchResult(rule.entry_id, rule.body, groups)
        return self._default

    def _present(self, text: str, folded: str) -> Optional[FrozenSet[str]]:
        """The required strings in the input, or None to test every rule."""
        if not self._needles or not filterable(text):
            return None
        return frozenset(needle for needle in self._needles if needle in folded)


def _requires(match: Match) -> Optional[Tuple[Requirements, ...]]:
    if match.operator != "regex":
        return None
    found = tuple(requirements(pattern) for pattern in match.patterns)
    return found if all(found) else None


# --- Operators ---


def _exact(match: Match) -> Test:
    values = frozenset(value.casefold() for value in match.values)
    return lambda text, folded: _NO_GROUPS if folded in values else None


def _contains(match: Match) -> Test:
    values = tuple(value.casefold() for value in match.values)
    return lambda text, folded: (
        _NO_GROUPS if any(value in folded for value in values) else None
    )


def _starts_with(match: Match) -> Test:
    values = tuple(value.casefold() for value in match.values)
    return lambda text, folded: _NO_GROUPS if folded.startswith(values) else None


def _regex(match: Match) -> Test:
    # Patterns are compiled with re.IGNORECASE and run on the original text, so
    # captured groups keep the user's casing.
    patterns = match.patterns

    def test(text: str, folded: str) -> Optional[Mapping[str, str]]:
        for pattern in patterns:
            found = pattern.search(text)
            if found:
                return found.groupdict(default="")
        return None

    return test


_BUILDERS = {
    "exact": _exact,
    "contains": _contains,
    "starts_with": _starts_with,
    "regex": _regex,
}


def _build_test(match: Match) -> Test:
    return _BUILDERS[match.operator](match)
