import re

import pytest

from llm_simulator.llm import prefilter
from llm_simulator.llm.prefilter import (
    CASE_TRAPS,
    filterable,
    requirements,
    satisfied,
)


def required(pattern):
    return requirements(re.compile(pattern, re.IGNORECASE))


@pytest.mark.parametrize(
    "pattern, expected",
    [
        ("hamlet", ({"hamlet"},)),
        ("HAMLET", ({"hamlet"},)),
        (r"\bhamlet\b", ({"hamlet"},)),
        # An optional or repeated tail ends the literal run before it.
        ("quotes?", ({"quote"},)),
        (r"symbol\w*", ({"symbol"},)),
        ("moby[- ]?dick", ({"moby"}, {"dick"})),
        (r"mr\.? hyde", ({" hyde"},)),
        # Lookaheads must all match: their requirements add up.
        (
            r"(?s)^(?=.*\bhamlet\b)(?=.*\b(quotes?|lines?)\b)",
            ({"hamlet"}, {"quote", "line"}),
        ),
        # One branch of an alternation must match: any of their strings.
        ("(cat|dog)", ({"cat", "dog"},)),
        ("(?:king lear|\\blear\\b)", ({"king lear", "lear"},)),
        # A branch that requires nothing makes the alternation require nothing.
        (r"(cat|\d+)", ()),
        # Repeats that must occur at least once keep their requirements.
        ("(?:abc)+", ({"abc"},)),
        ("(?:abc)*", ()),
        # Non-ASCII characters end a run; flags only weaken the test.
        ("bront[eë]", ({"bront"},)),
        ("café", ({"caf"},)),
        ("(?-i:ABC)", ({"abc"},)),
        # Too short, or nothing literal.
        ("ab", ()),
        (".*", ()),
        (r"\w+ing", ({"ing"},)),
        # Negative lookarounds and backreferences require nothing.
        ("(?!foo)bar", ({"bar"},)),
        (r"(abc)\1", ({"abc"},)),
    ],
)
def test_requirements(pattern, expected):
    assert required(pattern) == tuple(frozenset(group) for group in expected)


def test_alternation_takes_the_most_selective_group_of_each_branch():
    # A branch with one run keeps it whole; one that needs both "abc" and
    # "longer" contributes only "longer", the rarer of the two.
    assert required("(?:abc longer|xyz longest)") == (
        frozenset({"abc longer", "xyz longest"}),
    )
    assert required(r"(?:abc\s+longer|xyz\s+longest)") == (
        frozenset({"longer", "longest"}),
    )


def test_unknown_parse_requires_nothing(monkeypatch):
    def broken(*args):
        raise RuntimeError("parser changed")

    monkeypatch.setattr(prefilter._sre_parse, "parse", broken)
    assert required("hamlet") == ()


@pytest.mark.parametrize(
    "present, expected",
    [
        ({"hamlet", "quote"}, True),
        ({"hamlet", "line"}, True),
        ({"hamlet"}, False),
        ({"quote"}, False),
        (set(), False),
    ],
)
def test_satisfied(present, expected):
    needs = (frozenset({"hamlet"}), frozenset({"quote", "line"}))
    assert satisfied(needs, frozenset(present)) is expected


def test_nothing_required_is_always_satisfied():
    assert satisfied((), frozenset())


@pytest.mark.parametrize(
    "text, expected",
    [
        ("plain ascii", True),
        ("who’s the hero? 😀 café", True),
        *((f"line {trap}", False) for trap in CASE_TRAPS),
    ],
)
def test_filterable(text, expected):
    assert filterable(text) is expected


def test_case_traps_lists_every_non_ascii_match_of_an_ascii_character():
    # Anything re.IGNORECASE matches to ASCII but casefold may not turn into
    # it must be in CASE_TRAPS, or the filter could skip a matching rule.
    # Recomputed over every code point, so a Python upgrade that adds one
    # fails here.
    ascii_ = re.compile(r"[\x00-\x7f]", re.IGNORECASE)
    found = {
        chr(code)
        for code in range(0x80, 0x110000)
        if not 0xD800 <= code < 0xE000 and ascii_.fullmatch(chr(code))
    }
    assert found <= set(CASE_TRAPS)
