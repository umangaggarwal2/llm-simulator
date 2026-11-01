import dataclasses
import random
import re
from pathlib import Path

import pytest

from llm_simulator.llm.matcher import DEFAULT_ID, Matcher
from llm_simulator.llm.responses import (
    MATCH_OPERATORS,
    ResponsesConfig,
    load_responses,
)
from llm_simulator.llm import matcher as matcher_module

REPO_ROOT = Path(__file__).resolve().parents[2]


def make_matcher(*entries):
    config = ResponsesConfig.model_validate(
        {
            "version": 1,
            "default": {"content": "fallback"},
            "responses": [
                {"id": f"e{i}", "content": f"r{i}", **entry}
                for i, entry in enumerate(entries)
            ],
        }
    )
    return Matcher(config)


@pytest.mark.parametrize(
    "match, text, expected",
    [
        ({"exact": "hello"}, "hello", "e0"),
        ({"exact": "hello"}, "hello there", DEFAULT_ID),
        ({"exact": ["hi", "hello"]}, "hi", "e0"),
        ({"contains": "weather"}, "what's the weather like", "e0"),
        ({"contains": "weather"}, "what's the forecast", DEFAULT_ID),
        ({"contains": ["rain", "snow"]}, "will it snow", "e0"),
        ({"starts_with": "/echo"}, "/echo hi", "e0"),
        ({"starts_with": "/echo"}, "say /echo", DEFAULT_ID),
        ({"starts_with": ["/a", "/b"]}, "/b x", "e0"),
        ({"regex": r"\bmacbeth\b"}, "tell me about Macbeth", "e0"),
        ({"regex": r"\bmacbeth\b"}, "macbethian", DEFAULT_ID),
        ({"regex": ["^x", "y$"]}, "ay", "e0"),
    ],
)
def test_operators(match, text, expected):
    assert make_matcher({"match": match}).match(text).entry_id == expected


@pytest.mark.parametrize(
    "match, text",
    [
        ({"exact": "Hello"}, "HELLO"),
        ({"contains": "WEATHER"}, "the weather"),
        ({"starts_with": "/Echo"}, "/ECHO hi"),
        ({"regex": "Hamlet"}, "HAMLET themes"),
        ({"exact": "straße"}, "STRASSE"),
    ],
)
def test_matching_is_case_insensitive(match, text):
    assert make_matcher({"match": match}).match(text).entry_id == "e0"


def test_first_matching_entry_wins():
    matcher = make_matcher(
        {"match": {"contains": "hamlet"}},
        {"match": {"contains": "hamlet themes"}},
    )
    assert matcher.match("hamlet themes").entry_id == "e0"


def test_later_entry_matches_when_earlier_ones_do_not():
    matcher = make_matcher(
        {"match": {"contains": "macbeth"}},
        {"match": {"contains": "hamlet"}},
    )
    result = matcher.match("hamlet")
    assert result.entry_id == "e1"
    assert result.body.content == "r1"


def test_falls_back_to_default():
    result = make_matcher({"match": {"exact": "x"}}).match("y")
    assert result.entry_id == DEFAULT_ID
    assert result.body.content == "fallback"
    assert dict(result.groups) == {}


def test_empty_input_matches_only_default():
    matcher = make_matcher(
        {"match": {"contains": "a"}},
        {"match": {"starts_with": "b"}},
        {"match": {"regex": "c"}},
    )
    assert matcher.match("").entry_id == DEFAULT_ID


def test_no_entries_always_returns_default():
    assert make_matcher().match("anything").entry_id == DEFAULT_ID


def test_regex_returns_named_groups_with_original_casing():
    matcher = make_matcher({"match": {"regex": r"weather in (?P<city>\w+)"}})
    result = matcher.match("WEATHER IN Paris")
    assert result.entry_id == "e0"
    assert dict(result.groups) == {"city": "Paris"}


def test_regex_unmatched_optional_group_is_empty_string():
    matcher = make_matcher(
        {"match": {"regex": r"go (?P<where>\w+)(?: by (?P<how>\w+))?"}}
    )
    assert dict(matcher.match("go home").groups) == {"where": "home", "how": ""}


def test_regex_groups_come_from_the_pattern_that_matched():
    matcher = make_matcher(
        {"match": {"regex": [r"from (?P<a>\w+)", r"to (?P<a>\w+) (?P<b>\w+)"]}}
    )
    assert dict(matcher.match("to rome today").groups) == {"a": "rome", "b": "today"}


def test_non_regex_operators_return_no_groups():
    assert dict(make_matcher({"match": {"contains": "x"}}).match("x").groups) == {}


def test_every_schema_operator_has_a_builder():
    assert set(matcher_module._BUILDERS) == set(MATCH_OPERATORS)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("What are the themes in Macbeth?", "macbeth-themes"),
        ("hello there", "greeting"),
        ("thanks!", "thanks"),
        ("what is the capital of France", DEFAULT_ID),
    ],
)
def test_real_config(text, expected):
    matcher = Matcher(load_responses(REPO_ROOT / "config" / "responses.yml"))
    assert matcher.match(text).entry_id == expected


# --- Pre-filter ---


def never_called(text, folded):
    raise AssertionError("the pre-filter should have skipped this rule")


def without_running(matcher, index):
    """Make rule `index` fail the test if the matcher runs its regex."""
    rules = list(matcher._rules)
    rules[index] = dataclasses.replace(rules[index], test=never_called)
    matcher._rules = tuple(rules)
    return matcher


def test_rule_missing_a_required_string_is_not_run():
    matcher = without_running(
        make_matcher(
            {"match": {"regex": r"\bhamlet\b.*\bquotes?\b"}},
            {"match": {"regex": r"\bmacbeth\b"}},
        ),
        0,
    )
    assert matcher.match("quotes from Macbeth").entry_id == "e1"


def test_rule_with_its_strings_present_is_run_and_may_still_fail():
    matcher = make_matcher(
        {"match": {"regex": r"\bhamlet\b"}},
        {"match": {"regex": r"hamlet"}},
    )
    # "hamlet" is present, so e0 runs, but its word boundaries fail.
    assert matcher.match("hamlets").entry_id == "e1"


def test_rule_with_several_patterns_runs_when_any_could_match():
    matcher = make_matcher({"match": {"regex": ["hamlet", "macbeth"]}})
    assert matcher.match("macbeth").entry_id == "e0"


def test_rules_without_requirements_are_always_run():
    matcher = make_matcher(
        {"match": {"regex": r"\d{4}"}},
        {"match": {"contains": "year"}},
    )
    assert matcher.match("in 1603").entry_id == "e0"
    assert matcher.match("which year").entry_id == "e1"


def test_non_ascii_text_is_still_filtered():
    matcher = without_running(
        make_matcher(
            {"match": {"regex": r"hamlet"}},
            {"match": {"regex": r"macbeth"}},
        ),
        0,
    )
    assert matcher.match("who’s in Macbeth? 😀").entry_id == "e1"


@pytest.mark.parametrize(
    "pattern, text",
    [
        ("kelvin", "\u212aelvin"),
        (r"\bline\b", "l\u0131ne"),
        (r"\bline\b", "L\u0130NE"),
        ("class", "cla\u017f\u017f"),
    ],
)
def test_case_traps_match_as_the_regex_does(pattern, text):
    # re.IGNORECASE matches these characters to ASCII letters that their
    # casefold lacks, so the pre-filter must step aside for them.
    assert re.search(pattern, text, re.IGNORECASE)
    assert make_matcher({"match": {"regex": pattern}}).match(text).entry_id == "e0"


def unfiltered(matcher, text):
    """The first match the way it was found before the pre-filter."""
    folded = text.casefold()
    return next(
        (r.entry_id for r in matcher._rules if r.test(text, folded) is not None),
        DEFAULT_ID,
    )


def test_prefilter_never_changes_the_answer_on_the_sample_config():
    matcher = Matcher(load_responses(REPO_ROOT / "config" / "responses.yml"))
    rng = random.Random(0)
    needles = list(matcher._needles)
    # Near misses: required strings inside other words, and spellings the
    # regexes may or may not accept.
    near = ["learn", "clear", "hamlets", "quotation", "Quotes", "symbolism", "who's"]
    filler = "the a of and to in is what who tell me about please book story".split()
    traps = ["\u0130", "\u0131", "\u017f", "\u212a", "’", "é", "😀"]

    def word():
        pool = rng.choice((needles, near, filler, filler))
        token = rng.choice(pool)
        if rng.random() < 0.1:
            token = token.upper()
        if rng.random() < 0.05:
            at = rng.randrange(len(token) + 1)
            token = token[:at] + rng.choice(traps) + token[at:]
        return token

    texts = [
        " ".join(word() for _ in range(rng.randint(1, 30))) for _ in range(5000)
    ] + [" ".join(word() for _ in range(500)) for _ in range(20)]
    for text in texts:
        assert matcher.match(text).entry_id == unfiltered(matcher, text), text
