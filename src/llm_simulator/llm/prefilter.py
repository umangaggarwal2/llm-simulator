"""Literal pre-filtering for regex rules.

A regex can often match only text that contains certain plain strings: the
"hamlet" in (?=.*\\bhamlet\\b). Finding those strings once, from the regex's
parse tree, lets the matcher skip the regex after a substring test, which is
fifty to a hundred times cheaper than a search over a long prompt.

The analysis is conservative: a string is kept only where every match must
contain it, and anything not understood requires nothing. A rule may be
tried when it cannot match, but is never skipped when it could.
"""

import re
from typing import FrozenSet, List, Optional, Tuple

try:  # Python 3.11+
    from re import _constants as _sre_constants
    from re import _parser as _sre_parse
except ImportError:  # Python 3.10
    import sre_constants as _sre_constants
    import sre_parse as _sre_parse

_LITERAL = _sre_constants.LITERAL
_SUBPATTERN = _sre_constants.SUBPATTERN
_ASSERT = _sre_constants.ASSERT
_BRANCH = _sre_constants.BRANCH
_REPEATS = (_sre_constants.MAX_REPEAT, _sre_constants.MIN_REPEAT)
_LOOKAHEAD = 1

# Shorter strings occur in most text, so they filter nothing.
MIN_LENGTH = 3

# Every group must be present; a group is present when any of its strings is.
Requirements = Tuple[FrozenSet[str], ...]

# The non-ASCII characters that re.IGNORECASE matches to ASCII ones. Two of
# them casefold to something else ("ı" stays "ı", "İ" becomes "i̇"), so a
# regex can match "lıne" while the casefolded text lacks "line"; text with
# any of them is not filtered. test_prefilter checks the list against every
# code point.
CASE_TRAPS = "İıſK"
_CASE_TRAP = re.compile(f"[{CASE_TRAPS}]")


def requirements(pattern: re.Pattern) -> Requirements:
    """Lowercase ASCII strings a text must contain for `pattern` to match it.

    Empty when nothing is known. Valid only for text that filterable()
    accepts, tested against its casefolded form.
    """
    try:
        return tuple(_sequence(_sre_parse.parse(pattern.pattern, pattern.flags)))
    except Exception:  # an internal module; if it changes, filter nothing
        return ()


def filterable(text: str) -> bool:
    """Whether substring tests on text.casefold() are sound for this text."""
    return text.isascii() or _CASE_TRAP.search(text) is None


def satisfied(requirements: Requirements, present: FrozenSet[str]) -> bool:
    return all(not group.isdisjoint(present) for group in requirements)


def _sequence(items) -> List[FrozenSet[str]]:
    """Requirements of items matched one after another: all of them."""
    groups: List[FrozenSet[str]] = []
    run: List[str] = []

    def end_run() -> None:
        if len(run) >= MIN_LENGTH:
            groups.append(frozenset({"".join(run)}))
        run.clear()

    for op, av in items:
        if op is _LITERAL and av < 128:
            run.append(chr(av).lower())
            continue
        end_run()
        if op is _SUBPATTERN:
            groups += _sequence(av[-1])
        elif op is _ASSERT and av[0] == _LOOKAHEAD:
            groups += _sequence(av[1])
        elif op in _REPEATS and av[0] >= 1:
            groups += _sequence(av[2])
        elif op is _BRANCH:
            group = _any_of(av[1])
            if group:
                groups.append(group)
        # Anything else requires nothing: \b, ^, ., [...], \w, optional
        # repeats, negative lookarounds, backreferences, and any opcode a
        # later Python adds.
    end_run()
    return groups


def _any_of(branches) -> Optional[FrozenSet[str]]:
    """One branch must match: one group from each, joined; None when some
    branch requires nothing."""
    strings = set()
    for branch in branches:
        groups = _sequence(branch)
        if not groups:
            return None
        # The most selective group: the one whose shortest string is longest.
        strings |= max(groups, key=lambda group: min(map(len, group)))
    return frozenset(strings)
