"""What each request sends.

A Workload maps a request index to a RequestSpec. One run uses exactly one
source: a fixed prompt, synthetic prompts of a given length, or a JSONL file.
The runner only deals in indexes, and the client sends whatever spec it is
given, so a new source never touches either.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

from llm_simulator.llm.tokenizer import RegexTokenizer, Tokenizer
from llm_simulator.types import Message

# Letters-only words: one token each for the simulator's GPT-2-style
# tokenizer (with a leading space after the first), and mostly one token for
# real tokenizers too, so a prompt of k words is k content tokens.
_WORD_LIST = (
    "time person year way day thing man world life hand part child eye woman "
    "place work week case point number group problem fact story house river "
    "garden window letter music paper money table light water night city "
    "road field voice heart table market season answer question idea reason "
    "moment family friend country system program report course market price "
    "simple quiet bright early green heavy little modern happy strong clear "
    "open close carry follow begin bring build change consider continue "
    "describe develop explain finish happen improve include learn listen "
    "measure notice offer prepare produce provide reach remember return "
    "suggest travel wonder write"
)
WORDS = tuple(dict.fromkeys(_WORD_LIST.split()))


@dataclass(frozen=True)
class RequestSpec:
    prompt: str
    max_tokens: int


class Workload:
    """What request i sends. Built before the run; request() only looks it up
    or builds a prompt cheaply, outside any measured interval."""

    def __init__(self, description: str, make: Callable[[int], RequestSpec]):
        self.description = description
        self._make = make

    def request(self, index: int) -> RequestSpec:
        return self._make(index)


def fixed(prompt: str, max_tokens: int, prefix: str = "") -> Workload:
    """The same prompt for every request."""
    spec = RequestSpec(prefix + prompt, max_tokens)
    return Workload(
        _describe("fixed prompt", f"{max_tokens} tokens", prefix), lambda _: spec
    )


def synthetic(
    input_tokens: int,
    max_tokens: int,
    *,
    prefix: str = "",
    seed: int = 0,
    tokenizer: Tokenizer = RegexTokenizer(),
) -> Workload:
    """A different prompt per request, each exactly input_tokens prompt tokens.

    input_tokens is what usage.prompt_tokens reports on the simulator: the
    prefix, the words and the chat overhead together. Request i's words are
    drawn from random.Random(f"{seed}:{i}"), so runs are repeatable; distinct
    prompts keep a real server's prefix cache from flattering TTFT.
    """
    # Tokens with one word; every further word adds exactly one.
    base = tokenizer.count_messages([Message("user", prefix + WORDS[0])])
    if input_tokens < base:
        raise ValueError(
            f"input_tokens {input_tokens} is less than {base}: the chat overhead"
            + (" and the prefix" if prefix else "")
            + " plus one word"
        )
    words = input_tokens - base + 1

    def make(index: int) -> RequestSpec:
        rng = random.Random(f"{seed}:{index}")
        return RequestSpec(prefix + " ".join(rng.choices(WORDS, k=words)), max_tokens)

    return Workload(
        _describe(
            f"synthetic {input_tokens} input tokens", f"{max_tokens} tokens", prefix
        ),
        make,
    )


def from_jsonl(path: Path, max_tokens: int, prefix: str = "") -> Workload:
    """Prompts from a JSONL file, cycled when there are fewer than requests.

    Each non-blank line is {"prompt": str, "max_tokens": int (optional,
    default max_tokens)}. The whole file is checked before the run starts.
    """
    specs: List[RequestSpec] = []
    with open(path, encoding="utf-8") as lines:
        for number, line in enumerate(lines, start=1):
            if line.strip():
                specs.append(_parse_line(line, max_tokens, prefix, f"{path}:{number}"))
    if not specs:
        raise ValueError(f"{path}: no prompts")
    return Workload(
        _describe(
            f"dataset {Path(path).name} ({len(specs)} prompts)",
            f"{max_tokens} tokens unless the line sets max_tokens",
            prefix,
        ),
        lambda index: specs[index % len(specs)],
    )


def _parse_line(line: str, max_tokens: int, prefix: str, where: str) -> RequestSpec:
    try:
        entry = json.loads(line)
    except ValueError as exc:
        raise ValueError(f"{where}: invalid JSON: {exc}") from None
    if not isinstance(entry, dict):
        raise ValueError(f"{where}: expected an object")
    prompt = entry.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"{where}: missing 'prompt'")
    tokens = entry.get("max_tokens", max_tokens)
    # bool is an int in Python; true is not a token count.
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 1:
        raise ValueError(f"{where}: 'max_tokens' must be a positive integer")
    return RequestSpec(prefix + prompt, tokens)


def _describe(source: str, output: str, prefix: str) -> str:
    text = f"{source} → {output}"
    return text + f" · prefix {prefix!r}" if prefix else text
