"""Turns a chat request into the tokens to send. Pure and deterministic; no timing."""

import hashlib
from itertools import cycle, islice
from typing import List, Mapping, Optional, Sequence, Tuple

from llm_simulator.errors import ContextLengthExceeded
from llm_simulator.llm.matcher import Matcher
from llm_simulator.llm.responses import Synthetic, TemplatePart
from llm_simulator.llm.tokenizer import Tokenizer
from llm_simulator.types import GenerationPlan, GenerationRequest, Message

# Filler for synthetic responses. It ends with a space so that, where the
# tokens wrap around, the last sentence and the first stay separated.
SYNTHETIC_CORPUS = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod "
    "tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, "
    "quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo "
    "consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse "
    "cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non "
    "proident, sunt in culpa qui officia deserunt mollit anim id est laborum. "
)


class Generator:
    """Picks and tokenizes the response for a request, for one model.

    The same request with the same seed always produces the same plan, across
    processes and restarts.
    """

    def __init__(
        self,
        matcher: Matcher,
        tokenizer: Tokenizer,
        context_length: int,
        seed: int = 0,
    ):
        if context_length < 1:
            raise ValueError(f"context_length must be at least 1, got {context_length}")
        self._matcher = matcher
        self._tokenizer = tokenizer
        self._context_length = context_length
        self._seed = seed
        # Tokenized once; synthetic responses repeat these tokens to the exact length.
        self._corpus = tuple(tokenizer.encode(SYNTHETIC_CORPUS))

    def generate(self, request: GenerationRequest) -> GenerationPlan:
        raw = _last_user_message(request.messages)
        # Matching and selection ignore surrounding whitespace, so "hello\n" and
        # "hello" get the same answer; {last_user_message} keeps the raw text.
        text = raw.strip()

        prompt_tokens = self._tokenizer.count_messages(request.messages)
        # Checked before matching, so rejected requests cost nothing more.
        limit = self._completion_limit(prompt_tokens, request.max_tokens)

        result = self._matcher.match(text)
        seed = self._seed if request.seed is None else request.seed
        digest = _digest(seed, result.entry_id, text)

        choices = result.body.choices
        index = _hash_int(digest, 0) % len(choices)
        variant = choices[index]

        if variant.synthetic is not None:
            length = _synthetic_length(variant.synthetic, _hash_int(digest, 8))
            # Build only what will be sent; the full length still decides
            # whether the response was cut off.
            tokens = list(islice(cycle(self._corpus), min(length, limit)))
            truncated = length > limit
        else:
            variables = {
                **result.groups,
                "last_user_message": raw,
                "model": request.model,
            }
            tokens = self._tokenizer.encode(_render(variant.parts, variables))
            truncated = len(tokens) > limit
            tokens = tokens[:limit]

        return GenerationPlan(
            prompt_tokens=prompt_tokens,
            tokens=tuple(tokens),
            finish_reason="length" if truncated else "stop",
            entry_id=result.entry_id,
            variant=index,
        )

    def _completion_limit(self, prompt_tokens: int, max_tokens: Optional[int]) -> int:
        """How many tokens the completion may use, following OpenAI's rules."""
        if max_tokens is None:
            # Without max_tokens, the completion may use whatever context is left.
            if prompt_tokens >= self._context_length:
                raise ContextLengthExceeded(prompt_tokens, None, self._context_length)
            return self._context_length - prompt_tokens
        if prompt_tokens + max_tokens > self._context_length:
            raise ContextLengthExceeded(prompt_tokens, max_tokens, self._context_length)
        return max_tokens


def _last_user_message(messages: Sequence[Message]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return ""


def _digest(seed: int, entry_id: str, text: str) -> bytes:
    # NUL separators keep the fields from running into each other.
    return hashlib.sha256(f"{seed}\0{entry_id}\0{text}".encode("utf-8")).digest()


def _hash_int(digest: bytes, offset: int) -> int:
    # Variant selection uses bytes 0-7 and synthetic length bytes 8-15, so the
    # two choices are independent of each other.
    return int.from_bytes(digest[offset : offset + 8], "big")


def _synthetic_length(synthetic: Synthetic, h: int) -> int:
    span = synthetic.max_tokens - synthetic.min_tokens + 1
    return synthetic.min_tokens + h % span


def _render(parts: Tuple[TemplatePart, ...], variables: Mapping[str, str]) -> str:
    # Values are inserted as-is and never parsed, so user text containing
    # braces such as "{model}" comes through literally.
    pieces: List[str] = []
    for literal, name in parts:
        pieces.append(literal)
        if name is not None:
            pieces.append(variables[name])
    return "".join(pieces)
