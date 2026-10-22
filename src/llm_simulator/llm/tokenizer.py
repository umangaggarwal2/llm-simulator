"""Tokenizers: split text into the token pieces the simulator counts and streams."""

import re
from abc import ABC, abstractmethod
from typing import Iterable, List, Protocol


class Message(Protocol):
    role: str
    content: str


class Tokenizer(ABC):
    """Splits text into token pieces.

    Implementations must be lossless ("".join(encode(text)) == text), so
    streamed chunks rebuild the exact response, and deterministic, so plans and
    simulated metrics are reproducible across runs.
    """

    name: str

    # Chat-format overhead, following OpenAI's counting recipe: every message is
    # wrapped in template tokens, and the reply is primed with a few more.
    tokens_per_message = 3
    tokens_per_reply = 3

    @abstractmethod
    def encode(self, text: str) -> List[str]:
        """Split text into token pieces."""

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def count_messages(self, messages: Iterable[Message]) -> int:
        """Prompt tokens for a chat request, including chat-format overhead."""
        total = self.tokens_per_reply
        for message in messages:
            total += (
                self.tokens_per_message
                + self.count(message.role)
                + self.count(message.content)
            )
        return total


# Scripts BPE tokenizes at roughly one token per character. Without this, a run
# of them has no spaces to split on and would count as a single word.
_CJK = (
    "぀-ヿ"  # hiragana and katakana
    "㐀-䶿"  # CJK unified ideographs extension A
    "一-鿿"  # CJK unified ideographs
    "가-힯"  # Hangul syllables
    "豈-﫿"  # CJK compatibility ideographs
    "ｦ-ﾟ"  # half-width katakana
    "\U00020000-\U0002fa1f"  # CJK unified ideographs extensions B-F and supplement
)


class RegexTokenizer(Tokenizer):
    """Approximates GPT-style BPE pre-tokenization with a single regex.

    Splits words, punctuation, contractions and numbers the way GPT-2's
    pre-tokenizer does, with a leading space attached to the following token.
    CJK, kana and Hangul become one token per character, close to what BPE
    produces for them. Long or rare words stay whole instead of being split into
    sub-words, so counts run somewhat lower than a real BPE tokenizer's.
    """

    name = "regex"

    _PATTERN = re.compile(
        r"""
        '(?:[sdmt]|ll|ve|re)            # contractions: 's 'd 'm 't 'll 've 're
        | [ ]?[%(cjk)s]                 # CJK, kana and Hangul, one character per token
        | [ ]?(?:(?![%(cjk)s])[^\W\d_])+  # words (other letters), with an optional leading space
        | [ ]?\d{1,3}                   # numbers, in groups of up to three digits
        | [ ]?[^\s\w]+                  # runs of punctuation and symbols
        | \s+(?!\S)                     # whitespace, leaving the last space to start the next token
        | \s+                           # whitespace left over, such as a newline before a word
        | [\s\S]                        # anything else (e.g. underscores), so encoding is lossless
        """ % {"cjk": _CJK},
        re.IGNORECASE | re.VERBOSE,
    )

    def encode(self, text: str) -> List[str]:
        return self._PATTERN.findall(text)

    def count(self, text: str) -> int:
        return sum(1 for _ in self._PATTERN.finditer(text))


_TOKENIZERS = {RegexTokenizer.name: RegexTokenizer}


def get_tokenizer(name: str) -> Tokenizer:
    """Build a tokenizer by its config name, e.g. "regex"."""
    try:
        return _TOKENIZERS[name]()
    except KeyError:
        available = ", ".join(sorted(_TOKENIZERS))
        raise ValueError(
            f"unknown tokenizer '{name}'; available: {available}"
        ) from None
