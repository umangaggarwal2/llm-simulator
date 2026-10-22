import random
from dataclasses import dataclass

import pytest

from llm_simulator.llm.tokenizer import RegexTokenizer, Tokenizer, get_tokenizer


@dataclass
class Msg:
    role: str
    content: str


@pytest.fixture
def tokenizer():
    return RegexTokenizer()


EDGE_CASES = [
    "",
    " ",
    "   ",
    "\n\n",
    "\t indented",
    "  leading spaces",
    "trailing spaces   ",
    "line one\n\nline two",
    "snake_case_name",
    "naïve café",
    "東京タワー",
    "emoji 🙂🙂 here",
    "IT'S THEY'LL we've",
    "1234567890",
    "x = f(a_b) + 1;",
    "'quoted' and \"double\"",
]


@pytest.mark.parametrize("text", EDGE_CASES)
def test_encode_is_lossless(tokenizer, text):
    assert "".join(tokenizer.encode(text)) == text


def test_encode_is_lossless_on_random_text(tokenizer):
    rng = random.Random(1234)
    alphabet = "ab Z_'\n\t19,.!?-东京かナ한é🙂 、"
    for _ in range(5000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        assert "".join(tokenizer.encode(text)) == text


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "Hello, world! It's 2026.",
            ["Hello", ",", " world", "!", " It", "'s", " 202", "6", "."],
        ),
        ("They'll go", ["They", "'ll", " go"]),
        ("a  b", ["a", " ", " b"]),
        ("one\ntwo", ["one", "\n", "two"]),
        ("end   ", ["end", "   "]),
        ("snake_case", ["snake", "_", "case"]),
        ("f(x) == 1", ["f", "(", "x", ")", " ==", " 1"]),
    ],
)
def test_encode_splits_like_gpt(tokenizer, text, expected):
    assert tokenizer.encode(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("東京タワー", ["東", "京", "タ", "ワ", "ー"]),
        ("ひらがな", ["ひ", "ら", "が", "な"]),
        ("ｶﾀｶﾅ", ["ｶ", "ﾀ", "ｶ", "ﾅ"]),
        ("안녕 세계", ["안", "녕", " 세", "계"]),
        ("𠮷野家", ["𠮷", "野", "家"]),
        ("visit 東京 now", ["visit", " 東", "京", " now"]),
        ("abc東京def", ["abc", "東", "京", "def"]),
        ("東京、大阪。", ["東", "京", "、", "大", "阪", "。"]),
        ("Tokyo 2026年", ["Tokyo", " 202", "6", "年"]),
    ],
)
def test_encode_splits_cjk_per_character(tokenizer, text, expected):
    assert tokenizer.encode(text) == expected


def test_encode_is_deterministic(tokenizer):
    text = "To be, or not to be, that is the question."
    assert tokenizer.encode(text) == RegexTokenizer().encode(text)


@pytest.mark.parametrize("text", EDGE_CASES)
def test_count_matches_encode(tokenizer, text):
    assert tokenizer.count(text) == len(tokenizer.encode(text))


def test_count_messages_adds_chat_overhead(tokenizer):
    messages = [Msg("system", "Be brief."), Msg("user", "Hello, world!")]
    # system(1) + "Be brief."(3) + user(1) + "Hello, world!"(4), plus 3 per message and 3 for the reply.
    assert tokenizer.count_messages(messages) == (1 + 3) + (1 + 4) + 3 * 2 + 3


def test_count_messages_with_no_messages(tokenizer):
    assert tokenizer.count_messages([]) == tokenizer.tokens_per_reply


def test_get_tokenizer_returns_regex_tokenizer():
    tokenizer = get_tokenizer("regex")
    assert isinstance(tokenizer, RegexTokenizer)
    assert isinstance(tokenizer, Tokenizer)


def test_get_tokenizer_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown tokenizer 'bpe'; available: regex"):
        get_tokenizer("bpe")


def test_tokenizer_interface_requires_encode():
    class Incomplete(Tokenizer):
        name = "incomplete"

    with pytest.raises(TypeError):
        Incomplete()
