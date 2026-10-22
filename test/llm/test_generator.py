import hashlib
from pathlib import Path

import pytest

from llm_simulator.errors import ContextLengthExceeded
from llm_simulator.llm.generator import SYNTHETIC_CORPUS, Generator
from llm_simulator.llm.matcher import DEFAULT_ID, Matcher
from llm_simulator.llm.responses import ResponsesConfig, load_responses
from llm_simulator.llm.tokenizer import RegexTokenizer
from llm_simulator.types import GenerationRequest, Message

REPO_ROOT = Path(__file__).resolve().parents[2]
TOKENIZER = RegexTokenizer()
FIVE_VARIANTS = {"variants": ["zero", "one", "two", "three", "four"]}


def make_generator(*entries, default=None, context_length=10_000, seed=0):
    config = ResponsesConfig.model_validate(
        {
            "version": 1,
            "default": default or {"content": "fallback"},
            "responses": [{"id": f"e{i}", **entry} for i, entry in enumerate(entries)],
        }
    )
    return Generator(Matcher(config), TOKENIZER, context_length, seed)


def ask(text, *, model="qwen-mock", max_tokens=None, seed=None):
    return GenerationRequest(
        model=model,
        messages=(Message("user", text),),
        max_tokens=max_tokens,
        seed=seed,
    )


def expected_index(seed, entry_id, text, n):
    digest = hashlib.sha256(f"{seed}\0{entry_id}\0{text}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % n


# --- Variant selection ---


def test_same_request_gives_same_plan():
    generator = make_generator(default=FIVE_VARIANTS)
    assert generator.generate(ask("hello")) == generator.generate(ask("hello"))


def test_new_generator_gives_same_plan():
    first = make_generator(default=FIVE_VARIANTS).generate(ask("hello"))
    second = make_generator(default=FIVE_VARIANTS).generate(ask("hello"))
    assert first == second


def test_selection_follows_documented_hash():
    generator = make_generator(default=FIVE_VARIANTS, seed=42)
    for text in ["hello", "what is this", "東京"]:
        plan = generator.generate(ask(text))
        assert plan.variant == expected_index(42, DEFAULT_ID, text, 5)


def test_variants_spread_across_inputs():
    generator = make_generator(default=FIVE_VARIANTS)
    picked = {generator.generate(ask(f"question {i}")).variant for i in range(200)}
    assert picked == {0, 1, 2, 3, 4}


def test_request_seed_overrides_file_seed():
    from_file = make_generator(default=FIVE_VARIANTS, seed=42)
    from_request = make_generator(default=FIVE_VARIANTS, seed=0)
    for i in range(20):
        text = f"question {i}"
        assert from_file.generate(ask(text)) == from_request.generate(
            ask(text, seed=42)
        )


def test_different_seed_changes_some_variants():
    generator = make_generator(default=FIVE_VARIANTS)
    texts = [f"question {i}" for i in range(50)]
    seed_1 = [generator.generate(ask(t, seed=1)).variant for t in texts]
    seed_2 = [generator.generate(ask(t, seed=2)).variant for t in texts]
    assert seed_1 != seed_2


def test_entry_id_is_part_of_the_hash():
    generator = make_generator(
        {"match": {"starts_with": "a"}, **FIVE_VARIANTS},
        {"match": {"starts_with": "b"}, **FIVE_VARIANTS},
    )
    for text in ["a1", "b1"]:
        plan = generator.generate(ask(text))
        assert plan.variant == expected_index(0, plan.entry_id, text, 5)


# --- Input ---


def test_surrounding_whitespace_is_ignored_for_matching_and_selection():
    generator = make_generator(
        {"match": {"exact": "hello"}, **FIVE_VARIANTS}, default=FIVE_VARIANTS
    )
    plain = generator.generate(ask("hello"))
    padded = generator.generate(ask("  hello\n"))
    assert plain.entry_id == padded.entry_id == "e0"
    assert plain.variant == padded.variant


def test_uses_last_user_message():
    generator = make_generator(
        {"match": {"exact": "first"}, "content": "matched first"},
        {"match": {"exact": "second"}, "content": "matched second"},
    )
    request = GenerationRequest(
        model="m",
        messages=(
            Message("user", "first"),
            Message("assistant", "second"),
            Message("user", "second"),
            Message("assistant", "first"),
        ),
    )
    assert generator.generate(request).text == "matched second"


def test_no_user_message_matches_default():
    generator = make_generator({"match": {"contains": "a"}, "content": "x"})
    request = GenerationRequest(model="m", messages=(Message("system", "a"),))
    plan = generator.generate(request)
    assert plan.entry_id == DEFAULT_ID
    assert plan.text == "fallback"


# --- Templates ---


def test_builtin_variables():
    generator = make_generator(
        default={"content": "{model} heard [{last_user_message}]"}
    )
    plan = generator.generate(ask("  hi there\n", model="qwen-mock"))
    # {last_user_message} keeps the text exactly as sent.
    assert plan.text == "qwen-mock heard [  hi there\n]"


def test_regex_groups():
    generator = make_generator(
        {
            "match": {"regex": r"weather in (?P<city>\w+)"},
            "content": "Sunny in {city}.",
        }
    )
    assert generator.generate(ask("WEATHER IN Paris")).text == "Sunny in Paris."


def test_escaped_braces_render_literally():
    generator = make_generator(default={"content": "{{not a placeholder}}"})
    assert generator.generate(ask("x")).text == "{not a placeholder}"


def test_user_text_with_braces_is_not_expanded():
    generator = make_generator(default={"content": "You said: {last_user_message}"})
    assert generator.generate(ask("{model}")).text == "You said: {model}"


def test_empty_output_is_allowed():
    generator = make_generator(default={"content": "{last_user_message}"})
    request = GenerationRequest(model="m", messages=(Message("system", "x"),))
    plan = generator.generate(request)
    assert plan.tokens == ()
    assert plan.finish_reason == "stop"


# --- Synthetic ---


def test_synthetic_fixed_length_is_exact():
    generator = make_generator(default={"synthetic": {"tokens": 50}})
    plan = generator.generate(ask("x"))
    assert plan.completion_tokens == 50
    assert plan.finish_reason == "stop"


def test_synthetic_length_longer_than_corpus_is_exact_and_spaced():
    length = 3 * len(TOKENIZER.encode(SYNTHETIC_CORPUS)) + 7
    generator = make_generator(default={"synthetic": {"tokens": length}})
    plan = generator.generate(ask("x"))
    assert plan.completion_tokens == length
    assert "laborum. Lorem" in plan.text


def test_synthetic_range_follows_documented_hash():
    generator = make_generator(default={"synthetic": {"tokens": [10, 60]}})
    for i in range(30):
        text = f"load {i}"
        digest = hashlib.sha256(f"0\0{DEFAULT_ID}\0{text}".encode()).digest()
        expected = 10 + int.from_bytes(digest[8:16], "big") % 51
        assert generator.generate(ask(text)).completion_tokens == expected


def test_synthetic_range_varies_and_stays_in_bounds():
    generator = make_generator(default={"synthetic": {"tokens": [10, 60]}})
    lengths = {
        generator.generate(ask(f"load {i}")).completion_tokens for i in range(50)
    }
    assert len(lengths) > 1
    assert all(10 <= n <= 60 for n in lengths)


def test_synthetic_truncated_by_max_tokens():
    generator = make_generator(default={"synthetic": {"tokens": 100}})
    plan = generator.generate(ask("x", max_tokens=10))
    assert plan.completion_tokens == 10
    assert plan.finish_reason == "length"


# --- Truncation and context length ---


SIX_TOKENS = {"content": "one two three four five six"}


@pytest.mark.parametrize(
    "max_tokens, expected_tokens, expected_reason",
    [(3, 3, "length"), (6, 6, "stop"), (10, 6, "stop"), (None, 6, "stop")],
)
def test_max_tokens(max_tokens, expected_tokens, expected_reason):
    plan = make_generator(default=SIX_TOKENS).generate(ask("x", max_tokens=max_tokens))
    assert plan.completion_tokens == expected_tokens
    assert plan.finish_reason == expected_reason


def test_without_max_tokens_output_is_capped_at_remaining_context():
    request = ask("x")
    prompt = TOKENIZER.count_messages(request.messages)
    generator = make_generator(default=SIX_TOKENS, context_length=prompt + 4)
    plan = generator.generate(request)
    assert plan.tokens == ("one", " two", " three", " four")
    assert plan.finish_reason == "length"


def test_prompt_filling_the_context_is_rejected():
    request = ask("x")
    prompt = TOKENIZER.count_messages(request.messages)
    generator = make_generator(context_length=prompt)
    with pytest.raises(ContextLengthExceeded) as error:
        generator.generate(request)
    assert error.value.prompt_tokens == prompt
    assert error.value.max_tokens is None
    assert f"your messages resulted in {prompt} tokens" in str(error.value)


def test_prompt_plus_max_tokens_over_context_is_rejected():
    request = ask("x", max_tokens=5)
    prompt = TOKENIZER.count_messages(request.messages)
    generator = make_generator(context_length=prompt + 4)
    with pytest.raises(ContextLengthExceeded) as error:
        generator.generate(request)
    assert (
        f"you requested {prompt + 5} tokens ({prompt} in the messages, "
        "5 in the completion)" in str(error.value)
    )


def test_prompt_plus_max_tokens_exactly_at_context_is_allowed():
    request = ask("x", max_tokens=4)
    prompt = TOKENIZER.count_messages(request.messages)
    plan = make_generator(default=SIX_TOKENS, context_length=prompt + 4).generate(
        request
    )
    assert plan.completion_tokens == 4


def test_context_length_must_be_positive():
    config = ResponsesConfig.model_validate({"version": 1, "default": {"content": "x"}})
    with pytest.raises(ValueError, match="context_length must be at least 1"):
        Generator(Matcher(config), TOKENIZER, 0)


# --- Plan fields ---


def test_plan_fields():
    generator = make_generator(
        {"match": {"contains": "hamlet"}, "content": "To be, or not to be."}
    )
    request = GenerationRequest(
        model="m",
        messages=(Message("system", "Be brief."), Message("user", "quote hamlet")),
    )
    plan = generator.generate(request)
    assert plan.prompt_tokens == TOKENIZER.count_messages(request.messages)
    assert plan.entry_id == "e0"
    assert plan.variant == 0
    assert plan.text == "To be, or not to be."
    assert plan.tokens == tuple(TOKENIZER.encode("To be, or not to be."))
    assert plan.completion_tokens == len(plan.tokens)


def test_real_config():
    config = load_responses(REPO_ROOT / "config" / "responses.yml")
    generator = Generator(Matcher(config), TOKENIZER, 32768, config.seed)
    plan = generator.generate(ask("What are the themes in Macbeth?\n"))
    assert plan.entry_id == "macbeth-themes"
    assert plan.variant == 3
    assert plan.text.startswith("A lot of readers focus on fate and free will")
    assert plan.finish_reason == "stop"
