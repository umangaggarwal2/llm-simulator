import pytest

from llm_simulator.bench.workload import (
    WORDS,
    RequestSpec,
    fixed,
    from_jsonl,
    synthetic,
)
from llm_simulator.llm.tokenizer import RegexTokenizer
from llm_simulator.types import Message


def prompt_tokens(text):
    return RegexTokenizer().count_messages([Message("user", text)])


def test_words_are_one_token_each():
    tokenizer = RegexTokenizer()
    assert all(tokenizer.count(word) == 1 for word in WORDS)
    assert all(tokenizer.count(" " + word) == 1 for word in WORDS)
    assert len(set(WORDS)) == len(WORDS)


# --- fixed ---


def test_fixed_sends_the_same_prompt():
    workload = fixed("Hello.", 64, prefix="[short] ")
    assert (
        workload.request(0) == workload.request(99) == RequestSpec("[short] Hello.", 64)
    )
    assert workload.description == "fixed prompt → 64 tokens · prefix '[short] '"


# --- synthetic ---


@pytest.mark.parametrize(
    "prefix, tokens",
    [("", n) for n in (8, 9, 100, 512, 4096)]
    + [(prefix, n) for prefix in ("[short] ", "Context:") for n in (12, 100, 4096)],
)
def test_synthetic_prompts_have_exactly_the_requested_prompt_tokens(prefix, tokens):
    workload = synthetic(tokens, 32, prefix=prefix)
    for index in range(5):
        spec = workload.request(index)
        assert spec.prompt.startswith(prefix)
        assert prompt_tokens(spec.prompt) == tokens
        assert spec.max_tokens == 32


def test_each_request_gets_its_own_prompt():
    workload = synthetic(200, 10)
    prompts = {workload.request(index).prompt for index in range(50)}
    assert len(prompts) == 50


def test_prompts_are_repeatable_and_seeded():
    assert synthetic(100, 10).request(7) == synthetic(100, 10).request(7)
    assert synthetic(100, 10, seed=1).request(7) != synthetic(100, 10).request(7)


def test_too_few_input_tokens():
    with pytest.raises(ValueError, match="less than 8: the chat overhead plus one"):
        synthetic(7, 10)
    with pytest.raises(ValueError, match="and the prefix"):
        synthetic(9, 10, prefix="[short] ")


def test_synthetic_description():
    assert synthetic(512, 128).description == "synthetic 512 input tokens → 128 tokens"


# --- from_jsonl ---


def write(tmp_path, text):
    path = tmp_path / "prompts.jsonl"
    path.write_text(text)
    return path


def test_dataset_lines_are_cycled(tmp_path):
    path = write(
        tmp_path,
        '{"prompt": "one"}\n\n{"prompt": "two", "max_tokens": 5}\n{"prompt": "three"}\n',
    )
    workload = from_jsonl(path, 100, prefix="> ")
    assert [workload.request(index) for index in range(4)] == [
        RequestSpec("> one", 100),
        RequestSpec("> two", 5),
        RequestSpec("> three", 100),
        RequestSpec("> one", 100),
    ]
    assert workload.description.startswith("dataset prompts.jsonl (3 prompts)")


@pytest.mark.parametrize(
    "line, message",
    [
        ("{not json", "prompts.jsonl:2: invalid JSON"),
        ('["a list"]', "prompts.jsonl:2: expected an object"),
        ('{"text": "no prompt"}', "prompts.jsonl:2: missing 'prompt'"),
        ('{"prompt": ""}', "prompts.jsonl:2: missing 'prompt'"),
        ('{"prompt": "x", "max_tokens": 0}', "'max_tokens' must be a positive"),
        ('{"prompt": "x", "max_tokens": true}', "'max_tokens' must be a positive"),
    ],
)
def test_dataset_errors_name_the_line(tmp_path, line, message):
    path = write(tmp_path, '{"prompt": "fine"}\n' + line + "\n")
    with pytest.raises(ValueError, match=message):
        from_jsonl(path, 10)


def test_empty_dataset(tmp_path):
    with pytest.raises(ValueError, match="no prompts"):
        from_jsonl(write(tmp_path, "\n\n"), 10)
