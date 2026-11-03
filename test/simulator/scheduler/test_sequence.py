import pytest

from llm_simulator.simulator.scheduler.sequence import Sequence


def make_sequence(prompt=10, output=3):
    return Sequence(id=1, arrival=0.0, prompt_tokens=prompt, output_tokens=output)


def run(seq, *chunks):
    """Advance seq by each chunk; return which steps produced a token."""
    return [seq.advance(chunk) for chunk in chunks]


def test_a_new_sequence_must_compute_its_prompt():
    seq = make_sequence()
    assert (seq.known, seq.computed, seq.uncomputed) == (10, 0, 10)
    assert seq.prefilling


def test_prefill_produces_the_first_token():
    seq = make_sequence()
    assert run(seq, 10) == [True]
    assert (seq.generated, seq.known, seq.uncomputed) == (1, 11, 1)
    assert not seq.prefilling


def test_only_the_last_prefill_chunk_produces_a_token():
    seq = make_sequence()
    assert run(seq, 4, 4, 2) == [False, False, True]
    assert seq.generated == 1


def test_decode_steps_produce_one_token_each_until_finished():
    seq = make_sequence(output=3)
    assert run(seq, 10, 1, 1) == [True, True, True]
    assert seq.finished
    # The last token's KV is never computed: nothing attends to it.
    assert (seq.generated, seq.computed) == (3, 12)


def test_empty_output_finishes_after_prefill_without_a_token():
    seq = make_sequence(output=0)
    assert run(seq, 10) == [False]
    assert seq.finished


def test_preemption_recomputes_the_prompt_and_the_generated_tokens():
    seq = make_sequence(output=5)
    run(seq, 10, 1)  # two tokens generated
    seq.preempt()
    assert (seq.computed, seq.uncomputed, seq.preemptions) == (0, 12, 1)
    assert seq.prefilling
    # Recomputing produces the next token, not a repeat of the old ones.
    assert run(seq, 8, 4) == [False, True]
    assert seq.generated == 3


def test_preemption_mid_prefill_starts_the_prompt_again():
    seq = make_sequence()
    run(seq, 6)
    seq.preempt()
    assert (seq.uncomputed, seq.generated) == (10, 0)


@pytest.mark.parametrize("tokens", [0, 11])
def test_advance_rejects_more_than_uncomputed_or_nothing(tokens):
    with pytest.raises(ValueError):
        make_sequence().advance(tokens)


def test_advance_rejects_a_finished_sequence():
    seq = make_sequence(output=1)
    seq.advance(10)
    with pytest.raises(ValueError, match="finished"):
        seq.advance(1)


@pytest.mark.parametrize("prompt, output", [(0, 1), (1, -1)])
def test_invalid_sizes(prompt, output):
    with pytest.raises(ValueError):
        make_sequence(prompt, output)
