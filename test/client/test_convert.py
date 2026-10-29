import pytest

from llm_simulator.client.convert import PART_SEPARATOR, to_generation_request
from llm_simulator.client.exception import OpenAIError
from llm_simulator.client.schemas import ChatCompletionRequest
from llm_simulator.types import Message


def convert(*messages, **extra):
    return to_generation_request(
        ChatCompletionRequest(model="m", messages=list(messages), **extra)
    )


def text(value):
    return {"type": "text", "text": value}


def test_string_and_single_text_part_are_the_same_message():
    assert convert({"role": "user", "content": "hello"}) == convert(
        {"role": "user", "content": [text("hello")]}
    )


def test_text_parts_are_joined_without_a_separator():
    assert PART_SEPARATOR == ""
    request = convert({"role": "user", "content": [text("Summarise:"), text(" it")]})
    assert request.messages == (Message("user", "Summarise: it"),)


def test_every_role_accepts_parts():
    request = convert(
        {"role": "system", "content": [text("Be brief.")]},
        {"role": "user", "content": "hi"},
    )
    assert request.messages == (Message("system", "Be brief."), Message("user", "hi"))


@pytest.mark.parametrize("kind", ["image_url", "input_audio", "file", "refusal"])
def test_non_text_parts_are_rejected(kind):
    with pytest.raises(OpenAIError) as error:
        convert(
            {"role": "user", "content": "first"},
            {"role": "user", "content": [text("look"), {"type": kind}]},
        )
    assert error.value.status == 400
    assert error.value.param == "messages.[1].content.[1].type"
    assert kind in error.value.message


def test_text_part_needs_text():
    with pytest.raises(OpenAIError) as error:
        convert({"role": "user", "content": [{"type": "text"}]})
    assert error.value.param == "messages.[0].content.[0].text"
    assert error.value.message == (
        "Missing required parameter: 'messages.[0].content.[0].text'."
    )
