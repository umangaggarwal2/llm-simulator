import pytest

from llm_simulator.simulator import Engine


def test_engine_interface_requires_run():
    class Incomplete(Engine):
        name = "incomplete"

    with pytest.raises(TypeError):
        Incomplete()


def test_base_defaults():
    assert Engine.required_sections == ()
