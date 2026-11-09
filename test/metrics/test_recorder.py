from llm_simulator.metrics import (
    FanOutRecorder,
    MetricsRecorder,
    NoopRecorder,
    RequestTimings,
)
from llm_simulator.types import StepStats

TIMINGS = RequestTimings(e2e=0.02, queue=0.0, prefill=0.01, decode=0.01, tpot=0.01)


class CapturingRecorder(MetricsRecorder):
    def __init__(self, name, log):
        self.name = name
        self.log = log

    def request_started(self, model, prompt_tokens):
        self.log.append((self.name, "started", model, prompt_tokens))

    def first_token(self, model, ttft, queue_time):
        self.log.append((self.name, "first_token", model, ttft, queue_time))

    def token(self, model, itl, lag):
        self.log.append((self.name, "token", model, itl, lag))

    def request_finished(
        self, model, finish_reason, prompt_tokens, generation_tokens, timings
    ):
        self.log.append(
            (
                self.name,
                "finished",
                model,
                finish_reason,
                prompt_tokens,
                generation_tokens,
                timings,
            )
        )

    def step(self, model, stats):
        self.log.append((self.name, "step", model, stats))


def test_fan_out_forwards_every_call_to_every_recorder_in_order():
    log = []
    fan_out = FanOutRecorder([CapturingRecorder("a", log), CapturingRecorder("b", log)])
    fan_out.request_started("m", 8)
    fan_out.first_token("m", 0.01, 0.0)
    fan_out.token("m", None, 0.001)
    fan_out.request_finished("m", "stop", 8, 1, TIMINGS)
    assert log == [
        ("a", "started", "m", 8),
        ("b", "started", "m", 8),
        ("a", "first_token", "m", 0.01, 0.0),
        ("b", "first_token", "m", 0.01, 0.0),
        ("a", "token", "m", None, 0.001),
        ("b", "token", "m", None, 0.001),
        ("a", "finished", "m", "stop", 8, 1, TIMINGS),
        ("b", "finished", "m", "stop", 8, 1, TIMINGS),
    ]


STATS = StepStats(
    duration=0.005,
    compute=0.001,
    memory=0.005,
    overhead=0.0,
    sequences=1,
    prefill_tokens=0,
    decode_tokens=1,
    admitted=0,
    preempted=0,
    preempted_tokens=0,
    blocked=None,
    waiting=0,
    running=1,
    kv_blocks_used=1,
    kv_blocks_total=10,
)


def test_fan_out_forwards_steps():
    log = []
    fan_out = FanOutRecorder([CapturingRecorder("a", log), CapturingRecorder("b", log)])
    fan_out.step("m", STATS)
    assert log == [("a", "step", "m", STATS), ("b", "step", "m", STATS)]


def test_step_is_optional_for_recorders():
    # Recorders that only handle requests need not implement step.
    NoopRecorder().step("m", STATS)
