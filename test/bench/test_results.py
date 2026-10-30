import math
from array import array

import pytest

from llm_simulator.bench.config import BenchConfig
from llm_simulator.bench.records import RequestRecord, RunRecords, ServerSummary
from llm_simulator.bench.results import summarize

CONFIG = BenchConfig(model="m", concurrency=2, requests=4, max_tokens=4)


def ok(index, sent, ttft, gaps, end, prompt=10, requested=4, finish="length"):
    first = sent + ttft
    last = first + sum(gaps)
    return RequestRecord(
        index=index,
        sent=sent,
        end=end,
        first_token=first,
        last_token=last,
        itls=array("d", gaps),
        chunks=len(gaps) + 1,
        prompt_tokens=prompt,
        completion_tokens=len(gaps) + 1,
        finish_reason=finish,
        requested_tokens=requested,
        status=200,
    )


def failed(index, status, error):
    return RequestRecord(index=index, sent=0.0, end=0.1, status=status, error=error)


def run(records, started=0.0, ended=2.0):
    return summarize(RunRecords(records, started, ended), CONFIG, "fixed prompt")


def test_counts_and_finish_reasons():
    result = run(
        [
            ok(0, 0.0, 0.01, [0.004] * 3, 0.03),
            ok(1, 0.0, 0.02, [0.004], 0.03, finish="stop"),
            failed(2, 404, "no such model"),
            failed(3, 404, "no such model"),
        ]
    )
    assert (result.requests, result.ok, result.failed) == (4, 2, 2)
    assert result.finish_reasons == {"length": 1, "stop": 1}
    assert result.errors == {"HTTP 404: no such model": 2}


def test_latency_covers_successful_requests_only():
    result = run(
        [
            ok(0, 0.0, 0.010, [0.004, 0.006], 0.025),
            ok(1, 1.0, 0.030, [0.002, 0.002, 0.002], 1.04),
            failed(2, 0, "refused"),
        ]
    )
    ttft = result.latency["ttft"]
    assert ttft["count"] == 2
    assert ttft["mean"] == pytest.approx(0.020)
    # ITL pools every gap of every request.
    assert result.latency["itl"]["count"] == 5
    assert result.latency["itl"]["max"] == pytest.approx(0.006)
    # TPOT is one value per request: (last - first) / (chunks - 1).
    assert result.latency["tpot"]["count"] == 2
    assert result.latency["tpot"]["min"] == pytest.approx(0.002)
    assert result.latency["tpot"]["max"] == pytest.approx(0.005)
    assert result.latency["e2e"]["max"] == pytest.approx(0.04)


def test_single_chunk_requests_have_no_tpot():
    result = run([ok(0, 0.0, 0.01, [], 0.02, requested=1)])
    assert result.latency["tpot"]["count"] == 0
    assert result.latency["itl"]["count"] == 0
    assert result.latency["ttft"]["count"] == 1


def test_throughput_uses_the_window():
    result = run(
        [ok(0, 0.0, 0.01, [0.001] * 3, 0.1), ok(1, 0.5, 0.01, [0.001] * 3, 0.6)],
        started=0.0,
        ended=2.0,
    )
    assert result.duration == 2.0
    assert result.throughput.requests_per_second == 1.0
    assert result.throughput.output_tokens_per_second == 4.0  # 8 tokens / 2 s
    assert result.throughput.total_tokens_per_second == 14.0  # (20 + 8) / 2 s
    assert (result.prompt_tokens, result.output_tokens) == (20, 8)
    assert result.tokens_per_request["output"]["p50"] == 4


def test_shortfall_and_chunk_mismatch():
    short = ok(0, 0.0, 0.01, [0.001], 0.1, requested=4, finish="stop")
    packed = ok(1, 0.0, 0.01, [0.001], 0.1, requested=2)
    packed.completion_tokens = 2 * packed.chunks  # two tokens per chunk
    packed.requested_tokens = packed.completion_tokens
    result = run([short, packed, ok(2, 0.0, 0.01, [0.001] * 3, 0.1)])
    assert result.shortfall == 1
    assert result.chunk_mismatch == 1


def test_everything_failed():
    result = run([failed(0, 0, "refused"), failed(1, 500, "boom")])
    assert result.ok == 0
    assert result.latency["ttft"]["mean"] is None
    assert result.throughput.output_tokens_per_second == 0
    assert result.errors == {"no response: refused": 1, "HTTP 500: boom": 1}


def test_empty_window_does_not_divide_by_zero():
    result = run([], started=1.0, ended=1.0)
    assert result.throughput.requests_per_second == 0
    assert not math.isnan(result.throughput.total_tokens_per_second)


def stats(**values):
    return {"count": 1, "samples": 1, "min": None, "max": None, **values}


SECTION = {
    "latency_seconds": {
        "ttft": stats(mean=0.008, p50=0.008, p90=0.009, p95=0.009, p99=0.010),
        "itl": stats(mean=0.004, p50=0.004, p99=0.006),
        "tpot": stats(p50=0.004),
        "e2e": stats(p50=0.020),
        "lag": stats(p50=0.0005, p99=0.001, max=0.002),
        "ttft_simulated": stats(p50=0.007),
        "e2e_simulated": stats(p50=0.019),
    },
    "requests": {"finished": {"stop": 0, "length": 2, "cancelled": 0, "error": 0}},
    "tokens": {"prompt": 20, "generation": 8},
}


def with_server(records, summary):
    return summarize(RunRecords(records, 0.0, 2.0, server=summary), CONFIG, "w")


def two_ok():
    return [
        ok(0, 0.0, 0.010, [0.004] * 3, 0.025),
        ok(1, 0.0, 0.012, [0.004] * 3, 0.027),
    ]


def test_delivery_is_client_minus_server():
    result = with_server(two_ok(), ServerSummary("u", True, model=SECTION))
    comparison = result.server
    assert comparison.delivery["ttft"]["p50"] == pytest.approx(0.011 - 0.008)
    assert comparison.delivery["itl"]["p50"] == pytest.approx(0.0)
    # A statistic the server did not report has no delivery.
    assert comparison.delivery["tpot"]["mean"] is None
    assert comparison.server["lag"]["max"] == 0.002
    assert comparison.simulated["ttft"]["p50"] == 0.007
    assert comparison.counts_match
    assert comparison.server_output_tokens_per_second == 4.0  # 8 tokens / 2 s


def test_rejected_requests_are_not_expected_on_the_server():
    records = two_ok() + [failed(2, 400, "too long")]
    comparison = with_server(records, ServerSummary("u", True, model=SECTION)).server
    assert (comparison.server_requests, comparison.accepted_requests) == (2, 2)


def test_other_traffic_is_detected():
    section = {**SECTION, "requests": {"finished": {"stop": 5, "length": 2}}}
    comparison = with_server(two_ok(), ServerSummary("u", True, model=section)).server
    assert not comparison.counts_match


def test_older_summary_without_simulated_times():
    latency = dict(SECTION["latency_seconds"])
    del latency["ttft_simulated"], latency["e2e_simulated"]
    section = {**SECTION, "latency_seconds": latency}
    comparison = with_server(two_ok(), ServerSummary("u", True, model=section)).server
    assert comparison.simulated == {}


def test_unavailable_server_leaves_a_note():
    result = with_server(two_ok(), ServerSummary("u", False, reason="HTTP 404"))
    assert result.server is None
    assert result.server_note == "HTTP 404"


def test_not_compared():
    result = run(two_ok())
    assert (result.server, result.server_note) == (None, None)


def test_malformed_section_degrades_rather_than_failing():
    comparison = with_server(
        two_ok(), ServerSummary("u", True, model={"latency_seconds": 5})
    ).server
    assert comparison.delivery["ttft"]["p50"] is None
    assert comparison.server_requests == 0


# --- open loop ---

OPEN = BenchConfig(model="m", rate=2.0, max_concurrency=2)


def scheduled(record, at, waited=False):
    record.scheduled = at
    record.waited = waited
    return record


def test_arrivals_and_latency_from_schedule():
    records = [
        scheduled(ok(0, 0.5, 0.010, [0.004], 0.6), 0.5),
        scheduled(ok(1, 1.3, 0.010, [0.004], 1.4), 1.0, waited=True),
    ]
    result = summarize(RunRecords(records, 0.0, 2.0), OPEN, "w")
    arrivals = result.arrivals
    assert arrivals.target_rate == 2.0
    assert arrivals.offered_rate == pytest.approx(2 / 1.0)  # last scheduled at 1.0 s
    assert arrivals.achieved_rate == pytest.approx(2 / 1.3)  # last sent at 1.3 s
    assert not arrivals.keeping_up
    assert arrivals.waited == 1
    assert arrivals.queue_wait["max"] == pytest.approx(0.3)
    # From the send, both TTFTs are 10 ms; from the schedule, the waiting
    # request's user waited 300 ms longer.
    assert result.latency["ttft"]["max"] == pytest.approx(0.010)
    assert result.latency_from_schedule["ttft"]["max"] == pytest.approx(0.310)
    assert result.latency_from_schedule["e2e"]["min"] == pytest.approx(0.1)


def test_closed_loop_has_no_arrivals():
    result = run([ok(0, 0.0, 0.01, [0.001], 0.1)])
    assert (result.arrivals, result.latency_from_schedule) == (None, None)
