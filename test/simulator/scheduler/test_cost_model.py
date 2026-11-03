import pytest

from llm_simulator.simulator.scheduler.cost_model import RooflineCostModel, StepCost

# A 7B fp16 model on an H100: 13.5 GB of weights, 989 TFLOPS dense, 3,000 GB/s.
PARAMETERS = 6.74e9
WEIGHTS = 13.5e9
KV_PER_TOKEN = 524288
PEAK_FLOPS = 989e12
BANDWIDTH = 3000e9


def make_model(**overrides):
    params = dict(
        parameters=PARAMETERS,
        weight_bytes=WEIGHTS,
        kv_bytes_per_token=KV_PER_TOKEN,
        peak_flops=PEAK_FLOPS,
        memory_bandwidth=BANDWIDTH,
    )
    params.update(overrides)
    return RooflineCostModel(**params)


def test_batch_one_decode_is_memory_bound_by_the_weights():
    cost = make_model().step_cost([(1, 0)])
    assert cost.memory == pytest.approx((WEIGHTS + KV_PER_TOKEN) / BANDWIDTH)
    assert cost.duration == pytest.approx(0.0045, rel=1e-3)
    assert cost.bound == "memory"


def test_batching_short_decodes_is_almost_free():
    one = make_model().step_cost([(1, 100)]).duration
    many = make_model().step_cost([(1, 100)] * 64).duration
    # 64 times the tokens for about 25% more time: the weights are read once.
    assert many < 1.3 * one


def test_long_contexts_slow_every_decode():
    # 64 sequences at 2k context read 64 GB of KV on top of the weights.
    cost = make_model().step_cost([(1, 2048)] * 64)
    assert cost.duration == pytest.approx(
        (WEIGHTS + 64 * 2049 * KV_PER_TOKEN) / BANDWIDTH
    )
    assert cost.duration > 0.025


def test_a_full_budget_prefill_chunk_is_compute_bound():
    # 8,192 tokens: 2 x 6.74e9 x 8192 / 989e12 = 112 ms. Every decode that
    # shares this step waits as long.
    cost = make_model().step_cost([(8192, 0)])
    assert cost.compute == pytest.approx(2 * PARAMETERS * 8192 / PEAK_FLOPS)
    assert cost.duration == pytest.approx(0.1117, rel=1e-3)
    assert cost.bound == "compute"


def test_crossover_between_memory_and_compute_bound():
    # About peak_flops x weights / (2 x parameters x bandwidth) = 330 tokens.
    model = make_model()
    assert model.step_cost([(300, 0)]).bound == "memory"
    assert model.step_cost([(400, 0)]).bound == "compute"


def test_cached_context_of_a_prefill_chunk_is_read():
    model = make_model()
    first = model.step_cost([(512, 0)])
    later = model.step_cost([(512, 4096)])
    assert later.compute == first.compute
    assert later.memory - first.memory == pytest.approx(4096 * KV_PER_TOKEN / BANDWIDTH)


def test_efficiencies_scale_the_peak_figures():
    model = make_model(compute_efficiency=0.5, memory_efficiency=0.8)
    ideal = make_model()
    assert model.step_cost([(8192, 0)]).compute == pytest.approx(
        2 * ideal.step_cost([(8192, 0)]).compute
    )
    assert model.step_cost([(1, 0)]).memory == pytest.approx(
        ideal.step_cost([(1, 0)]).memory / 0.8
    )


def test_overhead_is_added_to_every_step_and_overlaps_nothing():
    base = make_model().step_cost([(1, 0)])
    with_overhead = make_model(step_overhead_s=0.002).step_cost([(1, 0)])
    assert with_overhead.duration == pytest.approx(base.duration + 0.002)
    assert (with_overhead.compute, with_overhead.memory) == (base.compute, base.memory)


def test_duration_and_bound():
    assert StepCost(compute=3, memory=2, overhead=1).duration == 4
    assert StepCost(compute=3, memory=2, overhead=1).bound == "compute"
    assert StepCost(compute=2, memory=2, overhead=0).bound == "memory"


@pytest.mark.parametrize("work", [[], [(0, 5)], [(1, -1)]])
def test_step_cost_rejects_invalid_work(work):
    with pytest.raises(ValueError):
        make_model().step_cost(work)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(parameters=0),
        dict(weight_bytes=0),
        dict(kv_bytes_per_token=0),
        dict(peak_flops=0),
        dict(memory_bandwidth=0),
        dict(compute_efficiency=0),
        dict(memory_efficiency=1.5),
        dict(step_overhead_s=-0.001),
    ],
)
def test_model_rejects_invalid_parameters(overrides):
    with pytest.raises(ValueError):
        make_model(**overrides)
