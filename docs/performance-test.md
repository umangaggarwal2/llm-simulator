# Performance and fidelity testing

How the simulator's timing is tested: what is measured, how a result is judged, and which experiments make up the scheduler's validation. Results are recorded in [architecture.md](architecture.md#scheduler-validation), under "Scheduler validation"; this document holds the strategy, the method and the plan.

## What is being tested

The simulator makes two kinds of timing claim, and they are tested differently.

- **Simulated time**: when the model says each token is due (`TokenEvent.sim_time`, the step durations, the summary's `ttft_simulated` and `e2e_simulated`). These come from the cost model and the scheduler's rules, so they can be predicted exactly before a run.
- **Observed time**: when a client actually receives each token. It equals simulated time plus `lag`, the simulator's own error (CPU work in our process, timer granularity). Observed time is what the systems under test experience, so lag is the simulator's fidelity error.

An experiment checks a **scheduler decision** with simulated values and reports observed values beside them, with the lag expected at that batch size. Mixing the two would let a timing problem in our process look like a scheduling problem, or hide one.

## Principles

1. **Predict before running.** Every experiment states its prediction, computed from the cost model and the policy's rules, and its pass criteria before any run. A result is then a test, not a description.
2. **One variable at a time.** Each model in [`config/validation/models.yml`](../config/validation/models.yml) differs from `base` (identical to `qwen-mock-batched`) in one field, and a test in `test/llm/test_models.py` checks that it does.
3. **Exact output lengths.** The server runs with the built-in `bench-responses`, so `--max-tokens` sets every completion's length exactly.
4. **When a prediction fails, stop.** Work out whether the prediction or the simulator is wrong before running the next experiment, and record which. A prediction is never adjusted to fit a result after the fact; if it was wrong, the correction and the reason are written down.
5. **An explanation is not proof.** When a result deviates and an explanation fits the numbers, it is recorded as inferred until a control experiment removes the suspected cause and changes nothing else (E1b is one).
6. **Check the instrument.** A profiler or a script is trusted only after it agrees with an independent measurement (see "Measuring CPU" below).

## Setup

Client and server on one machine, the server alone on its port:

```
LLM_SIM_MODELS=config/validation/models.yml LLM_SIM_RESPONSES=builtin:bench-responses \
LLM_SIM_PORT=8093 poetry run llmsimulator
```

In Docker, `SIM_CONFIG_DIR=../config/validation` points `bench-target` at the same models; the bench container's open-file limit is raised to 65,536 for high concurrency.

Each run writes JSON to `results/validation/` (gitignored). The JSON embeds the server's whole `/metrics/summary` section for the model, including the `scheduler` section, so every value used in a verdict can be found again.

## Where the numbers come from

| Value | Source | Use |
|---|---|---|
| Step time p50, p99, max; batch; KV usage; preemptions; blocked steps; step lag | Summary `scheduler` section, exact over the run | Simulated values for verdicts |
| Simulated TTFT and E2E | Summary `latency_seconds.ttft_simulated`, `e2e_simulated` | Simulated values for verdicts |
| Queue and prefill time | Summary `latency_seconds.queue`, `prefill` | Explaining TTFT |
| Observed TTFT, ITL, E2E, throughput | `llmsim-bench` client measurements | Reported beside simulated values |
| Token lag, step lag | Summary `latency_seconds.lag`, `scheduler.step_lag_seconds` | Fidelity |

**Predictions** use the roofline for the `base` model: 6.74B parameters, 13.48 GB of weights, 512 KiB of KV per token, 3,000 GB/s and 989 TFLOPS. A decode step at batch B with mean context C takes (13.48 GB + B × C × 512 KiB) ÷ 3,000 GB/s; a step's compute is 2 × 6.74B × tokens ÷ 989 TFLOPS, so 8,192 prompt tokens take 111.7 ms. For anything less simple than one formula, the prediction is computed with `RooflineCostModel` itself, step by step.

## Measuring CPU

Used when the question is where our process spends its time (E0 and the transport work):

- **Process CPU per token**: `utime + stime` from `/proc/<pid>/stat` before and after a run, divided by `llm_sim_generation_tokens_total`. No profiler involved; the reference for everything else.
- **py-spy, non-blocking only**: `py-spy record --gil --threads --nonblocking --rate 200 --format raw -- <server>`. The default blocking mode pauses the process for every sample; under this load it inflated CPU per token 1.7× and doubled one stage's share. With `ptrace_scope` at 1, py-spy has to start the server itself.
- **Stage attribution**: each sample goes to the stage of its deepest matching frame, after dropping the common root (`main` → uvicorn → `asyncio_run`); a stack with nothing left is uvloop's own C code. A profile is used only if its CPU per token matches `/proc` and its stage totals match timers inside the server.
- **Timers inside the server**: wrapping `Scheduler.schedule`, `Scheduler.complete`, the cost model and the queue handoff with `time.perf_counter`, in a launcher script that patches them before starting the app.
- **Garbage collection**: `gc.callbacks` timing every collection, written out by a background thread.

## The experiments

The scheduler's decisions, each with its own experiment. Status as of 2026-11-09.

| # | Decision | Status |
|---|---|---|
| E0 | Fidelity: where token lag comes from | Done: lag p99 ≈ 1.7 ms + 0.15 ms per sequence in the batch; about 90 µs of CPU per token, attributed by stage |
| E6 | The per-step overhead and memory efficiency settings | Pass: exact to under 0.001% |
| E1 | Batching is nearly free | Pass: step p50 within 0.3%; phase-locking found |
| E1b | Control: is phase-locking the cause of E1's TTFT? | Pending |
| E2 | Context slows every stream | Pass: step p50 within 0.13% |
| E3 | Chunked prefill | Pass: longest step exact for both budgets, long-prompt TTFT ratio 1.000; our request processing found in queue time |
| E4 | Admission waits for KV space | Pass: at most 9 running, no preemption, `kv` blocking, queue p50 within 1.4% |
| E5 | Preemption, and the watermark | Pass: within 3% of a virtual-time run of the scheduler; the watermark cut preemptions 9%; victims stalled up to 50 s |
| E7 | The scheduler's fidelity envelope | Partial: batch and step time match; TTFT 3.5–5 ms high and capacity 9% low, both from our request processing; lag p99 under 5 ms up to about 25 requests/s |

E6 runs first because if the configuration settings do not do exactly what they claim, no other result is worth reading. E7 runs last because it pushes towards saturation and changes several things at once, so it measures the envelope rather than isolating a cause.

### E0: where token lag comes from

- **Run**: closed loop at 1, 8, 32 and 64 streams; 1,000-token prompts, 200 output tokens.
- **Measures**: token lag against step lag at each batch size, then CPU per token by stage (see "Measuring CPU").
- **Outcome**: lag grows with batch size, not throughput, because a step releases one token to every sequence at once and our process delivers them one after another. The improvements this points to are T1–T7 in [architecture.md](architecture.md#fidelity-and-performance-the-per-token-path).

### E6: the settings are exact

- **Run**: one stream, 128-token prompts, 200 output tokens, 30 requests each on `base`, `overhead-1ms` and `memory-eff-80`.
- **Prediction**: step p50 4.5331, 5.5331 (+1 ms exactly) and 5.6664 ms (×1.25); simulated TTFT and E2E computed step by step.
- **Pass**: simulated values within 0.5%; observed ITL p50 within 0.3 ms.

### E1: batching is nearly free

- **Run**: closed-loop sweep at 1, 2, 4, 8, 16, 32, 64 streams; 128-token prompts, 200 output tokens; 20 s per level.
- **Prediction**: step ≈ (13.48 GB + B × 228 × 512 KiB) ÷ 3,000 GB/s, from 4.533 ms to 7.043 ms; output ≈ B ÷ step less about 0.5%, from 219 to 9,041 tokens/s; all memory-bound.
- **Pass**: step p50 within 5% at every level; batch close to the concurrency; no KV blocking or preemption; throughput within 10%.

### E1b: control for phase-locking (pending)

E1's TTFT grew to 100 ms at 64 streams with nothing blocked. The explanation that fits is phase-locking: equal-length requests in a closed loop finish together and arrive together, so one step prefills most of the batch at once. E1b removes that cause and nothing else, at 64 streams with 128-token prompts on `base`:

- **E1b-1, varied lengths**: closed loop, with a dataset whose lines are `{"input_tokens": 128, "max_tokens": N}` with N spread evenly over 100–300 (mean 200).
- **E1b-2, Poisson arrivals**: open loop at about 40 requests per second with fixed 200-token outputs, a batch of about 57.
- **Prediction if phase-locking is the cause**: simulated TTFT p50 about 10–15 ms in both, a largest step well below 82 ms, and throughput within about 2% of E1's prediction for that batch.
- **If either fails**: investigate before trusting E1's TTFT, since something else in the scheduler would be adding queueing.

### E2: context slows every stream

- **Run**: 32 streams, 256 output tokens, prompts of 128, 512, 1,024 and 2,048 tokens, one run each.
- **Prediction**: step ≈ (13.48 GB + 32 × (prompt + 128) × 512 KiB) ÷ 3,000 GB/s: 5.925, 8.072, 10.936 and 16.662 ms; KV at its maximum 10%, 20%, 34% and 61%; `budget` waits from 512 tokens (phase-locked prompts exceed 8,192 tokens) and a largest step of 111.7 ms.
- **Pass**: step p50 within 5%, decode steps memory-bound, no KV blocking or preemption.

### E3: chunked prefill trades spike height for spread

- **Run**: two `llmsim-bench` processes against one server at once, so each reports its own percentiles:
  - 16 steady decoders: `--concurrency 16 --input-tokens 128 --max-tokens 1000`, reading the server summary;
  - one stream of long prompts: `--concurrency 1 --input-tokens 8192 --max-tokens 1 --no-server-metrics`.
- **Models**: `base` (budget 8,192) and `budget-2048`.
- **Prediction**, computed step by step with the cost model (16 decoders at mid-life context, about 628 tokens):

  | | budget 8,192 | budget 2,048 |
  |---|---|---|
  | Steps per long prompt | 2 | 5 |
  | Decoders' largest ITL | 111.7 ms | 27.9 ms |
  | Long prompt's prefill time | 119.3 ms | 119.3 ms |
  | Decode-only step | 6.25 ms | 6.25 ms |

  The total prefill work does not change, only how it is sliced: a smaller budget gives lower but more frequent spikes at the same total cost, which is the trade-off behind vLLM's `max_num_batched_tokens`.
- **Pass**: decoders' ITL max within 10% of the prediction for each model; the long prompt's simulated TTFT about the same for both (ratio 1.0–1.15, allowing for the wait for a step already running).

### E4: admission waits for KV space

- **Run**: `small-kv` (1,260 blocks, 20,160 tokens, a watermark of 12 blocks); 32 streams; 2,000-token prompts, 200 output tokens.
- **Prediction**: each prompt needs 125 blocks, so at most 9 run at once (a tenth would leave fewer than 12 blocks free); fully grown they hold 9 × 138 = 1,242 blocks, so no preemption; about 23 waiting, blocked on `kv`; a decode step of about 7.8 ms, about 1.6 s per request, about 5.7 requests per second, so queue time p50 about 4 s and TTFT about the queue time plus 30 ms. To be recomputed step by step before running.
- **Pass**: running at most 9; no preemption; `kv` the dominant blocked reason; queue time within 15%.

### E5: preemption, and what the watermark changes

- **Run**: `small-kv` and `small-kv-no-watermark`; exactly 64 requests at once (`--concurrency 64 --requests 64 --warmup 0`), 200-token prompts, 2,000 output tokens, run until all finish; `--timeout 120`, since preempted requests wait long between tokens.
- **Prediction by hand**: all 64 are admitted (13 blocks each) but outgrow 1,260 blocks about 105 tokens into their output, so both variants preempt; each preemption throws away 200 to about 2,200 tokens.
- **Prediction in detail**: the pure core (`Scheduler`, `RooflineCostModel`) run in virtual time on the same 64 requests at t = 0, giving exact preemption counts, thrown-away tokens, batch mean and finish times for each variant. This checks that the driver, server and bench reproduce the scheduler's rules under heavy preemption; the rules themselves are covered by unit tests.
- **Pass**: preemptions and thrown-away tokens within 15% of the virtual run, the watermark variant lower, tokens per preemption within the hand bounds, batch mean and finish time within 5%.

### E7: the scheduler's fidelity envelope

- **Run**: open-loop sweep on `base` at 10, 25, 50, 75 and 100 requests per second; 512-token prompts, 64 output tokens (the workload of the `fixed_rate` envelope, for comparison).
- **Measures, rather than predicts**: batch, token lag p99, step lag p99 and queue time at each rate; where lag p99 crosses 5 ms (expected near a batch of 25, from E0); where the simulated GPU saturates. Each 512-token prompt costs about 7 ms of compute-bound GPU time, so 100 requests per second would spend 70% of the GPU on prefill alone; saturation is expected between 50 and 100.
- **Output**: a "Scheduler fidelity envelope" table under Known limits in [architecture.md](architecture.md), beside the `fixed_rate` one.

## Recording a result

For each experiment, the "Scheduler validation" section of [architecture.md](architecture.md) records: the command, the prediction, a table of predicted against measured values, the verdict, and anything the prediction missed with its explanation, marked inferred or confirmed. The status table above is updated with it.
