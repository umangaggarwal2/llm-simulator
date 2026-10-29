# Architecture

llm-simulator serves an OpenAI-compatible chat API backed by a deterministic fake model. It separates two questions that a real LLM server answers together:

- **What** does the model say? Answered by the **generator**, from `config/responses.yml`.
- **When** does each token arrive? Answered by the **engine**, from the model and hardware configuration.

Keeping them apart lets the response content stay simple and deterministic while the timing model grows from fixed delays to a full scheduler with batching and KV-cache pressure, without either side changing the other.

## Status

| Component | Module | Status |
|---|---|---|
| OpenAI-compatible API | `client/api.py`, `client/schemas.py`, `client/sse.py`, `client/convert.py`, `client/exception.py` | Done; serves responses from `responses.yml` |
| Arrival middleware and runtime dependency | `client/middleware.py`, `client/dependencies.py` | Done |
| Model deployments and startup wiring | `llm/models.py`, `config/models.yml`, `runtime.py`, `settings.py` | Done |
| Response schema and loader | `llm/responses.py`, [schema.md](schema.md) | Done |
| Tokenizer | `llm/tokenizer.py` | Done (`RegexTokenizer`) |
| Matcher | `llm/matcher.py` | Done |
| Shared types and domain errors | `types.py`, `errors.py` | Done |
| Generator | `llm/generator.py` | Done |
| Engine interface | `simulator/engine.py` | Done |
| `InstantEngine` | `simulator/instant.py` | Done |
| `FixedRateEngine` | `simulator/fixed_rate.py`, `simulator/_timing.py` | Done |
| Metrics: tracing, recorder, `/metrics` | `metrics/`, `client/api.py` | Done |
| Metrics: `/metrics/summary` with sample windows | `metrics/summary.py`, `metrics/window.py` | Done |
| Load generator | `bench/` | Planned |
| `SchedulerEngine` | `simulator/scheduler/` | Planned |

## Layers and dependencies

```
client ──▶ runtime.py ──▶ llm        ──┐
                 │                     ├──▶  types.py
                 └──────▶ simulator  ──┘
```

- `client` handles HTTP only: it converts API models to domain types and domain events back into OpenAI responses.
- `llm` decides the response content. It never deals with time.
- `simulator` decides when tokens are emitted. It never deals with response content.
- `runtime.py` is the composition root: the one module that assembles every layer, much like a Spring application context but with explicit construction. It reads the parsed `models.yml` and `responses.yml` and passes the generator and each engine plain parameters, so components never depend on configuration classes. The app builds it once at startup (singleton scope), handlers receive it through `Depends(get_runtimes)`, and the lifespan calls each engine's `start` and `stop`.
- `llm/models.py` holds the `models.yml` schema, including the hardware and serving sections. Only `runtime.py` reads it, so the simulator package never imports from `llm`.
- `types.py` holds the types that cross layers (`GenerationRequest`, `GenerationPlan`, `TokenEvent`) and imports nothing from the project, so `llm` and `simulator` never need to import each other.
- Lower layers know nothing about HTTP. They raise domain errors (for example `ResponsesConfigError` or `ContextLengthExceeded`) and `client` maps them to OpenAI-style error responses.

## Request flow

```
                    ┌──────────────────── startup (once) ──────────────────────┐
                    │ load responses.yml → ResponsesConfig (validated, frozen) │
                    │ load models.yml    → ModelsConfig                        │
                    │ build one Matcher, then per model: Tokenizer, Generator, │
                    │ Engine → app.state.runtimes                              │
                    │ lifespan: engine.start() for every model                 │
                    └──────────────────────────────────────────────────────────┘
HTTP request
    ▼
client/middleware.py ── ArrivalMiddleware: request.state.arrival = time.monotonic()
    ▼
client/api.py ── runtime = runtimes[body.model]        (Depends(get_runtimes); unknown → 404)
              ── ChatCompletionRequest → GenerationRequest   (client/convert.py)
    ▼  in a worker thread
llm/generator.py
    prompt_tokens  = tokenizer.count_messages(messages)
    result         = matcher.match(last user message)
    variant        = body.choices[sha256(seed, entry_id, input) mod n]
    tokens         = tokenizer.encode(render(variant))   or synthetic tokens
    tokens, reason = truncate(tokens, max_tokens)
    ▼  GenerationPlan(prompt_tokens, tokens, finish_reason, entry_id)
simulator/ (engine for the model)
    async for event in engine.run(plan, arrival)
    ▼  TokenEvent(index, text, sim_time)
client/sse.py ── SSE chunks            client/api.py ── one ChatCompletionResponse with Usage
```

Configuration is loaded when the app is created, not when a module is imported: `main()` runs uvicorn in factory mode (`create_app` is called at server start). A broken config stops the server before it accepts traffic, and tests build apps from small test configs with `create_app(Settings(...))`. File paths, host and port come from environment variables (`LLM_SIM_MODELS`, `LLM_SIM_RESPONSES`, `LLM_SIM_HOST`, `LLM_SIM_PORT`).

## HTTP layer

- **Arrival** is recorded by `ArrivalMiddleware`, a plain ASGI middleware that runs when the request's headers are parsed, before the body is read and validated. That is where a client's TTFT clock starts. `BaseHTTPMiddleware` (`@app.middleware("http")`) is avoided because it routes every streamed chunk through an extra buffer. Middleware is only used for this; LLM metrics will be recorded by the engine and the SSE stream, not by middleware.
- **Runtimes** are built once at startup, stored on `app.state`, and reach handlers through the `get_runtimes` dependency. Tests can replace them with `app.dependency_overrides[get_runtimes]`.
- **Errors before streaming.** The plan is generated before any response is sent, so an over-long request returns a proper `400` instead of breaking a stream that has already sent `200`. Generation runs in a worker thread so that slow matching does not stall other streams.
- **Pacing.** The SSE stream pulls each token from `engine.run()`, so a chunk is written only when the engine releases it; the wiring is the same for every engine. Non-streaming requests also wait for every token, so the response arrives when the simulated generation would finish. The stream sets `Cache-Control: no-cache` and `X-Accel-Buffering: no`, and no compression middleware is used, since either would buffer chunks.
- **Disconnects.** The engine's iterator is wrapped in `contextlib.aclosing`, so its cleanup runs when a client disconnects mid-stream.
- **OpenAI behaviour.** `model` is required. `seed`, `max_completion_tokens` (which takes precedence over the deprecated `max_tokens`) and `stream_options.include_usage` are supported; with `include_usage`, every chunk carries `"usage": null` and a final chunk with empty `choices` carries the totals. The stream starts with a `role: assistant` chunk. Parameters the simulator does not use, such as `temperature`, are accepted and ignored. `/v1/models` lists every configured model.
- **Message content.** A request message's `content` may be a string or an array of content parts, as in OpenAI's API. The schema (`ChatMessageParam`) turns a string into one text part before validation, so errors point at paths like `messages.[0].content.[1].type` rather than at each member of a `str | list` union; responses use a separate `ChatMessage` whose content is always a string. The schema accepts every part type OpenAI defines, and `to_generation_request` decides what the simulator serves: text parts are joined with no separator into the domain `Message`, so the tokenizer, matcher and generator see plain text, and any other part (`image_url`, `input_audio`, `file`, `refusal`) is rejected with a 400 naming the part, before anything is generated or traced. Rejecting rather than dropping them keeps prompt token counts honest.

## Model deployments (`config/models.yml`)

Each entry in `models` is a model deployed on a hardware profile, with its serving settings. Hardware profiles are named once under `hardware` and referenced by name; every model that references a profile gets its own simulated GPU of that type.

| Field | Belongs to | Used by |
|---|---|---|
| `name`, `context_length`, `tokenizer` | the model | generator, API |
| `kv_cache.bytes_per_token` | the model | scheduler |
| `hardware` (a profile name) | the GPU type | scheduler |
| `engine`, `performance`, `scheduler`, `latency` | this deployment | engines |

Sections are optional in the schema. Each engine declares the sections it needs (`Engine.required_sections`), and `build_runtimes` rejects a model whose engine needs a missing section. Unknown keys, duplicate names, unknown hardware references and unknown tokenizer or engine names all stop startup with an error naming the model.

## Components

### Tokenizer (`llm/tokenizer.py`)

Every layer measures in tokens: the generator truncates at `max_tokens`, the API reports `usage`, and the engine costs prefill and KV memory by token count. One shared tokenizer keeps those counts consistent; if the engine counted tokens differently from `usage`, simulated metrics would disagree with what clients see.

- `Tokenizer` is an abstract base class. Implementations provide `encode(text) -> list[str]`; `count` and `count_messages` have default implementations.
- Tokens are text pieces, not integer ids. Every consumer needs either the text (to stream it) or the count.
- Contract: encoding is **lossless** (`"".join(encode(t)) == t`) and **deterministic**.
- `count_messages` adds chat-format overhead following OpenAI's counting recipe: 3 tokens per message plus 3 to prime the reply.
- `RegexTokenizer` approximates GPT-2-style pre-tokenization: words with a leading space, punctuation runs, contractions, and numbers in groups of up to three digits. CJK, kana and Hangul are split one character per token. Long or rare words stay whole, so counts run somewhat below a real BPE tokenizer.
- `get_tokenizer(name)` builds a tokenizer from its config name. A `tiktoken` adapter can be added behind the same interface when exact counts matter.

### Response configuration (`llm/responses.py`)

Validates `responses.yml` against the schema in [schema.md](schema.md) and returns frozen Pydantic models.

- **Fail fast.** Every error (unknown keys, bad regexes, unknown template placeholders, duplicate ids, invalid ranges) is reported at startup, all at once, with the entry id. A mistake in a hand-edited file never surfaces as a 500 during a load test.
- **Normalized shapes.** Shorthand is expanded at load time: every body exposes `choices` as a tuple of variants, single match values become tuples, a fixed synthetic count becomes a `(n, n)` range, and regexes are precompiled with `re.IGNORECASE`.
- **Safe templates.** Only plain `{name}` placeholders are allowed. Templates are never passed to `str.format`, which could reach object attributes.

### Matcher (`llm/matcher.py`)

Finds the first `responses.yml` entry that matches the input and returns `MatchResult(entry_id, body, groups)`.

- Takes a plain string. The generator extracts the last user message once, trims leading and trailing whitespace for matching and hashing, and passes the untrimmed text to the `{last_user_message}` variable.
- Each entry is compiled once into a rule with a test function, built by a per-operator builder. `match()` casefolds the input once and loops over the rules with no per-operator branching.
- Regexes run against the original text, so captured groups keep the user's casing.
- Falls back to `default` (`entry_id == "default"`) when nothing matches.

**Known cost.** Matching is linear in input length × number of entries. Against the sample config (194 regex entries), an unmatched 120 KB prompt takes about 380 ms. Arrival-anchored timing (below) absorbs most of this, and generation can run off the event loop. A result cache keyed on the input, or a cap on how much input is matched, are options if measurements still require them.

### Generator (`llm/generator.py`)

Turns a `GenerationRequest` into a `GenerationPlan`. It is pure, synchronous and deterministic, and knows nothing about time.

```python
Generator(matcher, tokenizer, context_length, seed).generate(request) -> GenerationPlan
```

One generator is built per model, because the tokenizer and `context_length` depend on the model. The matcher is built once from `responses.yml` and shared by every model's generator.

- **Variant selection.** `sha256(seed, entry_id, input)`; bytes 0–7 of the digest, `mod len(choices)`, pick the variant. `seed` is the request's `seed` if given, otherwise the file's. `sha256` is used because Python's `hash()` is randomized per process. The exact algorithm is in [schema.md](schema.md#evaluation).
- **Templates.** Each template is parsed once at load (`parse_template` in `responses.py`) and stored on its `Variant`; validation checks the parsed parts and the generator renders them by joining. Values are inserted as-is, so user text containing braces is never expanded.
- **Synthetic responses produce tokens directly.** The corpus is tokenized once and its tokens are repeated to exactly N, so `tokens: 500` means 500 tokens under any tokenizer. A range's length comes from bytes 8–15 of the same digest.
- **Truncation.** Output is cut at `max_tokens`, with `finish_reason` set to `length`; otherwise `stop`.
- **Context length.** Follows OpenAI, against the selected model's `context_length`: a request is rejected when `prompt_tokens + max_tokens` exceeds it, or, without `max_tokens`, when the prompt alone fills it; otherwise the completion may use the remaining context. Rejection raises `ContextLengthExceeded` (in `errors.py`), which the API maps to `400 context_length_exceeded`. The check runs before matching, so rejected requests cost nothing more.

The API request model still needs a `seed` field. `GenerationRequest` carries the model name for the `{model}` placeholder.

## Timing model

### Arrival-anchored deadlines

Clients measure TTFT from when they send the request. On the server that interval contains real overhead (parsing, validation, matching, tokenizing) followed by the simulated delay.

Sleeping for relative durations (`sleep(ttft)`, then `sleep(itl)` per token) has two problems:

1. **Overhead leaks into metrics.** Observed TTFT becomes overhead plus simulated TTFT, and overhead varies from well under a millisecond to hundreds of milliseconds depending on the prompt.
2. **Errors accumulate.** Each sleep wakes slightly late. Over thousands of tokens the lateness adds up and measured throughput falls below the configured rate.

Instead, the API records `arrival` as early as possible and the engine sleeps until absolute deadlines:

```python
deadline_i = arrival + ttft + i * itl
await asyncio.sleep(max(0, deadline_i - time.monotonic()))
```

- Overhead is absorbed whenever it is shorter than the simulated delay before the first token.
- Lateness never accumulates: every token targets its own deadline.
- If a deadline has already passed, the token is sent immediately and the engine records **lag** (`now - deadline`). Non-zero lag means the measurements reflect the simulator's own limits rather than the configured hardware.

The engine owns this because only the engine knows the simulated schedule. The generator and the plan stay free of timing.

### Lag

Each `TokenEvent` carries `lag = max(0, release time − deadline)`: how late the engine released the token against its own timeline. It measures how faithfully the simulator carried out the configured strategy, not anything about the simulated hardware. Real servers have jitter too; if that is ever wanted, it belongs in the simulated timeline (seeded and reproducible), not in lag.

Token *i* is released at `arrival + sim_time(i) + lag(i)`, so, network aside:

```
observed TTFT     = simulated TTFT + lag(0)
observed ITL(i)   = simulated ITL  + (lag(i) − lag(i−1))      only changes in lag matter
observed E2E      = simulated E2E  + lag(n−1)                 the last lag, not the sum
```

| Source of lag | Typical size | Meaning |
|---|---|---|
| Timer granularity (asyncio waits in whole milliseconds on Linux) | about 1 ms | Normal |
| Event loop busy with other streams, or generation holding the GIL | 1–10 ms | Approaching the process's limits |
| A slow client (backpressure at the engine's `yield`) | any | Client-side; overdue tokens are then sent at once, keeping total time correct |
| More tokens per second than one process can deliver | grows | Measurements no longer reflect the configured hardware |
| Request overhead longer than the simulated TTFT | large on token 0 only | The overhead did not fit inside the TTFT |

Measured on a development machine with 250 tokens at a 4 ms ITL, wake-up lateness was about 0.8–1.0 ms at p50 and under 2.6 ms at worst for 1 to 200 concurrent streams, while end-to-end time stayed within 1.5 ms of the target. Individual gaps between tokens therefore vary by about ±1 ms, but mean ITL, TPOT and throughput are accurate. This jitter is accepted; waking early and spinning could reduce it at the cost of CPU. Lag will feed the metrics layer as a validity check for experiments.

### Where time is measured

For each token there are four possible points to measure, and each layer of the system measures its own:

```
deadline ──▶ released by engine ──▶ written to the socket ──▶ received by the client
 (sim_time)    (+ lag)                (+ serialization,          (+ network,
                                        transport wait)            client parsing)
```

| Layer | Measures | Tells you |
|---|---|---|
| Engine | **Simulated**: `sim_time` | What the configured model intended |
| Server metrics (`/metrics`) | **Server-observed**: release time, plus lag | Whether the simulator carried out its model faithfully |
| Load generator (`llmsim-bench`) | **Client-observed**: receive time | What a client experiences, including transport and network |

Server metrics use the release time rather than the socket write: it means the same for streaming and non-streaming requests (a non-streaming response is written once, at the end), it matches how serving engines such as vLLM time tokens, and release = arrival + `sim_time` + `lag`, so the lag histogram explains all of the difference between simulated and server-observed values. A slow client still shows up, because backpressure makes later tokens late. Transport and network costs are what separates server-observed from client-observed values; the load generator reports that gap.

### Engines

All engines implement the same interface:

```python
class Engine(ABC):
    def run(self, plan: GenerationPlan, arrival: float) -> AsyncIterator[TokenEvent]: ...
    async def start(self) -> None: ...   # no-op by default
    async def stop(self) -> None: ...    # no-op by default
```

`TokenEvent(index, text, sim_time, lag)` carries only timing information; `finish_reason` and `usage` come from the plan, which the API already holds. `sim_time` is in simulated seconds after arrival, so token 0's value is the simulated TTFT and the gaps between tokens are the ITLs. `start` and `stop` are called from the app's lifespan; they exist for engines with a background loop, such as the scheduler. The `simulator` package is organised like a Java package with one interface and several implementations: `engine.py` holds only the `Engine` base class, each engine has its own module (`instant.py`, `fixed_rate.py`, and a `scheduler/` sub-package to come), `_timing.py` holds `sleep_until`, which every timed engine shares, and `__init__.py` re-exports the public names, so callers write `from llm_simulator.simulator import FixedRateEngine`. The base module never imports its implementations, and the package never lists them; choosing one by name happens in `runtime.py`.

Engines take plain parameters (for example `FixedRateEngine(prefill_tokens_per_second, decode_tokens_per_second, scheduler_overhead_s)`), never configuration classes. `runtime.py` holds a builder per engine name (`ENGINES`) that maps a model's `models.yml` sections to those parameters. Each engine class declares `required_sections`, which are checked before its builder runs. The engine is chosen per model with `engine:` in `models.yml`.

When a client disconnects, the API closes the iterator and the engine releases the sequence in a `finally:` block. This frees nothing in the simpler engines but frees KV-cache capacity in the scheduler.

| Engine | Timing | Shared state | Purpose |
|---|---|---|---|
| `InstantEngine` | None; tokens are yielded immediately | None | Wiring and functional tests |
| `FixedRateEngine` | `ttft = overhead + prompt_tokens / prefill_tps`, `itl = 1 / decode_tps`; overhead is paid once per request | None; requests are independent | Simple baseline with configurable latency |
| `SchedulerEngine` | One shared step loop; each step's duration depends on the batch | Waiting queue, running batch, KV blocks | Load-dependent behaviour: queueing, batching, KV pressure |
| Virtual-time mode (possible later) | The scheduler run as a discrete-event simulation, without real sleeps | Same as `SchedulerEngine` | Capacity planning beyond what can be streamed in real time |

Each later engine changes how the simulated timeline is computed. Arrival-anchoring stays the same.

### Mapping to target metrics

For one request in `FixedRateEngine`, with prefill at 100,000 tok/s, decode at 250 tok/s, 2 ms scheduler overhead, 1,000 prompt tokens and 200 output tokens:

| Metric | Definition | Value |
|---|---|---|
| Queue latency | admitted − arrival | 0 (nothing queues) |
| Prefill latency | prompt_tokens / prefill_tps | 10 ms |
| TTFT | queue + overhead + prefill (the prefill pass produces the first token) | 12 ms |
| ITL | one decode step | 4 ms |
| Decode latency | (n − 1) × ITL | 796 ms |
| E2E latency | TTFT + decode latency | 808 ms |
| TPOT | (E2E − TTFT) / (n − 1) | 4 ms |

In `SchedulerEngine` the same metrics come out of the step loop, and they change with load.

## Scheduler and KV-cache model

KV memory is a shared, contended resource, so it only affects latency where requests compete. `FixedRateEngine` has no shared state and therefore cannot model it; the scheduler can.

**Capacity, computed at startup:**

```
kv_capacity_tokens = (hbm × gpu_memory_utilization − weight_bytes) / kv_bytes_per_token
kv_blocks          = kv_capacity_tokens / block_size
```

**Each step, in the style of vLLM:**

1. **Admit** waiting requests in arrival order while there are free blocks for the prompt (plus headroom) and fewer than `max_num_sequences` are running. If the head of the queue does not fit, it waits.
2. **Batch.** Each decoding sequence contributes one token; prefills fill the rest of `max_batch_tokens`.
3. **Grow.** A decoding sequence needs a new block every `block_size` tokens. If none is free, the newest sequence is preempted: its blocks are freed and it is requeued for recomputation.
4. **Duration** follows a roofline model: `max(compute, memory) + overhead`, where
   - `compute = 2 × params × scheduled_tokens / peak_flops`
   - `memory = (weight_bytes + Σ context_len × kv_bytes_per_token) / bandwidth`
5. **Emit.** Sleep until the step's absolute deadline, then deliver one token to every decoding sequence.

The step loop anchors its own clock (each step ends at the previous step's end plus this step's duration). Each request's arrival time is used for its queue latency and TTFT.

**How KV memory shows up in client-visible metrics:**

| KV effect | Real-world symptom | Simulated metric |
|---|---|---|
| Not enough free blocks at admission | Requests wait | Queue latency and TTFT rise |
| Blocks run out during decode | Preemption and recomputation | ITL spikes and higher E2E for the preempted sequence |
| More context in the batch | More bytes read per step | ITL rises for every sequence in the batch |

As an example, with 80 GB of HBM, 3,000 GB/s of bandwidth, 512 KiB of KV per token, 16 GB of weights and 90% memory utilization, KV capacity is about 107,000 tokens (roughly 26 concurrent 4k-context requests). A single short request decodes at about 5.3 ms per token; 26 requests at 4k context each read about 72 GB per step, so ITL rises to about 24 ms. Decode is memory-bound: compute takes well under a millisecond.

**Configuration the scheduler needs** beyond the README's example: model weight size (parameters and dtype, or `weights_gb`), `peak_tflops`, `block_size` (typically 16) and `gpu_memory_utilization` (typically 0.9).

**Status: designed, parked.** The metrics layer comes first, so the end-to-end product can be measured with the existing engines; the scheduler is the final piece that brings the metrics to full fidelity.

**Code layout.** Unlike the other engines, the scheduler is several separable parts, and only one of them deals with time. It is a sub-package whose pure core never imports `asyncio` or `time`:

```
simulator/scheduler/
  __init__.py      exports SchedulerEngine only
  kv_cache.py      BlockAllocator: capacity in blocks; allocate, grow and free a sequence's blocks
  cost_model.py    RooflineCostModel: step_duration(batch)
  policy.py        Scheduler: waiting and running queues, admission, batch building,
                   preemption; returns a StepPlan for the next step
  realtime.py      SchedulerEngine(Engine): the asyncio driver; step loop with sleep_until,
                   a queue per request, start/stop, frees blocks when a client disconnects
  virtual.py       later: a discrete-event driver for capacity planning, reusing the same core
```

- Dependencies point inward only: `realtime.py` → `policy.py` → `kv_cache.py`, `cost_model.py`.
- The core is tested step by step with no clock (admission, batching, preemption, KV accounting); only the driver needs the fake clock.
- The virtual-time mode reuses the policy, allocator and cost model with a different driver, so the two modes cannot drift apart.
- The driver is `realtime.py` rather than a second `engine.py`, so the interface module and an implementation never share a name.

## Metrics

```
 per request (short-lived)        per app (long-lived)                           storage
┌──────────────────────┐  hooks  ┌───────────────────────┐        ┌────────────────────┐   ┌──────────────────────────────┐
│ RequestTrace         │ ──────▶ │ MetricsRecorder (ABC) │        │ PrometheusRecorder │──▶│ CollectorRegistry            │──▶ GET /metrics
│ arrival, first/prev/ │         │  FanOutRecorder ──────┼──┬────▶│                    │   │ counters, gauges, histograms │
│ last release, count  │         │  NoopRecorder         │  │     └────────────────────┘   └──────────────────────────────┘
└──────────────────────┘         └───────────────────────┘  │     ┌────────────────────┐   ┌──────────────────────────────┐
                                                            └────▶│ SummaryRecorder    │──▶│ SampleWindows, RateWindows   │──▶ GET /metrics/summary
                                                                  └────────────────────┘   └──────────────────────────────┘
```

- **Tracing.** A `RequestTrace` follows one request. It keeps only running state (arrival, the first, previous and last release times, a token count), so its memory is constant however long the completion. It is owned by one request and needs no locking. The engine's event stream is wrapped so that every released token passes through the trace; engines know nothing about metrics.
- **When values are recorded.** Prompt tokens and `requests_running` when the request starts; TTFT and queue time at the first token; ITL and lag for every token; E2E, TPOT, prefill and decode time, request sizes and `requests_total{finish_reason}` when it ends. Prefill = first release − arrival − queue time; decode = last release − first release.
- **Cancelled requests.** TTFT, ITL, lag and token counters include them, because those tokens were produced. E2E, TPOT, prefill, decode and request-size distributions include only completed requests (`stop` or `length`); otherwise disconnects would pull them down. Cancellations are counted as `requests_total{finish_reason="cancelled"}`, and failures as `"error"`.
- **Recording.** `MetricsRecorder` is an interface, so the trace never touches a metrics library. `PrometheusRecorder` and `SummaryRecorder` are the two stores; `FanOutRecorder` forwards every call to both, so a trace reports once; `NoopRecorder` turns metrics off. There is one recorder per app, built in the composition root and given to handlers through a dependency. Its per-model label children are looked up once, at startup: a cached histogram observation costs about 0.8 µs against 1.7 µs with a label lookup, and every token makes two (ITL and lag). Token counters are incremented once per request.
- **Storage for `/metrics`.** Each app has its own `CollectorRegistry` in process memory, holding only current values: one number per counter or gauge, and bucket counts plus a sum and count per histogram, per model. Memory is constant regardless of traffic. History is not stored by the simulator: a Prometheus server scrapes `/metrics` and keeps the time series, and throughput comes from `rate()` over the counters, which also handles counters restarting from zero after a restart. A separate registry per app keeps tests and multiple apps from sharing or double-registering metrics.
- **Storage for `/metrics/summary`.** Percentiles estimated from histogram buckets are only accurate to about a bucket's width, which is too coarse for millisecond fidelity checks. `SummaryRecorder` keeps bounded sample windows instead: per model and metric, a `SampleWindow` ring buffer of the most recent raw values, giving exact percentiles over recent traffic with fixed memory.
  - **Two window sizes.** Per-request metrics (TTFT, E2E, TPOT, queue, prefill, decode, prompt and generation sizes) add one value per request and keep the last 10,000 requests. Per-token metrics (ITL, lag) add one value per token, pooled across requests, and keep the last 100,000 tokens: at about 6,000 tok/s, roughly the last 15–20 s. Both are settable (`LLM_SIM_SUMMARY_REQUEST_SAMPLES`, `LLM_SIM_SUMMARY_TOKEN_SAMPLES`).
  - **Ring buffer.** Values are stored in an `array('d')`: 8 bytes each, growing until full and then overwriting the oldest, so a model with no traffic uses almost nothing and a busy one at most about 2.2 MB. Each window also counts every value ever added, so the summary shows both `count` (recorded) and `samples` (still held). Recording a token in this store costs about 0.6 µs, against about 1.8 µs for Prometheus.
  - **Throughput.** A `RateWindow` per model counts generated tokens and finished requests in one-second buckets over the last 60 s (`LLM_SIM_SUMMARY_RATE_SECONDS`). Each bucket remembers its second, so stale ones are zeroed on reuse without a timer. Just after startup or a reset, the rate is taken over the time since, reported as `window_seconds`, with a floor of one second.
  - **Reading.** `snapshot()` copies every window on the event loop (about 0.5 ms with full windows, since each is a single memory copy), so recording and reading never interleave and no lock is needed. `summarize()` then sorts the copies and computes mean, min, p50, p90, p95, p99 and max, interpolating between ranks as numpy does; it runs in a worker thread (about 4.4 ms with full windows) so it does not delay token releases.
  - **Reset.** `POST /metrics/summary/reset` clears every window, rate and count, and restarts `since_reset_seconds`, so a benchmark can measure one run. `requests.running` is kept, since those requests are still in flight; a request in flight across a reset is recorded in the new period when it finishes, so reset while idle for a clean measurement. Prometheus counters are not reset: they must only go up.
- **Multiple workers.** With several worker processes, each has its own registry. `prometheus_client`'s multiprocess mode can be added if that is ever needed.

**Metric catalogue** (every metric is labelled with `model`):

| Type | Name | Records |
|---|---|---|
| Histogram | `llm_sim_time_to_first_token_seconds` | Server-observed TTFT |
| Histogram | `llm_sim_inter_token_latency_seconds` | Gap between consecutive tokens |
| Histogram | `llm_sim_request_time_per_output_token_seconds` | TPOT of completed requests |
| Histogram | `llm_sim_e2e_request_latency_seconds` | E2E of completed requests |
| Histogram | `llm_sim_request_queue_time_seconds` | Queue time (0 until the scheduler) |
| Histogram | `llm_sim_request_prefill_time_seconds`, `llm_sim_request_decode_time_seconds` | Derived as above |
| Histogram | `llm_sim_request_prompt_tokens`, `llm_sim_request_generation_tokens` | Sizes of completed requests |
| Histogram | `llm_sim_token_lag_seconds` | Fidelity: how late each token was released |
| Counter | `llm_sim_requests_total{finish_reason}` | `stop`, `length`, `cancelled`, `error` |
| Counter | `llm_sim_prompt_tokens_total`, `llm_sim_generation_tokens_total` | Throughput, via `rate()` |
| Gauge | `llm_sim_requests_running` | Active sequences |
| Info | `llm_sim_model_info{engine, context_length}` | Which engine each model runs |
| Later | `llm_sim_requests_waiting`, `llm_sim_kv_cache_usage_ratio`, `llm_sim_batch_size`, … | Read from the scheduler engine when metrics are scraped |

Names follow vLLM's, with an `llm_sim_` prefix and underscores instead of vLLM's colons, which Prometheus reserves for recording rules. Histogram buckets are spaced logarithmically, with ranges suited to each metric (lag goes down to 10 µs so that timer jitter is visible).

## Real-time delivery budget

Every real-time engine pays about one coroutine resume, one serialization and one socket write per token per stream; anchoring does not add to this. Measured in-process with 256 streams and no HTTP, a step loop delivers about 130,000 tokens/s when chunks are serialized through the Pydantic models and about 370,000 tokens/s with a prebuilt SSE template. HTTP framing and socket writes reduce this considerably.

**Measured end to end** (one server process, OpenAI SDK clients on the same 16-core machine, about 27 tokens per response, server-side values read from `/metrics`):

| Model, engine | Concurrent streams | TTFT | ITL | Lag mean / p99 | Delivered |
|---|---|---|---|---|---|
| `qwen-mock`, `fixed_rate` (configured: TTFT ≈ 2.1 ms, ITL 4 ms) | 1 | 2.6 ms | 4.00 ms | 0.8 / ≤ 1.3 ms | 244 tok/s |
| | 10 | 6.7 ms | 3.85 ms | 0.8 / ≤ 6.6 ms | 1,610 tok/s |
| | 50 | 29.6 ms | 2.97 ms | 5.7 / ≤ 50 ms | 5,843 tok/s |
| | 200 | 185 ms | 0.14 ms | 129 / ≤ 253 ms | 5,353 tok/s |
| `tiny-mock`, `instant` | 200 | 156 ms | 0.02 ms | 157 / ≤ 253 ms | 5,856 tok/s |

One process currently delivers about **5,500–6,000 tokens/s**. Producing tokens is not the limit: measured in-process for 200 streams, the engine releases a token in about 0.7 µs, tracing and recording add about 2.4 µs and SSE serialization about 6 µs, about 9 µs in all, against roughly 170 µs per token over HTTP. Most of the cost is delivering each chunk over HTTP. Because the stream is pulled by the writer, the engine releases the next token only after the previous chunk has been sent, so delivery cost appears as lag. By default uvicorn runs on the pure-Python `h11` parser and the standard asyncio loop; with `httptools` and `uvloop` installed (`uvicorn[standard]`), mean lag at 200 streams fell from 122 ms to 18.6 ms and mean TTFT from 183 ms to 56 ms. Clean throughput figures need the load generator: an ad-hoc test that started client processes inside its timing window measured start-up time rather than throughput. Up to that rate the simulator follows its configured timeline closely; beyond it, requests queue behind the event loop, TTFT grows, overdue tokens go out in bursts (ITL falls below the configured value), and lag shows it. For `qwen-mock`'s 250 tok/s per stream, that is roughly 20 concurrent streams per process. Raising the ceiling is the next performance task.

A process only needs to deliver tokens as fast as the GPU it simulates produces them, which is typically a few thousand to low tens of thousands of output tokens/s, so one process can simulate one replica. To stay within budget:

| Technique | Effect | Fidelity cost |
|---|---|---|
| One step loop for all streams | One timer per step; tokens are handed over with `queue.put_nowait` | None; real servers emit per step too |
| Prebuilt SSE chunk templates | About 2.8× cheaper per token than Pydantic serialization | None |
| uvloop | Faster event loop | None |
| Merge overdue tokens into one chunk | Bounded work when behind | Total time stays correct; ITL detail is lost only while overloaded |
| Multiple worker processes | Each simulates an independent replica | Matches data-parallel deployments; one GPU cannot be shared across processes |
| Virtual-time mode | No real sleeps | Results only; no live streaming |

## Open decisions

- **Step-duration model.** Roofline (recommended for `SchedulerEngine`), the empirical rates from the README (kept for `FixedRateEngine`), or roofline scaled by an efficiency factor calibrated against a real measurement.

## Build order

1. Metrics layer: tracing, recorders, Prometheus `/metrics` and `/metrics/summary` are done; next is the `llmsim-bench` load generator.
2. `SchedulerEngine` with KV blocks, admission, preemption and the roofline step model (designed above, parked until the metrics layer is in place).
