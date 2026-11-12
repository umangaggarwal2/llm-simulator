# Architecture

llm-simulator serves an OpenAI-compatible chat API backed by a deterministic fake model. It separates two questions that a real LLM server answers together:

- **What** does the model say? Answered by the **generator**, from `responses.yml` (built-in by default, `src/llm_simulator/defaults/`).
- **When** does each token arrive? Answered by the **engine**, from the model and hardware configuration.

Keeping them apart lets the response content stay simple and deterministic while the timing model grows from fixed delays to a full scheduler with batching and KV-cache pressure, without either side changing the other.

## Status

| Component | Module | Status |
|---|---|---|
| OpenAI-compatible API | `client/api.py`, `client/schemas.py`, `client/sse.py`, `client/convert.py`, `client/exception.py` | Done; serves responses from `responses.yml` |
| Arrival middleware and runtime dependency | `client/middleware.py`, `client/dependencies.py` | Done |
| Model deployments and startup wiring | `llm/models.py`, `defaults/models.yml`, `runtime.py`, `settings.py` | Done |
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
| Load generator: client, closed and open loop, workload, phases, client health, server comparison, sweeps | `bench/`, `defaults/bench-responses.yml` | Done |
| Docker images and compose (simulator, benchmark) | `docker/`, `.dockerignore` | Done |
| `SchedulerEngine` | `simulator/scheduler/` | Done: tiers 1–3, wiring (`qwen-mock-batched`) and step metrics; validation with `llmsim-bench` (S6) next |

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

Configuration is loaded when the app is created, not when a module is imported. `main()` parses the command-line flags, builds the settings (`Settings.from_args`: each flag given overrides its environment variable, `LLM_SIM_MODELS`, `LLM_SIM_RESPONSES`, `LLM_SIM_HOST`, `LLM_SIM_PORT`), calls `create_app` and only then starts uvicorn with the app. Building the app before uvicorn starts lets the flags reach it, and a broken or missing config stops the process with its message and exit code 1, before it accepts traffic. Without a file named by flag or variable, the settings point at the built-in configs installed with the package (`llm_simulator/defaults/`, looked up with `importlib.resources`); any file value may be `builtin:<name>` instead of a path, resolved in one place, `defaults.resolve`. Tests build apps from small test configs with `create_app(Settings(...))`, and `uvicorn llm_simulator.client.api:create_app --factory` still works for running uvicorn directly, configured from the environment.

## HTTP layer

- **Arrival** is recorded by `ArrivalMiddleware`, a plain ASGI middleware that runs when the request's headers are parsed, before the body is read and validated. That is where a client's TTFT clock starts. `BaseHTTPMiddleware` (`@app.middleware("http")`) is avoided because it routes every streamed chunk through an extra buffer. Middleware is only used for this; LLM metrics will be recorded by the engine and the SSE stream, not by middleware.
- **Runtimes** are built once at startup, stored on `app.state`, and reach handlers through the `get_runtimes` dependency. Tests can replace them with `app.dependency_overrides[get_runtimes]`.
- **Errors before streaming.** The plan is generated before any response is sent, so an over-long request returns a proper `400` instead of breaking a stream that has already sent `200`. Generation runs in a worker thread so that slow matching does not stall other streams.
- **Pacing.** The SSE stream pulls each token from `engine.run()`, so a chunk is written only when the engine releases it; the wiring is the same for every engine. Non-streaming requests also wait for every token, so the response arrives when the simulated generation would finish. The stream sets `Cache-Control: no-cache` and `X-Accel-Buffering: no`, and no compression middleware is used, since either would buffer chunks.
- **Disconnects.** The engine's iterator is wrapped in `contextlib.aclosing`, so its cleanup runs when a client disconnects mid-stream.
- **OpenAI behaviour.** `model` is required. `seed`, `max_completion_tokens` (which takes precedence over the deprecated `max_tokens`) and `stream_options.include_usage` are supported; with `include_usage`, every chunk carries `"usage": null` and a final chunk with empty `choices` carries the totals. The stream starts with a `role: assistant` chunk. Parameters the simulator does not use, such as `temperature`, are accepted and ignored. `/v1/models` lists every configured model.
- **Message content.** A request message's `content` may be a string or an array of content parts, as in OpenAI's API. The schema (`ChatMessageParam`) turns a string into one text part before validation, so errors point at paths like `messages.[0].content.[1].type` rather than at each member of a `str | list` union; responses use a separate `ChatMessage` whose content is always a string. The schema accepts every part type OpenAI defines, and `to_generation_request` decides what the simulator serves: text parts are joined with no separator into the domain `Message`, so the tokenizer, matcher and generator see plain text, and any other part (`image_url`, `input_audio`, `file`, `refusal`) is rejected with a 400 naming the part, before anything is generated or traced. Rejecting rather than dropping them keeps prompt token counts honest.

## Model deployments (`models.yml`)

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
- **Literal pre-filter** (`llm/prefilter.py`). At load time, each regex is parsed with Python's own regex parser (`sre_parse`, `re._parser` from 3.11) and walked for the plain strings every match must contain: literal runs of three or more ASCII characters, with lookaheads and required groups adding requirements (all of them) and alternations offering them (any branch). `(?=.*\bhamlet\b)(?=.*\b(quotes?|lines?)\b)` requires `hamlet` and one of `quote`, `line`. Per input, `match()` searches the casefolded text once for every distinct required string (115 for the sample config), then skips any rule whose requirements are not all present with a set test instead of running its regex. The analysis is conservative: anything it does not understand, including parser changes in a later Python, requires nothing, so a rule can be run needlessly but never skipped when it could match. One exception to substring tests is handled explicitly: `re.IGNORECASE` matches four non-ASCII characters (`İ ı ſ K`) to ASCII letters, and two of them casefold to something else, so text containing any of them is not filtered. A test checks that list against every code point, and another checks that the filter returns the same entry as unfiltered matching on thousands of generated prompts against the sample config.

**Known cost.** Matching is linear in input length. Regex searches dominate it: a lookahead pattern such as the one above backtracks over the whole input, about 45 µs per rule for a 3.3k-character prompt, while a substring test takes under 2 µs. Against the sample config (194 regex entries), the pre-filter cuts a 512-token prompt from 10.7 ms to 0.47 ms and an unmatched 120 KB prompt from 400 ms to 2 ms; short chat messages take under 0.1 ms either way. Matching runs in the threadpool but holds the GIL, so before the pre-filter it was most of the server's CPU under load with long prompts (about 88% of GIL time at 60 requests per second) and stalled the event loop. What remains is rules without requirements, which always run, and rules whose required string occurs inside other words (`lear` in "learn"), whose regex runs and then fails.

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

1. **Admit** waiting requests in arrival order (preempted ones first) while the whole prompt's blocks fit with a watermark of 1% of all blocks left free, and fewer than `max_num_sequences` are running. If the head of the queue does not fit, it waits. No request is admitted in a step that preempted one.
2. **Batch.** Running sequences go first, oldest first: a decoding sequence contributes one token, a sequence still in prefill its next chunk. Prefill chunks of newly admitted requests fill the rest of `max_batch_tokens`. Prefill is always chunked, so a prompt longer than the budget spreads over several steps while every other stream keeps decoding.
3. **Grow.** A sequence needs a new block every `block_size` tokens; growth may use the watermark. If no block is free, the newest admitted sequence is preempted: its blocks are freed and it goes back to the front of the waiting queue, to recompute its prompt and the tokens it already generated.
4. **Duration** is a roofline scaled by calibrated efficiencies: `max(compute, memory) + step_overhead`, where
   - `compute = 2 × params × scheduled_tokens / (peak_flops × compute_efficiency)`
   - `memory = (weight_bytes + Σ context_len × kv_bytes_per_token) / (bandwidth × memory_efficiency)`
   - `step_overhead` (`latency.step_overhead_ms`) is paid by every step, separately from `FixedRateEngine`'s per-request `scheduler_overhead_ms`. Both efficiencies default to 1.0.
5. **Emit.** Sleep until the step's absolute deadline, then deliver one token to every sequence whose known tokens are all computed.

Admission never uses a plan's true length, which a real server cannot know; the length is used only to stop the sequence, like an end-of-sequence token.

**The driver (`driver.py`).** The step loop anchors its own clock: each step ends at the previous step's end plus this step's duration, so a late wake-up shows as `lag` and never shifts the simulated timeline. Anchoring lets the loop fall behind real time, which has two consequences:

- Overdue steps do not sleep, so the loop yields (`await asyncio.sleep(0)`) after every step; otherwise catching up would starve every other coroutine.
- The next step can start, in simulated time, before a request that has already arrived. Requests therefore wait in an arrival queue and are handed to the policy only once the step clock has reached their arrival, so a request never joins a step that started before it arrived and queue time is never negative.

When the GPU is idle, the next step starts at the next request's arrival, not when the loop notices it, so time spent before the engine runs is absorbed as in `FixedRateEngine`. Each request has its own token queue and the loop never waits for clients; lag is measured when the consumer takes a token. Token 0 carries `queue_time` (first admission − arrival), which feeds the queue and prefill metrics. If the loop fails, every open and later request fails with `EngineFailed` instead of hanging.

**How KV memory shows up in client-visible metrics:**

| KV effect | Real-world symptom | Simulated metric |
|---|---|---|
| Not enough free blocks at admission | Requests wait | Queue latency and TTFT rise |
| Blocks run out during decode | Preemption and recomputation | ITL spikes and higher E2E for the preempted sequence |
| More context in the batch | More bytes read per step | ITL rises for every sequence in the batch |

As an example, with 80 GiB of HBM, 3,000 GB/s of bandwidth, 512 KiB of KV per token, a 6.74B-parameter fp16 model (13.48 GB of weights) and 90% memory utilization, KV capacity is 7,609 blocks of 16 tokens, about 121,700 tokens (roughly 29 concurrent 4k-context requests). A single short request decodes at about 4.5 ms per token; 29 requests at 4k context each read about 62 GB of KV per step on top of the weights, so ITL rises to about 25 ms. Decode is memory-bound: compute takes well under a millisecond.

**Units.** HBM is in GiB (`hbm_gib`, 2³⁰ bytes), as memory is built and as engines see it; bandwidth (`memory_bandwidth_gbps`) and `peak_tflops` are decimal, as vendors quote them; weights are `parameters_billions × 10⁹ × bytes_per_parameter` bytes.

**Configuration the scheduler needs** beyond `fixed_rate`'s: `peak_tflops` on the hardware profile, a `weights` section (`parameters_billions`, `bytes_per_parameter`, default 2), `kv_cache.block_size` (default 16), `gpu_memory_utilization` (default 0.9) and `watermark` (default 0.01), `latency.step_overhead_ms` (default 0) and an optional `efficiency` section (`compute`, `memory`, default 1.0). `runtime.py` derives the KV capacity, logs it, and refuses to start when `peak_tflops` is missing, the weights leave no room for KV, a full `context_length` does not fit the cache (as vLLM refuses), or `max_batch_tokens < max_num_sequences`.

**Status: designed; next to build.** It was parked until the metrics layer and load generator could measure the existing engines. Both are done, and so is the performance groundwork that gives one process the headroom recorded under Known limits.

**Code layout.** Unlike the other engines, the scheduler is several separable parts, and only one of them deals with time. It is a sub-package whose pure core never imports `asyncio` or `time`:

```
simulator/scheduler/
  __init__.py      exports SchedulerEngine only
  sequence.py      Sequence: a request's token bookkeeping (known, computed, generated);
                   one rule covers prefill, chunks, decode and recomputation
  kv_cache.py      kv_capacity_blocks; BlockAllocator: grow and free a sequence's blocks,
                   admission against the watermark
  cost_model.py    RooflineCostModel: step_cost(work) -> StepCost (compute, memory, overhead)
  policy.py        Scheduler: waiting and running queues; schedule() builds a StepPlan
                   (decodes, then prefill chunks, then admissions), complete() applies it
  driver.py        SchedulerEngine(Engine): the asyncio driver; step loop with sleep_until,
                   a queue per request, start/stop, frees blocks when a client disconnects
  virtual.py       later: a discrete-event driver for capacity planning, reusing the same core
```

- Three tiers, dependencies pointing down only: `driver.py` (time and I/O) → `policy.py` (decisions) → `sequence.py`, `kv_cache.py`, `cost_model.py` (resources and cost, independent of each other).
- The core is tested step by step with no clock (admission, batching, preemption, KV accounting); only the driver needs the fake clock.
- The virtual-time mode reuses the policy, allocator and cost model with a different driver, so the two modes cannot drift apart.
- The driver is `driver.py` rather than a second `engine.py`, so the interface module and an implementation never share a name. A virtual-time driver would sit beside it as `virtual.py`.

## Metrics

```
 sources                          per app (long-lived)                           storage
┌──────────────────────┐  request ┌───────────────────────┐        ┌────────────────────┐   ┌──────────────────────────────┐
│ RequestTrace         │  hooks   │ MetricsRecorder (ABC) │        │ PrometheusRecorder │──▶│ CollectorRegistry            │──▶ GET /metrics
│ one per request      │ ───────▶ │  FanOutRecorder ──────┼──┬────▶│                    │   │ counters, gauges, histograms │
└──────────────────────┘          │  NoopRecorder         │  │     └────────────────────┘   └──────────────────────────────┘
┌──────────────────────┐  step()  │                       │  │     ┌────────────────────┐   ┌──────────────────────────────┐
│ SchedulerEngine loop │ ───────▶ │                       │  └────▶│ SummaryRecorder    │──▶│ SampleWindows, RateWindows   │──▶ GET /metrics/summary
│ one per model        │ StepStats└───────────────────────┘        └────────────────────┘   └──────────────────────────────┘
└──────────────────────┘
```

There are two sources. A `RequestTrace` reports what one request experienced; the scheduler's step loop reports what the engine did, once per step, as a `StepStats` (in `types.py`) passed to an `on_step` callable. `runtime.py` binds that callable to `recorder.step` with the model's name, so the simulator never imports the metrics package. `step()` is a no-op by default: engines whose requests share nothing (`instant`, `fixed_rate`) have no steps to report. Both sources run on the event loop, so the recorders still need no locks.

- **Tracing.** A `RequestTrace` follows one request. It keeps only running state (arrival, the first, previous and last release times, a token count), so its memory is constant however long the completion. It is owned by one request and needs no locking. The engine's event stream is wrapped so that every released token passes through the trace; engines know nothing about metrics.
- **When values are recorded.** Prompt tokens and `requests_running` when the request starts; TTFT and queue time at the first token; ITL and lag for every token; E2E, TPOT, prefill and decode time, request sizes and `requests_total{finish_reason}` when it ends. Prefill = first release − arrival − queue time; decode = last release − first release.
- **Cancelled requests.** TTFT, ITL, lag and token counters include them, because those tokens were produced. E2E, TPOT, prefill, decode and request-size distributions include only completed requests (`stop` or `length`); otherwise disconnects would pull them down. Cancellations are counted as `requests_total{finish_reason="cancelled"}`, and failures as `"error"`.
- **Recording.** `MetricsRecorder` is an interface, so the trace never touches a metrics library. `PrometheusRecorder` and `SummaryRecorder` are the two stores; `FanOutRecorder` forwards every call to both, so a trace reports once; `NoopRecorder` turns metrics off. There is one recorder per app, built in the composition root and given to handlers through a dependency. Its per-model label children are looked up once, at startup: a cached histogram observation costs about 0.8 µs against 1.7 µs with a label lookup, and every token makes two (ITL and lag). Token counters are incremented once per request.
- **Storage for `/metrics`.** Each app has its own `CollectorRegistry` in process memory, holding only current values: one number per counter or gauge, and bucket counts plus a sum and count per histogram, per model. Memory is constant regardless of traffic. History is not stored by the simulator: a Prometheus server scrapes `/metrics` and keeps the time series, and throughput comes from `rate()` over the counters, which also handles counters restarting from zero after a restart. A separate registry per app keeps tests and multiple apps from sharing or double-registering metrics.
- **Storage for `/metrics/summary`.** Percentiles estimated from histogram buckets are only accurate to about a bucket's width, which is too coarse for millisecond fidelity checks. `SummaryRecorder` keeps bounded sample windows instead: per model and metric, a `SampleWindow` ring buffer of the most recent raw values, giving exact percentiles over recent traffic with fixed memory.
  - **The engine's timeline.** For completed requests, the trace also passes the scheduled time (`TokenEvent.sim_time`) of the first and last token in `RequestTimings`, and the summary keeps them as `ttft_simulated` and `e2e_simulated`. Observed minus simulated is that token's lag; with the client's figures beside them, the load generator separates the simulator running late from slow delivery. The pooled lag histogram cannot do this for TTFT, since it averages over every token. Prometheus ignores these fields, so `/metrics` is unchanged.
  - **Two window sizes.** Per-request metrics (TTFT, E2E, TPOT, queue, prefill, decode, prompt and generation sizes) add one value per request and keep the last 10,000 requests. Per-token metrics (ITL, lag) add one value per token, pooled across requests, and keep the last 100,000 tokens: at 13,000 tok/s, roughly the last 8 s. Both are settable (`LLM_SIM_SUMMARY_REQUEST_SAMPLES`, `LLM_SIM_SUMMARY_TOKEN_SAMPLES`).
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
| Histogram | `llm_sim_request_queue_time_seconds` | Queue time: first admission − arrival (0 for engines without a queue) |
| Histogram | `llm_sim_request_prefill_time_seconds`, `llm_sim_request_decode_time_seconds` | Derived as above |
| Histogram | `llm_sim_request_prompt_tokens`, `llm_sim_request_generation_tokens` | Sizes of completed requests |
| Histogram | `llm_sim_token_lag_seconds` | Fidelity: how late each token was released |
| Counter | `llm_sim_requests_total{finish_reason}` | `stop`, `length`, `cancelled`, `error` |
| Counter | `llm_sim_prompt_tokens_total`, `llm_sim_generation_tokens_total` | Throughput, via `rate()` |
| Gauge | `llm_sim_requests_running` | Requests in flight, including any waiting in a scheduler's queue |
| Info | `llm_sim_model_info{engine, context_length}` | Which engine each model runs |

**Scheduler metrics**, one update per step, exported only by models whose engine reports steps. They follow the step's workflow:

| Stage | Type | Name | Records |
|---|---|---|---|
| Admit | Gauge | `llm_sim_scheduler_sequences{state="waiting"\|"running"}` | Queue and batch size after the last step (vLLM's `num_requests_waiting` / `running`) |
| | Counter | `llm_sim_scheduler_blocked_steps_total{reason}` | Steps that left requests waiting: `kv`, `max_sequences`, `budget`, `preemption` |
| Batch | Histogram | `llm_sim_scheduler_batch_sequences` | Sequences per step |
| | Histogram | `llm_sim_scheduler_batch_tokens` | Tokens computed per step: how full the `max_batch_tokens` budget was |
| | Counter | `llm_sim_scheduler_step_tokens_total{phase="prefill"\|"decode"}` | Tokens computed, by phase; prefill includes recomputation |
| Grow / preempt | Gauge | `llm_sim_kv_cache_usage_ratio` | Share of KV blocks in use, 0–1 |
| | Gauge | `llm_sim_kv_cache_blocks` | KV capacity in blocks |
| | Counter | `llm_sim_preemptions_total` | Sequences preempted |
| | Counter | `llm_sim_preempted_tokens_total` | KV thrown away by preemption, to be computed again |
| Duration | Histogram | `llm_sim_scheduler_step_duration_seconds` | Simulated step time, the source of every ITL |
| | Counter | `llm_sim_scheduler_steps_total{bound="compute"\|"memory"}` | Steps by the limit that set their GPU time |
| | Counter | `llm_sim_scheduler_busy_seconds_total{part="gpu"\|"overhead"}` | Simulated busy time; `rate()` is the GPU's utilisation |

The gauges hold the state after each step completes, so an idle engine reads 0. Label children are created on a model's first step, so the metrics layer never checks engine names. `/metrics/summary` carries the same data as a `scheduler` section for each such model (steps, busy, overhead and memory-bound fractions, exact step-time and batch percentiles, token totals, waiting and KV usage now and at their maximum, preemptions, blocked steps); a reset clears the totals but keeps the current queue and cache state, from which the maxima start again. `llmsim-bench` prints it as a `scheduler` block after the server comparison.

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

Measured with the load generator (aiohttp client, 512-token prompts, 64-token responses), one simulator process delivers about **13,000–22,000 output tokens/s**, but keeps to its configured timeline only up to about 13,000 tok/s: a rate sweep held server lag p99 at 11 ms at 200 requests/s (13,000 tok/s) and fell more than a second behind at 400 requests/s (21,900 tok/s), where TTFT reached 855 ms and ITL collapsed to bursts; a concurrency sweep showed lag p99 rising from 1.4 ms at 1 stream to 25 ms at 32 and 64 ms at 128. An earlier estimate of 5,500–6,000 tok/s came from an ad-hoc client and partly measured that client's limit. Producing tokens is not the limit: measured in-process for 200 streams, the engine releases a token in about 0.7 µs, tracing and recording add about 2.4 µs and SSE serialization about 6 µs, about 9 µs in all, against roughly 170 µs per token over HTTP. Most of the cost is delivering each chunk over HTTP. Because the stream is pulled by the writer, the engine releases the next token only after the previous chunk has been sent, so delivery cost appears as lag. By default uvicorn runs on the pure-Python `h11` parser and the standard asyncio loop; with `httptools` and `uvloop` installed (`uvicorn[standard]`), mean lag at 200 streams fell from 122 ms to 18.6 ms and mean TTFT from 183 ms to 56 ms. Clean throughput figures need the load generator: an ad-hoc test that started client processes inside its timing window measured start-up time rather than throughput. Up to that rate the simulator follows its configured timeline closely; beyond it, requests queue behind the event loop, TTFT grows, overdue tokens go out in bursts (ITL falls below the configured value), and lag shows it. For `qwen-mock`'s 250 tok/s per stream, fidelity starts degrading at roughly 20–30 concurrent streams per process. Raising the ceiling is the next performance task, measured before and after with sweeps.

A process only needs to deliver tokens as fast as the GPU it simulates produces them, which is typically a few thousand to low tens of thousands of output tokens/s, so one process can simulate one replica. To stay within budget:

| Technique | Effect | Fidelity cost |
|---|---|---|
| One step loop for all streams | One timer per step; tokens are handed over with `queue.put_nowait` | None; real servers emit per step too |
| Prebuilt SSE chunk templates | About 2.8× cheaper per token than Pydantic serialization | None |
| uvloop | Faster event loop | None |
| Merge overdue tokens into one chunk | Bounded work when behind | Total time stays correct; ITL detail is lost only while overloaded |
| Multiple worker processes | Each simulates an independent replica | Matches data-parallel deployments; one GPU cannot be shared across processes |
| Virtual-time mode | No real sleeps | Results only; no live streaming |

## Load generator (`bench/`)

`llmsim-bench` measures latency as a client sees it: the third measurement layer, after the engine's simulated timeline and the server's release times. It speaks the OpenAI streaming protocol over HTTP with `aiohttp` (the optional `bench` extra), so it works against any OpenAI-compatible server, and imports no server code beyond pure helpers.

```
cli.py ──▶ sweep.py ──▶ (run one level)
   │
   ├──▶ runner.py ──▶ health.py          closed/open loop, phases; send and the server probe are passed in
   │       ▲
   ├──▶ client.py ──▶ aiohttp            one streaming request → RequestRecord
   ├──▶ server.py ──▶ aiohttp            reset/fetch /metrics/summary
   ├──▶ workload.py ──▶ llm/tokenizer    request index → RequestSpec
   ├──▶ results.py ──▶ metrics/window    records → BenchResult (describe)
   └──▶ report.py ──▶ results.py         table, sweep table, JSON
data (no dependencies): config.py, records.py
```

- **Modules.** `config.py` (`BenchConfig`, plain validated data) and `records.py` (`RequestRecord`) have no dependencies. `client.py` is the only module that touches the network: `stream_chat()` sends one request and returns a record, never raising for a failed request. `runner.py` decides when requests are sent and receives a `send` function, so it knows nothing about HTTP and is tested with fakes. `cli.py` reads arguments and wires the others, as `runtime.py` does for the server.
- **Client-observed timing.** The clock (`time.perf_counter`) is read once per received line. TTFT is the first chunk with non-empty `delta.content` (the opening role chunk carries `""` and is not a token); ITL is the gap between content chunks; E2E ends when `[DONE]` is read, when a real caller's request returns; `last_token` is kept so TPOT can match the server's definition. Token totals come from the usage chunk (`stream_options.include_usage`), because a server may pack several tokens into one chunk.
- **Client overhead.** The OpenAI SDK is not used: it builds a typed object per chunk. The aiohttp connection limit is set to the concurrency, since its default of 100 would silently cap the run. The read timeout applies between chunks, not to the whole request.
- **Analysis and report.** `results.summarize()` turns a run's records into a `BenchResult`; it is pure, so analysis is tested with hand-made records. Latency and throughput cover successful requests only, and failures are counted and grouped by status and message, since a request that failed part-way has no meaningful timing. TTFT, TPOT and E2E have one value per request; ITL pools every gap. Percentiles come from the same `describe()` as `/metrics/summary`, so client and server figures are computed alike. Throughput divides the successful requests' `usage` tokens by the run's window. Two warnings flag numbers that need care: responses that ended before `max_tokens`, and chunk counts that differ from `usage` (ITL is then per chunk). `report.py` prints the table and writes the JSON results file, which redacts the API key and records no hostname.
- **Workload.** A `Workload` maps a request index to a `RequestSpec(prompt, max_tokens)`; the runner deals only in indexes and the client sends whatever spec it is given. One run has one source: `synthetic()`, `from_jsonl()` (cycled, checked line by line before the run) or `fixed()`, with an optional prefix. Synthetic prompts are made of letters-only words, each one token for the simulator's tokenizer, sized so that `usage.prompt_tokens` equals the requested length exactly (the chat overhead of a single user message is 7 tokens). Each request gets its own prompt from `random.Random(f"{seed}:{index}")`: repeatable, and distinct so that a real server's prefix cache does not flatter TTFT. Prompts are built when requested, before the request's clock starts, so memory does not grow with the run.
- **Phases.** `run_level()` sends warm-up requests (default one per stream), waits for every one to finish, then measures a number of requests or a duration, and waits for the requests in flight. The window runs from the first measured send to the last measured completion. Warm-up requests use indexes −1, −2, …, so measured requests always start at index 0 and send the same prompts whatever the warm-up. The wait between warm-up and measuring leaves the server idle at that boundary, where L5 resets `/metrics/summary`. With a duration, a request counts if it was sent before the deadline. If no warm-up request succeeds, the run stops with the grouped errors.
  - **What warm-up removes, measured.** On localhost, warm-up made no measurable difference to TTFT: opening a connection costs well under a millisecond. The gap between client TTFT and the simulated value comes from two other sources. A floor of about 3 ms per request at one stream (HTTP stack, plan generation, client parsing): 5.7 ms measured against 2.1 ms simulated for an 8-token prompt. A further few milliseconds at 8 streams, because the closed loop's streams start together and one server event loop serves them in turn (11.5 ms at p50). Warm-up is still the right default against real servers, where connection setup, TLS and cold caches are not negligible.
- **Server comparison.** `server.py` resets the target's `/metrics/summary` at the idle boundary after warm-up (before the loop monitor starts, so its request is not counted as client lag) and reads the benchmarked model's section after the drain, so both sides cover the same requests. `results.compare()` puts the server's TTFT, ITL, TPOT and E2E, the engine's simulated TTFT and E2E, and the server's lag next to the client's figures; delivery is client minus server per statistic, a difference of percentiles because the summary has no per-request data to pair with the client's. The server's count of finished requests is checked against the client's accepted (HTTP 200) requests, since rejected requests are never traced; a mismatch means other traffic mixed in, and the report warns. The summary is read defensively, so an older or different shape loses columns rather than failing the run, and a target without the endpoint (vLLM, OpenAI) gets a "not available" note. Measured with 8 lock-stepped streams of 512-token prompts: TTFT p50 is 7.12 ms simulated, 8.02 ms on the server and 10.26 ms at the client; at p99 the server itself is 11 ms behind its schedule, so the TTFT tail at this concurrency is the simulator falling behind when requests arrive together, not delivery.
- **Open loop.** `open_loop()` sends a request at each arrival of a seeded Poisson process: `poisson_offsets()` sums exponential gaps (`random.expovariate`, inverse-transform sampling, mean 1/rate) from a random stream separate from the prompts'. Each arrival is slept for until its absolute time, so late wake-ups do not accumulate, and the scheduler always yields, even when an arrival is overdue, so requests already started run before more are added. Requests run as independent tasks: a slow server is not offered less load, unlike the closed loop. `max_concurrency` caps requests in flight with a semaphore; an arrival that finds it full waits, and the runner stamps each record with its scheduled time and whether it waited, leaving `send()` and the client unchanged. Latency is reported two ways: from the actual send (comparable with the server's figures) and from the scheduled time (what an arriving user waits); reporting only the first is coordinated omission, where the client's own queue hides an overloaded server. The schedule is checked against the sends: `offered` is the drawn schedule's rate, which varies in a short run, and `achieved` is the sends' rate. Warm-up stays a closed-loop burst, sized by `max_concurrency` or 8.
  - **Arrival distributions.** At the same mean rate, more variable arrivals queue more: by Kingman's approximation, wait ≈ ρ/(1 − ρ) × (CVa² + CVs²)/2 × service time. Evenly spaced arrivals (CV 0) understate latency; Poisson (CV 1) models independent users; burstier arrivals (CV > 1, such as vLLM's gamma burstiness) worsen tails further. Poisson is the only distribution implemented.
- **Sweeps.** `sweep.py` runs one config per value of `concurrency` or `rate`, in order, each as a full run with its own session and the same prompts, so differences come from the load. A level whose warm-up fails is recorded and the sweep continues. The terminal shows a progress line per level and a table; the JSON holds every level's full results.
- **Client health.** `LoopMonitor` asks to wake every 10 ms during the measure phase and records how late each wake-up was, together with the client's CPU time; `results.py` turns the samples into `ClientHealth`. An idle loop is about 1 ms late; above 5 ms at p99, the report warns that the client inflated its own latencies. The monitor collects and the analysis judges, like requests and their records. It found a real distortion in the project's own tests: with the server in a thread of the test process, the two share the GIL, and the client's loop was 10 ms late at p99 while the server streamed. Bench integration tests therefore run the server in its own process.
- **Closed loop.** `closed_loop()` keeps N requests in flight: each worker sends its next request when its previous one finishes. A slower server is offered less load, so this measures capacity at a fixed number of streams but hides queueing; the open loop (L6) shows it.
- **Controlled output length.** With the built-in `bench-responses`, every response is long synthetic text, so `max_completion_tokens` sets the exact output length; tagged prompts (`[short]`, `[long]`) select deterministic mixed lengths.
- **Build layers.** L1 core loop, L2 percentiles, throughput and JSON report, L3 workload (input length, JSONL, prefix), L4 phases and client health, L5 comparison with `/metrics/summary`, L6 open loop and sweeps: all done. The client stays single-process until the performance work; multi-process load comes after.

## Deployment (`docker/`)

- **Images.** One multi-stage `Dockerfile` builds two targets from the same locked dependencies: `server` (the simulator) and `bench` (adds the `bench` extra, aiohttp). A builder stage installs the dependencies from `poetry.lock` into a virtual environment, then installs the package as a wheel, since Poetry's editable install would point at a source tree the runtime images do not have; the runtime stages copy only the environment. Both run as an unprivileged user on `python:3.10-slim`, the version the tests run on.
- **Configuration.** The image carries no config files of its own: the installed package ships its built-in configs in `llm_simulator/defaults/`, which the server uses when no file is named. When `SIM_CONFIG_DIR` is set, compose mounts it read-only at `/config` and points `LLM_SIM_MODELS` and `LLM_SIM_RESPONSES` at it; unset, it leaves both empty, which the settings treat as unset. `bench-target` always uses `builtin:bench-responses`.
- **One worker.** Metrics live in process memory, so the image runs a single uvicorn worker; more workers would each keep their own `/metrics` and summary.
- **Benchmark topology.** The `bench` profile runs a separate `bench-target` simulator with the user's models but the image's `bench-responses.yml`, so benchmarks get exact output lengths without changing the content served to applications, and the load generator reaches it over the compose network. The benchmark's settings live in the compose file. Client and server share the host's CPUs; the load generator's client health check flags when that inflates its numbers.

## Known limits: the single-process fidelity envelope

How closely one simulator process keeps the engine's timeline under load, measured on 2026-10-30 and 2026-11-01. The setup:

- **Model and engine:** `qwen-mock` on `fixed_rate`, so the simulated TTFT is 7.12 ms and the simulated TPOT is 4.00 ms.
- **Load:** open-loop Poisson arrivals of synthetic 512-token prompts, asking for 64 output tokens. `bench-responses.yml` returns exactly 64.
- **Server stack:** uvicorn with httptools and uvloop, one worker.
- **Client:** `llmsim-bench`, which was not saturated in any run.

| Offered load | Output tok/s | Server lag p50 / p99 | Server TTFT p50 / p99 | TPOT p50 |
|---|---|---|---|---|
| 10 req/s | 600 | 0.37 / 1.4 ms | 7.5 / 8.6 ms | 4.00 ms |
| 50 req/s | 2,900 | 0.34 / 1.4 ms | 7.4 / 8.5 ms | 4.00 ms |
| 200 req/s | 13,100 | 0.38 / 3.7 ms | 7.6 / 31.8 ms | 4.00 ms |
| 250 req/s | 16,200 | 0.42 / 14.1 ms | 7.8 / 49.1 ms | 4.00 ms |
| 300 req/s | 19,400 | 0.61 / 34.0 ms | 9.0 / 78.4 ms | 3.98 ms |
| 350 req/s | 22,400 | 2.9 / 82.3 ms | 44.5 / 123.8 ms | 3.45 ms |
| 400 req/s | 25,300 | 16.2 / 187.6 ms | 126.5 / 228.4 ms | 2.21 ms |

- **What the table means.** One process holds the timeline to within a few milliseconds up to about **200 requests per second, or about 13,000 output tokens per second**. At about 250 req/s the TTFT tail starts to grow, and by about 350 req/s the median falls behind as well. The limits are the event loop and the GIL, not the client.
- **The first token is the most exposed.** It has the smallest time budget and the longest path. That path runs through request validation, the threadpool hop to `generate` and back, response start and the role chunk, all while queued behind every other stream's tokens.
- **Late tokens catch up.** Token deadlines are anchored to the request's arrival, so lateness never accumulates. Overdue tokens go out back to back. TPOT then reads below the configured 4 ms, while E2E stays within about 1–4% of the schedule.
- **The same load against a realistic `responses.yml` behaves the same.** With the sample config, 300 req/s gave this server-side error against the fixed-rate model:
  - **TTFT:** +2.6 ms at the median and +93 ms at p99
  - **E2E:** +1.3% mean
  - **TPOT:** 3.94 ms at the median

  Before the regex pre-filter, the same run collapsed at about 60 req/s.
- **The scheduler's envelope is much smaller.** A step releases a token to every sequence at once, and our process delivers them one after another, so the limit is set by batch size and per-request work rather than throughput. On the workload above (512-token prompts, 64 output tokens), lag p99 stays under 5 ms up to about 25 requests per second (a batch of about 10) against about 200 for `fixed_rate`; with 200-token outputs, up to about 25 concurrent streams (E0 and E7 under "Scheduler validation"). Our request processing also adds 3.5–5 ms of simulated queue time while the engine is busy, and at saturation it bunches admissions, which lowered capacity by 9% in E7.
- **How to read results past the envelope.** Lag measures the process, not the model. The `fidelity` line and the server comparison flag it. An engine with a capacity model, such as `SchedulerEngine`, gives meaningful results only while its simulated output rate stays inside this envelope. Beyond it, run more processes as more replicas, or use the planned virtual-time mode.
- **Attribution.** The per-token cost on the scheduler's path has been measured stage by stage (E0 under "Scheduler validation"); the improvements it points to are listed under "Fidelity and performance: the per-token path". First-token time on `fixed_rate` has not been broken down by stage.

## Scheduler validation

The strategy, the method, the full list of experiments and their status are in [performance-test.md](performance-test.md); this section records the results. Each experiment checks one scheduler decision against a prediction made before running it. Models come from `config/validation/models.yml`, where each variant differs from `base` (= `qwen-mock-batched`) in one field; the server runs with `config/bench-responses.yml`, so `--max-tokens` sets output lengths exactly.

### E0: where token lag comes from

Before trusting per-token results at high concurrency, the lag was split into its two parts: the step loop waking late (`step_lag_seconds`) and tokens waiting to be delivered after release (token lag − step lag). Closed loop, 1,000-token prompts, 200 output tokens, 192 requests per level, client and server on one machine (2026-11-09):

| concurrency | batch | step p50 | token lag p50 / p99 | step lag p50 / p99 | delivery p50 |
|---|---|---|---|---|---|
| 1 | 1.0 | 4.69 ms | 0.83 / 1.66 ms | 0.52 / 1.24 ms | 0.31 ms |
| 8 | 7.9 | 6.03 ms | 1.78 / 3.22 ms | 1.00 / 1.95 ms | 0.78 ms |
| 32 | 31.7 | 10.66 ms | 3.95 / 7.16 ms | 2.27 / 3.99 ms | 1.68 ms |
| 64 | 62.4 | 16.82 ms | 6.20 / 11.06 ms | 3.62 / 6.37 ms | 2.58 ms |

**Finding: lag grows with batch size, not with throughput.** Token lag p99 rises by about 0.15 ms per sequence in the batch, at only 3,100 output tokens per second at 64 streams, a quarter of what `fixed_rate` sustains with lag p99 under 5 ms. The cause is the shape of the work: `fixed_rate` spreads each request's tokens over its own timeline, while a step releases one token to every sequence at the same instant. Each token costs about 90 µs of event-loop CPU on this path (measured below; `fixed_rate`'s path is cheaper, with no step loop or queue handoff), so the batch's writes queue behind each other. Tokens late in that queue show it as delivery lag, and the step loop itself wakes late by about the same work, about 50 µs per sequence, because its timer fires only once those writes are done. The 0.5 ms at batch 1 is timer granularity.

**What it means.** Simulated values are unaffected: the timeline is anchored, so step times, `sim_time`, and the simulated TTFT and E2E in the summary are exact at every level, and observed ITL p50 matches the simulated step time (16.72 vs 16.82 ms at 64). What lag adds is noise on the observed per-token times, about 0.15 ms × batch size at p99. With the usual bar of lag p99 under 5 ms, observed per-token timings are faithful up to about 25 concurrent streams per process; beyond that, compare simulated values, or read observed ITL percentiles knowing they carry up to about 0.15 ms per stream of delivery jitter. Real engines have the same burst (vLLM's API server also delivers a step's tokens together), but its cost belongs to their process, not to the simulated model.

**Where the time goes.** The same load (64 streams, 30 s) profiled with py-spy in non-blocking `--gil` mode, checked against two independent measurements: process CPU from `/proc` (88.8–90.7 µs per token) and timers around the step loop inside the running server (13.4 µs per token). The profile agrees with both (90.3 and 12.8 µs). An earlier profile in py-spy's default blocking mode did not: pausing the process for every sample inflated CPU per token to 150 µs and doubled the step loop's share, so only non-blocking profiles are used here.

| Stage | µs per token | Share | Inside it |
|---|---|---|---|
| SSE chunk (`client/sse.py`) | 21.5 | 24% | Pydantic: `model_dump_json` 10, building the chunk models 5; the generator and f-string 6 |
| uvicorn `send` | 15.4 | 17% | Almost all in `send` itself, mostly the socket write in C |
| Trace and per-token metrics | 14.8 | 16% | Two `prometheus_client` histogram observations (ITL, lag) about 7; trace bookkeeping 5; summary windows 2 |
| Step loop (policy, cost, queue handoff) | 12.8 | 14% | `schedule` 5, `complete` 3.6, `put_nowait` 3.5 per sequence; 1.75 µs per sequence when measured hot in isolation, so cold caches between steps cost about 3× |
| Starlette `StreamingResponse` | 5.4 | 6% | |
| uvloop internals (C) | 5.0 | 6% | Callback dispatch, polling, timers |
| Engine stream (`driver.run`) | 4.9 | 5% | Queue get, `TokenEvent` |
| Per-request work | 4.8 (964 µs per request) | 5% | Parsing, validation, handler setup; the generator runs in a worker thread (about 0.8 ms per request) |
| asyncio scheduling | 4.0 | 4% | Task switches and wake-ups |
| Step metrics | 1.7 | 2% | |
| **Total** | **90** | | |

Garbage collection is not a significant cost: about 52 ms over the 30 s run (87, 8 and 1 collections of generations 0, 1 and 2). The one full collection took 13.7 ms, which accounts for the largest lag outliers.

The projection for the transport work follows from the table. Chunk templates (SSE from about 22 to about 3 µs) and lighter per-token metrics (about 15 to about 7) would bring the total from about 90 to about 60 µs; bypassing `StreamingResponse` and trimming the step loop perhaps to about 50. The socket write, uvloop and the task switch per token, about 25–30 µs together, remain. Lag p99 would roughly halve, so observed timings would stay within 5 ms up to about 40–50 streams rather than about 25: a real gain, but not the 3–4× that an earlier, unmeasured estimate suggested.

**Consequence for the build order.** This is the first measurement that ties the per-token transport cost to a fidelity limit the scheduler hits in practice: halving the cost per token roughly halves lag at a given batch size. It strengthens the case for the transport work planned after the scheduler.

### E6: the overhead and efficiency settings are exact

Checked first, because if the configuration knobs do not do exactly what they claim, no other result is worth reading. One stream, 128-token prompts, 200 output tokens, 30 requests per model (6,000 steps each), on `base`, `overhead-1ms` (`step_overhead_ms: 1`) and `memory-eff-80` (`efficiency.memory: 0.8`).

```
llmsim-bench --model <model> --base-url http://localhost:8093/v1 --concurrency 1 \
  --requests 30 --input-tokens 128 --max-tokens 200 --output results/validation/e6-<model>.json
```

**Prediction**, from the cost model before running: one prefill step of (128, 0), then decode steps of (1, 128) to (1, 326), all memory-bound. Overhead adds exactly 1 ms to every step; 80% memory efficiency multiplies the memory time by 1.25.

| Model | Step p50 predicted / simulated | Simulated TTFT p50 | Simulated E2E p50 | Observed ITL p50 | Verdict |
|---|---|---|---|---|---|
| `base` | 4.5331 / 4.5331 ms | 4.5157 ms (exact) | 906.62 ms (exact) | 4.671 ms (+0.14) | Pass |
| `overhead-1ms` | 5.5331 / 5.5331 ms | 5.5157 ms (exact) | 1,106.62 ms (exact) | 5.665 ms (+0.13) | Pass |
| `memory-eff-80` | 5.6664 / 5.6664 ms | 5.6446 ms (exact) | 1,133.27 ms (exact) | 5.758 ms (+0.09) | Pass |

Pass criteria were simulated values within 0.5% and observed ITL p50 within 0.3 ms. Simulated values match to the summary's precision (under 0.001%): the overhead adds exactly 1.000 ms per step and 200 ms per request, and the efficiency exactly ×1.25 on a memory-bound step. Observed ITL p50 sits 0.09–0.14 ms above the step time at batch 1, consistent with E0's lag at batch 1 (timer granularity), and lag p99 was 1.7 ms in all three runs.


### E1: batching is nearly free

A closed-loop concurrency sweep on `base`: 1 to 64 streams, 128-token prompts, 200 output tokens, 20 s per level.

```
llmsim-bench --model base --base-url http://localhost:8093/v1 --concurrency 1,2,4,8,16,32,64 \
  --duration 20 --input-tokens 128 --max-tokens 200 --output results/validation/e1.json
```

**Prediction.** Every step reads the weights once plus each sequence's KV; with contexts spread evenly over 129–327 tokens (mean 228), step ≈ (13.48 GB + B × 228 × 512 KiB) ÷ 3,000 GB/s, memory-bound throughout, and output throughput ≈ B ÷ step less about 0.5% for client turnaround.

| Streams | Step p50 predicted / simulated | Output tok/s predicted / measured | Batch | Memory-bound | Sim TTFT p50 | Step max |
|---|---|---|---|---|---|---|
| 1 | 4.533 / 4.533 ms | 219 / 220 | 1.0 | 100% | 4.5 ms | 4.6 ms |
| 2 | 4.573 / 4.572 ms | 435 / 434 | 2.0 | 100% | 10.0 ms | 4.6 ms |
| 4 | 4.653 / 4.652 ms | 855 / 852 | 4.0 | 100% | 10.0 ms | 5.3 ms |
| 8 | 4.812 / 4.812 ms | 1,654 / 1,638 | 7.9 | 99% | 13.6 ms | 12.2 ms |
| 16 | 5.131 / 5.134 ms | 3,103 / 3,046 | 15.8 | 99% | 20.8 ms | 22.7 ms |
| 32 | 5.768 / 5.776 ms | 5,520 / 5,294 | 31.3 | 99% | 44.6 ms | 45.4 ms |
| 64 | 7.043 / 7.064 ms | 9,041 / 8,391 | 61.9 | 99% | 100.7 ms | 82.2 ms |

**Verdict: pass.** Step p50 is within 0.3% at every level, no step was blocked by KV, the batch or the budget, and nothing was preempted. Batching works as the roofline says: 64 streams get 38× the throughput of one for 1.56× the step time.

**What the prediction missed: phase-locking.** TTFT and the largest steps grow with concurrency, which the steady-state prediction did not include. Every request has the same length and the loop is closed, so streams that start together finish together and send their next requests together; they never spread out. About every 200 steps, one step prefills most of the batch's new prompts at once: fully synchronised, that is B × 128 tokens, compute-bound at B × 1.75 ms. The observed step maximum is 74–88% of that at every level (12.2, 22.7, 45.4 and 82.2 ms at 8, 16, 32 and 64 streams), so the streams stay mostly in step, loosened only by client timing. Requests arriving during that step wait for it (queue p50 34 ms at 64) and then share their own prefill step (prefill p50 71 ms), which makes up the simulated TTFT of about 100 ms; the compute-bound steps take about 6% of GPU time, which is the 7% throughput shortfall at 64. This is correct behaviour for the workload, and a real engine under the same benchmark would phase-lock too. It suggests that fixed-length closed-loop benchmarks overstate TTFT compared with traffic whose lengths or arrival times vary, but the explanation is so far inferred from these numbers, not isolated. **Pending: E1b**, a control at 64 streams that breaks the synchronisation and changes nothing else, once with output lengths spread over 100–300 tokens (closed loop) and once with Poisson arrivals at about 40 requests per second (fixed lengths). If phase-locking is the cause, both should bring simulated TTFT p50 down to about 10–15 ms, the step maximum well below 82 ms and throughput to within about 2% of the prediction.

### E2: context slows every stream

32 streams on `base`, 256 output tokens, prompts of 128, 512, 1,024 and 2,048 tokens, 20 s each.

```
llmsim-bench --model base --base-url http://localhost:8093/v1 --concurrency 32 --duration 20 \
  --input-tokens <prompt> --max-tokens 256 --output results/validation/e2-<prompt>.json
```

**Prediction.** A decode step reads the weights plus 32 contexts averaging prompt + 128 tokens: step ≈ (13.48 GB + 32 × (prompt + 128) × 512 KiB) ÷ 3,000 GB/s, memory-bound. KV at full growth stays under capacity at every level (at most 73,728 of 121,744 tokens), so no KV blocking or preemption. As in E1 the streams phase-lock; from 512-token prompts up, 32 prompts arriving together exceed the 8,192-token budget, so admission is expected to wait on `budget` and the largest step to be one full prefill chunk, 111.7 ms.

| Prompt | Step p50 predicted / simulated | KV max predicted / measured | Step max | Blocked | Observed ITL p50 | Verdict |
|---|---|---|---|---|---|---|
| 128 | 5.925 / 5.924 ms | 10% / 10% | 52.4 ms | — | 6.00 ms | Pass |
| 512 | 8.072 / 8.081 ms | 20% / 20% | 111.7 ms | budget 5 | 8.06 ms | Pass |
| 1,024 | 10.936 / 10.950 ms | 34% / 34% | 111.7 ms | budget 6 | 10.96 ms | Pass |
| 2,048 | 16.662 / 16.683 ms | 61% / 61% | 111.7 ms | budget 7 | 16.71 ms | Pass |

**Verdict: pass.** Step p50 is within 0.13% at every length; 96–99% of steps are memory-bound (the rest are the prefill chunks); no KV blocking and no preemption. Every other prediction holds: KV usage at its maximum, `budget` waits from 512 tokens and not at 128 (32 × 128 = 4,096 tokens fits the budget), and a largest step of exactly 111.7 ms. Context is the dominant cost of decoding: 16× the prompt length makes every stream's step 2.8× slower at the same batch size. Observed ITL p50 is within 0.1 ms of the simulated step at 32 streams, with lag p99 of 5.0–8.7 ms as E0 predicts at this batch size; simulated TTFT is 58–218 ms, from the same phase-locking as E1.

### E3: chunked prefill trades spike height for spread

Two `llmsim-bench` processes against one server, so each reports its own percentiles: 16 steady decoders (128-token prompts, 1,000 output tokens, 30 s, reading the server summary) and, starting 3 s later, one stream of 8,192-token prompts with 1 output token (25 s, `--no-server-metrics`). On `base` (budget 8,192) and `budget-2048`.

```
llmsim-bench --model <model> --base-url http://localhost:8093/v1 --concurrency 16 --duration 30 \
  --input-tokens 128 --max-tokens 1000 --output results/validation/e3-<model>-decoders.json &
llmsim-bench --model <model> --base-url http://localhost:8093/v1 --concurrency 1 --duration 25 \
  --input-tokens 8192 --max-tokens 1 --no-server-metrics --output results/validation/e3-<model>-long.json
```

**Prediction**, computed step by step with the cost model (16 decoders at a mid-life context of about 628 tokens): with budget 8,192 a long prompt takes 2 steps, the first 111.7 ms; with 2,048 it takes 5 steps of at most 27.9 ms. The prefill work is the same either way, so the long prompt's prefill takes 119.3 ms with both. All 16 decoders are in every step, so the longest step is the decoders' longest ITL.

| | Budget 8,192 | Budget 2,048 |
|---|---|---|
| Longest step (decoders' ITL max), predicted / simulated | 111.7 / 111.7 ms | 27.9 / 27.9 ms |
| Decoders' observed ITL p50 / p99 / max | 6.27 / 112.7 / 115.7 ms | 6.31 / 30.5 / 36.4 ms |
| Long prompts' observed TTFT p50 / p99 | 138.5 / 147.6 ms | 138.5 / 149.4 ms |
| Long prompts served in 25 s | 178 | 178 |
| Decode tokens in the measured window | 63,936 | 63,936 |

**Verdict: pass.** The longest step is exactly as predicted for both budgets, and the long prompts' TTFT ratio is 1.000 (criterion: 1.0–1.15). A smaller budget lowers the spike every decoding stream sees by 4× and spreads it over more steps, at no cost in TTFT or throughput: the same prompts and the same decode tokens were served in the same time. This is the trade-off behind vLLM's `max_num_batched_tokens`.

**What the prediction missed: our request processing shows up as queue time while the engine is busy.** Long prompts' TTFT was 138.5 ms against a predicted prefill of 119.3 ms. In simulated time it is queue (p50 about 13 ms) plus prefill (p50 about 122 ms, within 2.3% of the prediction; the rest is the decoders' actual contexts). The queue time is our own work: arrival is stamped before the 53 KB body is read and validated and before the generator runs (2.3 ms for this prompt), and only then is the request handed to the engine, which adds it at the next step boundary. An idle engine absorbs that time, because it starts at the arrival; a busy one cannot add a request to a step already running, so the processing time, rounded up to a step boundary, becomes simulated queue time. Inferred from these numbers rather than isolated; it is the measured effect on TTFT of the per-request item T5 under "Fidelity and performance: the per-token path".

### E4: admission waits for KV space

`small-kv` (1,260 blocks of 16 tokens, a watermark of 12 blocks), 32 streams, 2,000-token prompts, 200 output tokens, 40 s.

```
llmsim-bench --model small-kv --base-url http://localhost:8093/v1 --concurrency 32 --duration 40 \
  --input-tokens 2000 --max-tokens 200 --output results/validation/e4.json
```

**Prediction**, from the block arithmetic and the cost model. A 2,000-token prompt needs 125 blocks; with 9 running, 135 are free, and a tenth would leave 10, below the watermark, so at most 9 run. Fully grown, a request holds ⌈2,199 ÷ 16⌉ = 138 blocks and 9 × 138 = 1,242 ≤ 1,260, so nothing is ever preempted. In steady state, admission is blocked by `kv` whenever anyone waits. A decode step with 9 contexts of about 2,100 tokens takes 7.8 ms; a step carrying one new prompt is compute-bound at 27.4 ms; 9 ÷ 200 = 4.5% of steps carry a prompt, so the mean step is 8.7 ms, about 115 steps per second, 1,040 output tokens and 5.2 requests per second. With about 23 waiting, Little's law gives a queue time of 23 ÷ 5.2 ≈ 4.4 s.

| | Predicted | Measured |
|---|---|---|
| Running at most | 9 | 9 (batch mean 8.84) |
| Preemptions | 0 | 0 |
| Blocked reason | `kv` whenever anyone waits | `kv` on 4,999 of 5,201 steps; no other reason |
| Waiting at most | 23 | 23 |
| KV usage at its peak | 1,242 of 1,260 blocks, 98.6% | 98.6% |
| Decode step p50 | 7.8 ms | 7.79 ms |
| Mean step | 8.7 ms | 8.83 ms (+1.5%) |
| Queue time p50 | 4.4 s | 4.46 s (+1.4%) |
| Throughput | 5.2 requests/s, 1,040 tokens/s | 5.01 requests/s, 1,001 tokens/s (−3.7%) |

**Verdict: pass** (criteria: running at most 9, no preemption, `kv` the dominant blocked reason, queue time within 15%). The KV cache, not the batch limit of 64 or the token budget, decides how many run: TTFT is almost all queueing (4.46 s of 4.57 s), the cost a real server pays when long prompts fill its cache.

**What the prediction missed.** It assumed requests enter one at a time as others finish, each with a 27.4 ms prefill step. They enter in groups instead: requests admitted together finish together and free their blocks together, the phase-locking of E1, so four are admitted at once and their 8,000 prompt tokens fill one 111.7 ms step (prefill p50 112 ms). The total prefill compute is the same, which is why the mean step and the throughput still land within 2–4% of the prediction.

### E5: preemption, and what the watermark changes

`small-kv` (watermark 1%) and `small-kv-no-watermark`: exactly 64 requests at once, one per stream, with no warm-up, 200-token prompts and 2,000 output tokens each, run until all finish. The chunk timeout is raised to 120 s because preempted requests can wait a long time between tokens.

```
llmsim-bench --model <model> --base-url http://localhost:8093/v1 --concurrency 64 --requests 64 \
  --warmup 0 --timeout 120 --input-tokens 200 --max-tokens 2000 --output results/validation/e5-<model>.json
```

**Prediction, in two parts.** By hand: all 64 are admitted (13 blocks each, 832 in all), but growing together they pass 1,260 blocks about 105 tokens into their output, and fully grown they would need 8,832, so both variants must preempt; each preemption throws away between 200 and about 2,200 tokens. For exact numbers, the pure scheduler core (`Scheduler` and `RooflineCostModel`, no server, no clock) was run in virtual time on the same 64 requests arriving at t = 0. That tests whether the driver, the server and the bench reproduce what the scheduler's rules dictate under heavy preemption; whether the rules themselves are right is the unit tests' job.

| | `small-kv`, predicted / measured | `small-kv-no-watermark`, predicted / measured |
|---|---|---|
| Requests completed | 64 / 64 | 64 / 64 |
| Preemptions | 129 / 131 (+1.6%) | 148 / 144 (−2.7%) |
| Tokens thrown away | 95,114 / 96,322 (+1.3%) | 106,835 / 103,903 (−2.7%) |
| Tokens per preemption | 737 / 735 | 722 / 722 |
| Batch mean | 15.35 / 15.38 | 15.37 / 15.39 |
| Last request finished (simulated) | 65.45 / 65.34 s | 65.55 / 65.40 s |
| E2E p50 (simulated) | 37.6 / 37.7 s | 37.7 / 37.8 s |

**Verdict: pass** (criteria: preemptions and thrown-away tokens within 15% of the virtual prediction, the watermark variant lower, tokens per preemption within the hand bounds, batch mean and finish time within 5%). Everything lands within 3%; the remaining difference is arrival order, since the bench's 64 requests arrive a few milliseconds apart rather than together, which changes who is newest at each preemption. KV usage peaked at 100% in both; admission waited on `kv` in about 6,150 steps and on `preemption` in the step after each eviction.

**The watermark** cut preemptions by 9% (13% in the virtual run) and the recomputed tokens with them, at no cost in time: recomputation is prefill, which is cheap while the GPU is memory-bound.

**What the prediction understated: how long a victim stalls.** The largest ITL was 50 s, against a predicted "seconds". A preempted request goes to the front of the queue but returns only when enough blocks are free, and the oldest requests hold the cache until they finish, which can take most of a request's lifetime. So while the median ITL is 7.9 ms, a few streams freeze for tens of seconds. This is what recompute preemption under heavy oversubscription costs, the same in vLLM's policy, and the case a gateway's or client's timeouts should be tested against.

### E7: the scheduler's fidelity envelope

An open-loop sweep on `base` at 10, 25, 50, 75 and 100 requests per second (Poisson arrivals; actually 10.0, 26.5, 53.3, 77.9 and 103.3), 512-token prompts and 64 output tokens, 30 s per rate: the workload of the `fixed_rate` envelope under Known limits.

```
llmsim-bench --model base --base-url http://localhost:8093/v1 --rate 10,25,50,75,100 --duration 30 \
  --timeout 120 --input-tokens 512 --max-tokens 64 --output results/validation/e7.json
```

**Prediction.** The simulated side from the pure scheduler core in virtual time, fed Poisson arrivals at the rates actually offered; the agreement can only be statistical, since the arrivals are different draws. Saturation is set by `max_num_sequences`: at a full batch of 64 a request takes about 0.75 s, so capacity is about 82 requests per second. Lag from E0's formula, 1.7 ms + 0.15 ms per sequence in the batch.

| Rate | Batch, virtual / real | Step p50, virtual / real | Sim TTFT p50, virtual / real | Queue p50, virtual / real | Lag p99, E0 formula / measured | Step lag p99 |
|---|---|---|---|---|---|---|
| 10.0 | 3.3 / 3.3 | 4.78 / 4.79 ms | 9.8 / 13.2 ms | 2.6 / 6.1 ms | 2.2 / 2.9 ms | 1.7 ms |
| 26.5 | 9.2 / 9.6 | 5.35 / 5.44 ms | 10.5 / 14.1 ms | 2.9 / 6.7 ms | 3.1 / 5.0 ms | 2.6 ms |
| 53.3 | 25.4 / 26.0 | 7.16 / 7.21 ms | 12.9 / 18.0 ms | 3.6 / 8.4 ms | 5.6 / 8.3 ms | 4.2 ms |
| 77.9 | 57.7 / 59.2 | 10.57 / 10.58 ms | 168 / 957 ms | 151 / 937 ms | 10.6 / 18.9 ms | 8.0 ms |
| 103.3 | 64 / 62.5 | 10.61 / 10.59 ms | 5.6 / 6.0 s | 5.6 / 6.0 s | 11.1 / 35.6 ms | 35.2 ms |

Measured throughput levelled off at 74 requests per second at both 77.9 and 103.3, with admission blocked by `max_sequences`; no KV blocking (KV peaked at 29%) and no preemption. `busy_fraction` stayed at 0.94–1.00, so the step loop kept up with simulated time at every rate: the ceiling is in the simulated timeline, not our process running behind.

**Verdict: partial.** Scheduling matches: batch within 4% and step p50 within 1.7% at every rate, and saturation comes from `max_num_sequences` as predicted. Two results fail, and both trace to our own process rather than the scheduler's rules:

- **TTFT is 3.5–5 ms higher than predicted at every rate** (criterion: within 15%; measured +35–44%). All of it is queue time: the E3 finding, our request processing counted as simulated queue time while the engine is busy. With short prompts and outputs, a few milliseconds is a large share.
- **Capacity is 9% lower: 74 requests per second against 81.7.** The mean step at saturation was 13.18 ms against 12.25 ms while the median was the same, so the extra time is in a minority of steps. In the real run new prompts reached the engine about three at a time (tokens per step p50 64, p90 1,597), while the virtual run spread them one per step (p50 575). One 512-token prompt riding with 64 decodes is free, because its compute (7.9 ms) hides under the step's memory time (10.6 ms); three together make the step compute-bound at 21.8 ms. Handing the virtual run's arrivals to the engine in 20–30 ms bunches reproduces the real run on every measure (capacity 70–77, mean step 12.95–14.15 ms, tokens per step p50 64 and p90 1,597), so grouped admission is confirmed as sufficient to explain the gap. That the grouping comes from our serialised request processing, and that equal-length requests keep it once saturated (the phase-locking of E1), is inferred.

**The envelope.** Observed per-token timing stays within 5 ms of the simulated timeline up to about 25 requests per second on this workload (a batch of about 10, 1,675 output tokens per second), against about 200 requests per second for `fixed_rate` on the same workload. E0's formula underestimates lag here by 30–70% below saturation and by 3× at it: with 64-token outputs each token carries more per-request work than with E0's 200, and a backlog of hundreds of waiting requests adds event-loop work of its own. Beyond the envelope, simulated step times and batches stay right; observed per-token times and simulated TTFT carry our process's overhead.
## Open decisions

None for the first `SchedulerEngine`. Decided on 2026-11-03: roofline × efficiency with a per-step overhead, always-chunked prefill, vLLM-style admission with a 1% watermark, and preemption by recomputation of the newest admitted sequence.

## Future plan

- **Prefix caching** (first extension of the scheduler): blocks hashed by their token content and shared between sequences with reference counts; freed blocks stay cached and are evicted least recently used. Skips prefill of shared prefixes such as system prompts and earlier chat turns, which lowers TTFT and raises effective KV capacity. It needs the prompt's token content, not only its length, to reach the scheduler.
- **Tensor parallelism**: weights and KV split over N GPUs, multiplying bandwidth and FLOPS by N, plus an all-reduce cost per layer in each step.
- **Not planned**: request priorities and speculative decoding.

### Fidelity and performance: the per-token path

**The problem.** A scheduler step releases one token to every sequence in the batch at the same simulated moment, as a real engine does. Our process then delivers them one after another on a single event loop, at about 90 µs of CPU each, so a token's observed time trails its simulated time by its place in that queue. Token lag p99 is about 1.7 ms + 0.15 ms per sequence in the batch (E0 under "Scheduler validation"): within the 5 ms fidelity bar up to about 25 concurrent streams per process, 11 ms at 64. Simulated values stay exact; what clients observe carries the jitter, and observed timing is what the systems under test experience, so this is a fidelity limit, not only a speed one.

**The improvements**, largest first, with the per-token costs measured in E0 (about 90 µs in total):

| # | Change | Cost now | Expected after | Notes |
|---|---|---|---|---|
| T1 | **SSE chunk templates**: build each request's constant JSON once and per token encode only the text, instead of building Pydantic models and calling `model_dump_json` | 21.5 µs | about 3 µs | The bytes sent must stay identical; a test compares them with today's output |
| T2 | **Lighter per-token metrics**: avoid `prometheus_client`'s per-call overhead for the two per-token histograms (ITL, lag), for example by counting buckets ourselves and exporting through a custom collector, and trim the trace's bookkeeping | 14.8 µs | about 7 µs | Exported values must not change |
| T3 | **Leaner response path**: write SSE bytes through the ASGI `send` directly instead of Starlette's `StreamingResponse` | 5.4 µs | about 1 µs | Disconnect detection, which Starlette does for us, must be kept |
| T4 | **Step loop trimming**: fewer objects per step (`Work`, `StepPlan`, tuples), `__slots__`, a cheaper handoff to each stream than an `asyncio.Queue` | 12.8 µs + 4.9 µs engine stream | perhaps half | Pure-core tests guard behaviour; its cost is mostly cold caches, so the gain is the least certain |
| T5 | **Per-request work**: parsing, validation and the worker-thread hop for the generator, about 1 ms on the event loop per request | about 5 µs per token at 200-token completions | | Matters more for short completions and high request rates. While the engine is busy, this time also reaches the simulated timeline: a request joins the first step after it reaches the engine, not the first step after it arrived, which added about 13 ms of queue time to 8,192-token prompts in E3 and 3.5–5 ms at every rate in E7, and at saturation bunches admissions so that prompts lose their free ride under memory-bound decode steps, which lowered capacity by 9% in E7. Making the work cheaper is the remedy; it cannot be backdated, because any step that started between the arrival and the hand-off is already running |
| T6 | **Garbage collection**: one full collection took 13.7 ms, the largest lag outlier; `gc.freeze()` after startup keeps the long-lived objects out of every later collection | — | shorter full collections | A one-line change; to be measured |
| T7 | **More processes**: several server processes, each its own simulated GPU, divide the burst by their number | — | — | Changes what one model means (each process is a replica) and needs Prometheus multiprocess mode; a later step |

The socket write, uvloop's own work and the task switch per token, about 25–30 µs together, remain after T1–T4. With T1 and T2 the total is projected at about 60 µs per token, and about 50 with T3 and T4, so lag p99 would roughly halve and observed timings would stay within 5 ms up to about 40–50 streams. Each change is measured the same way as E0: process CPU from `/proc`, a py-spy profile in non-blocking mode (blocking mode distorts it), and the lag sweep.

**Not an option**: spreading a step's releases over time to avoid the burst. It would smooth the observed numbers by misrepresenting batching. A realistic delivery spread, as a real API server has, could instead be modelled on purpose and configurable, once measured on a real one.

## Build order

1. Metrics layer: tracing, recorders, Prometheus `/metrics` and `/metrics/summary` are done. The `llmsim-bench` load generator is done (L1–L6).
2. Performance groundwork: `uvicorn[standard]` (httptools and uvloop) and the regex pre-filter are done; the resulting envelope is recorded under Known limits.
3. `SchedulerEngine` with KV blocks, admission, preemption and the roofline step model: the engine, its wiring and its per-step metrics are done; validation with `llmsim-bench` is next.
4. Transport work on the scheduler's token path: the per-token breakdown is measured (E0); the changes T1–T7 are listed under "Fidelity and performance: the per-token path", T1 and T2 first.
