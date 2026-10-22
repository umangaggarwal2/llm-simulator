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
| Engine interface and `InstantEngine` | `simulator/engine.py` | Done |
| `FixedRateEngine`, `SchedulerEngine` | `simulator/engine.py` | Planned |

## Layers and dependencies

```
client ──▶ runtime.py ──▶ llm        ──┐
                 │                     ├──▶  types.py
                 └──────▶ simulator  ──┘
```

- `client` handles HTTP only: it converts API models to domain types and domain events back into OpenAI responses.
- `llm` decides the response content. It never deals with time.
- `simulator` decides when tokens are emitted. It never deals with response content.
- `runtime.py` is the one module that assembles every layer. It reads the parsed `models.yml` and `responses.yml` and hands the generator and engine only the values they need.
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
simulator/engine.py
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

### Simulated and observed metrics

- **Simulated metrics** are what the engine intended, recorded for every token. These are what a metrics endpoint exposes.
- **Observed metrics** are what clients measure: simulated values plus network time plus any lag.

The difference is the simulator's fidelity error. With anchoring it should be close to zero apart from the network; the lag metric shows when it is not.

### Engines

All engines implement the same interface:

```python
class Engine(ABC):
    def run(self, plan: GenerationPlan, arrival: float) -> AsyncIterator[TokenEvent]: ...
    async def start(self) -> None: ...   # no-op by default
    async def stop(self) -> None: ...    # no-op by default
```

`TokenEvent(index, text, sim_time)` carries only timing information; `finish_reason` and `usage` come from the plan, which the API already holds. `sim_time` is in simulated seconds after arrival, so token 0's value is the simulated TTFT and the gaps between tokens are the ITLs. `start` and `stop` are called from the app's lifespan; they exist for engines with a background loop, such as the scheduler. Engines are built by name with `get_engine(name)`, chosen per model in `models.yml`.

When a client disconnects, the API closes the iterator and the engine releases the sequence in a `finally:` block. This frees nothing in the simpler engines but frees KV-cache capacity in the scheduler.

| Engine | Timing | Shared state | Purpose |
|---|---|---|---|
| `InstantEngine` | None; tokens are yielded immediately | None | Wiring and functional tests |
| `FixedRateEngine` | `ttft = overhead + prompt_tokens / prefill_tps + 1 / decode_tps`, `itl = 1 / decode_tps` | None; requests are independent | Simple baseline with configurable latency |
| `SchedulerEngine` | One shared step loop; each step's duration depends on the batch | Waiting queue, running batch, KV blocks | Load-dependent behaviour: queueing, batching, KV pressure |
| Virtual-time mode (possible later) | The scheduler run as a discrete-event simulation, without real sleeps | Same as `SchedulerEngine` | Capacity planning beyond what can be streamed in real time |

Each later engine changes how the simulated timeline is computed. Arrival-anchoring stays the same.

### Mapping to target metrics

For one request in `FixedRateEngine`, with prefill at 100,000 tok/s, decode at 250 tok/s, 2 ms scheduler overhead, 1,000 prompt tokens and 200 output tokens:

| Metric | Definition | Value |
|---|---|---|
| Queue latency | admitted − arrival | 0 (nothing queues) |
| Prefill latency | prompt_tokens / prefill_tps | 10 ms |
| TTFT | queue + overhead + prefill + first decode step | 16 ms |
| ITL | one decode step | 4 ms |
| Decode latency | (n − 1) × ITL | 796 ms |
| E2E latency | TTFT + decode latency | 812 ms |
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

## Real-time delivery budget

Every real-time engine pays about one coroutine resume, one serialization and one socket write per token per stream; anchoring does not add to this. Measured in-process with 256 streams and no HTTP, a step loop delivers about 130,000 tokens/s when chunks are serialized through the Pydantic models and about 370,000 tokens/s with a prebuilt SSE template. HTTP framing and socket writes reduce this several-fold, so the working budget is on the order of tens of thousands of tokens/s per process. This still needs measuring end to end.

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

1. `FixedRateEngine` with arrival-anchored deadlines and the lag metric, plus the `performance` section of `models.yml`.
2. `SchedulerEngine` with KV blocks, admission, preemption and the roofline step model.
