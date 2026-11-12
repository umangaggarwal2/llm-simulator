# llm-simulator
A GPU-free, deterministic stand-in for an OpenAI-compatible LLM server. Its latency and throughput respond to load the way a real serving engine's do, so the systems built on top of a model can find their own bottlenecks and failure modes without paying for model inference.

## What it is for

Testing everything that sits **in front of** a model: gateways, routers, load balancers, autoscalers, rate limiters, retry and timeout logic, streaming clients, and the dashboards and alerts that watch them. It answers "how does my system behave when the model slows down like this?", in CI, on a laptop or in Docker, with the same answer every run.

It is **not** a way to predict how fast a particular model will run on particular GPUs. Only the real engine, or a simulator built from profiled kernels, answers that.

## How faithful it is

| Aspect | Fidelity | Why |
|---|---|---|
| API: OpenAI request and response format, streaming, errors, `max_tokens` | High | Built to match the real API |
| Load trends: queueing under overload, TTFT growing with load, ITL rising with batch and context, prefill interference, KV-cache pressure and preemption | High, with the `scheduler` engine | It applies the same scheduling rules as vLLM; measured step times match the roofline within 0.3% from 1 to 64 streams, and admission, chunked prefill and preemption behave as predicted (see [docs/performance-test.md](docs/performance-test.md)) |
| Absolute numbers, such as "ITL is 23 ms at 40 users" | Medium; good once calibrated | The step-time model is an idealised roofline; one measurement against a real server anchors it |
| Content: what the model says | None, by design | Responses are scripted in `responses.yml` and deterministic |

The simulator models the resource constraints that drive client-visible latency and throughput: GPU memory, compute, memory bandwidth, KV-cache capacity, batching and scheduling. It does not measure real GPU utilization; those are modelled values derived from the configured hardware.

**Limits.** One process keeps faithful timing up to about 200 requests per second or 13,000 output tokens per second (see [Known limits](docs/architecture.md#known-limits-the-single-process-fidelity-envelope)); beyond that, run more replicas, as you would with real servers. With the `scheduler` engine the limit is set by batch size rather than throughput: a step releases a token to every sequence at once, and observed per-token timing stays within 5 ms of the simulated timeline up to about 25 concurrent streams per process with 200-token outputs, and up to about 25 requests per second (a batch of about 10) with 512-token prompts and 64-token outputs; simulated step times stay exact beyond that. Results are trustworthy while the `token_lag_seconds` metric, the simulator's own timing error, stays small; the `llmsim-bench` report flags it. How timing and fidelity are tested, and the results so far, are in [docs/performance-test.md](docs/performance-test.md).

# Installation

## Locally
Python 3.10 or later, with [Poetry](https://python-poetry.org/):

```bash
poetry install                 # the server, llmsimulator, and the load generator, llmsim-bench
poetry run llmsimulator         # serves on http://localhost:8080
poetry run llmsimulator --port 9000 --responses my-responses.yml
```

## From git, in another project or image
The package installs straight from the repository, with both commands, `llmsimulator` and `llmsim-bench`. Pin a tag or commit for repeatable builds; the command-line flags and built-in configs need a version after 0.1.0. The installed package carries the example configs, so a bare `llmsimulator` runs anywhere; pass only the files you want to replace:

```dockerfile
RUN pip install --no-cache-dir \
    "llm-simulator @ git+https://github.com/umangaggarwal2/llm-simulator.git@main"
COPY responses.yml /app/
EXPOSE 9000
CMD ["llmsimulator", "--port", "9000", "--responses", "/app/responses.yml"]
```

## Running in Docker
`docker/Dockerfile` builds two images: `llm-simulator` (the server) and `llm-simulator-bench` (the load generator). `docker/docker-compose.yml` runs them. Run every command from the repository root.

```bash
docker compose -f docker/docker-compose.yml up -d --build     # the simulator on http://localhost:8080
docker compose -f docker/docker-compose.yml logs -f simulator
docker compose -f docker/docker-compose.yml --profile '*' down    # everything, including a left-over bench-target
```

A plain `down` only sees services outside profiles: it would leave a `bench-target` from an interrupted benchmark running, and report the network as still in use. `--profile '*'` includes every profile.

**Your own configuration.** By default the simulator serves its built-in configs. Point `SIM_CONFIG_DIR` at a directory containing your `models.yml` and `responses.yml` (an absolute path) to serve those instead; it is mounted read-only. A config that fails validation stops the container with the same errors as running locally (`docker compose logs simulator`); after three failed restarts it stays stopped. `SIM_PORT` changes the host port.

```bash
SIM_CONFIG_DIR=/path/to/my-config docker compose -f docker/docker-compose.yml up -d
```

**Benchmarking.** The `bench` profile starts a second simulator, `bench-target`, with your `models.yml` but the image's built-in `bench-responses.yml`, so that output lengths are exact, and runs `llmsim-bench` against it over the compose network. The benchmark's settings are the `bench` service's `command` in `docker/docker-compose.yml` (by default a rate sweep of `qwen-mock` at 10, 50, 200 and 400 requests per second); edit them there. The report is printed, and the JSON results are written to `results/` on the host.

```bash
docker/bench.sh
```

The script builds the images if needed, runs the benchmark as your user (so `results/` is writable and the results file is yours), and then removes `bench-target`, also after a failure or Ctrl-C; a running `simulator` is left alone. It wraps the plain compose commands:

```bash
export HOST_UID=$(id -u) HOST_GID=$(id -g)
docker compose -f docker/docker-compose.yml --profile bench run --rm bench
docker compose -f docker/docker-compose.yml --profile bench rm --stop --force bench-target
```

`run --rm` removes the load generator's container when it exits, but Compose leaves the services it depended on, `bench-target` here, running.

The images run as an unprivileged user and a single uvicorn worker: metrics are kept in process memory, so several workers would each report only their own share. The server and the load generator share the host's CPUs; if the report says the client was saturated, its latencies are inflated.

# User Manual

The simulator is configured with two files: `models.yml` says **which models** it serves and **how fast** they answer, and `responses.yml` says **what** they answer. Clients talk to it exactly as they would to OpenAI. The sections below cover the API, both files, the load generator and the metrics.

Once the server is running (see Installation), any OpenAI client works with `base_url` pointed at it; no API key is checked:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
stream = client.chat.completions.create(
    model="qwen-mock",
    messages=[{"role": "user", "content": "Tell me about Hamlet"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

or with curl:

```bash
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "qwen-mock", "messages": [{"role": "user", "content": "Hello"}], "stream": true}'
```

`GET /v1/models` lists the configured models. The built-in `models.yml` has three: `qwen-mock` (fixed timing), `qwen-mock-batched` (the scheduler: latency depends on load) and `tiny-mock` (instant).

# API
The simulator serves `POST /v1/chat/completions` (streaming and non-streaming) and `GET /v1/models`, so OpenAI's SDKs and other OpenAI clients work unchanged with `base_url` pointed at it. A message's `content` may be a string or an array of text parts (`[{"type": "text", "text": "..."}]`); text parts are joined with no separator. The simulated models are text-only, so image, audio and file parts are rejected with a 400.

# Configuration YAML
`models.yml` lists the models the simulator serves; clients pick one with the request's `model` field. Hardware profiles are defined once and referenced by name, and each model that references a profile gets its own simulated GPU of that type. Memory is in GiB (`hbm_gib`); bandwidth (`memory_bandwidth_gbps`) and `peak_tflops` are decimal, as GPU vendors quote them.

```yaml
version: 1

hardware:
  h100-80gb:
    hbm_gib: 80
    memory_bandwidth_gbps: 3000
    peak_tflops: 989

models:
  - name: qwen-mock
    context_length: 32768
    tokenizer: regex          # default: regex
    engine: fixed_rate        # instant (default), fixed_rate or scheduler
    hardware: h100-80gb
    kv_cache:
      bytes_per_token: 524288
    performance:
      prefill_tokens_per_second: 100000
      decode_tokens_per_second: 250
    scheduler:
      max_batch_tokens: 8192
      max_num_sequences: 64
    latency:
      scheduler_overhead_ms: 2

  - name: tiny-mock           # only name and context_length are required
    context_length: 8192
```

Only `name` and `context_length` are required. The engine decides when tokens are sent:

| Engine | Timing | Needs |
|---|---|---|
| `instant` | Every token immediately | nothing |
| `fixed_rate` | First token after `scheduler_overhead_ms` + prompt tokens / `prefill_tokens_per_second`, then one token every 1 / `decode_tokens_per_second` | `performance` (`latency` optional) |
| `scheduler` | Continuous batching on one simulated GPU: requests share steps, wait when the KV cache or `max_num_sequences` is full, and are preempted when the cache runs out. Each step takes as long as a roofline model of the batch says. See `qwen-mock-batched` in the built-in [`models.yml`](src/llm_simulator/defaults/models.yml) for every field | `hardware` (with `peak_tflops`), `weights`, `kv_cache`, `scheduler` (`latency`, `efficiency` optional) |

The other sections are needed only by the engines that use them; the server refuses to start if a model's engine needs a section the model does not define, or if the file has an unknown key, a duplicate name or an unknown hardware profile.

The server reads its configuration from command-line flags and environment variables; a flag takes precedence over its variable. Relative paths resolve against the working directory. `llmsimulator --help` lists the flags, and `--version` prints the version.

| Flag | Variable | Default |
|---|---|---|
| `--models FILE` | `LLM_SIM_MODELS` | `builtin:models` |
| `--responses FILE` | `LLM_SIM_RESPONSES` | `builtin:responses` |
| `--host HOST` | `LLM_SIM_HOST` | `0.0.0.0` |
| `--port PORT` | `LLM_SIM_PORT` | `8080` |
| | `LLM_SIM_SUMMARY_REQUEST_SAMPLES` | `10000`: requests `/metrics/summary` keeps per-request values for |
| | `LLM_SIM_SUMMARY_TOKEN_SAMPLES` | `100000`: tokens it keeps per-token values (ITL, lag) for |
| | `LLM_SIM_SUMMARY_RATE_SECONDS` | `60`: seconds throughput is averaged over |

A file that cannot be read or does not validate stops the server before it starts, with a message naming the file and the problem.

**Built-in configs.** The package ships three files, so the server runs from any directory without configuration: `models` (the three demo models), `responses` (the classics conversation) and `bench-responses` (synthetic text for benchmarks). Any file option or variable accepts `builtin:<name>` in place of a path, for example `--responses builtin:bench-responses`. To start your own from one, print it: `llmsimulator --print-config models > my-models.yml`. In the repository they live in [`src/llm_simulator/defaults/`](src/llm_simulator/defaults/); an empty variable counts as unset.

# Response Configuration

`responses.yml` decides what the simulator says, never when: timing comes only from the model's engine in `models.yml`. The full reference, including validation rules and exactly how a variant is chosen, is [docs/schema.md](docs/schema.md); this section is a guide to writing one.

```yaml
version: 1
seed: 42                          # default seed for choosing variants

responses:                        # checked top to bottom; the first match wins
  - id: greeting
    match: { exact: ["hi", "hello"] }
    content: "Hello! How can I help you today?"

  - id: weather
    match: { regex: "weather in (?P<city>\\w+)" }
    variants:                     # one is chosen per request, deterministically
      - "It's sunny in {city} today."
      - "Expect light rain in {city}."

  - id: echo
    match: { starts_with: "/echo" }
    content: "You said: {last_user_message}"

  - id: long-answer
    match: { contains: "explain" }
    synthetic: { tokens: [200, 400] }   # filler text of 200 to 400 tokens

default:                          # required: used when nothing matches
  variants:
    - "I'm a simulated model ({model})."
    - "This is a canned response."
```

**Matching.** The input is the last `user` message, with surrounding whitespace removed. Each entry has one operator, and matching is case-insensitive:

| Operator | Matches when the input |
|---|---|
| `exact` | equals the value |
| `contains` | contains the value |
| `starts_with` | starts with the value |
| `regex` | contains a match for the pattern |

A list of values matches if any of them does. Order matters: put specific entries before general ones, because the first match wins.

**What to answer.** An entry, or `default`, has exactly one of:

- `content`: one fixed answer;
- `variants`: a list of answers, one chosen per request; each item is a string, or `{ synthetic: ... }`;
- `synthetic: { tokens: N }` or `{ tokens: [min, max] }`: lorem-ipsum filler of that many tokens, for when only the length matters.

**Templates.** `{last_user_message}` is the user's message as sent; `{model}` is the requested model; a regex's named groups, such as `(?P<city>\w+)` above, become `{city}`. Write `{{` and `}}` for literal braces.

**Determinism.** The same request always gets the same answer, across restarts: the variant, and the length within a synthetic range, come from a hash of the seed, the entry and the input. A request's `seed` parameter overrides the file's `seed`, so a client can ask for a different variant of the same prompt. The request's `max_tokens` (or `max_completion_tokens`) then cuts the answer, with `finish_reason` `length` if it did.

**Checking a file.** The server validates the file at startup and refuses to start with a message naming the entry for an unknown key, a duplicate `id`, a regex that does not compile, or a placeholder that does not exist.

**Tips.**

- For load tests, use `synthetic` answers longer than any `max_tokens` you send, so `max_tokens` sets the output length exactly: that is what the built-in [`bench-responses`](src/llm_simulator/defaults/bench-responses.yml) does, and what `llmsim-bench` assumes.
- A `regex` whose matches must contain a plain word (`\bhamlet\b`) is skipped cheaply when the input lacks it, so many entries stay fast on long prompts. Patterns made only of classes and wildcards (`^.{500,}`) run on every request; put them late.
- The built-in [`responses`](src/llm_simulator/defaults/responses.yml) is a larger example: a conversation about public-domain classics, with 1,000 answers. Pass your own file with `--responses` to replace it.

# Benchmarking
`llmsim-bench` sends streaming chat completions to any OpenAI-compatible server and measures them as a client sees them. It is installed with the package, alongside `llmsimulator`.

For a controlled output length, start the simulator with the benchmark responses, where every response is long synthetic text, so `--max-tokens` sets the exact number of output tokens:

```bash
poetry run llmsimulator --responses builtin:bench-responses
poetry run llmsim-bench --model qwen-mock --concurrency 8 --requests 100 \
    --input-tokens 512 --max-tokens 256 --output run.json
```

```
qwen-mock @ http://localhost:8080/v1 · closed loop · concurrency 8 · warm-up 8 · 100 requests · synthetic 512 input tokens → 256 tokens
requests    100 ok · 0 failed · finish length 100
duration    13.41 s
throughput  7.46 req/s · 1,909 output tok/s · 5,728 total tok/s
client      loop lag p50 0.45 ms · p99 1.28 ms · max 1.38 ms · CPU 26% · healthy
server      http://localhost:8080/metrics/summary · 100 requests (client 100 ✓)

latency           mean       min       p50       p90       p95       p99       max
  TTFT        11.46 ms   7.83 ms  10.26 ms  16.25 ms  18.65 ms  23.98 ms  25.14 ms
  ITL          4.00 ms   0.01 ms   3.81 ms   4.85 ms   4.97 ms   5.18 ms   8.08 ms
  TPOT         4.00 ms   3.95 ms   4.00 ms   4.01 ms   4.01 ms   4.01 ms   4.02 ms
  E2E          1.031 s   1.028 s   1.030 s   1.034 s   1.035 s   1.036 s   1.041 s
tokens/req        mean       min       p50       max
  prompt           512       512       512       512
  output           256       256       256       256

vs server       client    server simulated delivery*
  TTFT p50    10.26 ms   8.02 ms   7.12 ms   2.24 ms
  TTFT p99    23.98 ms  18.59 ms   7.12 ms   5.40 ms
  ITL p50      3.81 ms   3.84 ms         —  -0.03 ms
  ITL p99      5.18 ms   5.04 ms         —   0.14 ms
  TPOT p50     4.00 ms   4.00 ms         —   0.00 ms
  E2E p50      1.030 s   1.028 s   1.027 s   2.56 ms
  E2E p99      1.036 s   1.032 s   1.027 s   4.69 ms
  out tok/s      1,909     1,909
fidelity    server lag p50 0.67 ms · p99 1.43 ms · max 13.97 ms
* client percentile − server percentile
```

**Reading the comparison.** When the target serves `/metrics/summary` (the simulator does), the bench resets it after warm-up and reads it after the run, so both sides cover the same requests. `client` is when the bytes arrived, `server` is when the simulator released each token, and `simulated` is when its engine scheduled it. `server − simulated` is the simulator running late (its lag; the `fidelity` line shows lag over every token), and `delivery` (`client − server`) is the cost of getting tokens to the client: HTTP, the network and the client's own parsing. In the run above, the simulator was 0.9 ms late at the TTFT median and delivery added 2.2 ms; at p99 the server itself was 11 ms behind its schedule, because the 8 streams' requests arrive together. Delivery is a difference of percentiles, not of individual requests, and can be slightly negative for ITL when chunks arrive in bursts. If the server finished more requests than the client sent, a warning says other traffic mixed into its figures. Against a target without the endpoint (vLLM, OpenAI), the report says `server not available` and carries on; `--no-server-metrics` skips the comparison and `--server-metrics-url` points it elsewhere.

A run first sends `--warmup` unmeasured requests (default: one per concurrent stream), waits for them to finish, then measures `--requests` requests (default 100) or, with `--duration S`, every request sent in the next `S` seconds, and waits for the last ones to finish. If every warm-up request fails, the run stops there. The `client` line shows how late the client's own event loop woke and how much CPU it used; when the loop was late by more than 5 ms at p99, a warning says the client, not the server, inflated the latencies.

Latency and throughput cover successful requests; failures are counted and grouped by error. Warnings appear when responses end before `--max-tokens` or when a server packs several tokens into one chunk. `--output` writes the results as JSON (config with the API key redacted, environment, statistics) for comparing runs.

Prompts come from one source per run:

| Option | Prompts |
|---|---|
| `--input-tokens N` | A different synthetic prompt per request, each exactly `N` prompt tokens as the simulator's `usage` reports them (chat overhead included), repeatable with `--seed` |
| `--dataset FILE.jsonl` | One `{"prompt": "...", "max_tokens": 64}` per line (`max_tokens` optional), cycled if there are fewer lines than requests |
| `--prompt TEXT` | The same prompt for every request (the default) |

`--prompt-prefix` is prepended to every prompt; with `bench-responses.yml`, `"[short] "` and `"[long] "` select responses of 32–128 and 1024–4096 tokens. `--api-key` (default `$OPENAI_API_KEY`) is sent as a Bearer token; `--timeout` is the time allowed between chunks. The command exits with 1 if every request failed.

## Open loop
`--concurrency N` is a closed loop: N requests in flight, each worker sending its next request when its previous one finishes, so a slow server is offered less load and queueing stays hidden. `--rate R` is an open loop: requests arrive at R per second on a seeded Poisson schedule (exponential gaps, the standard model of independent users), whether or not earlier ones have finished. `--max-concurrency N` caps the requests in flight; an arrival that finds every slot busy waits for one.

```bash
poetry run llmsim-bench --model qwen-mock --rate 50 --max-concurrency 16 --duration 5 \
    --input-tokens 512 --max-tokens 64
```

```
qwen-mock @ http://localhost:8080/v1 · open loop · rate 50/s (Poisson, seed 0) · max concurrency 16 · warm-up 16 · 241 requests · synthetic 512 input tokens → 64 tokens
arrivals    target 50/s · offered 48.24/s · achieved 48.23/s · queue wait p50 0.88 ms · p99 55.35 ms · 66 waited

latency           mean       min       p50       p90       p95       p99       max
  TTFT         9.05 ms   7.76 ms   8.85 ms  10.07 ms  10.77 ms  12.20 ms  15.81 ms
  ...
  TTFT sched  16.72 ms   8.18 ms  10.13 ms  40.98 ms  45.44 ms  65.23 ms  84.48 ms
  E2E sched  268.91 ms 260.13 ms 262.25 ms 292.87 ms 297.49 ms 316.93 ms 337.31 ms
  queue wait   7.67 ms   0.13 ms   0.88 ms  31.74 ms  36.66 ms  55.35 ms  75.63 ms
```

Open-loop latency is shown two ways. `TTFT` and `E2E` are measured from when each request was actually sent, like the server's own figures and the comparison with them. `TTFT sched` and `E2E sched` are measured from the scheduled arrival, which is what an arriving user waits, including any wait for a slot. In the run above, 16 slots of about 0.26 s each serve about 61 requests per second, so 50 per second is about 82% utilisation: from the send TTFT p99 is 12 ms, but from the schedule it is 65 ms, because 66 requests queued for a slot. Measuring only from the send would hide that queue. The `arrivals` line compares the target rate with the schedule actually drawn (`offered`, which varies in a short run) and with the sends (`achieved`); a warning appears when requests waited or sends fell behind the schedule. Without `--max-concurrency`, nothing waits and the two measurements agree. The default warm-up is `--max-concurrency`, or 8 without a cap.

## Sweeps
Giving `--concurrency` or `--rate` several values runs the same benchmark at each, one after another, with the same prompts, and prints one row per level. Each level is a full run: warm-up, server reset, measurement, drain. A level that fails is reported and the sweep continues. `--output` writes every level's full results to one JSON file.

```bash
poetry run llmsim-bench --model qwen-mock --rate 10,50,200,400 --duration 5 \
    --input-tokens 512 --max-tokens 64
```

```
sweep rate · qwen-mock · synthetic 512 input tokens → 64 tokens · 5 s per level

rate              req/s  out tok/s   TTFT p50   TTFT p99    ITL p50    E2E p50    lag p99   dlv TTFT     client     errors achieved/s  sched p99     waited
10                 9.44        604    9.81 ms   11.71 ms    4.02 ms  262.04 ms    1.53 ms    2.02 ms         ok          0       9.94   13.06 ms          0
50                45.83      2,933    9.04 ms   14.44 ms    4.03 ms  261.24 ms    1.82 ms    1.29 ms         ok          0      48.23   16.16 ms          0
200              203.58     13,029    9.84 ms   52.87 ms    4.00 ms  262.02 ms   11.36 ms    1.70 ms         ok          0     214.40   53.14 ms          0
400              341.59     21,862  854.67 ms    1.287 s    0.02 ms  856.07 ms    1.181 s   36.60 ms         ok          0     415.93    1.288 s          0
```

Read a sweep for the level where latency breaks away while throughput stops keeping pace. Here the simulator keeps its timeline up to 200 requests per second (13,000 output tokens per second; server lag p99 11 ms). At 400 per second it falls more than a second behind (lag p99 1.18 s), TTFT jumps to 855 ms, and overdue tokens go out in bursts (ITL p50 0.02 ms). The client stayed healthy throughout, so the limit is the simulator's own process. Today's engines give every request its own timeline without a capacity limit, so this is the server process's ceiling, not model-style queueing; that arrives with the scheduler engine.

# Metrics

`GET /metrics` exposes the simulator's server-side metrics in Prometheus text format, labelled by `model`: TTFT, inter-token latency, TPOT, E2E, queue, prefill and decode time, request sizes, request and token counters, running requests, and `llm_sim_token_lag_seconds`, which shows how closely the simulator kept to its configured timeline. Models on the `scheduler` engine also export what each step did: queue and batch sizes, KV-cache usage, preemptions, why admission stopped, step durations and whether each step was compute- or memory-bound (`llm_sim_scheduler_*`, `llm_sim_kv_cache_*`, `llm_sim_preempt*`); `GET /metrics/summary` has the same as a `scheduler` section. Times are measured when the engine releases each token. See [docs/architecture.md](docs/architecture.md#metrics) for the full catalogue.

`GET /metrics/summary` returns the same measurements as JSON with exact percentiles (mean, min, p50, p90, p95, p99, max) over recent traffic: the last 10,000 requests for per-request values and the last 100,000 tokens for ITL and lag, plus throughput over the last 60 s. `POST /metrics/summary/reset` starts a fresh measurement period, for example before a benchmark run.

```json
{
  "since_reset_seconds": 2.34,
  "models": {
    "qwen-mock": {
      "engine": "fixed_rate",
      "requests": {"running": 0, "finished": {"stop": 3, "length": 0, "cancelled": 0, "error": 0}},
      "tokens": {"prompt": 24, "generation": 54},
      "throughput": {"window_seconds": 2.34, "requests_per_second": 1.28, "output_tokens_per_second": 23.05},
      "latency_seconds": {
        "ttft": {"count": 3, "samples": 3, "mean": 0.0073, "min": 0.0023, "p50": 0.0023, "p90": 0.0144, "p95": 0.0159, "p99": 0.0171, "max": 0.0174},
        "itl": {"...": "..."}, "tpot": {}, "e2e": {}, "queue": {}, "prefill": {}, "decode": {}, "lag": {}
      },
      "tokens_per_request": {"prompt": {}, "generation": {}}
    }
  }
}
```

`count` is how many values were recorded since the last reset; `samples` is how many the window still holds, which the statistics are computed from.

| Category       | Metric                       | Meaning                               |
| -------------- | ---------------------------- | ------------------------------------- |
| **Latency**    | TTFT                         | Time until first token                |
|                | TPOT                         | Average time per output token         |
|                | ITL                          | Time between individual output tokens |
|                | E2E latency                  | Total request latency                 |
|                | Queue latency                | Time waiting for scheduler            |
|                | Prefill latency              | Time processing input                 |
|                | Decode latency               | Time generating output                |
| **Throughput** | Requests/sec                 | Completed requests per second         |
|                | Input tok/sec                | Input processing throughput           |
|                | Output tok/sec               | Generation throughput                 |
|                | Aggregate tok/sec            | Input + output throughput             |
| **Serving**    | Active sequences             | Requests currently being served       |
|                | Batch size                   | Sequences in a scheduling iteration   |
|                | Scheduled tokens             | Tokens scheduled per iteration        |
| **Resources**  | KV-cache utilization         | KV capacity consumed                  |
|                | GPU memory utilization       | Simulated HBM consumption (planned; KV-cache utilization is the live memory measure today) |
|                | Compute utilization          | Simulated compute saturation (planned) |
|                | Memory bandwidth utilization | Simulated bandwidth saturation (planned) |

# Status and roadmap

Version 0.1.0. Until 1.0, the formats of `models.yml` and `responses.yml`, the metric names and the `llmsim-bench` options may still change.

## What works today

- **API**: `POST /v1/chat/completions`, streaming and non-streaming, with usage, `seed`, `max_tokens` and `max_completion_tokens`, text content parts and OpenAI-style errors; `GET /v1/models`. OpenAI's SDKs work unchanged.
- **Engines**: `instant`, `fixed_rate`, and `scheduler`, which simulates continuous batching on a GPU: a paged KV cache with admission against a watermark, always-chunked prefill, preemption by recomputation, and step times from a roofline model with configurable efficiency and per-step overhead.
- **Responses**: rule-based, deterministic answers from `responses.yml`, with variants, templates and synthetic text.
- **Metrics**: Prometheus `/metrics` and an exact-percentile `/metrics/summary`, per request and, for the scheduler, per step (batch, KV usage, preemptions, why requests waited).
- **Load generator**: `llmsim-bench`, closed and open loop, sweeps, datasets, client health checks and comparison with the server's own measurements.
- **Deployment**: Docker images and a compose setup for the simulator and benchmarks.
- **Validation**: each scheduler decision has been checked against a prediction made before running it; step times match the roofline within 0.3% from 1 to 64 streams. Results and method are in [docs/performance-test.md](docs/performance-test.md).

## Known limits

- **Concurrency per process.** Observed per-token timing stays within 5 ms of the simulated timeline up to about 25 concurrent streams with the `scheduler` engine, fewer with short outputs; simulated step times stay exact beyond that. Our own request handling also adds a few milliseconds of queue time while the engine is busy. See "Known limits" in [docs/architecture.md](docs/architecture.md).
- **Absolute numbers are uncalibrated.** The efficiency settings default to an ideal GPU, so latencies are optimistic until fitted to a measurement of a real server.
- **Slowness, not failures.** Queueing, preemption stalls and saturation are simulated; errors and dropped connections are not yet.
- **One process, one simulated GPU per model.** Run several processes for several replicas.

## Planned

Roughly in order of priority:

1. **Failure injection**: configurable error responses (429, 500, 503 with OpenAI-shaped bodies), streams that stop mid-response, and latency spikes, by rate or by rule, so retries, timeouts and circuit breakers can be tested.
2. **A faster per-token path**: cheaper SSE encoding, per-token metrics and request handling, which measurements suggest would roughly double the concurrency one process serves faithfully. The changes and their expected gains are listed under "Fidelity and performance: the per-token path" in [docs/architecture.md](docs/architecture.md).
3. **More of the API**: `stop`, `n`, tool calls, `response_format` and logprobs.
4. **Calibration**: fitting the efficiency and overhead settings to one run against a real server.
5. **Prefix caching**: shared prompt prefixes such as system prompts skip prefill, as in vLLM and SGLang.
6. **Resource metrics**: simulated GPU memory, compute and memory-bandwidth utilisation.
7. **Scale-out**: tensor parallelism in the cost model, several replicas behind a load balancer in compose, and a virtual-time mode for capacity planning beyond what can be streamed in real time.

Not planned: request priorities and speculative decoding.

Contributions and issues are welcome.

