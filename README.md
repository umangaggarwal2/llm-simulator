# llm-simulator
A deterministic fake LLM that streams tokens through OpenAI compatible API, provides custom GPU configurations and simulates metrics.

The simulator models an LLM serving system at the token level. It measures client-visible latency and throughput while simulating the resource constraints that drive those metrics: GPU memory, compute, memory bandwidth, KV-cache capacity, batching, and scheduling.

The simulator does not measure real GPU utilization, memory bandwidth, or compute utilization. These are modeled metrics derived from the configured hardware and serving-capacity parameters.

# Installation

# User Manual

# API
The simulator serves `POST /v1/chat/completions` (streaming and non-streaming) and `GET /v1/models`, so OpenAI's SDKs and other OpenAI clients work unchanged with `base_url` pointed at it. A message's `content` may be a string or an array of text parts (`[{"type": "text", "text": "..."}]`); text parts are joined with no separator. The simulated models are text-only, so image, audio and file parts are rejected with a 400.

# Configuration YAML
`config/models.yml` lists the models the simulator serves; clients pick one with the request's `model` field. Hardware profiles are defined once and referenced by name, and each model that references a profile gets its own simulated GPU of that type.

```yaml
version: 1

hardware:
  h100-80gb:
    hbm_gb: 80
    memory_bandwidth_gbps: 3000

models:
  - name: qwen-mock
    context_length: 32768
    tokenizer: regex          # default: regex
    engine: fixed_rate        # instant (default) or fixed_rate
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

The other sections are needed only by the engines that use them; the server refuses to start if a model's engine needs a section the model does not define, or if the file has an unknown key, a duplicate name or an unknown hardware profile.

The server reads its configuration from these environment variables:

| Variable | Default |
|---|---|
| `LLM_SIM_MODELS` | `config/models.yml` |
| `LLM_SIM_RESPONSES` | `config/responses.yml` |
| `LLM_SIM_HOST` | `0.0.0.0` |
| `LLM_SIM_PORT` | `8080` |
| `LLM_SIM_SUMMARY_REQUEST_SAMPLES` | `10000`: requests `/metrics/summary` keeps per-request values for |
| `LLM_SIM_SUMMARY_TOKEN_SAMPLES` | `100000`: tokens it keeps per-token values (ITL, lag) for |
| `LLM_SIM_SUMMARY_RATE_SECONDS` | `60`: seconds throughput is averaged over |

# Response Configuration
```
```

# Metrics

`GET /metrics` exposes the simulator's server-side metrics in Prometheus text format, labelled by `model`: TTFT, inter-token latency, TPOT, E2E, queue, prefill and decode time, request sizes, request and token counters, running requests, and `llm_sim_token_lag_seconds`, which shows how closely the simulator kept to its configured timeline. Times are measured when the engine releases each token. See [docs/architecture.md](docs/architecture.md#metrics) for the full catalogue.

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
|                | GPU memory utilization       | Simulated HBM consumption             |
|                | Compute utilization          | Simulated compute saturation          |
|                | Memory bandwidth utilization | Simulated bandwidth saturation        |

