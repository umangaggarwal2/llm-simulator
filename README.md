# llm-simulator
A deterministic fake LLM that streams tokens through OpenAI compatible API, provides custom GPU configurations and simulates metrics.

The simulator models an LLM serving system at the token level. It measures client-visible latency and throughput while simulating the resource constraints that drive those metrics: GPU memory, compute, memory bandwidth, KV-cache capacity, batching, and scheduling.

The simulator does not measure real GPU utilization, memory bandwidth, or compute utilization. These are modeled metrics derived from the configured hardware and serving-capacity parameters.

# Installation

# User Manual

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
    engine: instant           # default: instant
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

Only `name` and `context_length` are required. The other sections are needed only by the engines that use them; the server refuses to start if a model's engine needs a section the model does not define, or if the file has an unknown key, a duplicate name or an unknown hardware profile.

The server reads its configuration from these environment variables:

| Variable | Default |
|---|---|
| `LLM_SIM_MODELS` | `config/models.yml` |
| `LLM_SIM_RESPONSES` | `config/responses.yml` |
| `LLM_SIM_HOST` | `0.0.0.0` |
| `LLM_SIM_PORT` | `8080` |

# Response Configuration
```
```

# Metrics

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

