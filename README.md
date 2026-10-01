# llm-simulator
A deterministic fake LLM that streams tokens through OpenAI compatible API, provides custom GPU configurations and simulates metrics.

The simulator models an LLM serving system at the token level. It measures client-visible latency and throughput while simulating the resource constraints that drive those metrics: GPU memory, compute, memory bandwidth, KV-cache capacity, batching, and scheduling.

The simulator does not measure real GPU utilization, memory bandwidth, or compute utilization. These are modeled metrics derived from the configured hardware and serving-capacity parameters.

# Installation

# User Manual

# Configuration YAML
```
model:
  name: qwen-mock
  context_length: 32768

hardware:
  hbm_gb: 80
  memory_bandwidth_gbps: 3000

performance:
  prefill_tokens_per_second: 100000
  decode_tokens_per_second: 250

scheduler:
  max_batch_tokens: 8192
  max_num_sequences: 64

kv_cache:
  bytes_per_token: 524288

latency:
  scheduler_overhead_ms: 2
```

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

