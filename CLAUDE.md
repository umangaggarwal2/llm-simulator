# llm-simulator

A GPU-free, deterministic stand-in for an OpenAI-compatible LLM server whose
latency responds to load like a real serving engine. Read README.md for usage,
docs/architecture.md for the design, docs/performance-test.md for how timing is
validated, docs/schema.md for responses.yml.

## Commands

- Install: `poetry install -E bench`
- Run: `poetry run llmsimulator` (port 8080); flags `--host`, `--port`, `--models`, `--responses`
  override `LLM_SIM_HOST`, `LLM_SIM_PORT`, `LLM_SIM_MODELS`, `LLM_SIM_RESPONSES`
- Tests: `poetry run pytest -q`; skip the HTTP ones with `-m "not integration"`
- Format: `poetry run black src test`; `black --check src test` should pass before committing
- Benchmark: `poetry run llmsim-bench --model <name> --base-url http://localhost:8080/v1 ...`,
  against a server started with `--responses builtin:bench-responses`
- Built-in configs (`models`, `responses`, `bench-responses`) live in
  `src/llm_simulator/defaults/` and ship with the package; `config/validation/` holds the
  models used to validate the simulator

## Architecture rules

- `llm/` decides what is said and never deals with time; `simulator/` decides when and
  never deals with content. They share only `types.py`, which imports nothing from the project.
- `simulator/` never imports `metrics/`: engines report through callables (`on_step`)
  that `runtime.py` wires to the recorders.
- `runtime.py` is the composition root and the only reader of the parsed config;
  engines and components take plain parameters, never config classes.
- The scheduler has three tiers: `sequence.py`, `kv_cache.py` and `cost_model.py` (pure,
  independent), `policy.py` (pure decisions), `driver.py` (the only code with time and
  asyncio). Keep asyncio and clocks out of the pure tiers.

## Fidelity rules

- When in doubt, match real OpenAI API behaviour.
- Simulated time is anchored: a late wake-up shows as `lag` and must never shift the
  simulated timeline. Never smooth or spread a step's token releases to hide lag.
- Admission must never use a plan's real output length; a real server cannot know it.
- A metric's meaning must not change silently; renaming or redefining one is a breaking change.

## Conventions

- Match the surrounding code's naming and comment density; comments explain why.
- Tests: pure tiers are tested step by step without a clock; timed code uses an injected
  fake clock whose sleep yields to the event loop.
- Config schemas reject unknown keys; a config error must stop the server at startup
  with a message naming the entry.
- Update the docs in the same change as the code they describe.
- Performance claims need a prediction first and a measurement after; profile with
  py-spy `--nonblocking` only (blocking mode distorts results), and record results in
  docs/architecture.md under "Scheduler validation".
