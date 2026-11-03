"""SchedulerEngine: continuous batching over a shared, paged KV cache.

Three tiers, with dependencies pointing down only:

- kv_cache.py, cost_model.py, sequence.py: resources, step cost and
  per-request bookkeeping. Pure, and independent of each other.
- policy.py: admission, batch building and preemption. Pure.
- driver.py: the asyncio driver, the only part that deals with time.

The pure tiers never import asyncio or time, so they are tested step by step
without a clock. See "Scheduler and KV-cache model" in docs/architecture.md.
"""

from llm_simulator.simulator.scheduler.driver import EngineFailed, SchedulerEngine

__all__ = ["EngineFailed", "SchedulerEngine"]
