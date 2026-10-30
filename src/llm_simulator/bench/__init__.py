"""llmsim-bench: a load generator for OpenAI-compatible chat completion servers.

It measures latency as the client sees it (the third measurement layer, after
the engine's simulated timeline and the server's release times) and works
against any OpenAI-compatible endpoint, not only the simulator. Needs the
bench extra: poetry install -E bench.
"""
