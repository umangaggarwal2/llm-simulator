"""The llmsim-bench command: arguments in, a run, a report out.

The only module that reads arguments and the environment and wires the
others together, as runtime.py does for the server.
"""

import argparse
import asyncio
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, List, NamedTuple, Optional, Sequence

from llm_simulator.bench import workload as workloads
from llm_simulator.bench.config import BenchConfig
from llm_simulator.bench.records import RunRecords
from llm_simulator.bench.health import LoopMonitor
from llm_simulator.bench.report import (
    format_progress,
    format_result,
    format_sweep,
    write_json,
    write_sweep_json,
)
from llm_simulator.bench.results import BenchResult, failure, summarize
from llm_simulator.bench.runner import WarmupFailed, run_level
from llm_simulator.bench.sweep import levels, sweep
from llm_simulator.bench.workload import Workload

INSTALL_HINT = "llmsim-bench needs the bench extra: poetry install -E bench"


class Arguments(NamedTuple):
    # The run's config; for a sweep, its first level.
    config: BenchConfig
    # Where to write the results file, if anywhere.
    output: Optional[Path]
    # The swept parameter and its values, when one was given several values.
    sweep: Optional[str] = None
    values: Sequence[float] = ()


def _levels(kind: Callable[[str], float]) -> Callable[[str], List[float]]:
    """An argparse type for one value or a comma-separated list of them."""

    def parse(text: str) -> List[float]:
        try:
            return [kind(part) for part in text.split(",") if part.strip()]
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid list of numbers: {text!r}")

    return parse


def parse_args(argv: Optional[Sequence[str]] = None) -> Arguments:
    defaults = BenchConfig(model="-")
    parser = argparse.ArgumentParser(
        prog="llmsim-bench",
        description="Load-test an OpenAI-compatible chat completions server.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=defaults.base_url)

    load = parser.add_argument_group(
        "load", "a list of values (1,8,32) runs a sweep, one level per value"
    )
    shape = load.add_mutually_exclusive_group()
    shape.add_argument(
        "--concurrency",
        type=_levels(int),
        help="closed loop: requests in flight at once (default: 1)",
    )
    shape.add_argument(
        "--rate",
        type=_levels(float),
        help="open loop: requests per second, arriving on a Poisson schedule",
    )
    load.add_argument(
        "--max-concurrency",
        type=int,
        help="open loop: at most this many in flight; later arrivals wait",
    )
    length = load.add_mutually_exclusive_group()
    length.add_argument(
        "--requests",
        type=int,
        help="requests to measure (default: 100)",
    )
    length.add_argument(
        "--duration",
        type=float,
        help="seconds to measure for, instead of a number of requests",
    )
    load.add_argument(
        "--warmup",
        type=int,
        help="unmeasured requests sent first (default: one per stream, or 8)",
    )

    workload = parser.add_argument_group(
        "workload", "prompts come from --input-tokens, --dataset or --prompt"
    )
    workload.add_argument(
        "--max-tokens",
        type=int,
        default=defaults.max_tokens,
        help="max_completion_tokens of every request (a dataset line may override)",
    )
    source = workload.add_mutually_exclusive_group()
    source.add_argument(
        "--input-tokens",
        type=int,
        help="synthetic prompts of exactly this many prompt tokens on the simulator",
    )
    source.add_argument(
        "--dataset",
        type=Path,
        help='JSONL file, one {"prompt": ..., "max_tokens": ...} per line',
    )
    workload.add_argument(
        "--prompt", default=defaults.prompt, help="fixed prompt, used otherwise"
    )
    workload.add_argument(
        "--prompt-prefix", default="", help='prepended to every prompt, e.g. "[short] "'
    )
    workload.add_argument(
        "--seed", type=int, default=defaults.seed, help="seed for synthetic prompts"
    )

    connection = parser.add_argument_group("connection")
    connection.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="sent as a Bearer token (default: $OPENAI_API_KEY)",
    )
    connection.add_argument(
        "--timeout",
        type=float,
        default=defaults.timeout,
        help="seconds to wait for the next chunk",
    )

    server = parser.add_argument_group(
        "server comparison",
        "compares with the target's /metrics/summary, when it has one",
    )
    server.add_argument(
        "--no-server-metrics",
        action="store_true",
        help="do not reset or read the server's summary",
    )
    server.add_argument(
        "--server-metrics-url",
        help="the summary endpoint (default: the base URL with /v1 → /metrics/summary)",
    )

    parser.add_argument("--output", type=Path, help="write the results as JSON here")
    args = parser.parse_args(argv)
    parameter, values = None, ()
    for name in ("concurrency", "rate"):
        given = getattr(args, name)
        if given is not None:
            if not given:
                parser.error(f"--{name} needs at least one value")
            parameter, values = name, given
    try:
        config = BenchConfig(
            model=args.model,
            base_url=args.base_url,
            concurrency=values[0] if parameter == "concurrency" else None,
            rate=values[0] if parameter == "rate" else None,
            max_concurrency=args.max_concurrency,
            requests=args.requests,
            duration=args.duration,
            warmup=args.warmup,
            max_tokens=args.max_tokens,
            prompt=args.prompt,
            input_tokens=args.input_tokens,
            dataset=args.dataset,
            prompt_prefix=args.prompt_prefix,
            seed=args.seed,
            api_key=args.api_key,
            timeout=args.timeout,
            server_metrics=not args.no_server_metrics,
            server_metrics_url=args.server_metrics_url,
        )
        # Every level must be valid, not just the first.
        for value in values[1:]:
            levels(config, parameter, [value])[0]
    except ValueError as exc:
        parser.error(str(exc))
    if len(values) > 1:
        return Arguments(config, args.output, parameter, values)
    return Arguments(config, args.output)


def build_workload(config: BenchConfig) -> Workload:
    """The workload the config asks for; raises ValueError or OSError if it
    cannot be built (a too-small input_tokens, an unreadable dataset)."""
    prefix = config.prompt_prefix
    if config.input_tokens is not None:
        return workloads.synthetic(
            config.input_tokens, config.max_tokens, prefix=prefix, seed=config.seed
        )
    if config.dataset is not None:
        return workloads.from_jsonl(config.dataset, config.max_tokens, prefix)
    return workloads.fixed(config.prompt, config.max_tokens, prefix)


async def run(config: BenchConfig, workload: Workload) -> RunRecords:
    """Warm up, measure and return the measured records with their window.

    Raises WarmupFailed if no warm-up request succeeded.
    """
    from llm_simulator.bench.client import open_session, stream_chat
    from llm_simulator.bench.server import ServerMetrics, summary_url

    async with open_session(config) as session:
        server = None
        if config.server_metrics:
            url = config.server_metrics_url or summary_url(config.base_url)
            server = ServerMetrics(session, url, config.model)

        def send(index: int):
            return stream_chat(session, config, index, workload.request(index))

        closed = config.mode == "closed"
        return await run_level(
            send,
            concurrency=config.closed_concurrency if closed else None,
            rate=config.rate,
            max_concurrency=config.max_concurrency,
            seed=config.seed,
            requests=config.measured_requests,
            duration=config.duration,
            warmup=config.warmup_requests,
            warmup_concurrency=config.warmup_concurrency,
            monitor=LoopMonitor(),
            server=server,
        )


async def run_result(config: BenchConfig, workload: Workload) -> BenchResult:
    return summarize(await run(config, workload), config, workload.description)


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        import aiohttp  # noqa: F401
    except ImportError:
        print(INSTALL_HINT, file=sys.stderr)
        return 1
    args = parse_args(argv)
    config = args.config
    try:
        workload = build_workload(config)
    except (ValueError, OSError) as exc:
        print(f"llmsim-bench: error: {exc}", file=sys.stderr)
        return 2
    if args.sweep is not None:
        return _main_sweep(args, workload)
    try:
        result = asyncio.run(run_result(config, workload))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except WarmupFailed as exc:
        print(f"llmsim-bench: {exc}; nothing was measured", file=sys.stderr)
        for message, count in Counter(map(failure, exc.records)).most_common(5):
            print(f"  {count} × {message}", file=sys.stderr)
        return 1

    print(format_result(result))
    if args.output is not None:
        write_json(result, args.output)
        print(f"\nresults written to {args.output}")
    # Failing outright (for example, no server) is an error; some failed
    # requests are a result.
    return 0 if result.ok else 1


def _main_sweep(args: Arguments, workload: Workload) -> int:
    configs = levels(args.config, args.sweep, args.values)

    def progress(number, level):
        print(format_progress(number, len(configs), level), flush=True)

    try:
        done = asyncio.run(
            sweep(
                args.sweep,
                configs,
                lambda config: run_result(config, workload),
                progress,
            )
        )
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    print()
    print(format_sweep(done, workload.description))
    if args.output is not None:
        write_sweep_json(done, args.output)
        print(f"\nresults written to {args.output}")
    return 0 if any(level.result and level.result.ok for level in done) else 1


if __name__ == "__main__":
    sys.exit(main())
