#!/usr/bin/env bash
# Run the benchmark defined by the bench service in docker-compose.yml, then
# stop and remove the bench-target simulator it ran against. The simulator
# service, if running, is left alone.
#
#   docker/bench.sh            from anywhere; results land in results/
#
# Compose leaves dependencies running after `run`, and a plain `down` cannot
# see services in the bench profile, so without this the bench-target
# container would outlive the benchmark.
set -euo pipefail
cd "$(dirname "$0")/.."

compose=(docker compose -f docker/docker-compose.yml --profile bench)
# Run the load generator as you, so the results file is yours and results/
# is writable (see the README).
export HOST_UID="${HOST_UID:-$(id -u)}" HOST_GID="${HOST_GID:-$(id -g)}"

cleanup() {
    "${compose[@]}" rm --stop --force bench-target >/dev/null 2>&1 || true
    # Nothing else of the project running (no simulator): remove the network too.
    if [ -z "$(docker compose -f docker/docker-compose.yml --profile '*' ps -q)" ]; then
        docker compose -f docker/docker-compose.yml --profile '*' down >/dev/null 2>&1 || true
    fi
}
# Also on failure or Ctrl-C.
trap cleanup EXIT

"${compose[@]}" run --rm --build bench
