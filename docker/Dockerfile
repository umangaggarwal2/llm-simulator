# Images for llm-simulator: build from the repository root with
#   docker build -f docker/Dockerfile --target server -t llm-simulator .
#   docker build -f docker/Dockerfile --target bench  -t llm-simulator-bench .
# or let docker/docker-compose.yml build both.

# --- builder: locked dependencies and the package, in a self-contained venv ---
FROM python:3.10-slim AS builder
ENV POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN pip install "poetry>=2,<3"
WORKDIR /app
# Dependencies first, from the lock file: this layer is rebuilt only when
# the dependencies change, not on every source edit.
COPY pyproject.toml poetry.lock README.md ./
RUN poetry install --only main --no-root
# The package itself, as a wheel: poetry would install it in editable mode,
# pointing at /app/src, which the runtime images do not have.
COPY src ./src
RUN poetry build --format wheel \
    && .venv/bin/python -m pip install --no-deps dist/*.whl

# The load generator also needs the bench extra (aiohttp).
FROM builder AS builder-bench
RUN poetry install --only main --extras bench --no-root

# --- server: the simulator ---
FROM python:3.10-slim AS server
RUN useradd --create-home --uid 1000 sim
COPY --from=builder /app/.venv /app/.venv
# Default configs, used when none are mounted; also bench-responses.yml.
COPY config /app/config
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    LLM_SIM_HOST=0.0.0.0 \
    LLM_SIM_PORT=8080 \
    LLM_SIM_MODELS=/app/config/models.yml \
    LLM_SIM_RESPONSES=/app/config/responses.yml
USER sim
EXPOSE 8080
# The slim image has no curl; Python can make the request itself.
HEALTHCHECK --interval=5s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2)"]
# One process, one uvicorn worker: metrics live in process memory, so
# several workers would each keep their own /metrics and summary.
CMD ["llmsimulator"]

# --- bench: the load generator, llmsim-bench ---
FROM python:3.10-slim AS bench
RUN useradd --create-home --uid 1000 sim
COPY --from=builder-bench /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1
USER sim
ENTRYPOINT ["llmsim-bench"]
CMD ["--help"]
