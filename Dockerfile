# =============================================================================
# Stage 1: Builder (Python 3.12-slim + Poetry)
# =============================================================================
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip poetry

COPY pyproject.toml poetry.lock README.md ./
COPY src/ ./src/

RUN poetry install --only main

# =============================================================================
# Stage 2: Runtime (Python 3.12-slim)
# =============================================================================
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r appgroup && useradd -r -g appgroup -m appuser

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY scripts/ /app/scripts/

RUN mkdir -p /app/data/raw /app/data/processed /app/models /app/reports /app/mlruns \
    && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=25s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "credit_guard.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
