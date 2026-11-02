# Multi-stage Dockerfile for FastAPI Application
FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project specification files
COPY pyproject.toml uv.lock README.md ./

# Install dependencies into virtualenv
RUN uv sync --frozen --no-install-project

FROM python:3.11-slim AS runtime

WORKDIR /app

# Install runtime system libraries needed for PDF parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtualenv and app code from builder
COPY --from=builder /app/.venv /app/.venv
COPY app /app/app
COPY pyproject.toml README.md ./

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

