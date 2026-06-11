# ─────────────────────────────────────────────────────────────
#  Portfolio-Lab — Multi-stage Docker build
#  Stage 1: Frontend build (Bun + Vite)
#  Stage 2: Python runtime + static frontend
# ─────────────────────────────────────────────────────────────

# ---------- Stage 1: Frontend ----------
FROM oven/bun:1 AS frontend

WORKDIR /app

# Install frontend dependencies
COPY package.json bun.lock ./
RUN bun install --frozen-lockfile

# Copy frontend source and build
COPY public/ public/
RUN mkdir -p public/data
COPY src/  src/
COPY index.html tsconfig.json vite.config.ts ./
RUN bun run build

# ---------- Stage 2: Python runtime ----------
FROM python:3.12-slim

LABEL maintainer="portfolio-lab"
LABEL description="All-Season Portfolio Lab — signal pipeline + dashboard"

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl git build-essential cron && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Python deps — install first for layer caching
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-group ml

# Copy Python source
COPY src/ src/
COPY scripts/ scripts/
COPY Makefile crontab ./

# Copy built frontend from stage 1
COPY --from=frontend /app/dist /app/dist
COPY --from=frontend /app/public/data /app/public/data

# Copy entrypoint script
COPY scripts/docker-entrypoint.sh /app/scripts/docker-entrypoint.sh
RUN chmod +x /app/scripts/docker-entrypoint.sh

# Create data directories and non-root user
RUN mkdir -p /app/data /app/public/data /app/data/signals /app/data/cache /app/data/logs && \
    adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app/data /app/public/data

# Environment defaults
ENV PORTFOLIO_LAB_ENABLE_ML=0
ENV CRON_BACKEND=tasker
ENV PYTHONUNBUFFERED=1

# Health check — verify tasker API is serving task state.
HEALTHCHECK --interval=5m --timeout=30s --retries=3 --start-period=60s \
    CMD python -c "import json, urllib.request; data=json.load(urllib.request.urlopen('http://127.0.0.1:8000/api/tasker/status', timeout=5)); assert data.get('backend') == 'tasker'" || exit 1

EXPOSE 8000

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
