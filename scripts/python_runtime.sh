#!/bin/bash
# Shared Python launcher for scheduled portfolio-lab jobs.
#
# Cron/Hermes run with a sparse environment; use the project dependency
# runtime instead of whatever bare python3 happens to resolve to.
set -euo pipefail

PROJECT_DIR="${PORTFOLIO_LAB_PROJECT_DIR:-/root/projects/portfolio-lab}"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "[python_runtime] project directory not found: $PROJECT_DIR" >&2
    exit 127
fi

cd "$PROJECT_DIR"
export PORTFOLIO_LAB_ENABLE_ML="${PORTFOLIO_LAB_ENABLE_ML:-0}"

if [ -n "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/src:$PYTHONPATH"
else
    export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/src"
fi

if command -v uv >/dev/null 2>&1 && [ -f "$PROJECT_DIR/pyproject.toml" ]; then
    exec uv run python "$@"
fi

if [ -x "$PROJECT_DIR/.venv/bin/python" ]; then
    exec "$PROJECT_DIR/.venv/bin/python" "$@"
fi

echo "[python_runtime] uv not found and $PROJECT_DIR/.venv/bin/python is missing" >&2
exit 127
