#!/bin/bash
# Shared Python launcher for scheduled portfolio-lab jobs.
#
# Cron/Hermes run with a sparse environment; use the project dependency
# runtime instead of whatever bare python3 happens to resolve to.
# On cursor-box the venv python is musl-linked and /lib/ld-musl-x86_64.so.1
# is absent, so prefer the toolchain loader over bare exec / uv run.
#
# PROJECT_DIR defaults to this script's repo root (not a host-specific
# /root/projects/... path). Cron wrappers may still export
# PORTFOLIO_LAB_PROJECT_DIR for the sg01 layout.
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="${PORTFOLIO_LAB_PROJECT_DIR:-$REPO_ROOT}"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "[python_runtime] project directory not found: $PROJECT_DIR" >&2
    exit 127
fi

cd "$PROJECT_DIR"
export PORTFOLIO_LAB_ENABLE_ML="${PORTFOLIO_LAB_ENABLE_ML:-0}"
ulimit -c 0 2>/dev/null || true

if [ -n "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/src:$PYTHONPATH"
else
    export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/src"
fi

TOOLCHAIN_ROOT="${PORTFOLIO_LAB_TOOLCHAIN_ROOT:-/home/box/.local/share/portfolio-lab/toolchain}"
PYTHON_ROOT="${PORTFOLIO_LAB_PYTHON_ROOT:-$TOOLCHAIN_ROOT/python-root}"
BUILD_ROOT="${PORTFOLIO_LAB_ALPINE_BUILD_ROOT:-$TOOLCHAIN_ROOT/alpine-build-root}"
ALPINE_ROOT="${PORTFOLIO_LAB_ALPINE_ROOT:-$TOOLCHAIN_ROOT/alpine-root}"
MUSL_LOADER="${PORTFOLIO_LAB_MUSL_LOADER:-$BUILD_ROOT/lib/ld-musl-x86_64.so.1}"
LIB_DIRS="$PYTHON_ROOT/lib:$BUILD_ROOT/usr/lib:$BUILD_ROOT/lib:$ALPINE_ROOT/usr/lib:$ALPINE_ROOT/lib"

VENV_PY="$PROJECT_DIR/.venv/bin/python"
if [ -x "$MUSL_LOADER" ] && [ -e "$VENV_PY" ]; then
    # Only inject Alpine libs when invoking via the musl loader. The agent
    # uv / venv fallbacks must stay clean (cursor-box ~/.local/bin/uv wrapper
    # already breaks pytest when LD_LIBRARY_PATH is set).
    if [ -n "${LD_LIBRARY_PATH:-}" ]; then
        export LD_LIBRARY_PATH="$LIB_DIRS:$LD_LIBRARY_PATH"
    else
        export LD_LIBRARY_PATH="$LIB_DIRS"
    fi
    exec "$MUSL_LOADER" --library-path "$LD_LIBRARY_PATH" "$VENV_PY" "$@"
fi

AGENT_UV="${PORTFOLIO_LAB_AGENT_UV:-$PROJECT_DIR/scripts/agent_uv.sh}"
if [ -x "$AGENT_UV" ] && [ -f "$PROJECT_DIR/pyproject.toml" ]; then
    exec "$AGENT_UV" run python "$@"
fi

if command -v uv >/dev/null 2>&1 && [ -f "$PROJECT_DIR/pyproject.toml" ]; then
    exec uv run python "$@"
fi

if [ -x "$VENV_PY" ]; then
    exec "$VENV_PY" "$@"
fi

echo "[python_runtime] musl loader/uv missing and $VENV_PY is missing" >&2
exit 127
