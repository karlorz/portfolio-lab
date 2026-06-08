#!/usr/bin/env bash
# Repo-backed wrapper for the sg01 Hermes pytest watchdog.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$SCRIPT_DIR/pytest_watchdog.py" "$@"
