#!/usr/bin/env bash
# Resolve a clean standalone uv for agent test runs (`uv run` / pytest).
#
# On cursor-box, ~/.local/bin/uv is a relocatable wrapper that injects Alpine
# musl LD_LIBRARY_PATH / compilers so `uv sync` can build native wheels. That
# same injection breaks `uv run pytest` (glibc host python, or musl python
# launched without the toolchain loader). Prefer the standalone payload next
# to the wrapper, then PATH uv, and never re-enter the wrapper. Native wheel
# installs on cursor-box still call ~/.local/bin/uv sync directly.
set -euo pipefail

_is_uv_wrapper() {
    local f="$1"
    [ -f "$f" ] || return 1
    # Binary payloads are not wrappers.
    head -n 1 "$f" 2>/dev/null | grep -q '^#!' || return 1
    grep -q 'Portfolio Lab user-owned wrapper for uv\|STANDALONE_UV=' "$f" 2>/dev/null
}

_standalone_next_to_wrapper() {
    local wrapper="$1"
    local bin_dir prefix
    bin_dir="$(CDPATH= cd -- "$(dirname -- "$wrapper")" && pwd)"
    prefix="$(CDPATH= cd -- "$bin_dir/.." && pwd)"
    printf '%s\n' "$prefix/share/portfolio-lab/toolchain/standalone/uv"
}

_resolve_uv() {
    local candidate path_uv standalone

    if [ -n "${PORTFOLIO_LAB_UV:-}" ] && [ -x "${PORTFOLIO_LAB_UV}" ]; then
        printf '%s\n' "$PORTFOLIO_LAB_UV"
        return 0
    fi

    if [ -n "${PORTFOLIO_LAB_TOOLCHAIN_ROOT:-}" ]; then
        candidate="${PORTFOLIO_LAB_TOOLCHAIN_ROOT}/standalone/uv"
        if [ -x "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    fi

    candidate="${HOME}/.local/share/portfolio-lab/toolchain/standalone/uv"
    if [ -x "$candidate" ]; then
        printf '%s\n' "$candidate"
        return 0
    fi

    path_uv="$(command -v uv 2>/dev/null || true)"
    if [ -n "$path_uv" ] && [ -x "$path_uv" ]; then
        if _is_uv_wrapper "$path_uv"; then
            standalone="$(_standalone_next_to_wrapper "$path_uv")"
            if [ -x "$standalone" ]; then
                printf '%s\n' "$standalone"
                return 0
            fi
            echo "[agent_uv] PATH uv is a toolchain wrapper without a standalone payload: $path_uv" >&2
            echo "[agent_uv] set PORTFOLIO_LAB_UV to a clean uv binary, or install Astral uv." >&2
            exit 127
        fi
        printf '%s\n' "$path_uv"
        return 0
    fi

    echo "[agent_uv] uv not found. Install https://docs.astral.sh/uv/ or set PORTFOLIO_LAB_UV." >&2
    exit 127
}

UV_BIN="$(_resolve_uv)"

if _is_uv_wrapper "$UV_BIN"; then
    standalone="$(_standalone_next_to_wrapper "$UV_BIN")"
    if [ -x "$standalone" ]; then
        UV_BIN="$standalone"
    else
        echo "[agent_uv] refusing to exec the LD_LIBRARY_PATH-injecting uv wrapper: $UV_BIN" >&2
        exit 127
    fi
fi

exec "$UV_BIN" "$@"
