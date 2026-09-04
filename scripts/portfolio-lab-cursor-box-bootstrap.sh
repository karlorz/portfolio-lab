#!/bin/sh
# Portfolio Lab cursor-box user-owned dependency bootstrap runner.
# Strict BusyBox/POSIX /bin/sh syntax only.
#
# Supports stage-0 user-owned Python initialization when system Python is absent.
# dry-run and help are guaranteed non-mutating.
set -eu

PINNED_PYTHON_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20260901/cpython-3.11.16%2B20260901-x86_64-unknown-linux-musl-install_only.tar.gz"
PINNED_PYTHON_SHA256="dd9d55e2d83a90097b6c5c9d3b770e064f91278cf176a8f8369c11dcf1a4bc10"
EXPECTED_PYTHON_SHA256="$PINNED_PYTHON_SHA256"

TARGET_PREFIX="/home/box/.local"
STAGE0_ARCHIVE=""
COMMAND=""
IS_MUTATING=0
IS_DRY_RUN=0

# Parse arguments cleanly for both --prefix PATH and --prefix=PATH
prev_arg=""
for arg in "$@"; do
    if [ "$prev_arg" = "--prefix" ]; then
        TARGET_PREFIX="$arg"
        prev_arg=""
        continue
    fi
    case "$arg" in
        --prefix)
            prev_arg="--prefix"
            ;;
        --prefix=*)
            TARGET_PREFIX="${arg#--prefix=}"
            ;;
        --stage0-python-archive=*)
            STAGE0_ARCHIVE="${arg#--stage0-python-archive=}"
            ;;
        --override-sha256=python3=*)
            EXPECTED_PYTHON_SHA256="${arg#--override-sha256=python3=}"
            ;;
        --dry-run)
            IS_DRY_RUN=1
            ;;
        install|uninstall)
            COMMAND="$arg"
            IS_MUTATING=1
            ;;
        dry-run|verify|--help|-h|stage-0-info)
            COMMAND="$arg"
            ;;
    esac
done

# If uninstall was called with --dry-run, it is non-mutating
if [ "$COMMAND" = "uninstall" ] && [ "$IS_DRY_RUN" -eq 1 ]; then
    IS_MUTATING=0
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PORTFOLIO_LAB_BOOTSTRAP_PYTHON:-python3}"

# Determine if python3 is available
PY_AVAILABLE=0
if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PY_AVAILABLE=1
elif [ -x "$TARGET_PREFIX/bin/python3" ]; then
    PYTHON_BIN="$TARGET_PREFIX/bin/python3"
    PY_AVAILABLE=1
elif [ -x "$TARGET_PREFIX/share/portfolio-lab/toolchain/python-root/bin/python3" ]; then
    PYTHON_BIN="$TARGET_PREFIX/share/portfolio-lab/toolchain/python-root/bin/python3"
    PY_AVAILABLE=1
elif [ -x "$TARGET_PREFIX/share/portfolio-lab/toolchain/stage0-python/bin/python3" ]; then
    PYTHON_BIN="$TARGET_PREFIX/share/portfolio-lab/toolchain/stage0-python/bin/python3"
    PY_AVAILABLE=1
fi

if [ "$COMMAND" = "stage-0-info" ]; then
    echo "Stage-0 bootstrap status for prefix: $TARGET_PREFIX"
    echo "Python available: $PY_AVAILABLE"
    echo "Pinned Python standalone URL: $PINNED_PYTHON_URL"
    echo "Pinned Python SHA-256: $PINNED_PYTHON_SHA256"
    exit 0
fi

# When Python is absent:
if [ "$PY_AVAILABLE" -eq 0 ]; then
    if [ "$COMMAND" = "verify" ]; then
        echo "ERROR: Cannot verify toolchain on blank host: Python runtime is absent under $TARGET_PREFIX and system path." >&2
        exit 1
    fi

    if [ "$COMMAND" = "uninstall" ] && [ "$IS_DRY_RUN" -eq 1 ]; then
        # uninstall --dry-run without Python: non-mutating inspection of what would be removed
        echo "Uninstall dry-run for prefix: $TARGET_PREFIX (Python is absent)"
        MANIFEST_FILE="$TARGET_PREFIX/share/portfolio-lab/toolchain/bootstrap-manifest.json"
        if [ -f "$MANIFEST_FILE" ]; then
            echo "Manifest found at $MANIFEST_FILE"
        else
            echo "No manifest found at $TARGET_PREFIX; standard managed directories would be removed if present:"
            for d in bin share/portfolio-lab/toolchain; do
                if [ -d "$TARGET_PREFIX/$d" ]; then
                    echo "Would remove directory: $TARGET_PREFIX/$d"
                fi
            done
        fi
        exit 0
    fi

    if [ "$IS_MUTATING" -eq 0 ]; then
        # Non-mutating commands (dry-run, --help) MUST NOT mutate disk!
        # Emit informative dry-run output and guidance without downloading.
        cat <<EOF
{
  "prefix": "$TARGET_PREFIX",
  "bin_dir": "$TARGET_PREFIX/bin",
  "toolchain_dir": "$TARGET_PREFIX/share/portfolio-lab/toolchain",
  "alpine_root": "$TARGET_PREFIX/share/portfolio-lab/toolchain/alpine-root",
  "alpine_build_root": "$TARGET_PREFIX/share/portfolio-lab/toolchain/alpine-build-root",
  "python_root": "$TARGET_PREFIX/share/portfolio-lab/toolchain/python-root",
  "standalone_dir": "$TARGET_PREFIX/share/portfolio-lab/toolchain/standalone",
  "schema_version": "portfolio-lab-cursor-box-bootstrap/v2",
  "layout_revision": "v2.2-alpine-build-root-bunx",
  "stage_0": {
    "status": "ready",
    "system_python": "absent",
    "note": "Non-mutating dry-run in stage-0 mode. Run 'install' to securely initialize user-owned Python and the complete toolchain."
  }
}
EOF
        exit 0
    fi

    # Mutating command: stage-0 download & atomic initialization
    echo "System python3 not found; initializing user-owned Python runtime under $TARGET_PREFIX..." >&2
    STAGE0_DIR="$TARGET_PREFIX/share/portfolio-lab/toolchain/stage0-python"
    mkdir -p "$STAGE0_DIR"
    TMP_TAR="$STAGE0_DIR/python.tar.gz"

    if [ -n "$STAGE0_ARCHIVE" ] && [ -f "$STAGE0_ARCHIVE" ]; then
        cp "$STAGE0_ARCHIVE" "$TMP_TAR"
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "$TMP_TAR" "$PINNED_PYTHON_URL"
    elif command -v curl >/dev/null 2>&1; then
        curl -sSL -o "$TMP_TAR" "$PINNED_PYTHON_URL"
    else
        echo "ERROR: Neither wget, curl, nor pre-transferred --stage0-python-archive found to bootstrap Python." >&2
        exit 1
    fi

    ACTUAL_SHA="$(sha256sum "$TMP_TAR" | awk '{print $1}')"
    if [ "$ACTUAL_SHA" != "$EXPECTED_PYTHON_SHA256" ]; then
        echo "ERROR: Stage-0 Python checksum mismatch: expected $EXPECTED_PYTHON_SHA256, got $ACTUAL_SHA" >&2
        rm -f "$TMP_TAR"
        exit 1
    fi

    # Save archive copy for the Python installer if pre-transferred
    ARCHIVE_TO_PASS=""
    if [ -n "$STAGE0_ARCHIVE" ] && [ -f "$STAGE0_ARCHIVE" ]; then
        ARCHIVE_TO_PASS="$STAGE0_ARCHIVE"
    elif [ -f "$TMP_TAR" ]; then
        # Keep archive to pass to installer
        ARCHIVE_TO_PASS="$STAGE0_DIR/python-archive.tar.gz"
        cp "$TMP_TAR" "$ARCHIVE_TO_PASS"
    fi

    # Extract atomically
    tar -xzf "$TMP_TAR" -C "$STAGE0_DIR" --strip-components=1
    rm -f "$TMP_TAR"
    PYTHON_BIN="$STAGE0_DIR/bin/python3"
    echo "Stage-0 Python initialized at $PYTHON_BIN" >&2

    # If --stage0-python-archive was not already explicitly passed in "$@", append it
    if [ -z "$STAGE0_ARCHIVE" ] && [ -n "$ARCHIVE_TO_PASS" ]; then
        set -- "$@" "--stage0-python-archive=$ARCHIVE_TO_PASS"
    fi

    # Run Python installer and capture exit status
    EXIT_CODE=0
    "$PYTHON_BIN" "$SCRIPT_DIR/portfolio_lab_cursor_box_bootstrap.py" "$@" || EXIT_CODE=$?

    # Clean disposable stage-0 directory ONLY after a successful final install,
    # not any other command. Preserve explicitly supplied external archives outside STAGE0_DIR.
    if [ "$EXIT_CODE" -eq 0 ] && [ "$COMMAND" = "install" ]; then
        rm -rf "$STAGE0_DIR"
    fi

    exit "$EXIT_CODE"
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/portfolio_lab_cursor_box_bootstrap.py" "$@"
