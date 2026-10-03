#!/bin/sh
# Read-only probe: alpine-build-root musl loader + GNU Make.
# Fail closed if loader or make is missing/not executable, or if an ELF
# loader cannot run `make --version`. Wrapper path is advisory.
# Non-ELF loaders (hermetic shell-script fakes) skip the musl exec.
# MUST NOT mutate host paths; tests override locations via env.
set -eu

BUILD_ROOT="${PORTFOLIO_LAB_ALPINE_BUILD_ROOT:-$HOME/.local/share/portfolio-lab/toolchain/alpine-build-root}"
LOADER="$BUILD_ROOT/lib/ld-musl-x86_64.so.1"
MAKE_BIN="${PORTFOLIO_LAB_MAKE_BIN:-$BUILD_ROOT/usr/bin/make}"
WRAPPER="${PORTFOLIO_LAB_MAKE_WRAPPER:-$HOME/.local/bin/make}"

present_yn() {
    if [ -f "$1" ] && [ -x "$1" ]; then
        echo yes
    else
        echo no
    fi
}

# ELF magic 0x7f 'E' 'L' 'F'. Unknown/unreadable → not ELF (skip probe).
is_elf() {
    [ -f "$1" ] || return 1
    magic=$(od -An -N4 -tx1 "$1" 2>/dev/null | tr -d ' \n\t')
    [ "$magic" = "7f454c46" ]
}

LOADER_PRESENT="$(present_yn "$LOADER")"
MAKE_PRESENT="$(present_yn "$MAKE_BIN")"
WRAPPER_PRESENT="$(present_yn "$WRAPPER")"

echo "loader=${LOADER} present=${LOADER_PRESENT}"
echo "make_bin=${MAKE_BIN} present=${MAKE_PRESENT}"
echo "wrapper=${WRAPPER} present=${WRAPPER_PRESENT}"

status=ok
probe=skipped

if [ "$LOADER_PRESENT" != yes ] || [ "$MAKE_PRESENT" != yes ]; then
    status=fail
elif is_elf "$LOADER"; then
    # Probe is timeout-safe by being a single --version exec (no loop).
    LIB_PATH="$BUILD_ROOT/usr/lib:$BUILD_ROOT/lib"
    set +e
    probe_out=$("$LOADER" --library-path "$LIB_PATH" "$MAKE_BIN" --version 2>&1)
    probe_rc=$?
    set -e
    case "$probe_out" in
        *GNU\ Make*)
            if [ "$probe_rc" -eq 0 ]; then
                probe=ok
            else
                status=fail
                probe=fail
            fi
            ;;
        *)
            status=fail
            probe=fail
            ;;
    esac
    if [ "$probe" = fail ]; then
        echo "probe: failed rc=${probe_rc}" >&2
    fi
fi

echo "probe=${probe}"
echo "toolchain-make: ${status}"

if [ "$status" = ok ]; then
    exit 0
fi
exit 1
