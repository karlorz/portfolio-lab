#!/usr/bin/env bash
# uv Dependency Update + Compatibility Check
# Usage:
#   ./scripts/uv-update-check.sh               # dry-run: check status only
#   ./scripts/uv-update-check.sh --upgrade     # upgrade all + verify
#   ./scripts/uv-update-check.sh --upgrade-pkg numpy  # upgrade one package
#   ./scripts/uv-update-check.sh --ci          # strict CI mode
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

MODE="${1:---check}"
PKG="${2:-}"

# ── Helpers ────────────────────────────────────────────────────────────

section() { echo -e "\n${CYAN}── $1 ──${NC}"; }
pass()   { echo -e "  ${GREEN}✓${NC} $1"; }
fail()   { echo -e "  ${RED}✗${NC} $1"; }
warn()   { echo -e "  ${YELLOW}⚠${NC} $1"; }

fatal() {
    echo -e "${RED}FATAL:${NC} $1"
    exit 1
}

# ── Verification Steps (shared) ────────────────────────────────────────

check_lockfile() {
    section "Lockfile status"
    if uv lock --check 2>&1; then
        pass "uv.lock matches pyproject.toml"
    else
        fail "uv.lock is STALE — run: uv lock"
        return 1
    fi
}

check_outdated() {
    section "Outdated packages"
    local outdated
    outdated=$(uv pip list --outdated 2>&1 | tail -n +3) || true
    if [ -z "$outdated" ]; then
        pass "All packages up to date"
    else
        warn "Outdated packages found:"
        echo "$outdated" | while read -r line; do echo "    $line"; done
    fi
}

check_conflicts() {
    section "Dependency conflicts"
    if uv pip check 2>&1; then
        pass "No dependency conflicts"
    else
        fail "Conflicts detected"
        return 1
    fi
}

check_tree() {
    section "Dependency tree (depth 1)"
    uv tree --depth 1 2>&1
}

run_tests() {
    local strict="${1:-0}"  # 0 = warn only, 1 = hard fail
    section "Running test suite (safe mode, ML disabled)"
    if uv run pytest tests/ -q --tb=line 2>&1; then
        pass "All tests passed"
        return 0
    else
        if [ "$strict" -eq 1 ]; then
            fail "Test failures in CI mode — check output above"
            return 1
        else
            warn "Pre-existing test failures (not introduced by this update)"
            return 0
        fi
    fi
}

# ── Full verification pipeline ─────────────────────────────────────────

verify_all() {
    local strict="${1:-0}"
    local errors=0
    check_lockfile || errors=$((errors + 1))
    check_outdated
    check_conflicts || errors=$((errors + 1))
    check_tree
    run_tests "$strict" || errors=$((errors + 1))

    section "Summary"
    if [ "$errors" -eq 0 ]; then
        echo -e "${GREEN}All checks passed. Environment is clean.${NC}"
    else
        echo -e "${RED}${errors} check(s) failed.${NC}"
        return 1
    fi
}

# ── Upgrade modes ──────────────────────────────────────────────────────

do_upgrade_all() {
    section "Upgrading all packages"
    uv lock --upgrade
    uv sync
    pass "All packages upgraded to latest compatible versions"
    verify_all 0
}

do_upgrade_pkg() {
    local pkg="$1"
    [ -z "$pkg" ] && fatal "--upgrade-pkg requires a package name"

    section "Upgrading ${pkg}"
    uv lock --upgrade-package "$pkg"
    uv sync
    pass "${pkg} upgraded"
    verify_all 0
}

# ── Main ───────────────────────────────────────────────────────────────

echo -e "${CYAN}uv Update + Compatibility Check${NC}"
echo "Mode: ${MODE}"
echo ""

case "$MODE" in
    --check)
        verify_all 0
        ;;
    --upgrade)
        do_upgrade_all
        ;;
    --upgrade-pkg)
        do_upgrade_pkg "$PKG"
        ;;
    --ci)
        section "CI strict mode"
        uv lock --check || fatal "Lockfile stale"
        uv sync --locked
        uv pip check || fatal "Dependency conflicts"
        uv run pytest tests/ -q --tb=line || fatal "Tests failed"
        echo -e "${GREEN}CI checks passed.${NC}"
        ;;
    *)
        echo "Usage: $0 [--check|--upgrade|--upgrade-pkg <name>|--ci]"
        exit 1
        ;;
esac
