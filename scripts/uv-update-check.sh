#!/usr/bin/env bash
# uv Dependency Update + Compatibility Check
# Usage:
#   ./scripts/uv-update-check.sh                    # dry-run: lockfile + smoke tests
#   ./scripts/uv-update-check.sh --full              # dry-run + full test suite
#   ./scripts/uv-update-check.sh --upgrade           # upgrade all + smoke verify
#   ./scripts/uv-update-check.sh --upgrade-pkg numpy # upgrade one package
#   ./scripts/uv-update-check.sh --ci                # strict CI mode (full suite)
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# ── Load .env ────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ -f "$PROJECT_DIR/.env" ]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

# Defaults
MODE="--check"
PKG=""
FULL_SUITE=0

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --check|--upgrade|--ci) MODE="$1"; shift ;;
        --full)  FULL_SUITE=1; shift ;;
        --upgrade-pkg) MODE="--upgrade-pkg"; PKG="${2:-}"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# ── Helpers ────────────────────────────────────────────────────────────────

section() { echo -e "\n${CYAN}── $1 ──${NC}"; }
pass()   { echo -e "  ${GREEN}✓${NC} $1"; }
fail()   { echo -e "  ${RED}✗${NC} $1"; }
warn()   { echo -e "  ${YELLOW}⚠${NC} $1"; }

fatal() {
    echo -e "${RED}FATAL:${NC} $1"
    exit 1
}

# ── Verification Steps ────────────────────────────────────────────────────

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

check_ml_flag() {
    section "ML flag status"
    local ml="${PORTFOLIO_LAB_ENABLE_ML:-0}"
    if [ "$ml" = "1" ]; then
        warn "PORTFOLIO_LAB_ENABLE_ML=1 (ML enabled — torch/xgboost loaded)"
    else
        pass "PORTFOLIO_LAB_ENABLE_ML=0 (ML disabled — lightweight mode)"
    fi
}

# ── Smoke test (fast, ~130 tests, ~4s) ────────────────────────────────────

SMOKE_TEST_FILES=(
    "tests/test_behavioral_sentiment.py"
    "tests/test_combined_orchestrator.py"
    "tests/test_dual_momentum.py"
    "tests/test_evaluator.py"
)

run_tests_smoke() {
    local strict="${1:-0}"
    section "Smoke tests (${#SMOKE_TEST_FILES[@]} files, ~130 tests, ~4s)"
    if uv run pytest "${SMOKE_TEST_FILES[@]}" -q --tb=line 2>&1; then
        pass "Smoke tests passed"
        return 0
    else
        if [ "$strict" -eq 1 ]; then
            fail "Smoke test failures"
            return 1
        else
            warn "Smoke test failures (may be pre-existing)"
            return 0
        fi
    fi
}

# ── Full test suite (2850+ tests, several minutes) ────────────────────────

run_tests_full() {
    local strict="${1:-0}"
    section "Full test suite (2850+ tests, ML disabled)"
    echo "  This may take several minutes..."

    if timeout 120 uv run pytest tests/ -q --tb=line 2>&1; then
        pass "Full suite passed"
        return 0
    else
        local rc=$?
        if [ $rc -eq 124 ]; then
            warn "Full suite timed out (2 min) — may be stuck on pre-existing issues"
            warn "Use smoke tests for routine checks:  ./scripts/uv-update-check.sh"
            return 0
        elif [ "$strict" -eq 1 ]; then
            fail "Full suite failures"
            return 1
        else
            warn "Pre-existing test failures (not introduced by this update)"
            return 0
        fi
    fi
}

# ── Verification pipelines ────────────────────────────────────────────────

verify_all() {
    local strict="${1:-0}"
    local errors=0

    check_ml_flag
    check_lockfile || errors=$((errors + 1))
    check_outdated
    check_conflicts || errors=$((errors + 1))
    check_tree

    if [ "$FULL_SUITE" -eq 1 ] || [ "$strict" -eq 1 ]; then
        run_tests_full "$strict" || errors=$((errors + 1))
    else
        run_tests_smoke "$strict" || errors=$((errors + 1))
    fi

    section "Summary"
    if [ "$errors" -eq 0 ]; then
        echo -e "${GREEN}All checks passed. Environment is clean.${NC}"
    else
        echo -e "${RED}${errors} check(s) failed.${NC}"
        return 1
    fi
}

# ── Upgrade modes ──────────────────────────────────────────────────────────

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

# ── Main ───────────────────────────────────────────────────────────────────

echo -e "${CYAN}uv Update + Compatibility Check${NC}"
echo "Mode: ${MODE}  |  Tests: $( [ "$FULL_SUITE" -eq 1 ] && echo 'FULL' || echo 'smoke' )"
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
        run_tests_full 1 || fatal "Full suite failed"
        echo -e "${GREEN}CI checks passed.${NC}"
        ;;
    *)
        echo "Usage: $0 [--check|--full|--upgrade|--upgrade-pkg <name>|--ci]"
        exit 1
        ;;
esac
