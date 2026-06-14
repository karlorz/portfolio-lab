#!/bin/bash
# test-repo-guard.sh — Enforce that tests only run inside portfolio-lab.
# Refuses to run if the working directory is a deny-listed repo.
# Source this in any test runner to add directory-level protection.
#
# Usage:
#   source "$(dirname "$0")/test-repo-guard.sh"
#   guard_ensure_portfolio_lab    # exits if not in portfolio-lab
#
# This is defense-in-depth: even if an agent prompt fails to constrain
# the agent, this guard blocks test execution in the wrong repo.

set -euo pipefail

# Directories where tests MUST NOT run
DENY_LIST=(
    "/root/.hermes/hermes-agent"
    "/root/.hermes"
    "/root/hermes-agent"
    "/root/hermes"
)

REQUIRED_MARKER="portfolio-lab"
EXPECTED_PROJECT_DIR="${PORTFOLIO_LAB_PROJECT_DIR:-/root/projects/portfolio-lab}"

guard_ensure_portfolio_lab() {
    local cwd
    cwd="$(pwd -P 2>/dev/null || pwd)"

    # Check deny list
    for deny in "${DENY_LIST[@]}"; do
        if [[ "$cwd" == "$deny" || "$cwd" == "$deny"/* ]]; then
            echo "FATAL: test-repo-guard — refusing to run tests in $cwd" >&2
            echo "  This directory matches deny-list entry: $deny" >&2
            echo "  Tests must be run from $EXPECTED_PROJECT_DIR" >&2
            exit 78  # EX_CONFIG — configuration error
        fi
    done

    # Check for portfolio-lab marker file
    if [ ! -f "CLAUDE.md" ]; then
        echo "FATAL: test-repo-guard — CLAUDE.md marker not found in $cwd" >&2
        echo "  This may not be the portfolio-lab project directory." >&2
        exit 78
    fi

    # Verify CLAUDE.md contains portfolio-lab identifier
    if ! grep -q "$REQUIRED_MARKER" CLAUDE.md 2>/dev/null; then
        echo "FATAL: test-repo-guard — CLAUDE.md missing '$REQUIRED_MARKER' marker" >&2
        echo "  This does not appear to be the portfolio-lab project." >&2
        exit 78
    fi

    return 0
}
