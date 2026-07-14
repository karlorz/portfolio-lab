"""Regression: Makefile cron recipes must propagate non-zero job EXIT to make.

Tasker invokes ``make <target>`` and only sees make's exit code. Recipes that
run ``cron_update`` last always returned 0 even when the underlying job failed
or health was critical (false-green scheduler status).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = PROJECT_ROOT / "Makefile"

# Cron-style targets that capture EXIT then call cron_update (must end with exit $$EXIT).
# daily-pnl depends on mark-to-market; both must propagate.
CRON_EXIT_TARGETS = (
    "data",
    "dashboard",
    "eval",
    "research",
    "wiki-sync",
    "build",
    "sync",
    "overlay-signals",
    "overlay-dashboard",
    "health",
    "rebalance-health",
    "garch-risk",
    "mark-to-market",
    "daily-pnl",
    "unified-dashboard",
    "prune-logs",
)


def _recipe_body(makefile: str, target: str) -> str:
    """Return the recipe body for a simple Make target (tab-indented lines)."""
    # Allow blank lines between target: and first recipe (garch-risk quirk)
    pattern = rf"^{re.escape(target)}:[^\n]*\n((?:(?:\t|[ ]{{0}})[^\n]*\n)*)"
    # Prefer: target line then optional blank lines then tab recipes
    m = re.search(
        rf"^{re.escape(target)}:(?:[^\n]*)\n((?:\n|\t[^\n]*\n)+)",
        makefile,
        re.M,
    )
    if not m:
        raise AssertionError(f"Makefile target not found: {target}")
    body = m.group(1)
    # Stop at next non-empty non-tab line that looks like a new target/comment block end
    lines = []
    for line in body.splitlines(keepends=True):
        if line.startswith("\t") or line.strip() == "":
            lines.append(line)
        else:
            break
    return "".join(lines)


def test_health_recipe_exits_with_job_exit_code() -> None:
    """make health must end with exit $$EXIT so critical health is non-zero."""
    text = MAKEFILE.read_text()
    body = _recipe_body(text, "health")
    assert "src.monitor.health_check" in body
    assert "portfolio-lab-health" in body
    assert re.search(r"exit\s+\$\$EXIT\b", body), (
        "health recipe must propagate health_check exit via exit $$EXIT; "
        f"recipe tail:\n{''.join(body.splitlines()[-5:])}"
    )


@pytest.mark.parametrize("target", CRON_EXIT_TARGETS)
def test_cron_makefile_targets_propagate_exit(target: str) -> None:
    """All EXIT-capturing cron Makefile targets must end with exit $$EXIT."""
    text = MAKEFILE.read_text()
    body = _recipe_body(text, target)
    assert "EXIT=" in body or "EXIT2=" in body
    assert "$(CRON_UPDATE)" in body or "cron_update" in body
    assert re.search(r"exit\s+\$\$EXIT\b", body) or re.search(
        r"exit\s+\$\$EXIT2\b", body
    ), f"{target} recipe does not propagate EXIT:\n{body}"


def test_attribution_propagates_either_exit() -> None:
    """attribution runs two jobs; non-zero from either must surface."""
    text = MAKEFILE.read_text()
    body = _recipe_body(text, "attribution")
    assert "EXIT2=" in body
    assert re.search(r"exit\s+\$\$EXIT\b", body)
    assert re.search(r"exit\s+\$\$EXIT2\b", body)


def test_health_exit_codes_documented_in_makefile() -> None:
    """Nagios-style / timeout / OOM mapping stays documented near health."""
    text = MAKEFILE.read_text()
    # Status mapping shared by cron recipes
    assert 'STATUS="timeout"' in text
    assert 'STATUS="oom"' in text
    assert 'STATUS="error"' in text
    # Document propagation contract once (comment near health or top)
    assert "exit $$EXIT" in text
