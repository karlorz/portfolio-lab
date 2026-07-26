"""Batch JO / JN HT1 — dual-mode health timeout walls aligned ≥120s.

Session A plan (JO): tasker + hermes cron_guard + Makefile must not hard-kill
health at 60s when Makefile allows 120s. Outer wall ≥ inner wall.
Does not touch signals.json.target_allocations / order_router.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_ht1_makefile_health_timeout_at_least_120() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    m = re.search(
        r"timeout\s+(\d+)\s+\$\(PYTHON_RUNTIME\)\s+-m\s+src\.monitor\.health_check",
        makefile,
    )
    assert m is not None, "Makefile health timeout line missing"
    assert int(m.group(1)) >= 120, f"Makefile health timeout {m.group(1)} < 120"


def test_ht1_tasker_health_timeout_at_least_120() -> None:
    tasker = (ROOT / "config" / "tasker.yaml").read_text(encoding="utf-8")
    block = re.search(
        r"id:\s*portfolio-lab-health.*?timeout_seconds:\s*(\d+)",
        tasker,
        re.S,
    )
    assert block is not None, "portfolio-lab-health timeout_seconds missing"
    assert int(block.group(1)) >= 120, f"tasker health timeout {block.group(1)} < 120"


def test_ht1_hermes_shell_guard_at_least_120() -> None:
    shell = (ROOT / "scripts" / "cron" / "portfolio-lab-health-monitor.sh").read_text(
        encoding="utf-8"
    )
    m = re.search(r'cron_guard_start\s+"pf-health"\s+(\d+)', shell)
    assert m is not None, "cron_guard_start pf-health missing"
    assert int(m.group(1)) >= 120, f"hermes pf-health guard {m.group(1)} < 120"


def test_ht1_tasker_wall_ge_makefile_inner() -> None:
    """Outer tasker wall must not be below Makefile inner timeout."""
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    tasker = (ROOT / "config" / "tasker.yaml").read_text(encoding="utf-8")
    m = re.search(
        r"timeout\s+(\d+)\s+\$\(PYTHON_RUNTIME\)\s+-m\s+src\.monitor\.health_check",
        makefile,
    )
    t = re.search(
        r"id:\s*portfolio-lab-health.*?timeout_seconds:\s*(\d+)",
        tasker,
        re.S,
    )
    assert m and t
    assert int(t.group(1)) >= int(m.group(1)), (
        f"tasker {t.group(1)} < Makefile {m.group(1)}"
    )
