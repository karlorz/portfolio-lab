"""Tests for cron guard shell behavior."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CRON_GUARD = PROJECT_ROOT / "scripts" / "cron_guard.sh"


def test_cron_guard_end_reports_nonzero_exit_and_returns_it(tmp_path: Path) -> None:
    """cron_guard_end should log nonzero statuses after releasing its flock descriptor."""
    env = os.environ.copy()
    env["CRON_GUARD_LOCK_DIR"] = str(tmp_path / "locks")
    env["CRON_GUARD_MAX_LOAD"] = "999"

    result = subprocess.run(
        [
            "bash",
            "-c",
            f"""
source "{CRON_GUARD}"
cron_guard_start "unit-health" 60
set +e
cron_guard_end "unit-health" 42
status=$?
set -e
echo "end_status=$status"
exit "$status"
""",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 42
    assert "CRON_GUARD: unit-health FAILED exit=42" in output
    assert "end_status=42" in output


def test_cron_guard_end_does_not_emit_timeout_for_success(tmp_path: Path) -> None:
    """Stopping the watchdog during normal completion must not log a timeout."""
    env = os.environ.copy()
    env["CRON_GUARD_LOCK_DIR"] = str(tmp_path / "locks")
    env["CRON_GUARD_MAX_LOAD"] = "999"

    result = subprocess.run(
        [
            "bash",
            "-c",
            f"""
source "{CRON_GUARD}"
cron_guard_start "unit-success" 60
sleep 1
cron_guard_end "unit-success" 0
""",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "CRON_GUARD: unit-success COMPLETED" in output
    assert "Terminated" not in output
    assert "TIMEOUT" not in output
    assert "FORCE KILL" not in output
