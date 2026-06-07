"""Regression tests for the sg01 Hermes health cron wrapper."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HERMES_HEALTH_WRAPPER = Path("/root/.hermes/scripts/portfolio-lab-health-monitor.sh")
GUARD_SOURCE_LINE = "CRON_GUARD_MEMORY_MB=1024 source /root/projects/portfolio-lab/scripts/cron_guard.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _wrapper_with_stubbed_guard(tmp_path: Path) -> Path:
    if not HERMES_HEALTH_WRAPPER.exists():
        pytest.skip(f"{HERMES_HEALTH_WRAPPER} is not present on this host")

    wrapper_text = HERMES_HEALTH_WRAPPER.read_text()
    assert GUARD_SOURCE_LINE in wrapper_text

    guard_stub = tmp_path / "cron_guard_stub.sh"
    guard_stub.write_text(
        """#!/bin/bash
set -euo pipefail

cron_guard_start() {
    echo "guard_start $1"
    return 0
}

cron_guard_end() {
    echo "guard_end $1 $2"
    return "$2"
}
"""
    )

    wrapper_copy = tmp_path / "portfolio-lab-health-monitor.sh"
    wrapper_copy.write_text(wrapper_text.replace(GUARD_SOURCE_LINE, f"source {guard_stub}"))
    wrapper_copy.chmod(wrapper_copy.stat().st_mode | stat.S_IXUSR)
    return wrapper_copy


def _run_wrapper(tmp_path: Path, python_body: str) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    wrapper = _wrapper_with_stubbed_guard(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_args = tmp_path / "python-args.txt"

    _write_executable(
        bin_dir / "python3",
        f"""#!/bin/bash
printf '%s\\n' "$@" > "{python_args}"
{python_body}
""",
    )
    _write_executable(
        bin_dir / "tee",
        """#!/bin/bash
cat >/dev/null
exit 0
""",
    )

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["PYTHONPATH"] = ""

    result = subprocess.run(
        ["bash", str(wrapper)],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    args = python_args.read_text().splitlines() if python_args.exists() else []
    return result, args


def test_hermes_health_wrapper_invokes_current_health_check_module(tmp_path: Path) -> None:
    """The Hermes wrapper should run the current health_check module, not stale health.py."""
    result, args = _run_wrapper(tmp_path, "exit 0")

    assert result.returncode == 0, result.stderr
    assert args == ["-m", "src.monitor.health_check"]


def test_hermes_health_wrapper_reports_health_command_failure_to_guard(tmp_path: Path) -> None:
    """A failing health command should reach cron_guard_end even when output is piped through tee."""
    result, _ = _run_wrapper(tmp_path, "echo forced health failure >&2\nexit 42")

    output = result.stdout + result.stderr
    assert result.returncode == 42
    assert "guard_end pf-health 42" in output
