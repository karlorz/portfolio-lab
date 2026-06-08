"""Regression tests for the sg01 Hermes health cron wrapper."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HERMES_HEALTH_WRAPPER = Path("/root/.hermes/scripts/portfolio-lab-health-monitor.sh")
PROJECT_DIR_LINE = 'PROJECT_DIR="${PORTFOLIO_LAB_PROJECT_DIR:-/root/projects/portfolio-lab}"'
RUNTIME_LINE = 'PYTHON_RUNTIME="${PYTHON_RUNTIME:-$PROJECT_DIR/scripts/python_runtime.sh}"'


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _wrapper_with_stubbed_guard(tmp_path: Path) -> Path:
    if not HERMES_HEALTH_WRAPPER.exists():
        pytest.skip(f"{HERMES_HEALTH_WRAPPER} is not present on this host")

    wrapper_text = HERMES_HEALTH_WRAPPER.read_text()
    assert PROJECT_DIR_LINE in wrapper_text

    project = tmp_path / "project"
    scripts_dir = project / "scripts"
    cron_dir = scripts_dir / "cron"
    cron_dir.mkdir(parents=True)
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
    (cron_dir / "hermes_status.sh").write_text(
        """#!/bin/bash
record_hermes_cron_status() {
    echo "record_status $*"
}
"""
    )

    wrapper_copy = tmp_path / "portfolio-lab-health-monitor.sh"
    wrapper_copy.write_text(
        wrapper_text.replace(PROJECT_DIR_LINE, f'PROJECT_DIR="{project}"')
        .replace('CRON_GUARD_MEMORY_MB=1024 source "$PROJECT_DIR/scripts/cron_guard.sh"', f"source {guard_stub}")
    )
    wrapper_copy.chmod(wrapper_copy.stat().st_mode | stat.S_IXUSR)
    return wrapper_copy


def _run_wrapper(tmp_path: Path, python_body: str) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    wrapper = _wrapper_with_stubbed_guard(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_args = tmp_path / "python-args.txt"
    runtime_stub = tmp_path / "python_runtime.sh"

    _write_executable(
        runtime_stub,
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
    wrapper.write_text(wrapper.read_text().replace(RUNTIME_LINE, f'PYTHON_RUNTIME="${{PYTHON_RUNTIME:-{runtime_stub}}}"'))

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
