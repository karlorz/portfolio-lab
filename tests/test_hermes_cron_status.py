"""Tests for Hermes cron wrappers writing backend-aware status rows."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_WRAPPER = PROJECT_ROOT / "scripts" / "cron" / "portfolio-lab-dashboard.sh"
HERMES_STATUS_HELPER = PROJECT_ROOT / "scripts" / "cron" / "hermes_status.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _prepare_dashboard_wrapper_harness(
    tmp_path: Path,
    *,
    create_data_dir: bool,
    producer_exit: int,
) -> tuple[Path, Path, Path]:
    """Create a temp project with dashboard wrapper, status helper, and stubs."""
    project = tmp_path / "portfolio-lab"
    scripts_dir = project / "scripts"
    cron_dir = scripts_dir / "cron"
    cron_dir.mkdir(parents=True)
    if create_data_dir:
        (project / "data").mkdir()

    guard = scripts_dir / "cron_guard.sh"
    guard.write_text(
        """#!/bin/bash
set -euo pipefail

cron_guard_start() {
    return 0
}

cron_guard_end() {
    return "$2"
}
"""
    )

    if HERMES_STATUS_HELPER.exists():
        (cron_dir / "hermes_status.sh").write_text(HERMES_STATUS_HELPER.read_text())

    wrapper = cron_dir / "portfolio-lab-dashboard.sh"
    wrapper.write_text(DASHBOARD_WRAPPER.read_text().replace("/root/projects/portfolio-lab", str(project)))
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)

    update_args = tmp_path / "cron-update-args.txt"
    runtime = tmp_path / "python_runtime.sh"
    _write_executable(
        runtime,
        f"""#!/bin/bash
printf '%s\\n' "$*" >> "{update_args}"
if [ "$1" = "scripts/cron_update.py" ]; then
    exit 0
fi
exit {producer_exit}
""",
    )

    return project, runtime, update_args


def _run_wrapper(wrapper: Path, project: Path, runtime: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHON_RUNTIME"] = str(runtime)
    env["PORTFOLIO_LAB_PROJECT_DIR"] = str(project)

    return subprocess.run(
        ["bash", str(wrapper)],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_repo_hermes_wrapper_records_status_json_with_hermes_backend(tmp_path: Path) -> None:
    """A direct Hermes wrapper run should update cron_status.json as backend=hermes."""
    project, runtime, update_args = _prepare_dashboard_wrapper_harness(
        tmp_path,
        create_data_dir=True,
        producer_exit=0,
    )
    wrapper = project / "scripts" / "cron" / "portfolio-lab-dashboard.sh"

    result = _run_wrapper(wrapper, project, runtime)

    assert result.returncode == 0, result.stdout + result.stderr
    recorded_calls = update_args.read_text().splitlines()
    assert "src/dashboard/generator.py" in recorded_calls
    assert any(
        call.startswith("scripts/cron_update.py portfolio-lab-dashboard ok ")
        and call.endswith(" hermes")
        for call in recorded_calls
    )


def test_dashboard_wrapper_records_error_when_tee_stage_fails(tmp_path: Path) -> None:
    """A successful producer with a failed tee stage must not record ok."""
    project, runtime, update_args = _prepare_dashboard_wrapper_harness(
        tmp_path,
        create_data_dir=False,
        producer_exit=0,
    )
    wrapper = project / "scripts" / "cron" / "portfolio-lab-dashboard.sh"

    result = _run_wrapper(wrapper, project, runtime)

    assert result.returncode != 0
    assert "tee: data/dashboard.log" in result.stderr
    recorded_calls = update_args.read_text().splitlines()
    assert any(
        call.startswith("scripts/cron_update.py portfolio-lab-dashboard error ")
        and call.endswith(" hermes")
        for call in recorded_calls
    )


def test_dashboard_wrapper_preserves_producer_failure_precedence_when_tee_fails(
    tmp_path: Path,
) -> None:
    """Producer status remains authoritative when both producer and tee fail."""
    project, runtime, update_args = _prepare_dashboard_wrapper_harness(
        tmp_path,
        create_data_dir=False,
        producer_exit=42,
    )
    wrapper = project / "scripts" / "cron" / "portfolio-lab-dashboard.sh"

    result = _run_wrapper(wrapper, project, runtime)

    assert result.returncode == 42
    assert "tee: data/dashboard.log" in result.stderr
    recorded_calls = update_args.read_text().splitlines()
    assert any(
        call.startswith("scripts/cron_update.py portfolio-lab-dashboard error ")
        and call.endswith(" hermes")
        for call in recorded_calls
    )
