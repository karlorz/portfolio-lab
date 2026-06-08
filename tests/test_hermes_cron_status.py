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


def test_repo_hermes_wrapper_records_status_json_with_hermes_backend(tmp_path: Path) -> None:
    """A direct Hermes wrapper run should update cron_status.json as backend=hermes."""
    project = tmp_path / "portfolio-lab"
    scripts_dir = project / "scripts"
    cron_dir = scripts_dir / "cron"
    data_dir = project / "data"
    cron_dir.mkdir(parents=True)
    data_dir.mkdir()

    guard = scripts_dir / "cron_guard.sh"
    guard.write_text(
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
exit 0
""",
    )

    env = os.environ.copy()
    env["PYTHON_RUNTIME"] = str(runtime)

    result = subprocess.run(
        ["bash", str(wrapper)],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    recorded_calls = update_args.read_text().splitlines()
    assert "src/dashboard/generator.py" in recorded_calls
    assert any(
        call.startswith("scripts/cron_update.py portfolio-lab-dashboard ok ")
        and call.endswith(" hermes")
        for call in recorded_calls
    )
