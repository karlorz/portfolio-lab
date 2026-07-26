"""Regression tests for autonomous Hermes cron reporting."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUTONOMOUS_WRAPPER = PROJECT_ROOT / "scripts" / "cron" / "portfolio-lab-autonomous-agent.sh"
HERMES_STATUS_HELPER = PROJECT_ROOT / "scripts" / "cron" / "hermes_status.sh"
CONFIGURE_SCRIPT = PROJECT_ROOT / "scripts" / "cron" / "configure_autonomous_agent_job.py"
CRON_UPDATE_SCRIPT = PROJECT_ROOT / "scripts" / "cron_update.py"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _git(cmd: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *cmd],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _make_project_fixture(tmp_path: Path) -> Path:
    project = tmp_path / "portfolio-lab"
    scripts_dir = project / "scripts"
    cron_dir = scripts_dir / "cron"
    data_dir = project / "data"
    cron_dir.mkdir(parents=True)
    data_dir.mkdir()

    shutil.copy2(AUTONOMOUS_WRAPPER, cron_dir / AUTONOMOUS_WRAPPER.name)
    shutil.copy2(HERMES_STATUS_HELPER, cron_dir / HERMES_STATUS_HELPER.name)
    _write_executable(
        scripts_dir / "cron_guard.sh",
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
""",
    )

    _git(["init"], project)
    _git(["config", "user.email", "test@example.com"], project)
    _git(["config", "user.name", "Test User"], project)
    (project / "README.md").write_text("fixture\n")
    _git(["add", "README.md"], project)
    _git(["commit", "-m", "fixture"], project)
    return project


def test_autonomous_wrapper_records_actual_git_head_metadata(tmp_path: Path) -> None:
    """The guarded autonomous wrapper should cite the actual short HEAD."""
    project = _make_project_fixture(tmp_path)
    expected_head = _git(["rev-parse", "--short", "HEAD"], project)
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
    env["PORTFOLIO_LAB_PROJECT_DIR"] = str(project)
    env["PYTHON_RUNTIME"] = str(runtime)
    env["HOME"] = str(tmp_path)

    result = subprocess.run(
        ["bash", str(project / "scripts" / "cron" / AUTONOMOUS_WRAPPER.name)],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert f"GIT_HEAD: {expected_head}" in output
    assert "1a2b3c4" not in output
    assert any(
        call.startswith("scripts/cron_update.py portfolio-lab-autonomous-agent ok ")
        and call.endswith(f" hermes git_commit={expected_head}")
        for call in update_args.read_text().splitlines()
    )


def test_configure_autonomous_job_converts_agent_job_to_guarded_no_agent(tmp_path: Path) -> None:
    """A stale LLM-mode Hermes job should be rewritten to the guarded script path."""
    hermes_home = tmp_path / "hermes"
    cron_dir = hermes_home / "cron"
    cron_dir.mkdir(parents=True)
    jobs_path = cron_dir / "jobs.json"
    jobs_path.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "id": "7d989b13169e",
                        "name": "portfolio-lab-autonomous-agent",
                        "prompt": "final report should not become RuntimeError",
                        "script": None,
                        "no_agent": False,
                        "enabled_toolsets": ["terminal", "file", "web", "cronjob"],
                        "workdir": str(PROJECT_ROOT),
                    }
                ],
                "updated_at": "2026-06-08T00:00:00+08:00",
            }
        )
    )

    spec = importlib.util.spec_from_file_location("configure_autonomous_agent_job", CONFIGURE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    changed = module.configure_autonomous_agent_job(hermes_home=hermes_home, project_dir=PROJECT_ROOT)

    assert changed is True
    data = json.loads(jobs_path.read_text())
    job = data["jobs"][0]
    assert job["no_agent"] is True
    assert job["script"] == "portfolio-lab-autonomous-agent.sh"
    assert job["workdir"] == str(PROJECT_ROOT)
    assert job["enabled_toolsets"] is None
    installed = hermes_home / "scripts" / "portfolio-lab-autonomous-agent.sh"
    assert installed.exists()
    assert "git_commit=" in installed.read_text()


def test_cron_update_persists_git_commit_metadata(tmp_path: Path, monkeypatch) -> None:
    """The cron status row should retain report evidence passed by wrappers."""
    spec = importlib.util.spec_from_file_location("cron_update", CRON_UPDATE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cron_update.py",
            "portfolio-lab-autonomous-agent",
            "ok",
            "1",
            "hermes",
            "git_commit=abc1234",
        ],
    )

    module.main()

    status = json.loads((tmp_path / "data" / "cron_status.json").read_text())
    assert status["jobs"][0]["git_commit"] == "abc1234"


def test_cron_update_preserves_existing_tasker_backend(tmp_path: Path, monkeypatch) -> None:
    """Ad-hoc updates must not re-stamp job backend away from tasker SSOT."""
    spec = importlib.util.spec_from_file_location("cron_update", CRON_UPDATE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "cron_status.json").write_text(
        json.dumps(
            {
                "backend": "tasker",
                "jobs": [
                    {
                        "name": "portfolio-lab-health",
                        "status": "error",
                        "backend": "tasker",
                        "schedule": "0,30 * * * *",
                        "enabled": True,
                        "last_run": "old",
                    }
                ],
            }
        )
    )

    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "_default_backend", "manual")
    monkeypatch.setattr(
        sys,
        "argv",
        ["cron_update.py", "portfolio-lab-health", "ok", "1.5"],
    )

    module.main()

    status = json.loads((data_dir / "cron_status.json").read_text())
    job = status["jobs"][0]
    assert job["status"] == "ok"
    assert job["backend"] == "tasker"
    assert status.get("backend") == "tasker"
