"""Regression tests for scheduled Python runtime selection."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SCRIPT = PROJECT_ROOT / "scripts" / "python_runtime.sh"
MAKEFILE = PROJECT_ROOT / "Makefile"
REPO_CRON_SCRIPTS = [
    PROJECT_ROOT / "scripts" / "cron" / "portfolio-lab-app-build.sh",
    PROJECT_ROOT / "scripts" / "cron" / "portfolio-lab-dashboard.sh",
    PROJECT_ROOT / "scripts" / "cron" / "portfolio-lab-data-pipeline.sh",
    PROJECT_ROOT / "scripts" / "cron" / "portfolio-lab-research-agent.sh",
    PROJECT_ROOT / "scripts" / "cron" / "portfolio-lab-strategy-eval.sh",
    PROJECT_ROOT / "scripts" / "cron" / "portfolio-lab-wiki-sync.sh",
]
HERMES_SCRIPTS_DIR = Path("/root/.hermes/scripts")
HERMES_CRON_SCRIPT_NAMES = [
    "portfolio-lab-app-build.sh",
    "portfolio-lab-dashboard.sh",
    "portfolio-lab-data-pipeline.sh",
    "portfolio-lab-health-monitor.sh",
    "portfolio-lab-research-agent.sh",
    "portfolio-lab-strategy-eval.sh",
    "portfolio-lab-wiki-sync.sh",
]


def test_makefile_cron_targets_use_project_runtime_launcher() -> None:
    """Makefile cron targets should not run project Python through bare python3."""
    text = MAKEFILE.read_text()

    assert "PYTHON_RUNTIME := $(PROJECT_DIR)/scripts/python_runtime.sh" in text
    assert "python3 -m src." not in text
    assert "python3 scripts/" not in text
    assert "python3 $(CRON_UPDATE)" not in text


@pytest.mark.parametrize("script_path", REPO_CRON_SCRIPTS)
def test_repo_cron_wrappers_use_project_runtime_launcher(script_path: Path) -> None:
    """Repo cron shell wrappers should use the same runtime launcher as Makefile cron."""
    text = script_path.read_text()

    assert "PYTHON_RUNTIME=" in text
    assert "python3 src/" not in text
    assert "python3 -m src." not in text


@pytest.mark.parametrize("script_name", HERMES_CRON_SCRIPT_NAMES)
def test_hermes_cron_wrappers_use_project_runtime_launcher(script_name: str) -> None:
    """Hermes cron wrappers should not bypass the project dependency environment."""
    script_path = HERMES_SCRIPTS_DIR / script_name
    if not script_path.exists():
        pytest.skip(f"{script_path} is not present on this host")

    text = script_path.read_text()

    assert "PYTHON_RUNTIME=" in text
    assert "python3 src/" not in text
    assert "python3 -m src." not in text


def test_python_runtime_launcher_prefers_uv_run_python(tmp_path: Path) -> None:
    """The shared launcher should route Python through uv when uv is available."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "uv-calls.txt"
    uv_stub = bin_dir / "uv"
    uv_stub.write_text(
        f"""#!/bin/bash
printf '%s\\n' "$@" > "{calls}"
exit 0
"""
    )
    uv_stub.chmod(uv_stub.stat().st_mode | stat.S_IXUSR)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["PORTFOLIO_LAB_PROJECT_DIR"] = str(PROJECT_ROOT)

    result = subprocess.run(
        [str(RUNTIME_SCRIPT), "-m", "src.monitor.health_check"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert calls.read_text().splitlines() == ["run", "python", "-m", "src.monitor.health_check"]
