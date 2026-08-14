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
    PROJECT_ROOT / "scripts" / "cron" / "portfolio-lab-health-monitor.sh",
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


SCRIPT_SUFFIXES = {".sh", ".py", ".ts"}
PORTABILITY_SCAN_ALLOW = "# portability-scan allowlist: intentional production-path guard"


def _is_executable_script(path: Path) -> bool:
    """Shell/Python/TS code files; markdown and other docs are not scanned."""
    if path.suffix in SCRIPT_SUFFIXES:
        return True
    if path.suffix == "":
        return path.read_text(errors="ignore").startswith("#!")
    return False


def _script_files():
    scripts_dir = PROJECT_ROOT / "scripts"
    return sorted(
        path for path in scripts_dir.rglob("*") if path.is_file() and _is_executable_script(path)
    )


def _portability_violations(paths: list[Path]) -> list[str]:
    violations = []
    for script_path in paths:
        text = script_path.read_text(errors="ignore")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if (
                "/root/projects/portfolio-lab" in line
                and ":-/root/projects/portfolio-lab" not in line
                and PORTABILITY_SCAN_ALLOW not in line
            ):
                try:
                    display = script_path.relative_to(PROJECT_ROOT)
                except ValueError:
                    display = script_path
                violations.append(f"{display}:{line_number}")
    return violations


def _read_optional_host_script(script_path: Path) -> str:
    """Read host-installed wrapper scripts only when this runner can access them."""
    try:
        if not script_path.exists():
            pytest.skip(f"{script_path} is not present on this host")
        return script_path.read_text()
    except PermissionError:
        pytest.skip(f"{script_path} is not readable on this host")


def test_makefile_cron_targets_use_project_runtime_launcher() -> None:
    """Makefile cron targets should not run project Python through bare python3."""
    text = MAKEFILE.read_text()

    assert "PYTHON_RUNTIME := $(PROJECT_DIR)/scripts/python_runtime.sh" in text
    assert "python3 -m src." not in text
    assert "python3 scripts/" not in text
    assert "python3 $(CRON_UPDATE)" not in text


def test_makefile_exports_project_dir_for_runtime_launcher() -> None:
    """Makefile callers should set the runtime launcher project directory."""
    text = MAKEFILE.read_text()

    assert "PORTFOLIO_LAB_PROJECT_DIR ?= $(PROJECT_DIR)" in text
    assert "export PORTFOLIO_LAB_PROJECT_DIR" in text


def test_scripts_keep_sg01_project_path_only_as_env_fallback() -> None:
    """Hardcoded sg01 repo paths in executable scripts must be overrideable."""
    assert _portability_violations(_script_files()) == []


def test_portability_scan_covers_executable_code_not_markdown() -> None:
    """The scan targets shell/Python/TS scripts; markdown docs are out of scope."""
    scanned = {path.relative_to(PROJECT_ROOT) for path in _script_files()}
    assert Path("scripts/portfolio_lab_recovery.py") in scanned
    assert Path("scripts/deploy-lab-app.sh") in scanned
    assert Path("scripts/run-tests-safe") in scanned  # extensionless shebang script
    assert Path("scripts/LAB_APP_BACKUP_RESTORE.md") not in scanned
    assert Path("scripts/cron/README.md") not in scanned


def test_recovery_and_deploy_prod_path_guards_stay_literal_and_allowlisted() -> None:
    """Dev-mode/candidate safety guards keep the literal prod path; only the
    guard lines themselves carry the portability-scan allowlist marker."""
    for script_name in ("portfolio_lab_recovery.py", "deploy-lab-app.sh"):
        source = (PROJECT_ROOT / "scripts" / script_name).read_text(encoding="utf-8")
        for line_number, line in enumerate(source.splitlines(), start=1):
            if (
                "/root/projects/portfolio-lab" in line
                and ":-/root/projects/portfolio-lab" not in line
            ):
                assert PORTABILITY_SCAN_ALLOW in line, (
                    f"{script_name}:{line_number} hardcodes the prod path without the marker"
                )
    recovery = (PROJECT_ROOT / "scripts" / "portfolio_lab_recovery.py").read_text(encoding="utf-8")
    assert (
        'app_forbidden = "/root/projects/portfolio-lab" in (str(app_dir), str(raw_app_dir))'
        in recovery
    )
    assert (
        'die("dev mode rejects production paths (/root/projects/portfolio-lab, /var/www/portfolio-lab)")'
        in recovery
    )
    deploy = (PROJECT_ROOT / "scripts" / "deploy-lab-app.sh").read_text(encoding="utf-8")
    assert '[ "$APP_DIR" != "/root/projects/portfolio-lab" ]' in deploy


def test_portability_scan_allows_marked_prod_path_safety_guard(tmp_path: Path) -> None:
    """A safety guard that must name the prod path is allowed when it carries the marker."""
    guarded = tmp_path / "guard.py"
    guarded.write_text(
        f'app_forbidden = "/root/projects/portfolio-lab" in (str(app_dir), str(raw_app_dir))'
        f"  {PORTABILITY_SCAN_ALLOW}\n"
        f'die("dev mode rejects production paths (/root/projects/portfolio-lab)")'
        f"  {PORTABILITY_SCAN_ALLOW}\n",
        encoding="utf-8",
    )
    assert _portability_violations([guarded]) == []


def test_portability_scan_flags_new_hardcoded_prod_path_reference(tmp_path: Path) -> None:
    """Unmarked hardcoded prod-path assignment/reference in executable code is a violation."""
    offender = tmp_path / "offender.py"
    offender.write_text(
        'APP_DIR = "/root/projects/portfolio-lab"\n'
        'run(APP_DIR)\n',
        encoding="utf-8",
    )
    violations = _portability_violations([offender])
    assert len(violations) == 1
    assert violations[0].endswith("offender.py:1")

    fallback = tmp_path / "fallback.sh"
    fallback.write_text(
        'APP_DIR="${APP_DIR:-/root/projects/portfolio-lab}"\n',
        encoding="utf-8",
    )
    assert _portability_violations([fallback]) == []


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
    text = _read_optional_host_script(script_path)

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
