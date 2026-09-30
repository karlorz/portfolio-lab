"""Tests for S1 maintenance playbook entrypoint and runbooks."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = PROJECT_ROOT / "ops" / "maintenance" / "run.sh"
DAILY_RUNBOOK = PROJECT_ROOT / "ops" / "maintenance" / "daily.md"
WEEKLY_RUNBOOK = PROJECT_ROOT / "ops" / "maintenance" / "weekly.md"


def test_daily_runbook_must_not_rails_and_contracts():
    assert DAILY_RUNBOOK.is_file(), f"Missing {DAILY_RUNBOOK}"
    content = DAILY_RUNBOOK.read_text(encoding="utf-8")

    # MUST NOT rail checks
    assert "MUST NOT touch" in content
    assert "/home/box/.local/share/portfolio-lab/app" in content
    assert "MUST NOT call broker trade APIs or invoke `unlock_trade`" in content
    assert "MUST NOT drop stashes or delete branches without" in content
    assert "MUST NOT execute `pkill -f` or `killall` patterns" in content
    assert "MUST NOT start any second Tasker scheduler" in content
    assert "PORTFOLIO_LAB_ENABLE_ML" in content
    assert "must stay `0`" in content or "stays 0" in content

    # Output contract checks
    assert "data/ops-maintenance/last-daily.json" in content
    assert "host_health" in content
    assert "checks" in content
    assert "drift" in content
    assert "actions" in content
    assert "escalations" in content
    assert "ESCALATION:" in content
    assert "wiki_capture" in content


def test_weekly_runbook_must_not_rails_and_contracts():
    assert WEEKLY_RUNBOOK.is_file(), f"Missing {WEEKLY_RUNBOOK}"
    content = WEEKLY_RUNBOOK.read_text(encoding="utf-8")

    # MUST NOT rail checks
    assert "MUST NOT touch" in content
    assert "/home/box/.local/share/portfolio-lab/app" in content
    assert "MUST NOT call broker trade APIs or invoke `unlock_trade`" in content
    assert "MUST NOT drop stashes or delete branches without" in content
    assert "MUST NOT execute `pkill -f` or `killall` patterns" in content
    assert "MUST NOT start any second Tasker scheduler" in content
    assert "PORTFOLIO_LAB_ENABLE_ML" in content
    assert "archive/..." in content or "archive/" in content

    # Deep weekly audits
    assert "uv pip list --outdated" in content
    assert "make test-gate" in content
    assert "grok plugin update" in content

    # Output contract checks
    assert "data/ops-maintenance/last-weekly.json" in content
    assert "host_health" in content
    assert "checks" in content
    assert "drift" in content
    assert "actions" in content
    assert "escalations" in content
    assert "ESCALATION:" in content
    assert "wiki_capture" in content


def test_run_script_syntax():
    assert RUN_SCRIPT.is_file(), f"Missing {RUN_SCRIPT}"
    assert os.access(RUN_SCRIPT, os.X_OK), f"{RUN_SCRIPT} is not executable"

    res = subprocess.run(
        ["bash", "-n", str(RUN_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"bash -n failed: {res.stderr}"


def test_run_script_unknown_cycle_exits_nonzero():
    res = subprocess.run(
        [str(RUN_SCRIPT), "monthly"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode != 0
    assert "Usage:" in res.stderr
    assert "daily|weekly" in res.stderr


def test_run_script_no_args_exits_nonzero():
    res = subprocess.run(
        [str(RUN_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode != 0
    assert "Usage:" in res.stderr


def test_run_script_dry_run_daily(tmp_path):
    # Ensure a dummy grok binary exists if not already present
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_grok = bin_dir / "grok"
    fake_grok.write_text("#!/bin/sh\nexit 0\n")
    fake_grok.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"

    res = subprocess.run(
        [str(RUN_SCRIPT), "daily", "--dry-run"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert res.returncode == 0, f"run.sh failed: {res.stderr}"
    assert "--agent minimal" in res.stdout
    assert str(DAILY_RUNBOOK) in res.stdout
    assert "ops-maint-daily" in res.stdout


def test_run_script_dry_run_weekly(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_grok = bin_dir / "grok"
    fake_grok.write_text("#!/bin/sh\nexit 0\n")
    fake_grok.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"

    res = subprocess.run(
        [str(RUN_SCRIPT), "weekly", "--dry-run"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert res.returncode == 0, f"run.sh failed: {res.stderr}"
    assert "--agent minimal" in res.stdout
    assert str(WEEKLY_RUNBOOK) in res.stdout
    assert "ops-maint-weekly" in res.stdout


def test_run_script_fails_closed_when_grok_absent(tmp_path):
    empty_bin = tmp_path / "empty_bin"
    empty_bin.mkdir()

    env = dict(os.environ)
    env["PATH"] = str(empty_bin)
    env["OPS_SKIP_PATH_PREPEND"] = "1"

    res = subprocess.run(
        [str(RUN_SCRIPT), "daily", "--dry-run"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert res.returncode != 0
    assert "grok" in res.stderr
    assert "not found" in res.stderr


def test_run_script_fails_closed_when_runbook_missing(tmp_path):
    empty_ops = tmp_path / "empty_ops"
    empty_ops.mkdir()

    # Supply dummy grok so grok check passes
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_grok = bin_dir / "grok"
    fake_grok.write_text("#!/bin/sh\nexit 0\n")
    fake_grok.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["OPS_MAINT_DIR"] = str(empty_ops)

    res = subprocess.run(
        [str(RUN_SCRIPT), "daily", "--dry-run"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert res.returncode != 0
    assert "Runbook file not found" in res.stderr
    assert str(empty_ops / "daily.md") in res.stderr
