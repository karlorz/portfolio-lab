"""Tests for S1 maintenance playbook entrypoint and runbooks."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = PROJECT_ROOT / "ops" / "maintenance" / "run.sh"
DAILY_RUNBOOK = PROJECT_ROOT / "ops" / "maintenance" / "daily.md"
WEEKLY_RUNBOOK = PROJECT_ROOT / "ops" / "maintenance" / "weekly.md"


@pytest.fixture(autouse=True)
def _clear_price_cache():
    """Override tests/conftest.py so `uv run --with pytest --no-project` works.

    The session conftest autouse imports cachetools + src.signals.vpin_bvc;
    ops maintenance tests are read-only sh/runbook checks and do not need it.
    """
    yield


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

    # Autofix class registry and gate
    assert "Autofix Classes" in content
    assert "OPS_MAINT_AUTOFIX" in content
    assert "log-hygiene" in content
    assert "stale-lock-cleanup" in content
    assert "artifact-regen" in content

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

    # Autofix class registry and gate
    assert "Autofix Classes" in content
    assert "OPS_MAINT_AUTOFIX" in content
    assert "plugin-cache-refresh" in content
    assert "artifact-regen" in content

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
    assert "--model flash-max" in res.stdout
    assert str(DAILY_RUNBOOK) in res.stdout
    assert "ops-maint-daily" in res.stdout
    assert "last-daily.json" in res.stdout
    assert "report.py" in res.stdout


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
    assert "--model flash-max" in res.stdout
    assert str(WEEKLY_RUNBOOK) in res.stdout
    assert "ops-maint-weekly" in res.stdout
    assert "last-weekly.json" in res.stdout
    assert "report.py" in res.stdout


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


def test_run_script_dry_run_reports_env_file_status(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_grok = bin_dir / "grok"
    fake_grok.write_text("#!/bin/sh\nexit 0\n")
    fake_grok.chmod(0o755)

    env_file = tmp_path / "ops-maintenance.env"
    env_file.write_text("SKILLWIKI_MCP_TOKEN=dummy-test-token\n")

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["OPS_MAINT_ENV_FILE"] = str(env_file)

    res = subprocess.run(
        [str(RUN_SCRIPT), "daily", "--dry-run"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert res.returncode == 0, f"run.sh failed: {res.stderr}"
    assert f"Escalation Env: present ({env_file})" in res.stdout
    # The token value must never be printed
    assert "dummy-test-token" not in res.stdout
    assert "dummy-test-token" not in res.stderr

    env_absent = dict(env)
    env_absent["OPS_MAINT_ENV_FILE"] = str(tmp_path / "missing.env")
    res_absent = subprocess.run(
        [str(RUN_SCRIPT), "daily", "--dry-run"],
        capture_output=True,
        text=True,
        env=env_absent,
        check=False,
    )
    assert res_absent.returncode == 0, f"run.sh failed: {res_absent.stderr}"
    assert "Escalation Env: absent" in res_absent.stdout


CHECK_TOOLCHAIN_MAKE = PROJECT_ROOT / "ops" / "maintenance" / "check-toolchain-make.sh"
LIVE_TOOLCHAIN_PREFIX = "/home/box/.local/share/portfolio-lab"


def _toolchain_env(tmp_path: Path, extra: dict | None = None) -> dict[str, str]:
    """Point every override at tmp_path so the live alpine-build-root is never read."""
    env = dict(os.environ)
    env["HOME"] = str(tmp_path)
    env["PORTFOLIO_LAB_ALPINE_BUILD_ROOT"] = str(tmp_path / "alpine-build-root")
    env.pop("PORTFOLIO_LAB_MAKE_BIN", None)
    env.pop("PORTFOLIO_LAB_MAKE_WRAPPER", None)
    if extra:
        env.update(extra)
    return env


def _run_toolchain_check(
    tmp_path: Path, extra_env: dict | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["sh", str(CHECK_TOOLCHAIN_MAKE)],
        capture_output=True,
        text=True,
        env=_toolchain_env(tmp_path, extra_env),
        check=False,
        timeout=10,
    )


def _write_exec(path: Path, body: str = "#!/bin/sh\nexit 0\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _assert_hermetic(res: subprocess.CompletedProcess[str]) -> None:
    combined = f"{res.stdout}{res.stderr}"
    assert LIVE_TOOLCHAIN_PREFIX not in combined


def test_check_toolchain_make_syntax_and_executable():
    assert CHECK_TOOLCHAIN_MAKE.is_file(), f"Missing {CHECK_TOOLCHAIN_MAKE}"
    assert os.access(CHECK_TOOLCHAIN_MAKE, os.X_OK), f"{CHECK_TOOLCHAIN_MAKE} is not executable"
    text = CHECK_TOOLCHAIN_MAKE.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "#!/bin/sh"
    assert "PORTFOLIO_LAB_ALPINE_BUILD_ROOT" in text
    assert "PORTFOLIO_LAB_MAKE_BIN" in text
    assert "PORTFOLIO_LAB_MAKE_WRAPPER" in text
    res = subprocess.run(
        ["sh", "-n", str(CHECK_TOOLCHAIN_MAKE)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"sh -n failed: {res.stderr}"


def test_runbooks_mention_toolchain_make_check():
    daily = DAILY_RUNBOOK.read_text(encoding="utf-8")
    weekly = WEEKLY_RUNBOOK.read_text(encoding="utf-8")
    assert "check-toolchain-make.sh" in daily
    assert "alpine-build-root" in daily
    assert "check-toolchain-make.sh" in weekly
    assert "alpine-build-root" in weekly


def test_check_toolchain_make_ok_path(tmp_path):
    root = tmp_path / "alpine-build-root"
    loader = root / "lib" / "ld-musl-x86_64.so.1"
    make_bin = root / "usr" / "bin" / "make"
    _write_exec(loader)
    _write_exec(make_bin, "#!/bin/sh\necho 'GNU Make 4.4'\nexit 0\n")
    res = _run_toolchain_check(tmp_path)
    _assert_hermetic(res)
    assert res.returncode == 0, res.stdout + res.stderr
    assert f"loader={loader} present=yes" in res.stdout
    assert f"make_bin={make_bin} present=yes" in res.stdout
    assert f"wrapper={tmp_path / '.local' / 'bin' / 'make'} present=no" in res.stdout
    assert "probe=skipped" in res.stdout
    assert "toolchain-make: ok" in res.stdout


def test_check_toolchain_make_missing_loader(tmp_path):
    root = tmp_path / "alpine-build-root"
    make_bin = root / "usr" / "bin" / "make"
    _write_exec(make_bin)
    res = _run_toolchain_check(tmp_path)
    _assert_hermetic(res)
    assert res.returncode == 1
    assert f"loader={root / 'lib' / 'ld-musl-x86_64.so.1'} present=no" in res.stdout
    assert f"make_bin={make_bin} present=yes" in res.stdout
    assert "toolchain-make: fail" in res.stdout


def test_check_toolchain_make_missing_make_bin(tmp_path):
    root = tmp_path / "alpine-build-root"
    loader = root / "lib" / "ld-musl-x86_64.so.1"
    _write_exec(loader)
    res = _run_toolchain_check(tmp_path)
    _assert_hermetic(res)
    assert res.returncode == 1
    assert f"loader={loader} present=yes" in res.stdout
    assert f"make_bin={root / 'usr' / 'bin' / 'make'} present=no" in res.stdout
    assert "toolchain-make: fail" in res.stdout


def test_check_toolchain_make_non_executable(tmp_path):
    root = tmp_path / "alpine-build-root"
    loader = root / "lib" / "ld-musl-x86_64.so.1"
    make_bin = root / "usr" / "bin" / "make"
    loader.parent.mkdir(parents=True, exist_ok=True)
    make_bin.parent.mkdir(parents=True, exist_ok=True)
    loader.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    make_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    loader.chmod(0o644)
    make_bin.chmod(0o644)
    res = _run_toolchain_check(tmp_path)
    _assert_hermetic(res)
    assert res.returncode == 1
    assert f"loader={loader} present=no" in res.stdout
    assert f"make_bin={make_bin} present=no" in res.stdout
    assert "toolchain-make: fail" in res.stdout
