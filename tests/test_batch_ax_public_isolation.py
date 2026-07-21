"""Batch AX / H16: PUBLIC_DATA_DIR isolation for pytest dual-write safety."""

from __future__ import annotations

import os
from pathlib import Path


def test_conftest_bootstraps_public_data_dir_env():
    """Session bootstrap sets PUBLIC_DATA_DIR away from live WWW unless allowed."""
    if os.environ.get("PORTFOLIO_LAB_ALLOW_LIVE_PUBLIC", "0") == "1":
        return  # opt-in lane — skip isolation assertion
    public = os.environ.get("PUBLIC_DATA_DIR", "")
    assert public, "PUBLIC_DATA_DIR must be set under safe pytest"
    path = Path(public)
    assert path.is_dir()
    # Must not be the live operator tree or bare repo public/data
    assert "/var/www/portfolio-lab" not in str(path.resolve())
    # Isolated trees are under mktemp prefixes (Makefile or conftest)
    assert "plab-pytest-public" in str(path) or path != (
        Path(__file__).resolve().parents[1] / "public" / "data"
    )


def test_paths_public_data_dir_matches_env():
    from src import paths as paths_mod

    env_p = Path(os.environ["PUBLIC_DATA_DIR"]).resolve()
    assert Path(paths_mod.PUBLIC_DATA_DIR).resolve() == env_p


def test_dual_write_modules_rebinding_uses_isolated_dir():
    from src.monitor import health_check as hc
    from src.monitor import rebalance_health as rh

    env_p = Path(os.environ["PUBLIC_DATA_DIR"]).resolve()
    assert Path(hc.PUBLIC_DATA_DIR).resolve() == env_p
    assert Path(rh.PUBLIC_DATA_DIR).resolve() == env_p


def test_makefile_test_isolates_public_data_dir():
    mk = Path("Makefile").read_text(encoding="utf-8")
    # test: target body must mktemp + export PUBLIC_DATA_DIR
    assert "plab-pytest-public" in mk
    assert "PUBLIC_DATA_DIR" in mk
    assert "mktemp" in mk


def test_run_tests_safe_isolates_public_data_dir():
    src = Path("scripts/run-tests-safe").read_text(encoding="utf-8")
    assert "_isolate_public_data_dir" in src
    assert "PUBLIC_DATA_DIR" in src
    assert "plab-pytest-public" in src


def test_conftest_source_contracts():
    conf = Path("tests/conftest.py").read_text(encoding="utf-8")
    assert "_bootstrap_public_data_dir_isolation" in conf
    assert "_isolate_public_data_dir_modules" in conf
    assert "allow_live_public_data" in conf
    assert "PORTFOLIO_LAB_ALLOW_LIVE_PUBLIC" in conf
