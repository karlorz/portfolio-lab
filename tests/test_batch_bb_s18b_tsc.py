"""Batch BB: S18b optional suite cron docs + BondMomentum unknown-first cast."""

from __future__ import annotations

from pathlib import Path


def test_optional_suite_targets_not_in_cron_targets():
    from src.cron_compat import CRON_TARGETS, OPTIONAL_SUITE_TARGETS

    assert "portfolio-lab-test-unit" in OPTIONAL_SUITE_TARGETS
    assert "portfolio-lab-test-full" in OPTIONAL_SUITE_TARGETS
    assert "portfolio-lab-test-generator" in OPTIONAL_SUITE_TARGETS
    # Must not pollute production CRON_TARGETS / tasker / cron_status
    for job_id in OPTIONAL_SUITE_TARGETS:
        assert job_id not in CRON_TARGETS


def test_optional_suite_makefile_targets_exist():
    from src.cron_compat import OPTIONAL_SUITE_TARGETS

    mk = Path("Makefile").read_text(encoding="utf-8")
    for make_target in OPTIONAL_SUITE_TARGETS.values():
        # target: line
        assert f"{make_target}:" in mk or f"\n{make_target}:" in mk


def test_crontab_documents_s18b_commented_suite_jobs():
    ct = Path("crontab").read_text(encoding="utf-8")
    assert "S18b" in ct
    assert "test-unit" in ct
    assert "test-generator" in ct
    # Must remain commented (not live production cron)
    for line in ct.splitlines():
        if "make" in line and ("test-unit" in line or "test-generator" in line):
            assert line.lstrip().startswith("#"), f"suite cron must stay commented: {line}"


def test_cron_verify_still_passes_after_s18b():
    """Optional suite comments must not break Makefile↔crontab coverage."""
    import subprocess

    r = subprocess.run(
        ["uv", "run", "python", "scripts/cron_verify.py", "--crontab", "crontab"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert r.returncode == 0, r.stdout + r.stderr


def test_bond_momentum_unknown_first_cast():
    src = Path("src/components/BondMomentumPanel.tsx").read_text(encoding="utf-8")
    assert "as unknown as BondMomentumEnsemble" in src
    assert "as BondMomentumEnsemble)" not in src or "as unknown as BondMomentumEnsemble" in src
