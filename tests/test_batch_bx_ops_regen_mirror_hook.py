"""Batch BX: ops-regen + deploy soft-gate repo public/data mirror."""

from __future__ import annotations

from pathlib import Path


def test_ops_regen_includes_mirror_soft_gate():
    makefile = Path("Makefile").read_text(encoding="utf-8")
    assert "ops-regen:" in makefile
    body = makefile.split("ops-regen:")[1].split("# ──")[0]
    assert "dashboard" in body
    assert "wiki-sync" in body
    assert "health" in body
    # Batch BX soft gate — must not use hard-fail without ||
    assert "mirror-repo-public-data" in body
    assert "||" in body  # soft-fail pattern
    assert "soft-failed" in body or "non-blocking" in body


def test_deploy_lab_app_mirror_hook_soft_gate():
    src = Path("scripts/deploy-lab-app.sh").read_text(encoding="utf-8")
    assert "SKIP_MIRROR" in src
    assert "--skip-mirror" in src
    assert "mirror_repo_public_data_from_live" in src
    assert "mirror_repo_public_data.py" in src
    # soft: warn on failure, do not die
    assert "soft-failed" in src or "non-blocking" in src
    assert "mirror_repo_public_data_from_live" in src
    # Invoked from main before build so dist captures the fresh live mirror.
    main = src.split("main()")[1]
    assert "mirror_repo_public_data_from_live" in main
    assert "publish_dist" in main
    assert "build_frontend" in main
    assert main.find("mirror_repo_public_data_from_live") < main.find("build_frontend")
    assert main.find("build_frontend") < main.find("publish_dist")


def test_dashboard_delivery_ops_regen_still_has_core_steps():
    """Regression: AA/contracts still see dashboard+wiki+health in ops-regen."""
    makefile = Path("Makefile").read_text(encoding="utf-8")
    body = makefile.split("ops-regen:")[1].split("# ──")[0]
    assert "garch-risk" in body or "dashboard" in body
    assert "health" in body


def test_data_and_dashboard_include_mirror_soft_gate():
    """Batch CA: hourly data/dashboard success must soft-mirror live WWW → repo public/data.

    Repo public/data is gitignored static checkout; without post-job mirror it
    freezes while /var/www/portfolio-lab/data advances (c359/c360 false lag).
    Soft-fail (||) matches ops-regen BX — never block the owning job.
    """
    makefile = Path("Makefile").read_text(encoding="utf-8")
    # Extract recipe bodies (next target starts with .PHONY or bare target:)
    data_body = makefile.split("\ndata:")[1].split("\n.PHONY:")[0]
    dash_body = makefile.split("\ndashboard:")[1].split("\n# ──")[0]
    for body, name in ((data_body, "data"), (dash_body, "dashboard")):
        assert "mirror-repo-public-data" in body, f"{name} missing mirror soft-gate"
        assert "||" in body, f"{name} mirror must soft-fail with ||"
        assert "non-blocking" in body or "soft-failed" in body, f"{name} needs soft-fail message"
