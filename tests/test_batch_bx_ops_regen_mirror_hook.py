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
    # invoked from main after publish
    main = src.split("main()")[1]
    assert "mirror_repo_public_data_from_live" in main
    assert "publish_dist" in main
    # publish before mirror (order: publish then mirror)
    assert main.find("publish_dist") < main.find("mirror_repo_public_data_from_live")


def test_dashboard_delivery_ops_regen_still_has_core_steps():
    """Regression: AA/contracts still see dashboard+wiki+health in ops-regen."""
    makefile = Path("Makefile").read_text(encoding="utf-8")
    body = makefile.split("ops-regen:")[1].split("# ──")[0]
    assert "garch-risk" in body or "dashboard" in body
    assert "health" in body
