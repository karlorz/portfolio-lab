"""Fail-closed ENABLE/ALLOW-style env flags: literal 1 only."""

from __future__ import annotations

import pytest

from src.monitor.decision_registry import evaluate_promotion_candidate
from src.paths import resolve_runtime_public_data_dir

TRUTHY_OFF = ("", "0", "true", "True", "yes", "on", "2", "1 ")


@pytest.mark.parametrize("value", TRUTHY_OFF)
def test_allow_repo_public_data_off_unless_literal_one(tmp_path, value):
    live = tmp_path / "www-data"
    live.mkdir()
    root = tmp_path / "repo"
    (root / "public" / "data").mkdir(parents=True)
    got = resolve_runtime_public_data_dir(
        env={"PORTFOLIO_LAB_ALLOW_REPO_PUBLIC_DATA": value},
        live_public_data_dir=live,
        project_root=root,
        emit_log=False,
    )
    assert got == live


def test_allow_repo_public_data_on_for_literal_one(tmp_path):
    live = tmp_path / "www-data"
    live.mkdir()
    root = tmp_path / "repo"
    repo_public = root / "public" / "data"
    repo_public.mkdir(parents=True)
    got = resolve_runtime_public_data_dir(
        env={"PORTFOLIO_LAB_ALLOW_REPO_PUBLIC_DATA": "1"},
        live_public_data_dir=live,
        project_root=root,
        emit_log=False,
    )
    assert got == repo_public


@pytest.mark.parametrize("value", TRUTHY_OFF)
def test_promotion_require_wfe_off_unless_literal_one(monkeypatch, value):
    monkeypatch.setenv("PROMOTION_REQUIRE_WFE", value)
    result = evaluate_promotion_candidate(
        {
            "experiment_id": "exp-fail-closed",
            "metrics": {"sharpe": 1.5},
            "benchmark_metrics": {"sharpe": 1.0},
        }
    )
    assert "missing_walk_forward_wfe" not in result["failures"]


def test_promotion_require_wfe_on_for_literal_one(monkeypatch):
    monkeypatch.setenv("PROMOTION_REQUIRE_WFE", "1")
    result = evaluate_promotion_candidate(
        {
            "experiment_id": "exp-fail-closed",
            "metrics": {"sharpe": 1.5},
            "benchmark_metrics": {"sharpe": 1.0},
        }
    )
    assert "missing_walk_forward_wfe" in result["failures"]


def test_promotion_require_wfe_default_off(monkeypatch):
    monkeypatch.delenv("PROMOTION_REQUIRE_WFE", raising=False)
    result = evaluate_promotion_candidate(
        {
            "experiment_id": "exp-fail-closed",
            "metrics": {"sharpe": 1.5},
            "benchmark_metrics": {"sharpe": 1.0},
        }
    )
    assert "missing_walk_forward_wfe" not in result["failures"]
