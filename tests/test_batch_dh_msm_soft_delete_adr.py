"""Batch DH: multi_speed soft-delete walk-forward ADR + signal dict fix."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.backtest.multi_speed_momentum_backtest import (
    MultiSpeedMomentumBacktester,
    evaluate_msm_soft_delete_adr,
)
from src.dashboard.generator import DashboardGenerator


def test_compute_signal_accepts_dict_return() -> None:
    bt = MultiSpeedMomentumBacktester()
    engine = MagicMock()
    engine.get_signal_for_ticker.return_value = {"value": 0.5, "confidence": 0.8}
    bt._signal_engine = engine
    with patch(
        "src.backtest.multi_speed_momentum_backtest._MULTI_SPEED_AVAILABLE", True
    ):
        with patch(
            "src.backtest.multi_speed_momentum_backtest.MultiSpeedMomentum",
            return_value=engine,
        ):
            val = bt._compute_signal("SPY", "2024-06-01")
    assert val == 0.5


def test_adr_net_negative_keeps_soft_delete() -> None:
    result = SimpleNamespace(
        sharpe_ratio=0.75,
        baseline_sharpe=0.79,
        sharpe_improvement=-0.04,
        max_drawdown=-18.0,
        total_return=50.0,
        cagr=5.0,
    )
    adr = evaluate_msm_soft_delete_adr(result)
    assert adr["portfolio_gates_pass"] is False
    assert adr["auto_reenable"] is False
    assert "net_negative" in adr["adr_status"] or adr["adr_status"] == (
        "adr_net_negative_keep_soft_delete"
    )


def test_adr_pass_still_no_auto_reenable() -> None:
    result = SimpleNamespace(
        sharpe_ratio=0.85,
        baseline_sharpe=0.80,
        sharpe_improvement=0.05,
        max_drawdown=-15.0,
        total_return=80.0,
        cagr=8.0,
    )
    adr = evaluate_msm_soft_delete_adr(result)
    assert adr["portfolio_gates_pass"] is True
    assert adr["auto_reenable"] is False
    assert adr["adr_status"] == "adr_evidence_supports_review"


def test_adr_missing_artifact() -> None:
    adr = evaluate_msm_soft_delete_adr(
        result_path="/tmp/nonexistent_msm_backtest_xyz.json"
    )
    assert adr["portfolio_gates_pass"] is False
    assert adr["adr_status"] in {"evidence_missing", "evidence_unavailable"}


def test_zero_baseline_shadow_includes_adr() -> None:
    metrics = {
        "status": "healthy",
        "health_score": 0.55,
        "ic": 0.03,
        "ic_30d": 0.04,
        "ic_60d": 0.03,
        "ic_90d": 0.03,
        "reentry": {
            "reentry_eligible": True,
            "reentry_eps": 0.02,
            "reentry_blocked_reason": None,
            "horizons": {"ic_30d": 0.04, "ic_60d": 0.03, "ic_90d": 0.03},
        },
        "reentry_eligible": True,
    }
    with patch(
        "src.backtest.multi_speed_momentum_backtest.evaluate_msm_soft_delete_adr",
        return_value={
            "adr_status": "adr_net_negative_keep_soft_delete",
            "portfolio_gates_pass": False,
            "auto_reenable": False,
            "hint": "Net-negative keep soft-delete",
            "metrics": {"sharpe_improvement": -0.01},
        },
    ):
        shadow = DashboardGenerator._zero_baseline_shadow_checklist(
            "multi_speed_momentum", metrics
        )
    assert shadow["health_gates_pass"] is True
    assert shadow["portfolio_gates_pass"] is False
    assert shadow["shadow_reenable_ready"] is False
    assert "adr" in shadow
    assert shadow["adr"]["auto_reenable"] is False
