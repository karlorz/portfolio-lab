"""Batch EF: record smart-rebalance costs on OrderRouter execution fills.

Live friction: EA rebuilds YTD from order history because record_execution was
never called on broker fills — controller clock lagged until DX, costs polluted
until EA. Research: capture explicit costs on fill/execution event as SSOT.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.broker.order_router import (
    OrderPlan,
    OrderRouter,
    estimate_execution_cost_bps,
    record_rebalance_costs_from_orders,
)


def test_estimate_execution_cost_portfolio_bps() -> None:
    executed = [
        {"symbol": "SPY", "estimated_value": 46000.0, "status": "submitted"},
        {"symbol": "GLD", "estimated_value": 38000.0, "status": "submitted"},
        {"symbol": "TLT", "estimated_value": 16000.0, "status": "submitted"},
    ]
    # SPY 2*0.46 + GLD 5*0.38 + TLT 8*0.16 = 0.92+1.9+1.28 = 4.1
    bps = estimate_execution_cost_bps(executed, portfolio_value=100_000.0)
    assert abs(bps - 4.1) < 0.05


def test_record_from_orders_updates_controller(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.rebalancing.smart_rebalancer.DATA_DIR", tmp_path, raising=False
    )
    monkeypatch.setattr(
        "src.rebalancing.integration.DATA_DIR", tmp_path, raising=False
    )
    executed = [
        {
            "symbol": "SPY",
            "side": "SELL",
            "estimated_value": 3000.0,
            "status": "submitted",
            "timestamp": "2026-07-11T00:20:02+00:00",
        },
        {
            "symbol": "GLD",
            "side": "BUY",
            "estimated_value": 3000.0,
            "status": "submitted",
            "timestamp": "2026-07-11T00:20:02+00:00",
        },
    ]
    result = record_rebalance_costs_from_orders(
        executed,
        data_dir=tmp_path,
        portfolio_value=100_000.0,
        dry_run=False,
    )
    assert result["recorded"] is True
    assert result["cost_bps"] > 0
    assert result["symbols"] == ["GLD", "SPY"]
    state = json.loads((tmp_path / "smart_rebalance_state.json").read_text())
    assert state.get("ytd_costs")
    assert any(
        abs(float(c.get("cost_bps", 0)) - result["cost_bps"]) < 1e-6
        for c in state["ytd_costs"]
    )


def test_dry_run_does_not_record(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.rebalancing.smart_rebalancer.DATA_DIR", tmp_path, raising=False
    )
    result = record_rebalance_costs_from_orders(
        [{"symbol": "SPY", "estimated_value": 5000, "status": "dry_run"}],
        data_dir=tmp_path,
        dry_run=True,
    )
    assert result["recorded"] is False
    assert result["reason"] == "dry_run"
    assert not (tmp_path / "smart_rebalance_state.json").exists()


def test_execute_orders_records_on_live_submit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.rebalancing.smart_rebalancer.DATA_DIR", tmp_path, raising=False
    )
    monkeypatch.setattr(
        "src.rebalancing.integration.DATA_DIR", tmp_path, raising=False
    )
    (tmp_path / "kill_switch.json").write_text(
        json.dumps({"enabled": False}), encoding="utf-8"
    )

    router = OrderRouter(data_dir=str(tmp_path), paper=True)
    # Force ready without real Alpaca
    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_result.id = "ord-1"
    mock_result.status = "accepted"
    mock_client.submit_order.return_value = mock_result
    router.client = mock_client

    orders = [
        OrderPlan(
            symbol="SPY",
            side="BUY",
            qty=10,
            order_type="market",
            estimated_value=5000.0,
            reason="rebalance",
        )
    ]
    with patch.object(OrderRouter, "is_ready", return_value=True):
        with patch.object(
            OrderRouter,
            "_resolve_market_session",
            return_value={"allow_live_orders": True, "reason": "test"},
        ):
            report = router.execute_orders(
                orders, dry_run=False, kill_switch_check=False
            )

    assert report.get("orders_executed") == 1
    cost_meta = report.get("rebalance_cost_record") or {}
    assert cost_meta.get("recorded") is True
    assert cost_meta.get("cost_bps", 0) > 0
    state_path = tmp_path / "smart_rebalance_state.json"
    assert state_path.exists()
