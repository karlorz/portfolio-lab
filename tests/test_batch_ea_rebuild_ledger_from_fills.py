"""Batch EA: rebuild YTD cost ledger from order-fill event history.

Live friction after DZ: ytd still 63 bps from synthetic May test rows while
order events only show 2026-05-11 / 06-11 / 07-11 fills. Research: event-sourced
TCA projector rebuilds ledger from fills in notional bps; order log is SSOT.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.rebalancing.smart_rebalancer import (
    SmartRebalancingController,
    collect_unique_order_fills,
    estimate_day_cost_bps_from_fills,
    rebuild_ytd_costs_from_order_fills,
)


def _write_order_history(data_dir: Path, name: str, orders: list) -> None:
    (data_dir / name).write_text(
        json.dumps({"date": name.replace("order-history-", "").replace(".json", ""),
                    "recent_orders": orders}),
        encoding="utf-8",
    )


def test_collect_dedupes_snapshot_rewrites(tmp_path: Path) -> None:
    fill = {
        "symbol": "SPY",
        "side": "sell",
        "fill_value": 3006.84,
        "timestamp": "2026-07-11T00:20:02.531314+00:00",
        "reason": "rebalance_down",
    }
    # Same fill rewritten into multiple daily snapshots
    _write_order_history(tmp_path, "order-history-2026-07-11.json", [fill])
    _write_order_history(tmp_path, "order-history-2026-07-12.json", [fill])
    _write_order_history(tmp_path, "order-history-2026-07-13.json", [fill])
    fills = collect_unique_order_fills(tmp_path)
    assert len(fills) == 1
    assert fills[0]["symbol"] == "SPY"


def test_estimate_day_cost_portfolio_bps() -> None:
    fills = [
        {"symbol": "SPY", "fill_value": 46000.0, "side": "buy"},
        {"symbol": "GLD", "fill_value": 38000.0, "side": "buy"},
        {"symbol": "TLT", "fill_value": 16000.0, "side": "buy"},
    ]
    # portfolio 100k: 2*0.46 + 5*0.38 + 8*0.16 = 0.92+1.9+1.28 = 4.1
    bps = estimate_day_cost_bps_from_fills(fills, portfolio_value=100_000.0)
    assert abs(bps - 4.1) < 0.05


def test_rebuild_groups_by_event_day(tmp_path: Path) -> None:
    _write_order_history(
        tmp_path,
        "order-history-2026-07-11.json",
        [
            {
                "symbol": "SPY",
                "side": "buy",
                "fill_value": 46000,
                "timestamp": "2026-05-11T03:20:31+00:00",
            },
            {
                "symbol": "GLD",
                "side": "buy",
                "fill_value": 38000,
                "timestamp": "2026-05-11T03:20:31+00:00",
            },
            {
                "symbol": "TLT",
                "side": "buy",
                "fill_value": 16000,
                "timestamp": "2026-05-11T03:20:31+00:00",
            },
            {
                "symbol": "SPY",
                "side": "sell",
                "fill_value": 3000,
                "timestamp": "2026-07-11T00:20:02+00:00",
            },
            {
                "symbol": "GLD",
                "side": "buy",
                "fill_value": 3000,
                "timestamp": "2026-07-11T00:20:02+00:00",
            },
        ],
    )
    rows, meta = rebuild_ytd_costs_from_order_fills(
        tmp_path, year=2026, portfolio_value=100_000.0
    )
    assert meta["event_days"] == 2
    assert len(rows) == 2
    dates = {r["date"] for r in rows}
    assert "2026-05-11" in dates
    assert "2026-07-11" in dates
    total = sum(r["cost_bps"] for r in rows)
    assert total < 10.0  # far below polluted 63


def test_controller_rebuild_replaces_polluted_ledger(tmp_path: Path) -> None:
    state_path = tmp_path / "smart_rebalance_state.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": "smart-rebalance-state/v1",
                "ytd_costs": [
                    {"cost_bps": 15.0, "date": "2026-05-20", "symbols": ["SPY"]},
                    {"cost_bps": 12.0, "date": "2026-05-20", "symbols": ["SPY"]},
                    {"cost_bps": 5.0, "date": "2026-05-21", "symbols": []},
                ],
                "last_rebalance": "2026-05-21T00:00:00",
            }
        ),
        encoding="utf-8",
    )
    _write_order_history(
        tmp_path,
        "order-history-2026-07-11.json",
        [
            {
                "symbol": "SPY",
                "side": "sell",
                "fill_value": 3000,
                "timestamp": "2026-07-11T00:20:02+00:00",
            },
            {
                "symbol": "GLD",
                "side": "buy",
                "fill_value": 3000,
                "timestamp": "2026-07-11T00:20:02+00:00",
            },
        ],
    )
    ctrl = SmartRebalancingController(
        state_path=state_path, data_dir=tmp_path, load_state=True
    )
    result = ctrl.rebuild_cost_ledger_from_order_history(
        year=2026, portfolio_value=100_000.0, persist=True
    )
    assert result["rebuilt"] is True
    assert result["event_days"] == 1
    assert ctrl.cost_tracker.ytd_total_bps < 1.0
    assert ctrl.cost_tracker.is_over_budget() is False
    status = ctrl.get_status()
    assert status.get("ledger_source") == "order_fill_rebuild"
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    assert raw.get("ledger_source") == "order_fill_rebuild"
    assert raw.get("ytd_costs_superseded")  # old pollution archived
    assert len(raw["ytd_costs"]) == 1
