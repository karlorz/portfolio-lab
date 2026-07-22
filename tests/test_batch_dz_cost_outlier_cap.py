"""Batch DZ: quarantine single-trade cost outliers from YTD budget sum.

Live friction after DY: ytd still 163 bps with a 100 bps SPY row — config
safety.max_single_trade_cost_bps=15 already exists but was never applied to
ledger ingress. Research: max single-trade cost cap + isolate TCA outliers so
they do not distort budget burn; keep audit trail.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.rebalancing.smart_rebalancer import (
    CostBudgetTracker,
    SmartRebalancingController,
)


def test_sanitize_quarantines_above_cap() -> None:
    tracker = CostBudgetTracker(max_single_trade_cost_bps=15.0, ytd_year=2026)
    tracker.ytd_costs = [
        {"cost_bps": 12.0, "date": "2026-05-20", "symbols": ["SPY"]},
        {"cost_bps": 100.0, "date": "2026-05-21", "symbols": ["SPY"]},
        {"cost_bps": 15.0, "date": "2026-05-20", "symbols": ["SPY"]},  # at cap: keep
        {"cost_bps": 5.0, "date": "2026-05-21", "symbols": ["SPY", "GLD"]},
    ]
    report = tracker.sanitize_ledger(as_of_year=2026)
    assert report["quarantined_outlier"] == 1
    assert abs(tracker.ytd_total_bps - 32.0) < 1e-9  # 12+15+5
    assert len(tracker.ytd_costs) == 3
    assert len(tracker.quarantined_costs) == 1
    assert abs(tracker.quarantined_costs[0]["cost_bps"] - 100.0) < 1e-9
    assert tracker.quarantined_costs[0].get("reason") == "above_max_single_trade_cost_bps"
    assert abs(report["quarantined_bps"] - 100.0) < 1e-9


def test_add_cost_quarantines_outlier_ingress() -> None:
    tracker = CostBudgetTracker(max_single_trade_cost_bps=15.0)
    tracker.add_cost(8.0, "2026-07-01", ["SPY"])
    tracker.add_cost(100.0, "2026-07-01", ["SPY"])  # outlier
    assert abs(tracker.ytd_total_bps - 8.0) < 1e-9
    assert len(tracker.ytd_costs) == 1
    assert len(tracker.quarantined_costs) == 1


def test_load_state_quarantines_live_100bps_pattern(tmp_path: Path) -> None:
    state_path = tmp_path / "smart_rebalance_state.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": "smart-rebalance-state/v1",
                "ytd_costs": [
                    {"cost_bps": 12.0, "date": "2026-05-20", "symbols": ["SPY"]},
                    {"cost_bps": 100.0, "date": "2026-05-21", "symbols": ["SPY"]},
                    {"cost_bps": 15.0, "date": "2026-05-20", "symbols": ["SPY"]},
                    {"cost_bps": 5.0, "date": "2026-05-21", "symbols": ["SPY", "GLD"]},
                    {"cost_bps": 6.0, "date": "2026-05-21", "symbols": ["SPY", "GLD", "TLT"]},
                ],
                "last_rebalance": "2026-07-11T00:20:02+00:00",
            }
        ),
        encoding="utf-8",
    )
    ctrl = SmartRebalancingController(
        state_path=state_path, data_dir=tmp_path, load_state=True
    )
    # 12+15+5+6 = 38; 100 quarantined
    assert abs(ctrl.cost_tracker.ytd_total_bps - 38.0) < 1e-9
    assert ctrl.cost_tracker.is_over_budget() is False  # 38 < 50
    status = ctrl.get_status()
    assert status.get("ytd_outlier_quarantined_count") == 1
    assert abs(status.get("ytd_outlier_quarantined_bps") - 100.0) < 1e-9
    assert status.get("max_single_trade_cost_bps") == 15.0
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    assert any(c.get("cost_bps") == 100.0 for c in raw.get("quarantined_costs") or [])
    assert not any(
        abs(float(c.get("cost_bps", 0)) - 100.0) < 1e-9 for c in raw.get("ytd_costs") or []
    )


def test_at_cap_not_quarantined() -> None:
    tracker = CostBudgetTracker(max_single_trade_cost_bps=15.0)
    tracker.add_cost(15.0, "2026-06-01", ["SPY"])
    assert abs(tracker.ytd_total_bps - 15.0) < 1e-9
    assert tracker.quarantined_costs == []
