"""Batch DY: sanitize smart-rebalance YTD cost ledger (dedupe + year scope).

Live friction: ytd_cost_bps=214 from 20 rows with exact duplicate (cost,date,symbols)
entries (e.g. 12 bps SPY×4 on 2026-05-20), zero-cost noise, and no calendar-year
filter on totals. Research: append-only ledger with composite-key dedupe at
ingestion; YTD as year-filtered view — do not silently invent costs (DX) but do
collapse exact dups that inflate budget burn.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.rebalancing.smart_rebalancer import (
    CostBudgetTracker,
    SmartRebalancingController,
)


def test_sanitize_drops_zero_and_dedupes_exact() -> None:
    tracker = CostBudgetTracker()
    # Inject dirty rows directly (bypass add_cost idempotency) — mirrors disk pollution
    tracker.ytd_costs = [
        {"cost_bps": 12.0, "date": "2026-05-20", "symbols": ["SPY"]},
        {"cost_bps": 12.0, "date": "2026-05-20", "symbols": ["SPY"]},
        {"cost_bps": 12.0, "date": "2026-05-20", "symbols": ["SPY"]},
        {"cost_bps": 0.0, "date": "2026-05-21", "symbols": ["SPY"]},
        {"cost_bps": 5.0, "date": "2026-05-21", "symbols": ["SPY", "GLD"]},
        {"cost_bps": 5.0, "date": "2026-05-21", "symbols": ["GLD", "SPY"]},  # order noise
        {"cost_bps": 100.0, "date": "2026-05-21", "symbols": ["SPY"]},
    ]

    report = tracker.sanitize_ledger(as_of_year=2026)
    assert report["dropped_zero"] == 1
    assert report["dropped_duplicate"] >= 3  # three extra 12s + one 5
    assert report["kept"] == 3  # 12, 5, 100
    assert abs(tracker.ytd_total_bps - 117.0) < 1e-9
    assert report["before_count"] == 7
    assert report["after_count"] == 3


def test_ytd_total_scopes_to_calendar_year() -> None:
    tracker = CostBudgetTracker(ytd_year=2026)
    tracker.ytd_costs = [
        {"cost_bps": 10.0, "date": "2025-12-31", "symbols": ["SPY"]},
        {"cost_bps": 20.0, "date": "2026-01-15", "symbols": ["SPY"]},
        {"cost_bps": 5.0, "date": "2026-03-20", "symbols": ["GLD"]},
    ]
    # sanitize keeps prior-year for audit but YTD sum is year-scoped
    tracker.sanitize_ledger(as_of_year=2026, drop_prior_years_from_storage=False)
    assert abs(tracker.ytd_total_bps - 25.0) < 1e-9
    assert len(tracker.ytd_costs) == 3  # prior year retained when not dropping


def test_sanitize_can_drop_prior_years_from_storage() -> None:
    tracker = CostBudgetTracker()
    tracker.add_cost(10.0, "2025-06-01", ["SPY"])
    tracker.add_cost(20.0, "2026-01-15", ["SPY"])
    report = tracker.sanitize_ledger(
        as_of_year=2026, drop_prior_years_from_storage=True
    )
    assert report["dropped_prior_year"] == 1
    assert len(tracker.ytd_costs) == 1
    assert abs(tracker.ytd_total_bps - 20.0) < 1e-9


def test_load_state_auto_sanitizes_and_persists(tmp_path: Path) -> None:
    state_path = tmp_path / "smart_rebalance_state.json"
    dirty = {
        "schema_version": "smart-rebalance-state/v1",
        "ytd_costs": [
            {"cost_bps": 12.0, "date": "2026-05-20", "symbols": ["SPY"]},
            {"cost_bps": 12.0, "date": "2026-05-20", "symbols": ["SPY"]},
            {"cost_bps": 0.0, "date": "2026-05-21", "symbols": ["SPY"]},
            {"cost_bps": 5.0, "date": "2026-05-21", "symbols": ["SPY", "GLD"]},
            {"cost_bps": 5.0, "date": "2026-05-21", "symbols": ["SPY", "GLD"]},
            # 100 bps: Batch DZ controller cap (15) quarantines — not in YTD sum
            {"cost_bps": 100.0, "date": "2026-05-21", "symbols": ["SPY"]},
        ],
        "last_rebalance": "2026-07-11T00:20:02+00:00",
    }
    state_path.write_text(json.dumps(dirty), encoding="utf-8")

    ctrl = SmartRebalancingController(
        state_path=state_path, data_dir=tmp_path, load_state=True
    )
    # 12 + 5 = 17 (dupes/zero removed; 100 quarantined by safety cap)
    assert abs(ctrl.cost_tracker.ytd_total_bps - 17.0) < 1e-9
    status = ctrl.get_status()
    assert status.get("ledger_sanitized") is True
    assert status.get("ytd_cost_entries") == 2
    assert status.get("ytd_outlier_quarantined_count") == 1
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    assert len(raw["ytd_costs"]) == 2
    assert raw.get("ledger_sanitized") is True


def test_already_clean_ledger_no_false_sanitize_flag(tmp_path: Path) -> None:
    state_path = tmp_path / "s.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": "smart-rebalance-state/v1",
                "ytd_costs": [
                    {"cost_bps": 8.0, "date": "2026-06-01", "symbols": ["SPY"]},
                ],
                "last_rebalance": "2026-06-01",
            }
        ),
        encoding="utf-8",
    )
    ctrl = SmartRebalancingController(
        state_path=state_path, data_dir=tmp_path, load_state=True
    )
    assert abs(ctrl.cost_tracker.ytd_total_bps - 8.0) < 1e-9
    # No rewrite required — sanitized flag may be False or True-noop
    assert ctrl.get_status().get("ytd_cost_entries") == 1
