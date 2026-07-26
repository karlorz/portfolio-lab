"""Batch DB: direction deadband fix + international activation disclosure."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

from src.dashboard.generator import DashboardGenerator
from src.signals.health_tracker import SignalHealthTracker


def test_direction_from_signal_value_includes_clip_boundary() -> None:
    # regime_arb clips to ±0.2 — must be directional under new deadband
    assert SignalHealthTracker.direction_from_signal_value(0.2) == 1
    assert SignalHealthTracker.direction_from_signal_value(-0.2) == -1
    assert SignalHealthTracker.direction_from_signal_value(0.05) == 1
    assert SignalHealthTracker.direction_from_signal_value(-0.05) == -1
    assert SignalHealthTracker.direction_from_signal_value(0.04) == 0
    assert SignalHealthTracker.direction_from_signal_value(0.0) == 0


def test_log_prediction_simple_uses_inclusive_deadband() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "t.db"
        tracker = SignalHealthTracker(db)
        tracker.log_prediction_simple(
            source="cross_asset_regime_arb",
            signal_value=0.2,
            confidence=0.5,
            timestamp="2026-07-22T00:00:00",
        )
        tracker.log_prediction_simple(
            source="cross_asset_regime_arb",
            signal_value=0.052,
            confidence=0.5,
            timestamp="2026-07-22T00:01:00",
        )
        tracker.log_prediction_simple(
            source="cross_asset_regime_arb",
            signal_value=0.01,
            confidence=0.5,
            timestamp="2026-07-22T00:02:00",
        )
        conn = sqlite3.connect(db)
        dirs = [
            r[0]
            for r in conn.execute(
                "SELECT predicted_direction FROM signal_predictions ORDER BY timestamp"
            )
        ]
        conn.close()
        assert dirs == [1, 1, 0]


def test_repair_neutral_predicted_directions() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "t.db"
        tracker = SignalHealthTracker(db)
        # Simulate legacy bug: |value| high but direction 0
        with sqlite3.connect(db) as conn:
            conn.execute(
                """
                INSERT INTO signal_predictions
                (timestamp, source, signal_value, confidence, predicted_direction, metadata)
                VALUES
                ('2026-07-01T00:00:00', 'cross_asset_regime_arb', 0.12, 0.5, 0, '{}'),
                ('2026-07-02T00:00:00', 'cross_asset_regime_arb', -0.08, 0.5, 0, '{}'),
                ('2026-07-03T00:00:00', 'cross_asset_regime_arb', 0.01, 0.5, 0, '{}')
                """
            )
            conn.commit()
        n = tracker.repair_neutral_predicted_directions(source="cross_asset_regime_arb")
        assert n == 2
        with sqlite3.connect(db) as conn:
            rows = conn.execute(
                "SELECT signal_value, predicted_direction FROM signal_predictions "
                "ORDER BY timestamp"
            ).fetchall()
        assert rows[0] == (0.12, 1)
        assert rows[1] == (-0.08, -1)
        assert rows[2] == (0.01, 0)


def test_international_activation_disclosure_live_shape() -> None:
    expl = (
        "Intl Momentum: neutral, conf=low, EFA/SPY=-3.01pp, "
        "EEM/SPY=-9.75pp, VIX_filter=False"
    )
    act = DashboardGenerator._international_activation_disclosure(
        explanation=expl, value=0.0, confidence=0.0
    )
    assert "signal_type_neutral" in act["activation_gaps"]
    assert "confidence_below_0.5" in act["activation_gaps"]
    assert any("efa_rs_below" in g for g in act["activation_gaps"])
    assert any("eem_rs_below" in g for g in act["activation_gaps"])
    assert act["efa_threshold_pp"] == 5.0
    assert act["eem_threshold_pp"] == 8.0
    assert act["efa_vs_spy_pp"] == -3.01
    assert act["activation_hint"]


def test_inactive_signal_row_attaches_activation() -> None:
    statuses = DashboardGenerator._build_configured_source_status(
        regime="normal",
        source_breakdown=[
            {
                "source": "international_momentum",
                "weight": 0.0,
                "is_active": False,
                "value": 0.0,
                "confidence": 0.0,
                "inactive_explanation": (
                    "Intl Momentum: neutral, conf=low, EFA/SPY=-3.01pp, "
                    "EEM/SPY=-9.75pp, VIX_filter=False"
                ),
            }
        ],
    )
    intl = next(r for r in statuses if r["source"] == "international_momentum")
    assert intl["status"] == "inactive_signal"
    assert "activation" in intl
    assert intl["activation"]["efa_vs_spy_pp"] == -3.01
    assert "activation:" in intl["reason"]
