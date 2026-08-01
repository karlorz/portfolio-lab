"""Batch DI: alt soft-scale snapshot, provenance cohort, dual-pass label resolve."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import math
from datetime import datetime, timezone

from src.signals.alternative_data_signal import AlternativeDataSignalGenerator
from src.signals.health_tracker import SignalHealthTracker


def test_alt_snapshot_soft_scales_composite() -> None:
    gen = AlternativeDataSignalGenerator()
    composite = 0.2889
    soft = math.tanh(composite / 0.5)
    signal = SimpleNamespace(
        timestamp=datetime.now(timezone.utc).isoformat(),
        confidence=0.61,
        probability=0.8,
        regime="bull",
        raw_data={"composite_score": composite, "components": {}},
    )
    with patch.object(gen, "load_latest_signal", return_value=signal):
        snap = gen.get_signal_snapshot()
    assert abs(snap.value - (-soft)) < 1e-6  # SPY polarity map inverts sign
    assert snap.metadata.get("value_scale") == "tanh_0.5_spy_mapped"
    assert snap.metadata.get("composite_raw") == composite
    assert snap.metadata.get("polarity_policy") == "no_auto_invert_spy_mapped"
    assert abs(snap.value) < abs(composite) or abs(composite) < 0.5  # magnitude unchanged


def test_tail_risk_uses_tanh_scale() -> None:
    gen = AlternativeDataSignalGenerator()
    # Synthetic low-vol path: short vol << long vol → positive soft value
    prices = [100.0 + 0.01 * i for i in range(520)]
    with patch.object(gen, "_get_prices", return_value=prices):
        c = gen._tail_risk_signal()
    assert c.raw_inputs.get("scale") == "tanh_0.6"
    assert c.value < 1.0


def test_provenance_readiness_awaiting_label_lag() -> None:
    t = object.__new__(SignalHealthTracker)
    r = SignalHealthTracker.post_fix_provenance_readiness(
        t,
        "alternative_data",
        n_provenance_stamped=3,
        n_provenance_labeled=0,
    )
    assert r["ready"] is False
    assert r["status"] == "awaiting_label_lag"
    assert r["cohort_kind"] == "provenance_batch"


def test_count_provenance_prefers_provenance_when_no_polarity() -> None:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "t.db"
        tracker = SignalHealthTracker(db)
        for i in range(3):
            tracker.log_prediction_simple(
                source="alternative_data",
                signal_value=0.3,
                confidence=0.5,
                timestamp=f"2026-07-22T0{i}:00:00",
                metadata={"provenance_batch": "df", "value_scale": "tanh_0.5"},
            )
        stats = tracker.count_provenance_rows("alternative_data")
        assert stats["n_polarity_stamped"] == 0
        assert stats["n_with_provenance"] == 3
        assert stats["cohort_readiness"]["status"] == "awaiting_label_lag"
        assert stats["cohort_readiness"].get("cohort_kind") == "provenance_batch"


def test_resolve_dual_pass_when_newest_all_skipped() -> None:
    t = SignalHealthTracker.__new__(SignalHealthTracker)
    calls = {"oldest": 0}

    def fake_list(limit=30, oldest_first=False):
        if oldest_first:
            return ["2026-06-01"]
        return ["2026-07-22", "2026-07-21"]

    def fake_fwd(d):
        if d in ("2026-07-22", "2026-07-21"):
            return None
        return 0.01

    def fake_update(returns, date):
        return 5 if date == "2026-06-01" else 0

    def resolve_impl(max_days=30, oldest_first=False):
        # re-bind real method body via tracker instance methods
        return SignalHealthTracker.resolve_pending_labels(
            t, max_days=max_days, oldest_first=oldest_first
        )

    t.list_unresolved_prediction_dates = fake_list  # type: ignore[method-assign]
    t._spy_forward_return = fake_fwd  # type: ignore[method-assign]
    t.update_actual_directions = fake_update  # type: ignore[method-assign]
    t.db_path = None  # unused

    # Call real method
    summary = SignalHealthTracker.resolve_pending_labels(t, max_days=5, oldest_first=False)
    assert summary.get("dual_pass_oldest") is True
    assert summary["predictions_updated"] == 5
    assert summary["dates_resolved"] >= 1
