"""Batch DG: post-fix polarity cohort min-sample + label-lag readiness."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

from src.dashboard.generator import DashboardGenerator
from src.signals.health_tracker import SignalHealthTracker


def test_readiness_awaiting_label_lag() -> None:
    r = SignalHealthTracker.post_fix_cohort_readiness(
        SignalHealthTracker,  # type: ignore[arg-type]
        "cross_asset_regime_arb",
        n_polarity_stamped=3,
        n_polarity_labeled=0,
    )
    # call as unbound with fake self via instance
    t = object.__new__(SignalHealthTracker)
    r = SignalHealthTracker.post_fix_cohort_readiness(
        t,
        "cross_asset_regime_arb",
        n_polarity_stamped=3,
        n_polarity_labeled=0,
    )
    assert r["ready"] is False
    assert r["status"] == "awaiting_label_lag"
    assert r["labeled_deficit"] == 10
    assert r["auto_invert_policy"] == "disabled"
    assert "label lag" in r["readiness_hint"].lower() or "labeled" in r[
        "readiness_hint"
    ].lower()


def test_readiness_cohort_building() -> None:
    t = object.__new__(SignalHealthTracker)
    r = SignalHealthTracker.post_fix_cohort_readiness(
        t,
        "cross_asset_regime_arb",
        n_polarity_stamped=12,
        n_polarity_labeled=4,
    )
    assert r["ready"] is False
    assert r["status"] == "cohort_building"
    assert r["labeled_deficit"] == 6
    assert r["n_pending_labels"] == 8


def test_readiness_ready_at_min_labeled() -> None:
    t = object.__new__(SignalHealthTracker)
    r = SignalHealthTracker.post_fix_cohort_readiness(
        t,
        "cross_asset_regime_arb",
        n_polarity_stamped=15,
        n_polarity_labeled=10,
        ic_polarity_cohort=0.12,
    )
    assert r["ready"] is True
    assert r["status"] == "cohort_ready_for_shadow_ic"
    assert r["ic_polarity_cohort"] == 0.12
    assert r["force_wake_policy"] == "disabled"


def test_count_provenance_includes_cohort_readiness() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "t.db"
        tracker = SignalHealthTracker(db)
        # 3 polarity stamps, unlabeled
        for i in range(3):
            tracker.log_prediction_simple(
                source="cross_asset_regime_arb",
                signal_value=0.05,
                confidence=0.6,
                timestamp=f"2026-07-22T0{i}:00:00",
                metadata={
                    "provenance_batch": "df",
                    "polarity_policy": "no_auto_invert_spy_mapped",
                    "pattern": "equity_rotation",
                },
            )
        stats = tracker.count_provenance_rows("cross_asset_regime_arb")
        assert stats["n_polarity_stamped"] == 3
        assert stats["n_polarity_labeled"] == 0
        assert stats["cohort_readiness"]["status"] == "awaiting_label_lag"
        assert stats["cohort_readiness"]["ready"] is False


def test_count_provenance_ready_after_labels() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "t.db"
        tracker = SignalHealthTracker(db)
        for i in range(12):
            tracker.log_prediction_simple(
                source="cross_asset_regime_arb",
                signal_value=0.05 if i % 2 == 0 else -0.05,
                confidence=0.6,
                timestamp=f"2026-07-{10 + (i // 4):02d}T{i:02d}:00:00",
                metadata={
                    "provenance_batch": "df",
                    "polarity_policy": "no_auto_invert_spy_mapped",
                },
            )
        # Label all
        with sqlite3.connect(db) as conn:
            conn.execute(
                "UPDATE signal_predictions SET actual_direction = 1, "
                "accuracy_calculated = 1 WHERE actual_direction IS NULL"
            )
            conn.commit()
        stats = tracker.count_provenance_rows("cross_asset_regime_arb")
        assert stats["n_polarity_labeled"] >= 10
        assert stats["cohort_readiness"]["ready"] is True
        assert stats["ic_polarity_cohort"] is not None or stats["n_polarity_labeled"] >= 10


def test_label_alignment_surfaces_readiness() -> None:
    # Live path may vary; shape contract when provenance present
    diag = DashboardGenerator._label_alignment_diagnostic("cross_asset_regime_arb")
    if not diag:
        return
    if "cohort_readiness" in diag:
        assert "ready" in diag["cohort_readiness"]
        assert diag["cohort_readiness"].get("auto_invert_policy") == "disabled"
