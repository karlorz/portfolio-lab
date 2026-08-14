"""Batch DF: prediction metadata provenance for post-fix shadow IC cohorts."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.signals.health_tracker import SignalHealthTracker
from src.signals.signal_snapshot import SignalSnapshot
from src.strategy.ensemble_voter import EnsembleVoter, SignalReading, SignalSource


def test_snapshot_to_reading_carries_metadata() -> None:
    snap = SignalSnapshot(
        source="cross_asset_regime_arb",
        timestamp="2026-07-22T00:00:00",
        value=0.05,
        confidence=0.6,
        explanation="equity rotation",
        metadata={
            "pattern": "equity_rotation",
            "polarity_policy": "no_auto_invert_spy_mapped",
            "equity_regime": "bull",
        },
    )
    reading = snap.to_signal_reading()
    assert reading.metadata is not None
    assert reading.metadata.get("pattern") == "equity_rotation"
    assert reading.metadata.get("polarity_policy") == "no_auto_invert_spy_mapped"


def test_log_prediction_simple_stores_metadata() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "t.db"
        tracker = SignalHealthTracker(db)
        tracker.log_prediction_simple(
            source="cross_asset_regime_arb",
            signal_value=0.05,
            confidence=0.6,
            timestamp="2026-07-22T00:00:00",
            metadata={
                "provenance_batch": "df",
                "polarity_policy": "no_auto_invert_spy_mapped",
                "pattern": "equity_rotation",
            },
        )
        conn = sqlite3.connect(db)
        meta = conn.execute(
            "SELECT metadata FROM signal_predictions WHERE source='cross_asset_regime_arb'"
        ).fetchone()[0]
        conn.close()
        parsed = json.loads(meta)
        assert parsed["provenance_batch"] == "df"
        assert parsed["polarity_policy"] == "no_auto_invert_spy_mapped"


def test_apply_weights_logs_metadata() -> None:
    voter = EnsembleVoter.__new__(EnsembleVoter)
    reading = SignalReading(
        source=SignalSource.CROSS_ASSET_REGIME_ARB,
        timestamp="t",
        value=0.05,
        confidence=0.6,
        weight=0.0,
        regime_fit="all",
        explanation="Equity (bull) diverging",
        is_active=True,
        metadata={"pattern": "equity_rotation", "polarity_policy": "no_auto_invert_spy_mapped"},
    )
    mock_tracker = MagicMock()
    with patch(
        "src.strategy.ensemble_voter_vote._get_health_tracker", return_value=mock_tracker
    ):
        out = voter._apply_weights_to_readings(
            {SignalSource.CROSS_ASSET_REGIME_ARB: reading},
            {SignalSource.CROSS_ASSET_REGIME_ARB: 0.1},
        )
    assert len(out) == 1
    mock_tracker.log_prediction_simple.assert_called_once()
    kwargs = mock_tracker.log_prediction_simple.call_args.kwargs
    assert kwargs["metadata"]["pattern"] == "equity_rotation"
    assert kwargs["metadata"]["provenance_batch"] == "df"
    assert kwargs["metadata"]["polarity_policy"] == "no_auto_invert_spy_mapped"


def test_count_provenance_rows_empty_cohort() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "t.db"
        tracker = SignalHealthTracker(db)
        # legacy empty metadata
        tracker.log_prediction_simple(
            source="cross_asset_regime_arb",
            signal_value=0.1,
            confidence=0.5,
            timestamp="2026-07-01T00:00:00",
            metadata={},
        )
        # stamped
        tracker.log_prediction_simple(
            source="cross_asset_regime_arb",
            signal_value=-0.1,
            confidence=0.5,
            timestamp="2026-07-02T00:00:00",
            metadata={
                "provenance_batch": "df",
                "polarity_policy": "no_auto_invert_spy_mapped",
            },
        )
        stats = tracker.count_provenance_rows("cross_asset_regime_arb", lookback_days=90)
        assert stats["n_rows"] == 2
        assert stats["n_with_provenance"] == 1
        assert stats["n_polarity_stamped"] == 1
        assert stats["policy"] == "shadow_ic_post_fix_no_auto_invert"
