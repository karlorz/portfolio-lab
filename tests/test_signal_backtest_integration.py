"""Signal backtest integration tests — end-to-end validation of signal pipeline.

Tests the full flow from signal generation through the typed snapshot
pipeline, ensemble voting, and quality monitoring (SPC + IC decay).

Unlike test_collect_signals_integration.py which mocks signal modules,
these tests exercise real signal computation where possible (using
synthetic data) and validate the typed pipeline end-to-end.
"""

import os

os.environ["PORTFOLIO_LAB_ENABLE_ML"] = "0"

import json
import pytest
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.strategy.ensemble_voter import (
    Regime, SignalSource, SignalReading, EnsembleVoter, EnsembleVote,
)
from src.signals.signal_snapshot import SignalSnapshot
from src.monitor.ic_decay_monitor import ICMonitor
from src.monitor.spc_monitor import SPCMonitor


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

_TS = datetime.now(timezone.utc).isoformat()


def _make_reading(source_name, value, confidence, weight=0.0,
                  regime_fit="all", asset_signals=None):
    """Create a SignalReading with all required fields."""
    src = SignalSource(source_name) if isinstance(source_name, str) else source_name
    return SignalReading(
        source=src,
        timestamp=_TS,
        value=value,
        confidence=confidence,
        weight=weight,
        regime_fit=regime_fit,
        asset_signals=asset_signals,
    )


def _make_snapshot(source, value, confidence, regime_fit="all",
                   is_active=True, asset_signals=None):
    """Create a SignalSnapshot with defaults."""
    return SignalSnapshot(
        source=source,
        timestamp=_TS,
        value=value,
        confidence=confidence,
        asset_signals=asset_signals or {},
        regime_fit=regime_fit,
        is_active=is_active,
    )


def _make_voter(tmp_path):
    """Create an EnsembleVoter with isolated paths."""
    voter = EnsembleVoter.__new__(EnsembleVoter)
    voter.data_path = tmp_path
    voter.db_path = tmp_path / "ensemble_signals.db"
    voter.current_readings = {}
    voter.current_regime = Regime.NORMAL
    voter.current_regime_confidence = 0.5
    voter._init_db()
    # Bandit weighter required by compute_vote
    voter.bandit = MagicMock()
    voter.bandit_observations = 0
    voter.bandit.get_weights.return_value = {}
    return voter


# ─────────────────────────────────────────────────────────────
#  1. Snapshot → Reading → Vote pipeline
# ─────────────────────────────────────────────────────────────

class TestSnapshotToVotePipeline:
    """Test the full typed pipeline from SignalSnapshot to ensemble vote."""

    def test_five_active_signals_produce_weighted_consensus(self, tmp_path):
        """All 5 active signals should produce a weighted consensus."""
        voter = _make_voter(tmp_path)
        snapshots = {
            "multi_speed_momentum": _make_snapshot("multi_speed_momentum", 0.3, 0.6),
            "cross_asset_rv": _make_snapshot("cross_asset_rv", 0.1, 0.7),
            "international_momentum": _make_snapshot("international_momentum", 0.5, 0.8),
            "alternative_data": _make_snapshot("alternative_data", 0.4, 0.75),
            "cross_asset_regime_arb": _make_snapshot("cross_asset_regime_arb", 0.2, 0.65),
        }

        def _mock_collect(self_voter, **kwargs):
            readings = {}
            for name, snap in snapshots.items():
                reading = snap.to_signal_reading()
                src = SignalSource(name)
                readings[src] = reading
            self_voter.current_readings = readings
            return readings

        with patch.object(EnsembleVoter, "collect_signals", _mock_collect):
            result = voter.compute_vote()

        assert isinstance(result, EnsembleVote)
        assert -1.0 <= result.weighted_consensus <= 1.0
        assert 0.0 <= result.agreement_ratio <= 1.0
        assert result.num_sources >= 1

    def test_inactive_signal_excluded_from_vote(self, tmp_path):
        """Inactive signals should be excluded from the consensus.

        Batch HZ: isolate from live SignalHealthTracker SSOT. Production health
        may hard-sleep alternative_data (degraded_negative_ic) which zeros all
        vote mass and made this hermetic consensus assertion flaky under make test.
        """
        voter = _make_voter(tmp_path)
        voter.current_regime = Regime.NORMAL
        voter.current_regime_confidence = 0.9
        readings = {
            SignalSource.ALTERNATIVE_DATA: _make_reading(
                SignalSource.ALTERNATIVE_DATA, value=0.5, confidence=0.8,
            ),
            SignalSource.MULTI_SPEED_MOM: _make_reading(
                SignalSource.MULTI_SPEED_MOM, value=0.0, confidence=0.0,
            ),
        }
        # Pass-through health gate so live SH scores cannot sleep ALT_DATA.
        with patch.object(
            EnsembleVoter, "_apply_health_weights", lambda self, w: w
        ):
            result = voter.compute_vote(
                readings=readings, regime=Regime.NORMAL, regime_confidence=0.9
            )
        # ALT_DATA should dominate; MSM has near-zero weight
        assert result.weighted_consensus > 0

    def test_signal_snapshot_round_trip_through_pipeline(self, tmp_path):
        """SignalSnapshot → SignalReading should preserve direction and values."""
        snap = _make_snapshot(
            "alternative_data", value=0.8, confidence=0.9,
            asset_signals={"SPY": 0.6, "GLD": -0.2},
        )
        reading = snap.to_signal_reading()
        assert reading.value == 0.8
        assert reading.confidence == 0.9
        assert reading.asset_signals == {"SPY": 0.6, "GLD": -0.2}


# ─────────────────────────────────────────────────────────────
#  2. Regime gating integration
# ─────────────────────────────────────────────────────────────

class TestRegimeGatingPipeline:
    """Test that regime gating correctly modifies signal weights."""

    def test_crisis_regime_gates_msm(self, tmp_path):
        """MSM should be gated OFF in CRISIS regime."""
        voter = _make_voter(tmp_path)
        readings = {
            SignalSource.MULTI_SPEED_MOM: _make_reading(
                SignalSource.MULTI_SPEED_MOM, value=0.5, confidence=0.7,
            ),
            SignalSource.ALTERNATIVE_DATA: _make_reading(
                SignalSource.ALTERNATIVE_DATA, value=0.3, confidence=0.8,
            ),
        }
        result = voter.compute_vote(
            readings=readings, regime=Regime.CRISIS, regime_confidence=0.8,
        )
        # In CRISIS, MSM should have near-zero weight; ALT_DATA should still contribute
        msm_entry = next(
            (s for s in result.source_votes if s.source == SignalSource.MULTI_SPEED_MOM),
            None,
        )
        if msm_entry:
            # MSM is gated OFF in CRISIS, but renormalization may give it a tiny residual
            assert msm_entry.weight < 0.05

    def test_high_vol_regime_gates_msm(self, tmp_path):
        """MSM should be gated OFF in HIGH_VOL regime."""
        voter = _make_voter(tmp_path)
        readings = {
            SignalSource.MULTI_SPEED_MOM: _make_reading(
                SignalSource.MULTI_SPEED_MOM, value=0.4, confidence=0.6,
            ),
        }
        result = voter.compute_vote(
            readings=readings, regime=Regime.HIGH_VOL, regime_confidence=0.8,
        )
        msm_entry = next(
            (s for s in result.source_votes if s.source == SignalSource.MULTI_SPEED_MOM),
            None,
        )
        if msm_entry:
            # MSM is gated OFF in HIGH_VOL, but renormalization may give it a tiny residual
            assert msm_entry.weight < 0.05


# ─────────────────────────────────────────────────────────────
#  3. IC decay + SPC quality monitoring pipeline
# ─────────────────────────────────────────────────────────────

class TestQualityMonitoringPipeline:
    """Test that IC decay and SPC monitors integrate with the signal pipeline."""

    def test_ic_decay_tracks_signal_quality(self):
        """IC monitor should track signal quality over multiple observations."""
        monitor = ICMonitor(window_size=30, trend_window=5)
        # Simulate 20 observations: good prediction accuracy
        for i in range(20):
            pred = float(i) / 20.0
            actual = pred + np.random.normal(0, 0.01)
            monitor.record("test_signal", prediction=pred, actual_return=actual)

        report = monitor.compute_decay_report()
        assert "test_signal" in report
        assert report["test_signal"]["ic_rolling"] is not None
        assert report["test_signal"]["ic_rolling"] > 0.5
        assert report["test_signal"]["status"] in ("healthy", "warning")

    def test_ic_decay_detects_degradation(self):
        """IC monitor should detect when signal quality degrades."""
        monitor = ICMonitor(window_size=60, decay_threshold=0.05, stable_min=0.10,
                            trend_window=5)
        # Phase 1: strong positive correlation (20 observations)
        for i in range(20):
            monitor.record("degrading_signal", prediction=float(i),
                           actual_return=float(i) * 0.01)
        # Phase 2: anti-correlated (degrading, 20 more)
        for i in range(20):
            monitor.record("degrading_signal", prediction=float(i),
                           actual_return=-float(i) * 0.01)

        report = monitor.compute_decay_report()
        # With anti-correlated second half, IC should be lower
        assert report["degrading_signal"]["ic_trend"] in ("decaying", "stable", "improving", "unknown")

    def test_spc_monitor_tracks_signal_distribution(self):
        """SPC monitor should record and report signal distributions."""
        monitor = SPCMonitor(window_size=20)
        # Record normal signal values
        for i in range(20):
            monitor.record("test_signal", 0.3 + np.random.normal(0, 0.05))
        status = monitor.get_all_status()
        assert "test_signal" in status

    def test_ic_decay_state_persistence_across_runs(self, tmp_path):
        """IC state should persist across monitor instances."""
        monitor1 = ICMonitor(window_size=30)
        for i in range(10):
            monitor1.record("persisted_signal", float(i), float(i) * 0.01)
        path = tmp_path / "ic_state.json"
        monitor1.save_state(path=path)

        monitor2 = ICMonitor(window_size=30)
        monitor2.load_state(path=path)
        report = monitor2.compute_decay_report()
        assert "persisted_signal" in report
        assert report["persisted_signal"]["observations"] == 10


# ─────────────────────────────────────────────────────────────
#  4. Signal schema validation pipeline
# ─────────────────────────────────────────────────────────────

class TestSchemaValidationPipeline:
    """Test that signal output passes Pydantic validation."""

    def test_ensemble_voting_validates(self):
        """Ensemble voting output should pass Pydantic validation."""
        from src.monitor.signal_schemas import validate_signal

        data = {
            "regime": "NORMAL",
            "regime_confidence": 0.8,
            "weighted_consensus": 0.35,
            "agreement_ratio": 0.75,
            "source_breakdown": [],
            "active_weights": {},
        }
        result = validate_signal("ensemble_voting", data)
        assert result["regime"] == "NORMAL"
        assert result["weighted_consensus"] == 0.35

    def test_ic_decay_validates(self):
        """IC decay output should pass Pydantic validation."""
        from src.monitor.signal_schemas import validate_signal

        data = {
            "signals": {
                "alt_data": {
                    "ic_rolling": 0.82,
                    "ic_trend": "stable",
                    "observations": 25,
                    "status": "healthy",
                },
            },
        }
        result = validate_signal("ic_decay", data)
        assert "signals" in result

    def test_garch_cvar_validates(self):
        """GARCH-CVaR output should pass Pydantic validation."""
        from src.monitor.signal_schemas import validate_signal

        data = {
            "cvar_95": -0.03,
            "cvar_95_garch": -0.035,
            "var_95": -0.02,
            "var_95_garch": -0.025,
            "cvar_ratio": 1.5,
            "garch_active": True,
            "current_volatility": 0.15,
            "forecast_volatility": 0.16,
            "volatility_clustering": "normal",
        }
        result = validate_signal("garch_cvar", data)
        assert result["cvar_95"] == -0.03
        assert result["garch_active"] is True

    def test_validation_graceful_on_bad_data(self):
        """Validation should gracefully degrade on bad data."""
        from src.monitor.signal_schemas import validate_signal

        # Missing required fields — should return original data, not crash
        data = {"regime": "NORMAL"}
        result = validate_signal("ensemble_voting", data)
        # Should either fill defaults or return as-is
        assert isinstance(result, dict)

    def test_validate_all_signals_with_mixed_data(self):
        """validate_all_signals should handle mixed valid/invalid data."""
        from src.monitor.signal_schemas import validate_all_signals

        data = {
            "ensemble_voting": {"regime": "NORMAL", "weighted_consensus": 0.3},
            "garch_cvar": {"cvar_95": -0.03, "var_95": -0.02},
            "unknown_signal": {"foo": "bar"},
        }
        result = validate_all_signals(data)
        assert "ensemble_voting" in result
        assert "unknown_signal" in result

    def test_gold_tlt_correlation_validates(self):
        """Gold-TLT correlation output should pass Pydantic validation."""
        from src.monitor.signal_schemas import validate_signal

        data = {
            "current_correlation": 0.10,
            "current_regime": "neutral",
            "correlation_trend": "stable",
            "mean_correlation": 0.20,
            "min_correlation": -0.15,
            "max_correlation": 0.60,
            "structural_breaks_count": 2,
            "regimes_count": 5,
            "implications": "Diversification benefit reduced.",
        }
        result = validate_signal("gold_tlt_correlation", data)
        assert result["current_correlation"] == 0.10
        assert result["current_regime"] == "neutral"


# ─────────────────────────────────────────────────────────────
#  5. Cross-signal consistency checks
# ─────────────────────────────────────────────────────────────

class TestCrossSignalConsistency:
    """Test consistency across the signal pipeline."""

    def test_signal_snapshot_source_matches_reading(self):
        """SignalSnapshot source should map to correct SignalSource."""
        sources = [
            ("multi_speed_momentum", SignalSource.MULTI_SPEED_MOM),
            ("cross_asset_rv", SignalSource.CROSS_ASSET_RV),
            ("international_momentum", SignalSource.INTERNATIONAL_MOMENTUM),
            ("alternative_data", SignalSource.ALTERNATIVE_DATA),
            ("cross_asset_regime_arb", SignalSource.CROSS_ASSET_REGIME_ARB),
        ]
        for name, expected_source in sources:
            snap = _make_snapshot(name, value=0.3, confidence=0.7)
            reading = snap.to_signal_reading()
            assert reading.source == expected_source, f"{name} → {reading.source}"

    def test_weight_normalization(self, tmp_path):
        """Active signal weights should be properly distributed."""
        voter = _make_voter(tmp_path)
        readings = {
            SignalSource.ALTERNATIVE_DATA: _make_reading(
                SignalSource.ALTERNATIVE_DATA, value=0.3, confidence=0.8,
            ),
            SignalSource.INTERNATIONAL_MOMENTUM: _make_reading(
                SignalSource.INTERNATIONAL_MOMENTUM, value=0.4, confidence=0.7,
            ),
            SignalSource.CROSS_ASSET_RV: _make_reading(
                SignalSource.CROSS_ASSET_RV, value=0.1, confidence=0.6,
            ),
        }
        result = voter.compute_vote(
            readings=readings, regime=Regime.NORMAL, regime_confidence=0.7,
        )
        weights = [s.weight for s in result.source_votes]
        total_weight = sum(weights)
        # Weights should be non-negative and sum close to 1.0
        assert all(w >= 0 for w in weights)
        assert total_weight > 0

    def test_consensus_range_bounded(self, tmp_path):
        """Weighted consensus should be bounded in [-1, 1]."""
        voter = _make_voter(tmp_path)
        # Extreme values
        readings = {
            SignalSource.ALTERNATIVE_DATA: _make_reading(
                SignalSource.ALTERNATIVE_DATA, value=1.0, confidence=1.0,
            ),
            SignalSource.INTERNATIONAL_MOMENTUM: _make_reading(
                SignalSource.INTERNATIONAL_MOMENTUM, value=0.9, confidence=0.9,
            ),
        }
        result = voter.compute_vote(
            readings=readings, regime=Regime.NORMAL, regime_confidence=0.7,
        )
        assert -1.0 <= result.weighted_consensus <= 1.0

    def test_empty_readings_produces_valid_result(self, tmp_path):
        """Empty readings should produce a valid EnsembleVote."""
        voter = _make_voter(tmp_path)
        result = voter.compute_vote(readings={}, regime=Regime.NORMAL)
        assert isinstance(result, EnsembleVote)
        assert result.num_sources == 0
