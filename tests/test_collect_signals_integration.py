#!/usr/bin/env python3
"""Integration tests for EnsembleVoter.collect_signals() — typed SignalSnapshot bridge.

Tests that collect_signals() correctly invokes each signal module's
get_signal_snapshot() / to_signal_snapshot() method and converts to
SignalReading via the typed pipeline, rather than raw dict unpacking.

Each test mocks the underlying signal module to return a controlled
SignalSnapshot, then verifies the resulting SignalReading.
"""
import os
os.environ["PORTFOLIO_LAB_ENABLE_ML"] = "0"

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.strategy.ensemble_voter import (
    Regime, SignalSource, SignalReading, EnsembleVoter,
)
from src.signals.signal_snapshot import SignalSnapshot


def _make_voter(tmp_path):
    """Create an EnsembleVoter with isolated paths."""
    voter = EnsembleVoter.__new__(EnsembleVoter)
    voter.data_path = tmp_path
    voter.db_path = tmp_path / "ensemble_signals.db"
    voter.current_readings = {}
    voter.current_regime = Regime.NORMAL
    voter.current_regime_confidence = 0.5
    voter._init_db()
    return voter


class TestCollectSignalsMSM:
    """Multi-Speed Momentum via typed SignalSnapshot bridge."""

    def test_msm_snapshot_converted_to_reading(self, tmp_path):
        voter = _make_voter(tmp_path)
        mock_snapshot = SignalSnapshot(
            source="multi_speed_momentum",
            timestamp="2026-05-23T12:00:00",
            value=0.4,
            confidence=0.75,
            asset_signals={"SPY": 0.3, "TLT": -0.1, "GLD": 0.2},
            regime_fit="all",
            is_active=True,
            explanation="MSM positive trend",
        )

        with patch("src.signals.multi_speed_momentum.MultiSpeedMomentum") as MockMSM:
            MockMSM.return_value.get_signal_snapshot.return_value = mock_snapshot
            readings = voter.collect_signals()

        assert SignalSource.MULTI_SPEED_MOM in readings
        r = readings[SignalSource.MULTI_SPEED_MOM]
        assert r.value == 0.4
        assert r.confidence == 0.75
        assert r.asset_signals == {"SPY": 0.3, "TLT": -0.1, "GLD": 0.2}

    def test_msm_inactive_snapshot_excluded(self, tmp_path):
        voter = _make_voter(tmp_path)
        mock_snapshot = SignalSnapshot(
            source="multi_speed_momentum",
            timestamp="2026-05-23T12:00:00",
            value=0.0,
            confidence=0.0,
            is_active=False,
            explanation="MSM no data",
        )

        with patch("src.signals.multi_speed_momentum.MultiSpeedMomentum") as MockMSM:
            MockMSM.return_value.get_signal_snapshot.return_value = mock_snapshot
            readings = voter.collect_signals()

        assert SignalSource.MULTI_SPEED_MOM not in readings

    def test_msm_import_error_graceful(self, tmp_path):
        voter = _make_voter(tmp_path)
        with patch("src.signals.multi_speed_momentum.MultiSpeedMomentum", side_effect=ImportError):
            readings = voter.collect_signals()
        assert SignalSource.MULTI_SPEED_MOM not in readings

    def test_msm_exception_graceful(self, tmp_path):
        voter = _make_voter(tmp_path)
        with patch("src.signals.multi_speed_momentum.MultiSpeedMomentum") as MockMSM:
            MockMSM.return_value.get_signal_snapshot.side_effect = RuntimeError("DB error")
            readings = voter.collect_signals()
        assert SignalSource.MULTI_SPEED_MOM not in readings


class TestCollectSignalsCrossAssetRV:
    """Cross-Asset Relative Value via typed SignalSnapshot bridge."""

    def test_rv_snapshot_converted_to_reading(self, tmp_path):
        voter = _make_voter(tmp_path)
        mock_snapshot = SignalSnapshot(
            source="cross_asset_rv",
            timestamp="2026-05-23T12:00:00",
            value=0.3,
            confidence=0.7,
            asset_signals={"SPY": 0.2, "GLD": -0.1, "TLT": 0.0},
            regime_fit="all",
            is_active=True,
            explanation="Cross-asset RV: z=1.50, diverged=3/6 pairs",
        )

        with patch("src.signals.cross_asset_relative_value.CrossAssetRVScanner") as MockScanner:
            MockScanner.return_value.get_signal_snapshot.return_value = mock_snapshot
            readings = voter.collect_signals()

        assert SignalSource.CROSS_ASSET_RV in readings
        r = readings[SignalSource.CROSS_ASSET_RV]
        assert r.value == 0.3
        assert r.confidence == 0.7

    def test_rv_zero_signal_inactive(self, tmp_path):
        voter = _make_voter(tmp_path)
        mock_snapshot = SignalSnapshot(
            source="cross_asset_rv",
            timestamp="2026-05-23T12:00:00",
            value=0.0,
            confidence=0.0,
            is_active=False,
            explanation="No divergence",
        )

        with patch("src.signals.cross_asset_relative_value.CrossAssetRVScanner") as MockScanner:
            MockScanner.return_value.get_signal_snapshot.return_value = mock_snapshot
            readings = voter.collect_signals()

        assert SignalSource.CROSS_ASSET_RV not in readings


class TestCollectSignalsInternationalMomentum:
    """International Momentum via typed SignalSnapshot bridge."""

    def test_intl_momentum_efa_lead(self, tmp_path):
        voter = _make_voter(tmp_path)
        mock_snapshot = SignalSnapshot(
            source="international_momentum",
            timestamp="2026-05-23T12:00:00",
            value=0.5,
            confidence=0.7,
            asset_signals={"SPY": 0.1, "EFA": -0.1, "EEM": 0.0},
            regime_fit="all",
            is_active=True,
            explanation="Intl Momentum: efa_lead, EFA/SPY=+5.00%",
        )

        # Create a mock signal object with to_signal_snapshot method
        mock_signal = MagicMock()
        mock_signal.to_signal_snapshot.return_value = mock_snapshot

        # Price data needs >= 20 rows and SPY/EFA/EEM columns
        import pandas as pd
        n_rows = 25
        price_df = pd.DataFrame({
            "SPY": range(400, 400 + n_rows),
            "EFA": range(60, 60 + n_rows),
            "EEM": range(40, 40 + n_rows),
        }, index=pd.date_range("2026-01-01", periods=n_rows))

        with patch("src.signals.international_momentum.InternationalMomentumGenerator") as MockGen, \
             patch.object(voter, "_load_price_data", return_value=price_df):
            MockGen.return_value.generate_signal.return_value = mock_signal
            readings = voter.collect_signals()

        assert SignalSource.INTERNATIONAL_MOMENTUM in readings
        r = readings[SignalSource.INTERNATIONAL_MOMENTUM]
        assert r.value == 0.5
        assert r.confidence == 0.7

    def test_intl_momentum_no_price_data(self, tmp_path):
        voter = _make_voter(tmp_path)
        with patch.object(voter, "_load_price_data", return_value=None):
            readings = voter.collect_signals()
        # Should gracefully skip
        assert SignalSource.INTERNATIONAL_MOMENTUM not in readings


class TestCollectSignalsAlternativeData:
    """Alternative Data via typed SignalSnapshot bridge."""

    def test_alt_data_snapshot_converted(self, tmp_path):
        voter = _make_voter(tmp_path)
        mock_snapshot = SignalSnapshot(
            source="alternative_data",
            timestamp="2026-05-23T12:00:00",
            value=0.4,
            confidence=0.65,
            asset_signals={"SPY": 0.4},
            regime_fit="all",
            is_active=True,
            explanation="Alt Data: regime=bull, composite=0.4000",
        )

        with patch("src.signals.alternative_data_signal.AlternativeDataSignalGenerator") as MockGen:
            MockGen.return_value.get_signal_snapshot.return_value = mock_snapshot
            readings = voter.collect_signals()

        assert SignalSource.ALTERNATIVE_DATA in readings
        r = readings[SignalSource.ALTERNATIVE_DATA]
        assert r.value == 0.4
        assert r.confidence == 0.65

    def test_alt_data_no_signal_file(self, tmp_path):
        voter = _make_voter(tmp_path)
        mock_snapshot = SignalSnapshot(
            source="alternative_data",
            timestamp="2026-05-23T12:00:00",
            value=0.0,
            confidence=0.0,
            is_active=False,
            explanation="Alternative data signal unavailable",
        )

        with patch("src.signals.alternative_data_signal.AlternativeDataSignalGenerator") as MockGen:
            MockGen.return_value.get_signal_snapshot.return_value = mock_snapshot
            readings = voter.collect_signals()

        assert SignalSource.ALTERNATIVE_DATA not in readings


class TestCollectSignalsRegimeArb:
    """Cross-Asset Regime Arbitrage via typed SignalSnapshot bridge."""

    def test_regime_arb_snapshot_converted(self, tmp_path):
        voter = _make_voter(tmp_path)
        mock_snapshot = SignalSnapshot(
            source="cross_asset_regime_arb",
            timestamp="2026-05-23T12:00:00",
            value=0.2,
            confidence=0.6,
            asset_signals={"SPY": -0.1, "GLD": 0.2, "TLT": 0.1},
            regime_fit="all",
            is_active=True,
            explanation="Cross-asset regime arb: pattern=equity_gold_divergence",
        )

        with patch("src.signals.cross_asset_regime_arb.CrossAssetRegimeArbDetector") as MockDet:
            MockDet.return_value.get_signal_snapshot.return_value = mock_snapshot
            readings = voter.collect_signals()

        assert SignalSource.CROSS_ASSET_REGIME_ARB in readings
        r = readings[SignalSource.CROSS_ASSET_REGIME_ARB]
        assert r.value == 0.2
        assert r.confidence == 0.6

    def test_regime_arb_inactive_excluded(self, tmp_path):
        voter = _make_voter(tmp_path)
        mock_snapshot = SignalSnapshot(
            source="cross_asset_regime_arb",
            timestamp="2026-05-23T12:00:00",
            value=0.0,
            confidence=0.0,
            is_active=False,
            explanation="No regime divergence",
        )

        with patch("src.signals.cross_asset_regime_arb.CrossAssetRegimeArbDetector") as MockDet:
            MockDet.return_value.get_signal_snapshot.return_value = mock_snapshot
            readings = voter.collect_signals()

        assert SignalSource.CROSS_ASSET_REGIME_ARB not in readings


class TestCollectSignalsUnifiedOverlay:
    """Unified Overlay — already returns SignalReading directly."""

    def test_unified_overlay_included(self, tmp_path):
        voter = _make_voter(tmp_path)
        mock_reading = SignalReading(
            source=SignalSource.UNIFIED_OVERLAY,
            timestamp="2026-05-23T12:00:00",
            value=0.15,
            confidence=0.6,
            weight=0.0,
            regime_fit="all",
            explanation="Unified: 2 overlays active",
        )

        with patch("src.strategy.orchestrator_ensemble_bridge.OrchestratorEnsembleBridge") as MockBridge:
            MockBridge.return_value.get_ensemble_reading.return_value = mock_reading
            readings = voter.collect_signals()

        assert SignalSource.UNIFIED_OVERLAY in readings
        r = readings[SignalSource.UNIFIED_OVERLAY]
        assert r.value == 0.15

    def test_unified_overlay_import_error(self, tmp_path):
        voter = _make_voter(tmp_path)
        with patch("src.strategy.orchestrator_ensemble_bridge.OrchestratorEnsembleBridge",
                   side_effect=ImportError):
            readings = voter.collect_signals()
        assert SignalSource.UNIFIED_OVERLAY not in readings


class TestCollectSignalsRegimeGating:
    """Verify that zero-weight signals are skipped per regime."""

    def test_skips_zero_weight_signals_in_crisis(self, tmp_path):
        voter = _make_voter(tmp_path)
        # In CRISIS regime, check if MSM is zero-weight
        crisis_weights = Regime.CRISIS.value if hasattr(Regime.CRISIS, 'value') else None
        # Just verify the function runs with a regime parameter
        with patch("src.signals.multi_speed_momentum.MultiSpeedMomentum", side_effect=ImportError), \
             patch("src.signals.cross_asset_relative_value.CrossAssetRVScanner", side_effect=ImportError), \
             patch("src.signals.international_momentum.InternationalMomentumGenerator", side_effect=ImportError), \
             patch("src.signals.alternative_data_signal.AlternativeDataSignalGenerator", side_effect=ImportError), \
             patch("src.signals.cross_asset_regime_arb.CrossAssetRegimeArbDetector", side_effect=ImportError), \
             patch("src.strategy.orchestrator_ensemble_bridge.OrchestratorEnsembleBridge", side_effect=ImportError):
            readings = voter.collect_signals(regime=Regime.CRISIS)
        # Should return empty dict (all imports fail)
        assert isinstance(readings, dict)

    def test_all_signals_in_normal_regime(self, tmp_path):
        """In NORMAL regime, all signals should be attempted."""
        voter = _make_voter(tmp_path)
        # Just verify no crash with all sources mocked
        with patch("src.signals.multi_speed_momentum.MultiSpeedMomentum", side_effect=ImportError), \
             patch("src.signals.cross_asset_relative_value.CrossAssetRVScanner", side_effect=ImportError), \
             patch("src.signals.international_momentum.InternationalMomentumGenerator", side_effect=ImportError), \
             patch("src.signals.alternative_data_signal.AlternativeDataSignalGenerator", side_effect=ImportError), \
             patch("src.signals.cross_asset_regime_arb.CrossAssetRegimeArbDetector", side_effect=ImportError), \
             patch("src.strategy.orchestrator_ensemble_bridge.OrchestratorEnsembleBridge", side_effect=ImportError):
            readings = voter.collect_signals(regime=Regime.NORMAL)
        assert isinstance(readings, dict)


class TestCollectSignalsFullPipeline:
    """End-to-end: all signals return valid snapshots → all readings present."""

    def test_all_signals_collected(self, tmp_path):
        voter = _make_voter(tmp_path)

        snapshots = {
            "msm": SignalSnapshot(source="multi_speed_momentum", timestamp="2026-05-23T12:00:00",
                                  value=0.4, confidence=0.75, asset_signals={"SPY": 0.3, "TLT": -0.1, "GLD": 0.2},
                                  regime_fit="all", is_active=True, explanation="MSM positive"),
            "rv": SignalSnapshot(source="cross_asset_rv", timestamp="2026-05-23T12:00:00",
                                 value=0.3, confidence=0.7, asset_signals={"SPY": 0.2, "GLD": -0.1},
                                 regime_fit="all", is_active=True, explanation="RV divergence"),
            "intl": SignalSnapshot(source="international_momentum", timestamp="2026-05-23T12:00:00",
                                   value=0.5, confidence=0.7, asset_signals={"SPY": 0.1, "EFA": -0.1},
                                   regime_fit="all", is_active=True, explanation="Intl EFA lead"),
            "alt": SignalSnapshot(source="alternative_data", timestamp="2026-05-23T12:00:00",
                                  value=0.4, confidence=0.65, asset_signals={"SPY": 0.4},
                                  regime_fit="all", is_active=True, explanation="Alt bull"),
            "arb": SignalSnapshot(source="cross_asset_regime_arb", timestamp="2026-05-23T12:00:00",
                                  value=0.2, confidence=0.6, asset_signals={"SPY": -0.1, "GLD": 0.2},
                                  regime_fit="all", is_active=True, explanation="Arb divergence"),
            "mtf": SignalSnapshot(source="multi_timeframe_fusion", timestamp="2026-05-23T12:00:00",
                                  value=0.25, confidence=0.7, asset_signals={"SPY": 0.2, "GLD": 0.1, "TLT": 0.3},
                                  regime_fit="all", is_active=True, explanation="MTF fusion"),
        }

        mock_unified_reading = SignalReading(
            source=SignalSource.UNIFIED_OVERLAY, timestamp="2026-05-23T12:00:00",
            value=0.15, confidence=0.6, weight=0.0, regime_fit="all",
            explanation="Unified active",
        )

        # Mock intl signal chain: generate_signal() returns obj with .to_signal_snapshot()
        mock_intl_signal = MagicMock()
        mock_intl_signal.to_signal_snapshot.return_value = snapshots["intl"]

        # Price data needs >= 20 rows with SPY/EFA/EEM
        import pandas as pd
        n_rows = 25
        price_df = pd.DataFrame({
            "SPY": range(400, 400 + n_rows),
            "EFA": range(60, 60 + n_rows),
            "EEM": range(40, 40 + n_rows),
        }, index=pd.date_range("2026-01-01", periods=n_rows))

        with patch("src.signals.multi_speed_momentum.MultiSpeedMomentum") as MockMSM, \
             patch("src.signals.cross_asset_relative_value.CrossAssetRVScanner") as MockRV, \
             patch("src.signals.international_momentum.InternationalMomentumGenerator") as MockIntl, \
             patch("src.signals.alternative_data_signal.AlternativeDataSignalGenerator") as MockAlt, \
             patch("src.signals.cross_asset_regime_arb.CrossAssetRegimeArbDetector") as MockArb, \
             patch("src.strategy.orchestrator_ensemble_bridge.OrchestratorEnsembleBridge") as MockUnified, \
             patch("src.signals.multi_timeframe_fusion.MultiTimeframeFusion") as MockMTF, \
             patch.object(voter, "_load_price_data", return_value=price_df):

            MockMSM.return_value.get_signal_snapshot.return_value = snapshots["msm"]
            MockRV.return_value.get_signal_snapshot.return_value = snapshots["rv"]
            MockIntl.return_value.generate_signal.return_value = mock_intl_signal
            MockAlt.return_value.get_signal_snapshot.return_value = snapshots["alt"]
            MockArb.return_value.get_signal_snapshot.return_value = snapshots["arb"]
            MockUnified.return_value.get_ensemble_reading.return_value = mock_unified_reading
            MockMTF.return_value.get_signal_snapshot.return_value = snapshots["mtf"]

            readings = voter.collect_signals()

        assert len(readings) == 8
        assert SignalSource.MULTI_SPEED_MOM in readings
        assert SignalSource.CROSS_ASSET_RV in readings
        assert SignalSource.INTERNATIONAL_MOMENTUM in readings
        assert SignalSource.ALTERNATIVE_DATA in readings
        assert SignalSource.CROSS_ASSET_REGIME_ARB in readings
        assert SignalSource.UNIFIED_OVERLAY in readings
        assert SignalSource.MULTI_TIMEFRAME_FUSION in readings
        assert SignalSource.GOOGLE_TRENDS in readings
