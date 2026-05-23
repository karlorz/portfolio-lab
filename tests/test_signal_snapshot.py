#!/usr/bin/env python3
"""Tests for SignalSnapshot — canonical signal output type."""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.signals.signal_snapshot import SignalSnapshot


class TestSignalSnapshotCreation:
    def test_basic_creation(self):
        snap = SignalSnapshot(
            source="multi_speed_momentum",
            timestamp="2026-05-23T12:00:00",
            value=0.5,
            confidence=0.8,
        )
        assert snap.source == "multi_speed_momentum"
        assert snap.value == 0.5
        assert snap.confidence == 0.8
        assert snap.is_active is True
        assert snap.asset_signals == {}
        assert snap.regime_fit == "normal"
        assert snap.explanation == ""
        assert snap.metadata == {}

    def test_full_creation(self):
        snap = SignalSnapshot(
            source="cross_asset_rv",
            timestamp="2026-05-23T12:00:00",
            value=-0.3,
            confidence=0.6,
            asset_signals={"SPY": -0.2, "GLD": 0.1, "TLT": 0.3},
            regime_fit="crisis",
            is_active=True,
            explanation="Equity weakness, bond strength",
            metadata={"raw_score": -0.45, "lookback": 63},
        )
        assert snap.value == -0.3
        assert snap.asset_signals["SPY"] == -0.2
        assert snap.metadata["raw_score"] == -0.45

    def test_bearish_signal(self):
        snap = SignalSnapshot(source="test", timestamp="now", value=-1.0, confidence=0.9)
        assert snap.value == -1.0

    def test_bullish_signal(self):
        snap = SignalSnapshot(source="test", timestamp="now", value=1.0, confidence=0.9)
        assert snap.value == 1.0

    def test_inactive_signal(self):
        snap = SignalSnapshot(source="test", timestamp="now", value=0.0, confidence=0.0, is_active=False)
        assert snap.is_active is False


class TestToSignalReading:
    def test_converts_to_signal_reading(self):
        snap = SignalSnapshot(
            source="multi_speed_momentum",
            timestamp="2026-05-23T12:00:00",
            value=0.5,
            confidence=0.8,
            asset_signals={"SPY": 0.3, "GLD": -0.1},
            regime_fit="low_vol",
            explanation="Strong momentum",
        )
        reading = snap.to_signal_reading()
        from src.strategy.ensemble_voter import SignalSource
        assert reading.source == SignalSource.MULTI_SPEED_MOM
        assert reading.value == 0.5
        assert reading.confidence == 0.8
        assert reading.weight == 0.0  # Set later
        assert reading.regime_fit == "low_vol"
        assert reading.asset_signals == {"SPY": 0.3, "GLD": -0.1}
        assert reading.explanation == "Strong momentum"

    def test_source_name_resolution_by_value(self):
        snap = SignalSnapshot(
            source="cross_asset_rv",
            timestamp="now",
            value=0.1,
            confidence=0.5,
        )
        reading = snap.to_signal_reading()
        from src.strategy.ensemble_voter import SignalSource
        assert reading.source == SignalSource.CROSS_ASSET_RV

    def test_unknown_source_raises(self):
        snap = SignalSnapshot(
            source="nonexistent_signal",
            timestamp="now",
            value=0.0,
            confidence=0.0,
        )
        with pytest.raises(ValueError, match="No SignalSource enum"):
            snap.to_signal_reading()

    def test_all_signal_sources_resolvable(self):
        """Every SignalSource enum value should be resolvable."""
        from src.strategy.ensemble_voter import SignalSource
        for member in SignalSource:
            snap = SignalSnapshot(
                source=member.value,
                timestamp="now",
                value=0.0,
                confidence=0.5,
            )
            reading = snap.to_signal_reading()
            assert reading.source == member

    def test_empty_asset_signals_becomes_none(self):
        snap = SignalSnapshot(
            source="multi_speed_momentum",
            timestamp="now",
            value=0.5,
            confidence=0.8,
        )
        reading = snap.to_signal_reading()
        assert reading.asset_signals is None


class TestFromDict:
    def test_from_dict_with_standard_keys(self):
        data = {
            "source": "cross_asset_regime_arb",
            "timestamp": "2026-05-23T12:00:00",
            "signal_value": 0.3,
            "confidence": 0.7,
            "asset_signals": {"SPY": 0.2, "GLD": -0.1},
            "is_active": True,
            "explanation": "Regime divergence",
        }
        snap = SignalSnapshot.from_dict(data)
        assert snap.source == "cross_asset_regime_arb"
        assert snap.value == 0.3
        assert snap.confidence == 0.7
        assert snap.asset_signals == {"SPY": 0.2, "GLD": -0.1}
        assert snap.is_active is True

    def test_from_dict_with_alternative_keys(self):
        """Some modules use 'value' instead of 'signal_value', 'active' instead of 'is_active'."""
        data = {
            "signal_name": "test_signal",
            "value": -0.5,
            "overall_conviction": 0.6,
            "active": True,
        }
        snap = SignalSnapshot.from_dict(data)
        assert snap.source == "test_signal"
        assert snap.value == -0.5
        assert snap.confidence == 0.6
        assert snap.is_active is True

    def test_from_dict_preserves_extra_keys_as_metadata(self):
        data = {
            "source": "test",
            "signal_value": 0.1,
            "confidence": 0.5,
            "pattern": "contango",
            "lookback_days": 63,
        }
        snap = SignalSnapshot.from_dict(data)
        assert snap.metadata["pattern"] == "contango"
        assert snap.metadata["lookback_days"] == 63

    def test_from_dict_with_composite_key(self):
        """Crypto momentum uses 'composite' for signal value."""
        data = {
            "source": "crypto_momentum",
            "composite": 0.8,
            "confidence": 0.9,
        }
        snap = SignalSnapshot.from_dict(data)
        assert snap.value == 0.8

    def test_from_dict_defaults(self):
        data = {}
        snap = SignalSnapshot.from_dict(data)
        assert snap.source == "unknown"
        assert snap.value == 0.0
        assert snap.confidence == 0.5
        assert snap.is_active is True
        assert snap.timestamp  # Should be auto-generated


class TestRoundTrip:
    def test_dict_round_trip_preserves_core_fields(self):
        """from_dict -> to_signal_reading should preserve core signal data."""
        data = {
            "source": "multi_speed_momentum",
            "timestamp": "2026-05-23T12:00:00",
            "signal_value": 0.4,
            "confidence": 0.7,
            "asset_signals": {"SPY": 0.3, "GLD": 0.1, "TLT": -0.2},
            "explanation": "Positive momentum across speeds",
        }
        snap = SignalSnapshot.from_dict(data)
        reading = snap.to_signal_reading()
        assert reading.value == 0.4
        assert reading.confidence == 0.7
        assert reading.asset_signals == {"SPY": 0.3, "GLD": 0.1, "TLT": -0.2}
        assert reading.explanation == "Positive momentum across speeds"

    def test_multiple_signals_to_vote(self):
        """Simulate the full pipeline: 3 signals -> SignalReadings -> compute_vote."""
        from src.strategy.ensemble_voter import SignalSource, SignalReading, EnsembleVoter, Regime

        snapshots = [
            SignalSnapshot(
                source="multi_speed_momentum",
                timestamp="2026-05-23T12:00:00",
                value=0.5, confidence=0.8,
                asset_signals={"SPY": 0.4, "GLD": 0.1, "TLT": -0.3},
                regime_fit="normal",
            ),
            SignalSnapshot(
                source="cross_asset_rv",
                timestamp="2026-05-23T12:00:00",
                value=-0.2, confidence=0.6,
                asset_signals={"SPY": -0.1, "GLD": 0.2, "TLT": 0.1},
                regime_fit="normal",
            ),
            SignalSnapshot(
                source="international_momentum",
                timestamp="2026-05-23T12:00:00",
                value=0.3, confidence=0.7,
                asset_signals={"SPY": 0.2, "GLD": 0.1, "TLT": -0.1},
                regime_fit="low_vol",
            ),
        ]

        readings = {s.to_signal_reading().source: s.to_signal_reading() for s in snapshots}
        voter = EnsembleVoter()
        vote = voter.compute_vote(readings, regime=Regime.NORMAL)
        assert vote.weighted_consensus != 0  # Should have a non-zero consensus
        assert "SPY" in vote.source_votes or len(vote.source_votes) > 0


class TestModuleIntegration:
    """Test that signal modules can produce SignalSnapshot via to_signal_snapshot()."""

    def test_crypto_momentum_to_signal_snapshot(self):
        """CryptoCompositeSignal.to_signal_snapshot() produces valid snapshot."""
        from src.signals.crypto_momentum import CryptoCompositeSignal, CryptoAssetSignal

        btc = CryptoAssetSignal(
            symbol="BTC-USD", price=68000, momentum_6m=0.5, momentum_3m=0.3,
            momentum_1m=0.1, vol_30d=0.6, vol_90d=0.5, vol_regime="normal",
            signal_state="long", target_weight=0.60, confidence=0.8,
        )
        eth = CryptoAssetSignal(
            symbol="ETH-USD", price=3500, momentum_6m=0.4, momentum_3m=0.2,
            momentum_1m=0.05, vol_30d=0.7, vol_90d=0.55, vol_regime="normal",
            signal_state="long", target_weight=0.40, confidence=0.7,
        )
        composite = CryptoCompositeSignal(
            timestamp="2026-05-23T12:00:00",
            btc_signal=btc, eth_signal=eth,
            composite_weight=0.05, vol_scale_factor=1.0,
            funding_source="gld", gld_reduction=0.05,
            signal_state="long", confidence=0.75,
            is_valid=True, reason="Positive momentum, normal vol regime",
        )
        snap = composite.to_signal_snapshot()
        assert snap.source == "crypto_momentum"
        assert snap.value == 1.0  # 0.05 / 0.05 = 1.0
        assert snap.confidence == 0.75
        assert snap.is_active is True
        assert snap.asset_signals["GLD"] == -0.05

    def test_crypto_momentum_invalid_snapshot(self):
        """Invalid crypto signal produces zero-value snapshot."""
        from src.signals.crypto_momentum import CryptoCompositeSignal, CryptoAssetSignal

        btc = CryptoAssetSignal(
            symbol="BTC-USD", price=68000, momentum_6m=-0.2, momentum_3m=-0.1,
            momentum_1m=-0.05, vol_30d=1.5, vol_90d=1.2, vol_regime="extreme",
            signal_state="flat", target_weight=0.0, confidence=0.2,
        )
        eth = CryptoAssetSignal(
            symbol="ETH-USD", price=3500, momentum_6m=-0.3, momentum_3m=-0.15,
            momentum_1m=-0.1, vol_30d=1.8, vol_90d=1.4, vol_regime="extreme",
            signal_state="flat", target_weight=0.0, confidence=0.1,
        )
        composite = CryptoCompositeSignal(
            timestamp="2026-05-23T12:00:00",
            btc_signal=btc, eth_signal=eth,
            composite_weight=0.0, vol_scale_factor=0.0,
            funding_source="gld", gld_reduction=0.0,
            signal_state="flat", confidence=0.15,
            is_valid=False, reason="Extreme vol, negative momentum",
        )
        snap = composite.to_signal_snapshot()
        assert snap.value == 0.0  # Invalid -> neutral
        assert snap.is_active is False

    def test_crypto_snapshot_to_reading_round_trip(self):
        """Crypto snapshot produces valid data — not in SignalSource enum yet."""
        from src.signals.crypto_momentum import CryptoCompositeSignal, CryptoAssetSignal

        btc = CryptoAssetSignal(
            symbol="BTC-USD", price=68000, momentum_6m=0.3, momentum_3m=0.2,
            momentum_1m=0.1, vol_30d=0.5, vol_90d=0.45, vol_regime="normal",
            signal_state="long", target_weight=0.60, confidence=0.7,
        )
        eth = CryptoAssetSignal(
            symbol="ETH-USD", price=3500, momentum_6m=0.2, momentum_3m=0.1,
            momentum_1m=0.05, vol_30d=0.6, vol_90d=0.5, vol_regime="normal",
            signal_state="long", target_weight=0.40, confidence=0.6,
        )
        composite = CryptoCompositeSignal(
            timestamp="2026-05-23T12:00:00",
            btc_signal=btc, eth_signal=eth,
            composite_weight=0.03, vol_scale_factor=0.8,
            funding_source="gld", gld_reduction=0.03,
            signal_state="long", confidence=0.65,
            is_valid=True, reason="Moderate momentum",
        )
        snap = composite.to_signal_snapshot()
        # Crypto is not in the SignalSource enum — to_signal_reading raises
        with pytest.raises(ValueError, match="No SignalSource enum"):
            snap.to_signal_reading()
        # But the snapshot itself is valid for dashboard/diagnostics
        assert snap.value > 0
        assert snap.confidence == 0.65

    def test_cross_asset_regime_arb_to_snapshot(self):
        """CrossAssetRegimeArbDetector.get_signal_snapshot() returns valid snapshot."""
        from src.signals.cross_asset_regime_arb import CrossAssetRegimeArbDetector

        # Mock the DB-dependent scan method
        arb = CrossAssetRegimeArbDetector.__new__(CrossAssetRegimeArbDetector)

        # Mock scan to return None (no data scenario)
        arb.scan = MagicMock(return_value=None)
        snap = arb.get_signal_snapshot()
        assert snap.source == "cross_asset_regime_arb"
        assert snap.value == 0.0
        assert snap.is_active is False


class TestNewModuleSnapshots:
    """Test recently added get_signal_snapshot() methods."""

    def test_cross_asset_rv_snapshot_from_dict(self):
        """CrossAssetRVScanner.get_signal_snapshot() uses from_dict internally."""
        from src.signals.cross_asset_relative_value import CrossAssetRVScanner
        scanner = CrossAssetRVScanner.__new__(CrossAssetRVScanner)
        # Mock get_ensemble_signal to return a standard dict
        scanner.get_ensemble_signal = MagicMock(return_value={
            "signal_value": 0.3,
            "confidence": 0.7,
            "timestamp": "2026-05-23T12:00:00",
            "asset_signals": {"SPY": 0.2, "GLD": -0.1, "TLT": 0.0},
            "avg_z_score": 1.5,
            "num_diverged": 3,
            "total_pairs": 6,
        })
        snap = scanner.get_signal_snapshot()
        assert snap.source == "cross_asset_rv"
        assert snap.value == 0.3
        assert snap.confidence == 0.7
        assert snap.is_active is True
        assert "Cross-asset RV" in snap.explanation

    def test_cross_asset_rv_snapshot_zero_signal(self):
        """Zero signal_value → is_active=False."""
        from src.signals.cross_asset_relative_value import CrossAssetRVScanner
        scanner = CrossAssetRVScanner.__new__(CrossAssetRVScanner)
        scanner.get_ensemble_signal = MagicMock(return_value={
            "signal_value": 0.0,
            "confidence": 0.0,
            "timestamp": "2026-05-23T12:00:00",
            "asset_signals": {},
        })
        snap = scanner.get_signal_snapshot()
        assert snap.is_active is False

    def test_alternative_data_snapshot_no_file(self):
        """AlternativeDataSignalGenerator.get_signal_snapshot() with no data file."""
        from src.signals.alternative_data_signal import AlternativeDataSignalGenerator
        gen = AlternativeDataSignalGenerator.__new__(AlternativeDataSignalGenerator)
        gen.load_latest_signal = MagicMock(return_value=None)
        snap = gen.get_signal_snapshot()
        assert snap.source == "alternative_data"
        assert snap.value == 0.0
        assert snap.is_active is False

    def test_international_momentum_snapshot(self):
        """InternationalMomentumSignal.to_signal_snapshot() converts correctly."""
        from src.signals.international_momentum import InternationalMomentumSignal
        signal = InternationalMomentumSignal(
            timestamp="2026-05-23T12:00:00",
            signal_type="efa_lead",
            confidence=0.7,
            confidence_level="medium",
            efa_momentum_6m=15.0,
            eem_momentum_6m=5.0,
            spy_momentum_6m=10.0,
            efa_vs_spy=5.0,
            eem_vs_spy=-5.0,
            spy_shift=0.1,
            efa_shift=-0.1,
            eem_shift=0.0,
            max_allocation_efa=0.05,
            max_allocation_eem=0.03,
            holding_period_days=30,
            data_fresh=True,
            vix_filter_active=False,
            correlation_override=False,
        )
        snap = signal.to_signal_snapshot()
        assert snap.source == "international_momentum"
        assert snap.value == pytest.approx(0.5)  # clip(5.0/10.0, -0.5, 0.5)
        assert snap.confidence == 0.7
        assert snap.is_active is True
        assert "SPY" in snap.asset_signals

    def test_international_momentum_snapshot_neutral(self):
        """Neutral international momentum → zero value."""
        from src.signals.international_momentum import InternationalMomentumSignal
        signal = InternationalMomentumSignal(
            timestamp="2026-05-23T12:00:00",
            signal_type="neutral",
            confidence=0.3,
            confidence_level="low",
            efa_momentum_6m=0.0,
            eem_momentum_6m=0.0,
            spy_momentum_6m=0.0,
            efa_vs_spy=0.0,
            eem_vs_spy=0.0,
            spy_shift=0.0,
            efa_shift=0.0,
            eem_shift=0.0,
            max_allocation_efa=0.05,
            max_allocation_eem=0.03,
            holding_period_days=30,
            data_fresh=True,
            vix_filter_active=True,
            correlation_override=False,
        )
        snap = signal.to_signal_snapshot()
        assert snap.value == 0.0
        assert snap.is_active is False  # vix_filter_active=True

    def test_cross_asset_rv_snapshot_to_reading(self):
        """Cross-asset RV snapshot → SignalReading round-trip."""
        from src.signals.cross_asset_relative_value import CrossAssetRVScanner
        scanner = CrossAssetRVScanner.__new__(CrossAssetRVScanner)
        scanner.get_ensemble_signal = MagicMock(return_value={
            "signal_value": 0.3,
            "confidence": 0.7,
            "timestamp": "2026-05-23T12:00:00",
            "asset_signals": {"SPY": 0.2, "GLD": -0.1, "TLT": 0.0},
        })
        snap = scanner.get_signal_snapshot()
        reading = snap.to_signal_reading()
        from src.strategy.ensemble_voter import SignalSource
        assert reading.source == SignalSource.CROSS_ASSET_RV
        assert reading.value == 0.3

    def test_multi_speed_momentum_get_signal_snapshot(self):
        """MultiSpeedMomentum.get_signal_snapshot() produces valid snapshot."""
        from src.signals.multi_speed_momentum import MultiSpeedMomentum
        with patch.object(MultiSpeedMomentum, "get_signal_for_ticker") as mock_ticker:
            mock_ticker.side_effect = lambda t, d=None: (
                {"value": 0.3, "confidence": 0.8} if t == "SPY"
                else {"value": -0.1, "confidence": 0.7} if t == "TLT"
                else {"value": 0.2, "confidence": 0.75}
            )
            msm = MultiSpeedMomentum()
            snap = msm.get_signal_snapshot(tickers=["SPY", "TLT", "GLD"])

        assert snap.source == "multi_speed_momentum"
        assert snap.is_active is True
        assert "SPY" in snap.asset_signals
        assert snap.asset_signals["SPY"] == 0.3
        assert snap.asset_signals["TLT"] == -0.1
        assert snap.confidence == pytest.approx(0.75, abs=0.05)

    def test_multi_speed_momentum_snapshot_no_data(self):
        """MSM snapshot with no data produces inactive snapshot."""
        from src.signals.multi_speed_momentum import MultiSpeedMomentum
        with patch.object(MultiSpeedMomentum, "get_signal_for_ticker", return_value=None):
            msm = MultiSpeedMomentum()
            snap = msm.get_signal_snapshot(tickers=["SPY", "TLT", "GLD"])

        assert snap.is_active is False
        assert snap.value == 0.0

    def test_multi_speed_momentum_snapshot_to_reading(self):
        """MSM snapshot converts to SignalReading via typed bridge."""
        from src.signals.multi_speed_momentum import MultiSpeedMomentum
        from src.strategy.ensemble_voter import SignalSource
        with patch.object(MultiSpeedMomentum, "get_signal_for_ticker") as mock_ticker:
            mock_ticker.return_value = {"value": 0.5, "confidence": 0.9}
            msm = MultiSpeedMomentum()
            snap = msm.get_signal_snapshot(tickers=["SPY"])
            reading = snap.to_signal_reading()

        assert reading.source == SignalSource.MULTI_SPEED_MOM
        assert reading.value == 0.5


class TestUnifiedOrchestratorSignals:
    """Test to_signal_snapshot() for signals consumed by the unified orchestrator."""

    def test_bond_duration_signal_snapshot(self):
        """BondDurationSignal.to_signal_snapshot() converts correctly."""
        from src.signals.bond_duration_signal import BondDurationSignal
        signal = BondDurationSignal(
            timestamp="2026-05-23T12:00:00",
            yield_10y=4.5, yield_2y=4.3, spread_10y2y=0.2,
            curve_regime="steep", real_rate=1.8, real_rate_regime="attractive",
            rate_6m_ago=4.0, rate_change_6m=0.5, rate_direction="rising",
            tlt_weight=0.15, ief_weight=0.10, shy_weight=0.05,
            effective_duration=12.5, position="long",
            confidence=0.7, is_valid=True, reason="Steep curve, attractive real rates",
        )
        snap = signal.to_signal_snapshot()
        assert snap.source == "bond_duration_signal"
        assert snap.value == 0.5  # "long" maps to 0.5
        assert snap.confidence == 0.7
        assert snap.is_active is True
        assert snap.asset_signals["TLT"] == 0.15
        assert "steep" in snap.explanation

    def test_bond_duration_signal_invalid(self):
        """Invalid bond duration signal produces inactive snapshot."""
        from src.signals.bond_duration_signal import BondDurationSignal
        signal = BondDurationSignal(
            timestamp="2026-05-23T12:00:00",
            yield_10y=4.0, yield_2y=4.0, spread_10y2y=0.0,
            curve_regime="flat", real_rate=0.5, real_rate_regime="neutral",
            rate_6m_ago=4.0, rate_change_6m=0.0, rate_direction="stable",
            tlt_weight=0.10, ief_weight=0.10, shy_weight=0.10,
            effective_duration=5.0, position="blend",
            confidence=0.3, is_valid=False, reason="Flat curve, no signal",
        )
        snap = signal.to_signal_snapshot()
        assert snap.is_active is False
        assert snap.value == 0.0  # "blend" maps to 0.0

    def test_calendar_seasonality_signal_snapshot(self):
        """CalendarSeasonalitySignal.to_signal_snapshot() converts correctly."""
        from src.signals.calendar_seasonality import CalendarSeasonalitySignal
        signal = CalendarSeasonalitySignal(
            assessment_date="2026-05-23",
            day_of_week="Friday", is_trading_day=True,
            active_windows=["tom_window", "opex_week"],
            urgency_modifier=0.8,
            tom_modifier=0.3, holiday_modifier=0.0,
            quarter_end_modifier=0.0, monday_modifier=0.0,
            fomc_modifier=0.0, december_modifier=0.0, opex_modifier=0.2,
            recommendation="proceed", effect="positive",
            next_window="month_end", next_window_date="2026-05-29",
            days_to_next_window=6, confidence=0.65,
        )
        snap = signal.to_signal_snapshot()
        assert snap.source == "calendar_seasonality"
        assert snap.value == 0.3  # "positive" maps to 0.3
        assert snap.confidence == 0.65
        assert snap.is_active is True
        assert snap.asset_signals == {}  # Calendar signal is portfolio-wide

    def test_calendar_seasonality_non_trading_day(self):
        """Non-trading day produces inactive snapshot."""
        from src.signals.calendar_seasonality import CalendarSeasonalitySignal
        signal = CalendarSeasonalitySignal(
            assessment_date="2026-05-24",
            day_of_week="Saturday", is_trading_day=False,
            active_windows=[],
            urgency_modifier=0.0,
            tom_modifier=0.0, holiday_modifier=0.0,
            quarter_end_modifier=0.0, monday_modifier=0.0,
            fomc_modifier=0.0, december_modifier=0.0, opex_modifier=0.0,
            recommendation="delay", effect="neutral",
            next_window="monday", next_window_date="2026-05-26",
            days_to_next_window=2, confidence=0.3,
        )
        snap = signal.to_signal_snapshot()
        assert snap.is_active is False

    def test_collar_signal_snapshot(self):
        """CollarSignal.to_signal_snapshot() converts correctly."""
        from src.signals.collar_signal import CollarSignal, CollarStrikes
        strikes = CollarStrikes(
            underlying_price=590.0, call_strike=610.0, put_strike=560.0,
            call_premium=3.50, put_premium=4.20, net_premium=-0.70,
            call_delta=0.30, put_delta=-0.25, vix_level=18.0,
            regime="normal", days_to_expiry=30, is_cashless=False,
            collar_cost_pct=0.0012,
        )
        signal = CollarSignal(
            timestamp="2026-05-23T12:00:00",
            signal_state="collared",
            call_strike=610.0, put_strike=560.0, underlying_price=590.0,
            expected_monthly_yield=0.008, max_upside_pct=0.034,
            max_downside_pct=-0.051,
            vix_level=18.0, regime="normal",
            strikes=strikes,
            collar_notional_pct=0.50, spy_shift=0.05,
            confidence=0.75, is_valid=True,
            reason="Normal VIX, collared position",
        )
        snap = signal.to_signal_snapshot()
        assert snap.source == "collar_signal"
        assert snap.value == 0.3  # "collared" maps to 0.3
        assert snap.confidence == 0.75
        assert snap.is_active is True
        assert snap.asset_signals["SPY"] == -0.05  # -spy_shift

    def test_collar_signal_unhedged(self):
        """Unhedged collar produces zero value."""
        from src.signals.collar_signal import CollarSignal, CollarStrikes
        strikes = CollarStrikes(
            underlying_price=590.0, call_strike=0, put_strike=0,
            call_premium=0, put_premium=0, net_premium=0,
            call_delta=0, put_delta=0, vix_level=25.0,
            regime="high_vol", days_to_expiry=0, is_cashless=True,
            collar_cost_pct=0,
        )
        signal = CollarSignal(
            timestamp="2026-05-23T12:00:00",
            signal_state="unhedged",
            call_strike=0, put_strike=0, underlying_price=590.0,
            expected_monthly_yield=0.0, max_upside_pct=0.0,
            max_downside_pct=0.0,
            vix_level=25.0, regime="high_vol",
            strikes=strikes,
            collar_notional_pct=0.0, spy_shift=0.0,
            confidence=0.3, is_valid=False,
            reason="High VIX, no collar",
        )
        snap = signal.to_signal_snapshot()
        assert snap.value == 0.0  # "unhedged" maps to 0.0
        assert snap.is_active is False