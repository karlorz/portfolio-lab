"""Tests for credit spread signal generator v3.14."""
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch


class TestCreditSignalType:
    """Credit regime signal type enum."""

    def test_enum_values(self):
        from src.signals.credit_spread_signal import CreditSignalType
        assert CreditSignalType.RISK_ON.value == "risk_on"
        assert CreditSignalType.RISK_OFF.value == "risk_off"
        assert CreditSignalType.NEUTRAL.value == "neutral"


class TestAllocationShift:
    """Allocation shift direction enum."""

    def test_enum_values(self):
        from src.signals.credit_spread_signal import AllocationShift
        assert AllocationShift.INCREASE.value == "increase"
        assert AllocationShift.DECREASE.value == "decrease"
        assert AllocationShift.HOLD.value == "hold"


class TestAllocationRecommendation:
    """AllocationRecommendation frozen dataclass."""

    def test_recommendation_increase(self):
        from src.signals.credit_spread_signal import AllocationRecommendation, AllocationShift
        rec = AllocationRecommendation(
            symbol="SPY", current_weight=0.46, recommended_weight=0.48,
            shift=AllocationShift.INCREASE, shift_percent=2.0,
            rationale="Risk-on: increase equity"
        )
        assert rec.symbol == "SPY"
        assert rec.shift == AllocationShift.INCREASE
        assert rec.shift_percent == 2.0

    def test_recommendation_hold(self):
        from src.signals.credit_spread_signal import AllocationRecommendation, AllocationShift
        rec = AllocationRecommendation(
            symbol="GLD", current_weight=0.38, recommended_weight=0.38,
            shift=AllocationShift.HOLD, shift_percent=0.0,
            rationale="Neutral"
        )
        assert rec.shift == AllocationShift.HOLD
        assert rec.current_weight == rec.recommended_weight

    def test_recommendation_frozen(self):
        """Frozen dataclass — should not allow mutation."""
        from src.signals.credit_spread_signal import AllocationRecommendation, AllocationShift
        rec = AllocationRecommendation(
            symbol="SPY", current_weight=0.46, recommended_weight=0.48,
            shift=AllocationShift.INCREASE, shift_percent=2.0,
            rationale="Test"
        )
        with pytest.raises(Exception):
            rec.symbol = "QQQ"


class TestCreditSpreadSignal:
    """CreditSpreadSignal dataclass."""

    def test_signal_creation_risk_on(self):
        from src.signals.credit_spread_signal import (
            CreditSpreadSignal, CreditSignalType, AllocationRecommendation, AllocationShift
        )
        rec = AllocationRecommendation(
            symbol="SPY", current_weight=0.46, recommended_weight=0.48,
            shift=AllocationShift.INCREASE, shift_percent=2.0, rationale="Test"
        )
        signal = CreditSpreadSignal(
            timestamp=datetime(2026, 5, 14),
            signal_type=CreditSignalType.RISK_ON,
            confidence=0.75,
            spread_absolute=2.3,
            spread_zscore=0.5,
            trend_direction="widening",
            persistence_days=5,
            volatility_regime="normal",
            is_active=True,
            recommendations=[rec],
            summary="Credit spreads widening — risk-on for equities"
        )
        assert signal.signal_type == CreditSignalType.RISK_ON
        assert signal.is_active is True
        assert signal.confidence == 0.75
        assert signal.spread_absolute == 2.3
        assert len(signal.recommendations) == 1

    def test_signal_neutral(self):
        from src.signals.credit_spread_signal import CreditSpreadSignal, CreditSignalType
        signal = CreditSpreadSignal(
            timestamp=datetime(2026, 5, 14),
            signal_type=CreditSignalType.NEUTRAL,
            confidence=0.3, spread_absolute=0.5, spread_zscore=0.0,
            trend_direction="stable", persistence_days=1,
            volatility_regime="low", is_active=False,
            recommendations=[], summary="Neutral"
        )
        assert signal.signal_type == CreditSignalType.NEUTRAL
        assert signal.is_active is False

    def test_signal_risk_off(self):
        from src.signals.credit_spread_signal import CreditSpreadSignal, CreditSignalType
        signal = CreditSpreadSignal(
            timestamp=datetime(2026, 5, 14),
            signal_type=CreditSignalType.RISK_OFF,
            confidence=0.8, spread_absolute=-1.5, spread_zscore=-1.2,
            trend_direction="tightening", persistence_days=7,
            volatility_regime="elevated", is_active=True,
            recommendations=[], summary="Credit tightening — risk-off"
        )
        assert signal.signal_type == CreditSignalType.RISK_OFF


class TestSignalGeneratorCore:
    """CreditSpreadSignalGenerator core logic."""

    @pytest.fixture
    def generator(self):
        from src.signals.credit_spread_signal import CreditSpreadSignalGenerator
        gen = CreditSpreadSignalGenerator(vix_level=15.0)
        gen.fetcher = MagicMock()
        return gen

    def test_signal_disabled_vix_above_cutoff(self):
        from src.signals.credit_spread_signal import CreditSpreadSignalGenerator
        gen = CreditSpreadSignalGenerator(vix_level=40.0)
        disabled, reason = gen._is_signal_disabled()
        assert disabled is True

    def test_signal_not_disabled_low_vix(self, generator):
        disabled, reason = generator._is_signal_disabled()
        assert disabled is False

    def test_signal_not_disabled_at_cutoff(self):
        """VIX at exactly cutoff (35) should NOT disable the signal."""
        from src.signals.credit_spread_signal import CreditSpreadSignalGenerator
        gen = CreditSpreadSignalGenerator(vix_level=35.0)
        disabled, reason = gen._is_signal_disabled()
        assert disabled is False

    def test_determine_signal_type_risk_on(self, generator):
        from src.signals.credit_spread_signal import CreditSignalType
        sig_type, active, reason = generator._determine_signal_type(
            spread=0.02, persistence=5, confidence=0.5
        )
        assert isinstance(sig_type, CreditSignalType)
        assert isinstance(active, bool)

    def test_determine_signal_type_low_persistence(self, generator):
        from src.signals.credit_spread_signal import CreditSignalType
        sig_type, active, reason = generator._determine_signal_type(
            spread=0.02, persistence=1, confidence=0.5
        )
        assert sig_type == CreditSignalType.NEUTRAL
        assert active is False

    def test_determine_signal_type_disabled_by_vix(self):
        from src.signals.credit_spread_signal import CreditSpreadSignalGenerator, CreditSignalType
        gen = CreditSpreadSignalGenerator(vix_level=40.0)
        gen.fetcher = MagicMock()
        sig_type, active, reason = gen._determine_signal_type(
            spread=0.02, persistence=5, confidence=0.5
        )
        assert sig_type == CreditSignalType.NEUTRAL
        assert active is False

    def test_determine_signal_type_low_confidence(self, generator):
        from src.signals.credit_spread_signal import CreditSignalType
        sig_type, active, reason = generator._determine_signal_type(
            spread=0.02, persistence=5, confidence=0.2
        )
        # Low confidence + spread above threshold → should be neutral
        assert sig_type == CreditSignalType.NEUTRAL

    def test_generate_recommendations_neutral(self, generator):
        from src.signals.credit_spread_signal import CreditSignalType, AllocationShift
        recs = generator._generate_recommendations(CreditSignalType.NEUTRAL)
        assert len(recs) > 0
        for r in recs:
            assert r.shift == AllocationShift.HOLD
            assert r.current_weight == r.recommended_weight

    def test_generate_recommendations_risk_on(self, generator):
        from src.signals.credit_spread_signal import CreditSignalType
        recs = generator._generate_recommendations(CreditSignalType.RISK_ON)
        assert len(recs) > 0
        assert any(r.symbol == "SPY" for r in recs)

    def test_generate_recommendations_risk_off(self, generator):
        from src.signals.credit_spread_signal import CreditSignalType
        recs = generator._generate_recommendations(CreditSignalType.RISK_OFF)
        assert len(recs) > 0
        for r in recs:
            assert 0.05 <= r.recommended_weight <= 0.80

    def test_recommendations_clamped(self, generator):
        from src.signals.credit_spread_signal import CreditSignalType
        recs = generator._generate_recommendations(CreditSignalType.RISK_ON)
        for r in recs:
            assert 0.05 <= r.recommended_weight <= 0.80, \
                f"{r.symbol}: {r.recommended_weight} outside [0.05, 0.80]"

    def test_base_weights_defined(self, generator):
        assert len(generator.BASE_WEIGHTS) > 0
        assert "SPY" in generator.BASE_WEIGHTS

    def test_vix_cutoff_defined(self, generator):
        assert generator.VIX_CUTOFF > 0

    def test_min_persistence_days_defined(self, generator):
        assert generator.MIN_PERSISTENCE_DAYS > 0


class TestGetCreditSignal:
    """Module-level get_credit_signal function."""

    def test_get_credit_signal_returns_dict(self):
        from src.signals.credit_spread_signal import get_credit_signal
        with patch('src.signals.credit_spread_signal.CreditSpreadSignalGenerator') as mock_gen_class:
            from src.signals.credit_spread_signal import (
                CreditSpreadSignal, CreditSignalType
            )
            mock_gen = MagicMock()
            mock_gen.generate_signal.return_value = CreditSpreadSignal(
                timestamp=datetime(2026, 5, 14),
                signal_type=CreditSignalType.RISK_ON,
                confidence=0.7, spread_absolute=1.5, spread_zscore=0.5,
                trend_direction="widening", persistence_days=4,
                volatility_regime="normal", is_active=True,
                recommendations=[], summary="Test signal"
            )
            mock_gen_class.return_value = mock_gen

            result = get_credit_signal(vix_level=15.0)
            assert isinstance(result, dict)


class TestMain:
    """CLI entry point."""

    def test_main_help(self, monkeypatch, capsys):
        import sys
        from src.signals.credit_spread_signal import main
        test_args = ["credit_spread_signal", "--help"]
        monkeypatch.setattr(sys, 'argv', test_args)
        try:
            main()
        except SystemExit:
            pass
        captured = capsys.readouterr()
        assert "credit" in captured.out.lower() or "Credit" in captured.out
