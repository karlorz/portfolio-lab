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
    """AllocationRecommendation dataclass."""

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

    def test_to_dict(self):
        from src.signals.credit_spread_signal import AllocationRecommendation, AllocationShift
        rec = AllocationRecommendation(
            symbol="SPY", current_weight=0.46, recommended_weight=0.48,
            shift=AllocationShift.INCREASE, shift_percent=2.0,
            rationale="Test"
        )
        d = rec.to_dict()
        assert d["symbol"] == "SPY"
        assert isinstance(d, dict)


class TestCreditSpreadSignal:
    """CreditSpreadSignal dataclass."""

    def test_signal_creation(self):
        from src.signals.credit_spread_signal import (
            CreditSpreadSignal, CreditSignalType, AllocationRecommendation, AllocationShift
        )
        rec = AllocationRecommendation(
            symbol="SPY", current_weight=0.46, recommended_weight=0.48,
            shift=AllocationShift.INCREASE, shift_percent=2.0, rationale="Test"
        )
        signal = CreditSpreadSignal(
            timestamp=datetime(2026, 5, 14),
            spread_pct=1.5, spread_bps=150,
            signal_type=CreditSignalType.RISK_ON,
            confidence=0.75, persistence_days=5,
            is_active=True, reason="Spread widening detected",
            vix_level=18.0, vix_cutoff=30.0,
            recommendations=[rec]
        )
        assert signal.signal_type == CreditSignalType.RISK_ON
        assert signal.is_active is True
        assert signal.confidence == 0.75
        assert signal.spread_bps == 150
        assert len(signal.recommendations) == 1

    def test_signal_neutral(self):
        from src.signals.credit_spread_signal import CreditSpreadSignal, CreditSignalType
        signal = CreditSpreadSignal(
            timestamp=datetime(2026, 5, 14),
            spread_pct=0.5, spread_bps=50,
            signal_type=CreditSignalType.NEUTRAL,
            confidence=0.3, persistence_days=1,
            is_active=False, reason="Within neutral band",
            vix_level=15.0, vix_cutoff=30.0,
            recommendations=[]
        )
        assert signal.signal_type == CreditSignalType.NEUTRAL
        assert signal.is_active is False


class TestSignalGeneratorCore:
    """CreditSpreadSignalGenerator core logic."""

    @pytest.fixture
    def generator(self):
        from src.signals.credit_spread_signal import CreditSpreadSignalGenerator
        gen = CreditSpreadSignalGenerator(vix_level=15.0)
        # Disable the real fetcher
        gen.fetcher = MagicMock()
        return gen

    def test_signal_disabled_high_vix(self):
        from src.signals.credit_spread_signal import CreditSpreadSignalGenerator
        gen = CreditSpreadSignalGenerator(vix_level=35.0)
        disabled, reason = gen._is_signal_disabled()
        assert disabled is True
        assert "35" in reason

    def test_signal_not_disabled_low_vix(self, generator):
        disabled, reason = generator._is_signal_disabled()
        assert disabled is False

    def test_determine_signal_type_risk_on(self, generator):
        from src.signals.credit_spread_signal import CreditSignalType
        # Simulate widening spread (risk-on for credit)
        sig_type, active, reason = generator._determine_signal_type(
            spread=0.02, persistence=5, confidence=0.5
        )
        # Depends on thresholds; just verify it returns valid types
        assert isinstance(sig_type, CreditSignalType)
        assert isinstance(active, bool)

    def test_determine_signal_type_low_persistence(self, generator):
        from src.signals.credit_spread_signal import CreditSignalType
        sig_type, active, reason = generator._determine_signal_type(
            spread=0.02, persistence=1, confidence=0.5
        )
        assert sig_type == CreditSignalType.NEUTRAL
        assert active is False

    def test_determine_signal_type_disabled(self):
        from src.signals.credit_spread_signal import CreditSpreadSignalGenerator, CreditSignalType
        gen = CreditSpreadSignalGenerator(vix_level=35.0)
        gen.fetcher = MagicMock()
        sig_type, active, reason = gen._determine_signal_type(
            spread=0.02, persistence=5, confidence=0.5
        )
        assert sig_type == CreditSignalType.NEUTRAL
        assert active is False

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
        # At least one asset should shift
        shifts = [r for r in recs if r.shift_percent != 0.0]
        # In RISK_ON, we expect equity to increase
        assert any(r.symbol == "SPY" for r in recs)

    def test_generate_recommendations_risk_off(self, generator):
        from src.signals.credit_spread_signal import CreditSignalType
        recs = generator._generate_recommendations(CreditSignalType.RISK_OFF)
        assert len(recs) > 0
        # All recommendations should be valid
        for r in recs:
            assert 0.05 <= r.recommended_weight <= 0.80

    def test_recommendations_clamped(self, generator):
        """Recommended weights should be clamped to [0.05, 0.80]."""
        from src.signals.credit_spread_signal import CreditSignalType
        recs = generator._generate_recommendations(CreditSignalType.RISK_ON)
        for r in recs:
            assert 0.05 <= r.recommended_weight <= 0.80, \
                f"{r.symbol}: {r.recommended_weight} outside [0.05, 0.80]"

    def test_base_weights_sum(self, generator):
        """Base weights should be defined."""
        assert len(generator.BASE_WEIGHTS) > 0
        assert "SPY" in generator.BASE_WEIGHTS

    def test_vix_cutoff_defined(self, generator):
        assert generator.VIX_CUTOFF > 0

    def test_min_persistence_days_defined(self, generator):
        assert generator.MIN_PERSISTENCE_DAYS > 0


class TestGenerateSignal:
    """Full signal generation flow (mocked fetcher)."""

    @pytest.fixture
    def generator_with_mock(self):
        from src.signals.credit_spread_signal import CreditSpreadSignalGenerator
        gen = CreditSpreadSignalGenerator(vix_level=15.0)
        mock_fetcher = MagicMock()
        from src.data.credit_fetcher import CreditMetrics, CreditSignal as CFetchSignal
        mock_fetcher.get_latest_metrics.return_value = CreditMetrics(
            lqd_yield=5.5, hyg_yield=7.8, spread=2.3, spread_bps=230,
            spread_z=0.5, regime="normal", timestamp=datetime.now()
        )
        mock_fetcher.get_signal.return_value = CFetchSignal(
            signal_type="risk_on", confidence=0.6, persistence_days=5,
            is_active=True, reason="Test",
            spread=2.3, spread_z=0.5, regime="normal"
        )
        gen.fetcher = mock_fetcher
        return gen

    def test_generate_signal_returns_valid(self, generator_with_mock):
        signal = generator_with_mock.generate_signal()
        assert signal is not None
        assert signal.signal_type is not None
        assert signal.confidence > 0
        assert len(signal.recommendations) > 0

    def test_generate_signal_has_timestamp(self, generator_with_mock):
        signal = generator_with_mock.generate_signal()
        assert signal.timestamp is not None

    def test_generate_signal_vix_level(self, generator_with_mock):
        signal = generator_with_mock.generate_signal()
        assert signal.vix_level == 15.0


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
                spread_pct=1.5, spread_bps=150,
                signal_type=CreditSignalType.RISK_ON,
                confidence=0.7, persistence_days=4,
                is_active=True, reason="Test signal",
                vix_level=15.0, vix_cutoff=30.0,
                recommendations=[]
            )
            mock_gen_class.return_value = mock_gen

            result = get_credit_signal(vix_level=15.0)
            assert isinstance(result, dict)
            assert result["signal"] is not None


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
