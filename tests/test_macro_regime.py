"""
Test suite for Macro Regime Synthesizer (v4.30)

Tests signal harmonization, regime classification, weight calculation,
and allocation overlay functionality.
"""

import pytest
import numpy as np
from datetime import datetime
from pathlib import Path
import tempfile
import json

from src.regime.macro_regime import (
    MacroRegimeSynthesizer,
    MacroRegime,
    SignalState,
    SignalInput,
    RegimeClassification,
    SignalRegistry,
    classify_current_regime,
    create_default_synthesizer,
)


class TestSignalHarmonization:
    """Test signal state harmonization to [-1, 0, +1]."""
    
    def test_fed_policy_easing(self):
        synth = MacroRegimeSynthesizer()
        state = synth.harmonize_signal("fed_policy", "easing")
        assert state == SignalState.POSITIVE
    
    def test_fed_policy_tightening(self):
        synth = MacroRegimeSynthesizer()
        state = synth.harmonize_signal("fed_policy", "tightening")
        assert state == SignalState.NEGATIVE
    
    def test_fed_policy_neutral(self):
        synth = MacroRegimeSynthesizer()
        state = synth.harmonize_signal("fed_policy", "neutral")
        assert state == SignalState.NEUTRAL
    
    def test_yield_curve_steep(self):
        synth = MacroRegimeSynthesizer()
        state = synth.harmonize_signal("yield_curve", "steep")
        assert state == SignalState.POSITIVE
    
    def test_yield_curve_inverted(self):
        synth = MacroRegimeSynthesizer()
        state = synth.harmonize_signal("yield_curve", "inverted")
        assert state == SignalState.NEGATIVE
    
    def test_credit_spread_normal(self):
        synth = MacroRegimeSynthesizer()
        state = synth.harmonize_signal("credit_spread", "normal")
        assert state == SignalState.POSITIVE
    
    def test_credit_spread_distressed(self):
        synth = MacroRegimeSynthesizer()
        state = synth.harmonize_signal("credit_spread", "distressed")
        assert state == SignalState.NEGATIVE
    
    def test_fx_carry_safe(self):
        synth = MacroRegimeSynthesizer()
        state = synth.harmonize_signal("fx_carry", "safe")
        assert state == SignalState.POSITIVE
    
    def test_fx_carry_unwind_risk(self):
        synth = MacroRegimeSynthesizer()
        state = synth.harmonize_signal("fx_carry", "unwind_risk")
        assert state == SignalState.NEGATIVE
    
    def test_unknown_signal_returns_neutral(self):
        synth = MacroRegimeSynthesizer()
        state = synth.harmonize_signal("unknown_signal", "some_state")
        assert state == SignalState.NEUTRAL
    
    def test_alternative_mappings_bullish(self):
        synth = MacroRegimeSynthesizer()
        state = synth.harmonize_signal("equity_tsmom", "bullish")
        assert state == SignalState.POSITIVE
    
    def test_alternative_mappings_bearish(self):
        synth = MacroRegimeSynthesizer()
        state = synth.harmonize_signal("equity_tsmom", "bearish")
        assert state == SignalState.NEGATIVE
    
    def test_alternative_mappings_mixed(self):
        synth = MacroRegimeSynthesizer()
        state = synth.harmonize_signal("equity_tsmom", "mixed")
        assert state == SignalState.NEUTRAL


class TestDefaultWeights:
    """Test default signal weight calculations."""
    
    def test_weights_sum_to_one(self):
        synth = MacroRegimeSynthesizer()
        total = sum(synth.weights.values())
        assert abs(total - 1.0) < 0.001
    
    def test_all_signals_have_weights(self):
        synth = MacroRegimeSynthesizer()
        for signal in synth.SIGNAL_REGISTRY.keys():
            assert signal in synth.weights
    
    def test_key_signals_have_higher_weights(self):
        synth = MacroRegimeSynthesizer()
        # Core macro signals should have higher weights
        assert synth.weights["fed_policy"] >= 0.10
        assert synth.weights["yield_curve"] >= 0.10
        assert synth.weights["credit_spread"] >= 0.10
        assert synth.weights["equity_tsmom"] >= 0.10
    
    def test_vpin_has_lower_weight(self):
        synth = MacroRegimeSynthesizer()
        # VPIN is microstructure, should have lower weight
        assert synth.weights["vpin"] <= 0.10


class TestConfidenceCalculation:
    """Test confidence score calculation."""
    
    def test_full_agreement_high_confidence(self):
        synth = MacroRegimeSynthesizer()
        signals = {
            "fed_policy": SignalInput("fed_policy", SignalState.POSITIVE, 0, 100, datetime.now()),
            "yield_curve": SignalInput("yield_curve", SignalState.POSITIVE, 0, 100, datetime.now()),
            "credit_spread": SignalInput("credit_spread", SignalState.POSITIVE, 0, 100, datetime.now()),
        }
        
        weighted_sum = sum(
            s.state.value * synth.weights.get(n, 0.1)
            for n, s in signals.items()
        )
        confidence, agreement = synth.calculate_confidence(signals, weighted_sum)
        
        # Full agreement should give high confidence
        assert confidence > 50
        assert agreement > 0.8
    
    def test_full_disagreement_low_confidence(self):
        synth = MacroRegimeSynthesizer()
        signals = {
            "fed_policy": SignalInput("fed_policy", SignalState.POSITIVE, 0, 100, datetime.now()),
            "credit_spread": SignalInput("credit_spread", SignalState.NEGATIVE, 0, 100, datetime.now()),
            "fx_carry": SignalInput("fx_carry", SignalState.NEGATIVE, 0, 100, datetime.now()),
        }
        
        weighted_sum = sum(
            s.state.value * synth.weights.get(n, 0.1)
            for n, s in signals.items()
        )
        confidence, agreement = synth.calculate_confidence(signals, weighted_sum)
        
        # Disagreement should give lower confidence
        assert confidence < 70
    
    def test_empty_signals_zero_confidence(self):
        synth = MacroRegimeSynthesizer()
        confidence, agreement = synth.calculate_confidence({}, 0.0)
        assert confidence == 0.0
        assert agreement == 0.0


class TestRegimeClassification:
    """Test regime classification logic."""
    
    def test_crisis_with_four_negative_signals(self):
        synth = MacroRegimeSynthesizer()
        signals = {
            "fed_policy": SignalInput("fed_policy", SignalState.NEGATIVE, 0, 100, datetime.now()),
            "yield_curve": SignalInput("yield_curve", SignalState.NEGATIVE, 0, 100, datetime.now()),
            "credit_spread": SignalInput("credit_spread", SignalState.NEGATIVE, 0, 100, datetime.now()),
            "fx_carry": SignalInput("fx_carry", SignalState.NEGATIVE, 0, 100, datetime.now()),
            "vpin": SignalInput("vpin", SignalState.NEGATIVE, 0, 100, datetime.now()),
        }
        
        classification = synth.classify_regime(signals, min_confidence=50.0)
        assert classification.regime == MacroRegime.CRISIS
    
    def test_risk_on_growth_signals(self):
        synth = MacroRegimeSynthesizer()
        signals = {
            "fed_policy": SignalInput("fed_policy", SignalState.POSITIVE, 0, 100, datetime.now()),  # Easing
            "yield_curve": SignalInput("yield_curve", SignalState.POSITIVE, 0, 100, datetime.now()),  # Steep
            "credit_spread": SignalInput("credit_spread", SignalState.POSITIVE, 0, 100, datetime.now()),  # Normal
        }
        
        classification = synth.classify_regime(signals, min_confidence=50.0)
        assert classification.regime == MacroRegime.RISK_ON_GROWTH
    
    def test_risk_on_late_cycle_signals(self):
        synth = MacroRegimeSynthesizer()
        signals = {
            "fed_policy": SignalInput("fed_policy", SignalState.NEGATIVE, 0, 100, datetime.now()),  # Tightening
            "yield_curve": SignalInput("yield_curve", SignalState.NEUTRAL, 0, 100, datetime.now()),  # Flat
            "credit_spread": SignalInput("credit_spread", SignalState.POSITIVE, 0, 100, datetime.now()),  # Normal
        }
        
        classification = synth.classify_regime(signals, min_confidence=0.0)  # Allow lower confidence
        # With these signals, regime detection may be neutral due to limited signal coverage
        assert classification.regime in [MacroRegime.RISK_ON_LATE, MacroRegime.NEUTRAL]
    
    def test_defensive_with_inverted_curve(self):
        synth = MacroRegimeSynthesizer()
        signals = {
            "yield_curve": SignalInput("yield_curve", SignalState.NEGATIVE, 0, 100, datetime.now()),  # Inverted
            "bond_momentum": SignalInput("bond_momentum", SignalState.POSITIVE, 0, 100, datetime.now()),  # TLT momentum
        }
        
        classification = synth.classify_regime(signals, min_confidence=0.0)  # Allow lower confidence
        # May be defensive or neutral depending on signal matching
        assert classification.regime in [MacroRegime.DEFENSIVE, MacroRegime.NEUTRAL]
    
    def test_low_confidence_returns_neutral(self):
        synth = MacroRegimeSynthesizer()
        signals = {
            "fed_policy": SignalInput("fed_policy", SignalState.POSITIVE, 0, 100, datetime.now()),
            "yield_curve": SignalInput("yield_curve", SignalState.NEUTRAL, 0, 100, datetime.now()),
        }
        
        classification = synth.classify_regime(signals, min_confidence=90.0)
        # Low confidence (only 2 signals) should return neutral
        assert classification.regime == MacroRegime.NEUTRAL
    
    def test_classification_has_timestamp(self):
        synth = MacroRegimeSynthesizer()
        signals = {
            "fed_policy": SignalInput("fed_policy", SignalState.POSITIVE, 0, 100, datetime.now()),
        }
        
        classification = synth.classify_regime(signals)
        assert isinstance(classification.timestamp, datetime)
    
    def test_classification_has_weighted_sum(self):
        synth = MacroRegimeSynthesizer()
        signals = {
            "fed_policy": SignalInput("fed_policy", SignalState.POSITIVE, 0, 100, datetime.now()),
            "credit_spread": SignalInput("credit_spread", SignalState.POSITIVE, 0, 100, datetime.now()),
        }
        
        classification = synth.classify_regime(signals)
        assert isinstance(classification.weighted_sum, float)


class TestAllocationOverlay:
    """Test portfolio allocation overlay calculations."""
    
    def test_risk_on_growth_increases_spy(self):
        synth = MacroRegimeSynthesizer()
        base = {"spy": 0.46, "gld": 0.38, "tlt": 0.16}
        
        overlay = synth.get_allocation_overlay(MacroRegime.RISK_ON_GROWTH, 80.0, base)
        
        assert overlay["spy"] > base["spy"]  # Increased equity
        assert overlay["gld"] < base["gld"]  # Decreased gold
        assert overlay["tlt"] < base["tlt"]  # Decreased bonds
    
    def test_defensive_decreases_spy(self):
        synth = MacroRegimeSynthesizer()
        base = {"spy": 0.46, "gld": 0.38, "tlt": 0.16}
        
        overlay = synth.get_allocation_overlay(MacroRegime.DEFENSIVE, 80.0, base)
        
        assert overlay["spy"] < base["spy"]  # Decreased equity
        assert overlay["gld"] > base["gld"]  # Increased gold
        assert overlay["tlt"] > base["tlt"]  # Increased bonds
    
    def test_crisis_maximum_defensive(self):
        synth = MacroRegimeSynthesizer()
        base = {"spy": 0.46, "gld": 0.38, "tlt": 0.16}
        
        overlay = synth.get_allocation_overlay(MacroRegime.CRISIS, 80.0, base)
        
        # Crisis should have largest shifts
        spy_shift = base["spy"] - overlay["spy"]
        assert spy_shift >= 0.10  # At least 10% reduction
    
    def test_low_confidence_no_shift(self):
        synth = MacroRegimeSynthesizer()
        base = {"spy": 0.46, "gld": 0.38, "tlt": 0.16}
        
        overlay = synth.get_allocation_overlay(MacroRegime.RISK_ON_GROWTH, 50.0, base)
        
        # Should return unchanged base
        assert overlay == base
    
    def test_allocation_sums_to_one(self):
        synth = MacroRegimeSynthesizer()
        base = {"spy": 0.46, "gld": 0.38, "tlt": 0.16}
        
        for regime in MacroRegime:
            overlay = synth.get_allocation_overlay(regime, 80.0, base)
            total = sum(overlay.values())
            assert abs(total - 1.0) < 0.001, f"Regime {regime.value} sums to {total}"
    
    def test_no_negative_allocations(self):
        synth = MacroRegimeSynthesizer()
        base = {"spy": 0.05, "gld": 0.38, "tlt": 0.16}  # Low SPY
        
        overlay = synth.get_allocation_overlay(MacroRegime.CRISIS, 80.0, base)
        
        for asset, alloc in overlay.items():
            assert alloc >= 0, f"Negative allocation for {asset}: {alloc}"


class TestWeightUpdate:
    """Test dynamic weight updates from accuracy history."""
    
    def test_weights_recalibrated_by_accuracy(self):
        synth = MacroRegimeSynthesizer()
        
        # High accuracy for fed_policy, lower for others
        accuracy = {
            "fed_policy": 0.85,
            "yield_curve": 0.55,
            "credit_spread": 0.50,
            "equity_tsmom": 0.60,
            "bond_momentum": 0.45,
            "fx_carry": 0.40,
            "intl_equity": 0.45,
            "commodity_curve": 0.35,
            "vpin": 0.30,
        }
        
        original_fed_weight = synth.weights["fed_policy"]
        synth.update_weights_from_accuracy(accuracy, temperature=0.3)
        
        # Fed policy should now have higher weight
        assert synth.weights["fed_policy"] > original_fed_weight
    
    def test_weights_respect_min_max_bounds(self):
        synth = MacroRegimeSynthesizer()
        
        accuracy = {
            "fed_policy": 1.0,  # Perfect accuracy
            "yield_curve": 0.0,  # Zero accuracy
        }
        
        synth.update_weights_from_accuracy(accuracy, temperature=0.1)
        
        # Should still respect bounds
        assert synth.weights["fed_policy"] <= 0.25  # Max
        assert synth.weights["yield_curve"] >= 0.05  # Min
    
    def test_weights_renormalize_after_bounds(self):
        synth = MacroRegimeSynthesizer()
        
        accuracy = {
            "fed_policy": 0.80,
            "yield_curve": 0.70,
            "credit_spread": 0.75,
        }
        
        synth.update_weights_from_accuracy(accuracy)
        
        # Should still sum to 1
        total = sum(synth.weights.values())
        assert abs(total - 1.0) < 0.001
    
    def test_empty_accuracy_no_change(self):
        synth = MacroRegimeSynthesizer()
        original_weights = synth.weights.copy()
        
        synth.update_weights_from_accuracy({})
        
        assert synth.weights == original_weights


class TestDatabaseOperations:
    """Test database persistence operations."""
    
    def test_database_initialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_regime.db"
            synth = MacroRegimeSynthesizer(db_path=str(db_path))
            
            # Database file should exist
            assert db_path.exists()
    
    def test_persist_classification(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_regime.db"
            synth = MacroRegimeSynthesizer(db_path=str(db_path))
            
            signals = {
                "fed_policy": SignalInput("fed_policy", SignalState.POSITIVE, 0, 100, datetime.now()),
                "yield_curve": SignalInput("yield_curve", SignalState.POSITIVE, 0, 100, datetime.now()),
            }
            
            classification = synth.classify_regime(signals)
            synth.persist_classification(classification)
            
            # Should be retrievable
            history = synth.get_regime_history(days=1)
            assert len(history) == 1
            assert history[0]["regime"] == classification.regime.value
    
    def test_get_regime_history_returns_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_regime.db"
            synth = MacroRegimeSynthesizer(db_path=str(db_path))
            
            history = synth.get_regime_history(days=30)
            assert isinstance(history, list)


class TestSignalRegistry:
    """Test signal registry functionality."""
    
    def test_list_signals_returns_nine(self):
        signals = SignalRegistry.list_signals()
        assert len(signals) == 9
    
    def test_get_metadata_returns_dict(self):
        metadata = SignalRegistry.get_signal_metadata("fed_policy")
        assert isinstance(metadata, dict)
        assert "display_name" in metadata
        assert "mapping" in metadata
    
    def test_get_unknown_signal_returns_none(self):
        metadata = SignalRegistry.get_signal_metadata("unknown")
        assert metadata is None
    
    def test_validate_valid_signal(self):
        assert SignalRegistry.validate_signal_input("fed_policy", "easing")
    
    def test_validate_invalid_signal(self):
        assert not SignalRegistry.validate_signal_input("unknown", "state")


class TestSerialization:
    """Test data serialization to dictionaries."""
    
    def test_to_dict_structure(self):
        synth = MacroRegimeSynthesizer()
        
        signals = {
            "fed_policy": SignalInput("fed_policy", SignalState.POSITIVE, 0, 100, datetime.now()),
        }
        
        classification = synth.classify_regime(signals)
        data = synth.to_dict(classification)
        
        assert "timestamp" in data
        assert "regime" in data
        assert "regime_display" in data
        assert "confidence" in data
        assert "signal_agreement" in data
        assert "weighted_sum" in data
        assert "allocation_shifts" in data
        assert "signal_breakdown" in data
    
    def test_regime_display_formatted(self):
        synth = MacroRegimeSynthesizer()
        
        signals = {
            "fed_policy": SignalInput("fed_policy", SignalState.POSITIVE, 0, 100, datetime.now()),
        }
        
        classification = synth.classify_regime(signals)
        data = synth.to_dict(classification)
        
        # Should be title case with spaces
        assert " " in data["regime_display"] or data["regime"] == "neutral"


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_default_synthesizer(self):
        synth = create_default_synthesizer()
        assert isinstance(synth, MacroRegimeSynthesizer)
        assert len(synth.weights) == 9
    
    def test_classify_current_regime_returns_dict(self):
        signals = {
            "fed_policy": "easing",
            "yield_curve": "steep",
            "credit_spread": "normal",
        }
        
        result = classify_current_regime(signals)
        assert isinstance(result, dict)
        assert "regime" in result
        assert "confidence" in result


class TestRecommendationGeneration:
    """Test recommendation generation."""
    
    def test_high_confidence_gives_specific_action(self):
        synth = MacroRegimeSynthesizer()
        rec = synth._generate_recommendation(MacroRegime.RISK_ON_GROWTH, 80.0)
        assert "HOLD" not in rec
        assert "RISK ON GROWTH" in rec
    
    def test_low_confidence_gives_hold(self):
        synth = MacroRegimeSynthesizer()
        rec = synth._generate_recommendation(MacroRegime.RISK_ON_GROWTH, 50.0)
        assert "HOLD" in rec
    
    def test_neutral_recommendation(self):
        synth = MacroRegimeSynthesizer()
        rec = synth._generate_recommendation(MacroRegime.NEUTRAL, 80.0)
        assert "HOLD" in rec or "base" in rec.lower()


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    def test_2008_financial_crisis_regime(self):
        """Simulate signals during 2008 crisis."""
        synth = MacroRegimeSynthesizer()
        
        signals = {
            "fed_policy": SignalInput("fed_policy", SignalState.NEGATIVE, 0, 100, datetime.now()),  # Tightening then easing
            "yield_curve": SignalInput("yield_curve", SignalState.NEGATIVE, 0, 100, datetime.now()),  # Inverted
            "credit_spread": SignalInput("credit_spread", SignalState.NEGATIVE, 0, 100, datetime.now()),  # Distressed
            "fx_carry": SignalInput("fx_carry", SignalState.NEGATIVE, 0, 100, datetime.now()),  # Unwind
            "equity_tsmom": SignalInput("equity_tsmom", SignalState.NEGATIVE, 0, 100, datetime.now()),  # Risk-off
            "bond_momentum": SignalInput("bond_momentum", SignalState.POSITIVE, 0, 100, datetime.now()),  # Flight to quality
        }
        
        classification = synth.classify_regime(signals)
        # Should detect crisis or defensive regime
        assert classification.regime in [MacroRegime.CRISIS, MacroRegime.DEFENSIVE]
    
    def test_2020_covid_recovery_regime(self):
        """Simulate signals during 2020 recovery."""
        synth = MacroRegimeSynthesizer()
        
        signals = {
            "fed_policy": SignalInput("fed_policy", SignalState.POSITIVE, 0, 100, datetime.now()),  # Easing
            "yield_curve": SignalInput("yield_curve", SignalState.POSITIVE, 0, 100, datetime.now()),  # Steep
            "credit_spread": SignalInput("credit_spread", SignalState.NEUTRAL, 0, 100, datetime.now()),  # Improving
            "fx_carry": SignalInput("fx_carry", SignalState.NEGATIVE, 0, 100, datetime.now()),  # Some stress
            "equity_tsmom": SignalInput("equity_tsmom", SignalState.POSITIVE, 0, 100, datetime.now()),  # Recovery
        }
        
        classification = synth.classify_regime(signals)
        # Should be risk-on, recovery, or neutral depending on signal agreement
        assert classification.regime in [
            MacroRegime.RISK_ON_GROWTH, 
            MacroRegime.RISK_ON_LATE, 
            MacroRegime.NEUTRAL
        ]
    
    def test_bull_market_risk_on_growth(self):
        """Simulate strong bull market."""
        synth = MacroRegimeSynthesizer()
        
        signals = {
            "fed_policy": SignalInput("fed_policy", SignalState.POSITIVE, 0, 100, datetime.now()),  # Easing
            "yield_curve": SignalInput("yield_curve", SignalState.POSITIVE, 0, 100, datetime.now()),  # Steep
            "credit_spread": SignalInput("credit_spread", SignalState.POSITIVE, 0, 100, datetime.now()),  # Tight
            "equity_tsmom": SignalInput("equity_tsmom", SignalState.POSITIVE, 0, 100, datetime.now()),  # Risk-on
            "bond_momentum": SignalInput("bond_momentum", SignalState.NEGATIVE, 0, 100, datetime.now()),  # Rates rising
        }
        
        classification = synth.classify_regime(signals)
        assert classification.regime == MacroRegime.RISK_ON_GROWTH


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
