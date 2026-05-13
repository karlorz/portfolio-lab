#!/usr/bin/env python3
"""
Tests for regime_sentiment.py — RegimeSentiment enum, CombinedRegimeSignal dataclass,
RegimeSentimentIntegrator (score mapping, weight adjustment, regime classification,
circuit breaker, position scaling, allocation tilts), and RegimeSentimentPipeline.
"""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.strategy.regime_sentiment import (
    RegimeSentiment,
    CombinedRegimeSignal,
    RegimeSentimentIntegrator,
    RegimeSentimentPipeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(**overrides):
    defaults = dict(
        timestamp=datetime.now().isoformat(),
        technical_regime="bullish_momentum",
        technical_confidence=0.75,
        sentiment_regime="neutral",
        sentiment_confidence=0.60,
        combined_score=0.35,
        combined_regime="risk_on",
        technical_weight=0.70,
        sentiment_weight=0.30,
        circuit_breaker_level="green",
        position_scaling_factor=0.95,
        equity_tilt=0.50,
        bond_duration_tilt=0.0,
        gold_tilt=0.0,
    )
    defaults.update(overrides)
    return CombinedRegimeSignal(**defaults)


def _make_mock_sentiment(regime_signal="neutral", confidence=0.60):
    """Create a mock AggregatedSentiment object."""
    mock = MagicMock()
    mock.regime_signal = regime_signal
    mock.confidence = confidence
    return mock


# ---------------------------------------------------------------------------
# RegimeSentiment Enum Tests
# ---------------------------------------------------------------------------

class TestRegimeSentimentEnum:

    def test_values(self):
        assert RegimeSentiment.EXTREME_BULLISH.value == "extreme_bullish"
        assert RegimeSentiment.BULLISH.value == "bullish"
        assert RegimeSentiment.NEUTRAL.value == "neutral"
        assert RegimeSentiment.BEARISH.value == "bearish"
        assert RegimeSentiment.EXTREME_BEARISH.value == "extreme_bearish"


# ---------------------------------------------------------------------------
# CombinedRegimeSignal Tests
# ---------------------------------------------------------------------------

class TestCombinedRegimeSignal:

    def test_to_dict_keys(self):
        s = _make_signal()
        d = s.to_dict()
        assert "timestamp" in d
        assert "technical_regime" in d
        assert "combined_score" in d
        assert "circuit_breaker_level" in d
        assert "equity_tilt" in d

    def test_to_dict_values(self):
        s = _make_signal(combined_score=0.45, combined_regime="risk_on")
        d = s.to_dict()
        assert d["combined_score"] == 0.45
        assert d["combined_regime"] == "risk_on"


# ---------------------------------------------------------------------------
# RegimeSentimentIntegrator — init
# ---------------------------------------------------------------------------

class TestIntegratorInit:

    def test_default_weights(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.technical_weight == 0.70
        assert integrator.sentiment_weight == 0.30

    def test_custom_weights(self):
        integrator = RegimeSentimentIntegrator(technical_weight=0.80, sentiment_weight=0.20)
        assert integrator.technical_weight == 0.80
        assert integrator.sentiment_weight == 0.20

    def test_weights_normalized(self):
        integrator = RegimeSentimentIntegrator(technical_weight=2.0, sentiment_weight=1.0)
        total = integrator.technical_weight + integrator.sentiment_weight
        assert total == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# RegimeSentimentIntegrator — map_sentiment_to_score
# ---------------------------------------------------------------------------

class TestMapSentiment:

    def test_risk_on(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.map_sentiment_to_score("risk_on") == 0.5

    def test_neutral(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.map_sentiment_to_score("neutral") == 0.0

    def test_risk_off(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.map_sentiment_to_score("risk_off") == -0.5

    def test_extreme_risk_off(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.map_sentiment_to_score("extreme_risk_off") == -0.8

    def test_unknown_returns_zero(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.map_sentiment_to_score("unknown") == 0.0


# ---------------------------------------------------------------------------
# RegimeSentimentIntegrator — map_technical_to_score
# ---------------------------------------------------------------------------

class TestMapTechnical:

    def test_bullish_momentum(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.map_technical_to_score("bullish_momentum") == 0.7

    def test_crisis(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.map_technical_to_score("crisis") == -0.8

    def test_neutral_trending(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.map_technical_to_score("neutral_trending") == 0.2

    def test_unknown_returns_zero(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.map_technical_to_score("unknown_regime") == 0.0

    def test_all_regimes_have_scores(self):
        integrator = RegimeSentimentIntegrator()
        for regime in ["bullish_momentum", "neutral_trending", "volatile_chop",
                        "bearish_momentum", "crisis", "recovery", "expansion", "contraction"]:
            score = integrator.map_technical_to_score(regime)
            assert -1 <= score <= 1


# ---------------------------------------------------------------------------
# RegimeSentimentIntegrator — adjust_weights
# ---------------------------------------------------------------------------

class TestAdjustWeights:

    def test_default_weights(self):
        integrator = RegimeSentimentIntegrator()
        tech_w, sent_w = integrator.adjust_weights(0.8, 0.6)
        assert tech_w == 0.70
        assert sent_w == 0.30

    def test_low_tech_high_sent(self):
        integrator = RegimeSentimentIntegrator()
        tech_w, sent_w = integrator.adjust_weights(0.4, 0.8)
        assert tech_w == 0.50
        assert sent_w == 0.50

    def test_very_low_tech(self):
        integrator = RegimeSentimentIntegrator()
        tech_w, sent_w = integrator.adjust_weights(0.2, 0.5)
        assert tech_w == 0.40
        assert sent_w == 0.60

    def test_boundary_tech_confidence(self):
        integrator = RegimeSentimentIntegrator()
        # Exactly at 0.5 → strict < 0.5 check falls through to default
        tech_w, sent_w = integrator.adjust_weights(0.5, 0.8)
        assert tech_w == 0.70  # Default weights
        assert sent_w == 0.30


# ---------------------------------------------------------------------------
# RegimeSentimentIntegrator — classify_combined_regime
# ---------------------------------------------------------------------------

class TestClassifyRegime:

    def test_extreme_risk_on(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.classify_combined_regime(0.7) == "extreme_risk_on"

    def test_risk_on(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.classify_combined_regime(0.4) == "risk_on"

    def test_neutral(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.classify_combined_regime(0.0) == "neutral"

    def test_risk_off(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.classify_combined_regime(-0.4) == "risk_off"

    def test_extreme_risk_off(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.classify_combined_regime(-0.7) == "extreme_risk_off"

    def test_boundaries(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.classify_combined_regime(0.6) == "extreme_risk_on"
        assert integrator.classify_combined_regime(0.3) == "risk_on"
        assert integrator.classify_combined_regime(-0.3) == "risk_off"
        assert integrator.classify_combined_regime(-0.6) == "extreme_risk_off"


# ---------------------------------------------------------------------------
# RegimeSentimentIntegrator — determine_circuit_breaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:

    def test_green(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.determine_circuit_breaker(0.5) == "green"

    def test_yellow(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.determine_circuit_breaker(0.0) == "yellow"

    def test_orange(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.determine_circuit_breaker(-0.3) == "orange"

    def test_red(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.determine_circuit_breaker(-0.6) == "red"

    def test_boundaries(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.determine_circuit_breaker(0.2) == "green"
        assert integrator.determine_circuit_breaker(-0.2) == "yellow"
        assert integrator.determine_circuit_breaker(-0.5) == "orange"


# ---------------------------------------------------------------------------
# RegimeSentimentIntegrator — calculate_position_scaling
# ---------------------------------------------------------------------------

class TestPositionScaling:

    def test_extreme_risk_on(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.calculate_position_scaling("extreme_risk_on") == 1.0

    def test_risk_on(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.calculate_position_scaling("risk_on") == 0.95

    def test_neutral(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.calculate_position_scaling("neutral") == 0.85

    def test_risk_off(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.calculate_position_scaling("risk_off") == 0.70

    def test_extreme_risk_off(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.calculate_position_scaling("extreme_risk_off") == 0.50

    def test_unknown_defaults(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.calculate_position_scaling("unknown") == 0.85


# ---------------------------------------------------------------------------
# RegimeSentimentIntegrator — calculate_allocation_tilts
# ---------------------------------------------------------------------------

class TestAllocationTilts:

    def test_risk_on_tilts(self):
        integrator = RegimeSentimentIntegrator()
        eq, bond, gold = integrator.calculate_allocation_tilts(0.5, "risk_on")
        assert eq > 0  # Positive equity tilt
        assert bond == 0.0  # Neutral bond
        assert gold == 0.0  # No gold hedge

    def test_risk_off_tilts(self):
        integrator = RegimeSentimentIntegrator()
        eq, bond, gold = integrator.calculate_allocation_tilts(-0.4, "risk_off")
        assert eq < 0  # Negative equity tilt
        assert bond == -0.5  # Shorten duration
        assert gold == 0.7  # Gold hedge

    def test_extreme_risk_off(self):
        integrator = RegimeSentimentIntegrator()
        eq, bond, gold = integrator.calculate_allocation_tilts(-0.8, "extreme_risk_off")
        assert eq <= 0
        assert bond == -0.5
        assert gold == 0.7

    def test_equity_tilt_clipped(self):
        integrator = RegimeSentimentIntegrator()
        eq, _, _ = integrator.calculate_allocation_tilts(1.0, "extreme_risk_on")
        assert eq <= 1.0
        eq_neg, _, _ = integrator.calculate_allocation_tilts(-1.0, "extreme_risk_off")
        assert eq_neg >= -1.0

    def test_mild_negative_gold(self):
        integrator = RegimeSentimentIntegrator()
        _, _, gold = integrator.calculate_allocation_tilts(-0.3, "neutral")
        assert gold == 0.4  # Score < -0.2 but not risk_off


# ---------------------------------------------------------------------------
# RegimeSentimentIntegrator — combine_signals
# ---------------------------------------------------------------------------

class TestCombineSignals:

    def test_returns_signal(self):
        integrator = RegimeSentimentIntegrator()
        sentiment = _make_mock_sentiment("neutral", 0.6)
        signal = integrator.combine_signals("bullish_momentum", 0.8, sentiment)
        assert isinstance(signal, CombinedRegimeSignal)

    def test_combined_score_calculation(self):
        integrator = RegimeSentimentIntegrator()
        sentiment = _make_mock_sentiment("neutral", 0.6)
        signal = integrator.combine_signals("bullish_momentum", 0.8, sentiment)
        # tech_score=0.7, sent_score=0.0, weights=0.7/0.3
        expected = 0.70 * 0.7 + 0.30 * 0.0
        assert signal.combined_score == pytest.approx(expected, abs=0.01)

    def test_regime_fields(self):
        integrator = RegimeSentimentIntegrator()
        sentiment = _make_mock_sentiment("risk_on", 0.8)
        signal = integrator.combine_signals("bullish_momentum", 0.8, sentiment)
        assert signal.technical_regime == "bullish_momentum"
        assert signal.sentiment_regime == "risk_on"

    def test_circuit_breaker_in_signal(self):
        integrator = RegimeSentimentIntegrator()
        sentiment = _make_mock_sentiment("neutral", 0.6)
        signal = integrator.combine_signals("bullish_momentum", 0.8, sentiment)
        assert signal.circuit_breaker_level in ("green", "yellow", "orange", "red")

    def test_tilts_in_signal(self):
        integrator = RegimeSentimentIntegrator()
        sentiment = _make_mock_sentiment("neutral", 0.6)
        signal = integrator.combine_signals("bullish_momentum", 0.8, sentiment)
        assert -1 <= signal.equity_tilt <= 1
        assert -1 <= signal.bond_duration_tilt <= 1
        assert -1 <= signal.gold_tilt <= 1


# ---------------------------------------------------------------------------
# RegimeSentimentPipeline — get_current_allocation_weights
# ---------------------------------------------------------------------------

class TestAllocationWeights:

    def test_default_base(self):
        pipeline = RegimeSentimentPipeline.__new__(RegimeSentimentPipeline)
        pipeline.integrator = RegimeSentimentIntegrator()
        signal = _make_signal(equity_tilt=0.0, gold_tilt=0.0, bond_duration_tilt=0.0)
        weights = pipeline.get_current_allocation_weights(signal)
        assert "SPY" in weights
        assert "GLD" in weights
        assert "TLT" in weights
        assert sum(weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_positive_equity_tilt(self):
        pipeline = RegimeSentimentPipeline.__new__(RegimeSentimentPipeline)
        pipeline.integrator = RegimeSentimentIntegrator()
        signal = _make_signal(equity_tilt=0.5, gold_tilt=0.0, bond_duration_tilt=0.0)
        weights = pipeline.get_current_allocation_weights(signal)
        # SPY should increase from base 0.46
        assert weights["SPY"] > 0.46

    def test_negative_equity_tilt(self):
        pipeline = RegimeSentimentPipeline.__new__(RegimeSentimentPipeline)
        pipeline.integrator = RegimeSentimentIntegrator()
        signal = _make_signal(equity_tilt=-0.5, gold_tilt=0.0, bond_duration_tilt=0.0)
        weights = pipeline.get_current_allocation_weights(signal)
        assert weights["SPY"] < 0.46

    def test_gold_hedge(self):
        pipeline = RegimeSentimentPipeline.__new__(RegimeSentimentPipeline)
        pipeline.integrator = RegimeSentimentIntegrator()
        signal = _make_signal(equity_tilt=0.0, gold_tilt=0.7, bond_duration_tilt=0.0)
        weights = pipeline.get_current_allocation_weights(signal)
        assert weights["GLD"] > 0.38

    def test_custom_base(self):
        pipeline = RegimeSentimentPipeline.__new__(RegimeSentimentPipeline)
        pipeline.integrator = RegimeSentimentIntegrator()
        signal = _make_signal(equity_tilt=0.0, gold_tilt=0.0, bond_duration_tilt=0.0)
        base = {"SPY": 0.50, "GLD": 0.30, "TLT": 0.20}
        weights = pipeline.get_current_allocation_weights(signal, base_allocation=base)
        assert sum(weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_weights_clamped(self):
        pipeline = RegimeSentimentPipeline.__new__(RegimeSentimentPipeline)
        pipeline.integrator = RegimeSentimentIntegrator()
        # Extreme tilt should be clamped
        signal = _make_signal(equity_tilt=1.0, gold_tilt=1.0, bond_duration_tilt=1.0)
        weights = pipeline.get_current_allocation_weights(signal)
        assert weights["SPY"] <= 0.70
        assert weights["GLD"] <= 0.50
        assert weights["TLT"] <= 0.25
