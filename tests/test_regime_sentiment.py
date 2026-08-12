#!/usr/bin/env python3
"""
Tests for regime_sentiment.py — RegimeSentiment enum, CombinedRegimeSignal dataclass,
RegimeSentimentIntegrator (score mapping, weight adjustment, regime classification,
circuit breaker, position scaling, allocation tilts), and RegimeSentimentPipeline.
"""
import json

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

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

    def test_timestamp_is_utc_timezone_aware(self):
        integrator = RegimeSentimentIntegrator()
        sentiment = _make_mock_sentiment("neutral", 0.6)
        signal = integrator.combine_signals("bullish_momentum", 0.8, sentiment)

        ts = datetime.fromisoformat(signal.timestamp)
        assert ts.tzinfo is not None
        assert ts.utcoffset() == timezone.utc.utcoffset(ts)


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


# ---------------------------------------------------------------------------
# __all__ export validation
# ---------------------------------------------------------------------------

class TestExports:
    """Verify __all__ exports."""

    def test_all_exports_present(self):
        import src.strategy.regime_sentiment as mod
        for name in mod.__all__:
            assert hasattr(mod, name), f"Missing export: {name}"

    def test_all_count(self):
        import src.strategy.regime_sentiment as mod
        assert len(mod.__all__) == 4


# ---------------------------------------------------------------------------
# RegimeSentiment enum extended
# ---------------------------------------------------------------------------

class TestRegimeSentimentExtended:
    """Extended RegimeSentiment enum tests."""

    def test_all_five_values(self):
        assert len(RegimeSentiment) == 5

    def test_extreme_values(self):
        assert RegimeSentiment.EXTREME_BULLISH.value == "extreme_bullish"
        assert RegimeSentiment.EXTREME_BEARISH.value == "extreme_bearish"


# ---------------------------------------------------------------------------
# RegimeSentimentIntegrator extended
# ---------------------------------------------------------------------------

class TestIntegratorExtended:
    """Extended integrator tests."""

    def _make_integrator(self):
        return RegimeSentimentIntegrator()

    def test_classify_combined_regime_boundaries(self):
        integrator = self._make_integrator()
        # Uses risk_on/risk_off terminology, not bullish/bearish
        assert integrator.classify_combined_regime(0.8) in ("risk_on", "extreme_risk_on")
        assert integrator.classify_combined_regime(-0.8) in ("risk_off", "extreme_risk_off")
        assert integrator.classify_combined_regime(0.0) == "neutral"

    def test_position_scaling_normal(self):
        integrator = self._make_integrator()
        scale = integrator.calculate_position_scaling("neutral")
        assert isinstance(scale, float)
        assert scale > 0

    def test_map_sentiment_known_values(self):
        integrator = self._make_integrator()
        for regime in ["extreme_bullish", "bullish", "neutral", "bearish", "extreme_bearish"]:
            score = integrator.map_sentiment_to_score(regime)
            assert -1 <= score <= 1

    def test_map_technical_known_values(self):
        integrator = self._make_integrator()
        for regime in ["early_expansion", "late_expansion", "contraction", "recovery", "neutral"]:
            score = integrator.map_technical_to_score(regime)
            assert -1 <= score <= 1

    def test_circuit_breaker_levels(self):
        integrator = self._make_integrator()
        for score in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            cb = integrator.determine_circuit_breaker(score)
            assert isinstance(cb, str)

    def test_adjust_weights_returns_tuple(self):
        integrator = self._make_integrator()
        tech_w, sent_w = integrator.adjust_weights(0.7, 0.6)
        assert isinstance(tech_w, float)
        assert isinstance(sent_w, float)

    def test_calculate_allocation_tilts_returns_tuple(self):
        integrator = self._make_integrator()
        eq, bond, gold = integrator.calculate_allocation_tilts(0.5, "risk_on")
        assert isinstance(eq, float)
        assert isinstance(bond, float)
        assert isinstance(gold, float)


# ---------------------------------------------------------------------------
# RegimeSentimentPipeline extended
# ---------------------------------------------------------------------------

class TestPipelineExtended:
    """Extended pipeline tests."""

    def test_init_with_data_dir(self, tmp_path):
        pipeline = RegimeSentimentPipeline(data_dir=tmp_path)
        assert pipeline.data_dir == tmp_path

    def test_save_signal(self, tmp_path):
        pipeline = RegimeSentimentPipeline(data_dir=tmp_path)
        signal = _make_signal()
        pipeline.save_signal(signal)
        # Should create a file
        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) >= 1


# ---------------------------------------------------------------------------
# demo callable
# ---------------------------------------------------------------------------

class TestDemo:
    """Test demo function callable."""

    def test_demo_callable(self):
        from src.strategy.regime_sentiment import demo
        assert callable(demo)


# ===================================================================
# NEW TEST SECTIONS — Dataclass validation, edge cases, constants,
# boundary conditions, CLI guard, export completeness
# ===================================================================


# ---------------------------------------------------------------------------
# CombinedRegimeSignal — dataclass field validation
# ---------------------------------------------------------------------------

class TestCombinedRegimeSignalDataclass:
    """Validate dataclass fields, types, defaults, and to_dict."""

    def test_is_dataclass(self):
        import dataclasses
        assert dataclasses.is_dataclass(CombinedRegimeSignal)

    def test_field_count(self):
        import dataclasses
        fields = dataclasses.fields(CombinedRegimeSignal)
        assert len(fields) == 14

    def test_field_names_complete(self):
        import dataclasses
        names = [f.name for f in dataclasses.fields(CombinedRegimeSignal)]
        expected = [
            "timestamp", "technical_regime", "technical_confidence",
            "sentiment_regime", "sentiment_confidence", "combined_score",
            "combined_regime", "technical_weight", "sentiment_weight",
            "circuit_breaker_level", "position_scaling_factor",
            "equity_tilt", "bond_duration_tilt", "gold_tilt",
        ]
        for name in expected:
            assert name in names, f"Missing field: {name}"
        assert len(names) == len(expected)

    def test_field_types(self):
        import dataclasses
        fields = {f.name: f.type for f in dataclasses.fields(CombinedRegimeSignal)}
        assert fields["timestamp"] is str
        assert fields["technical_regime"] is str
        assert fields["technical_confidence"] is float
        assert fields["sentiment_regime"] is str
        assert fields["sentiment_confidence"] is float
        assert fields["combined_score"] is float
        assert fields["combined_regime"] is str
        assert fields["technical_weight"] is float
        assert fields["sentiment_weight"] is float
        assert fields["circuit_breaker_level"] is str
        assert fields["position_scaling_factor"] is float
        assert fields["equity_tilt"] is float
        assert fields["bond_duration_tilt"] is float
        assert fields["gold_tilt"] is float

    def test_no_defaults(self):
        import dataclasses
        fields = dataclasses.fields(CombinedRegimeSignal)
        for f in fields:
            assert f.default is dataclasses.MISSING, (
                f"Field {f.name} should not have default"
            )
            assert f.default_factory is dataclasses.MISSING, (
                f"Field {f.name} should not have default_factory"
            )

    def test_to_dict_contains_all_fields(self):
        s = _make_signal()
        d = s.to_dict()
        assert len(d) == 14
        expected_keys = [
            "timestamp", "technical_regime", "technical_confidence",
            "sentiment_regime", "sentiment_confidence", "combined_score",
            "combined_regime", "technical_weight", "sentiment_weight",
            "circuit_breaker_level", "position_scaling_factor",
            "equity_tilt", "bond_duration_tilt", "gold_tilt",
        ]
        for key in expected_keys:
            assert key in d, f"Missing key: {key}"

    def test_to_dict_values_match(self):
        s = _make_signal(
            timestamp="2024-06-01T12:00:00",
            technical_regime="crisis",
            technical_confidence=0.5,
            sentiment_regime="risk_off",
            sentiment_confidence=0.7,
            combined_score=-0.4,
            combined_regime="risk_off",
            technical_weight=0.6,
            sentiment_weight=0.4,
            circuit_breaker_level="orange",
            position_scaling_factor=0.7,
            equity_tilt=-0.6,
            bond_duration_tilt=-0.5,
            gold_tilt=0.7,
        )
        d = s.to_dict()
        assert d["timestamp"] == "2024-06-01T12:00:00"
        assert d["technical_regime"] == "crisis"
        assert d["technical_confidence"] == 0.5
        assert d["sentiment_regime"] == "risk_off"
        assert d["sentiment_confidence"] == 0.7
        assert d["combined_score"] == -0.4
        assert d["combined_regime"] == "risk_off"
        assert d["technical_weight"] == 0.6
        assert d["sentiment_weight"] == 0.4
        assert d["circuit_breaker_level"] == "orange"
        assert d["position_scaling_factor"] == 0.7
        assert d["equity_tilt"] == -0.6
        assert d["bond_duration_tilt"] == -0.5
        assert d["gold_tilt"] == 0.7

    def test_dataclass_repr(self):
        s = _make_signal(timestamp="test_ts")
        rep = repr(s)
        assert "CombinedRegimeSignal" in rep
        assert "timestamp=" in rep
        assert "combined_score=" in rep

    def test_dataclass_eq(self):
        s1 = _make_signal(timestamp="t1", combined_score=0.5)
        s2 = _make_signal(timestamp="t1", combined_score=0.5)
        assert s1 == s2

    def test_dataclass_neq(self):
        s1 = _make_signal(timestamp="t1", combined_score=0.5)
        s2 = _make_signal(timestamp="t1", combined_score=0.6)
        assert s1 != s2


# ---------------------------------------------------------------------------
# RegimeSentiment enum — field validation
# ---------------------------------------------------------------------------

class TestRegimeSentimentEnumValidation:
    """Deep enum validation: members, names, uniqueness, types."""

    def test_enum_member_count(self):
        assert len(RegimeSentiment) == 5

    def test_enum_names_set(self):
        expected = {"EXTREME_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "EXTREME_BEARISH"}
        names = {m.name for m in RegimeSentiment}
        assert names == expected

    def test_enum_values_unique(self):
        values = [m.value for m in RegimeSentiment]
        assert len(values) == len(set(values))

    def test_enum_members_are_instance(self):
        for member in RegimeSentiment:
            assert isinstance(member, RegimeSentiment)

    def test_enum_values_are_str(self):
        for member in RegimeSentiment:
            assert isinstance(member.value, str)


# ---------------------------------------------------------------------------
# RegimeSentimentIntegrator — constants validation
# ---------------------------------------------------------------------------

class TestIntegratorConstants:
    """Validate module-level constants exist with expected types/ranges."""

    def test_default_technical_weight_value(self):
        assert RegimeSentimentIntegrator.DEFAULT_TECHNICAL_WEIGHT == 0.70

    def test_default_sentiment_weight_value(self):
        assert RegimeSentimentIntegrator.DEFAULT_SENTIMENT_WEIGHT == 0.30

    def test_default_weights_sum_to_one(self):
        total = (RegimeSentimentIntegrator.DEFAULT_TECHNICAL_WEIGHT
                 + RegimeSentimentIntegrator.DEFAULT_SENTIMENT_WEIGHT)
        assert total == pytest.approx(1.0)

    def test_default_weight_types(self):
        assert isinstance(RegimeSentimentIntegrator.DEFAULT_TECHNICAL_WEIGHT, float)
        assert isinstance(RegimeSentimentIntegrator.DEFAULT_SENTIMENT_WEIGHT, float)

    def test_risk_on_thresholds_positive(self):
        assert RegimeSentimentIntegrator.EXTREME_RISK_ON_THRESHOLD > 0
        assert RegimeSentimentIntegrator.RISK_ON_THRESHOLD > 0

    def test_risk_off_thresholds_negative(self):
        assert RegimeSentimentIntegrator.RISK_OFF_THRESHOLD < 0
        assert RegimeSentimentIntegrator.EXTREME_RISK_OFF_THRESHOLD < 0

    def test_thresholds_ordered_descending(self):
        assert (RegimeSentimentIntegrator.EXTREME_RISK_ON_THRESHOLD
                > RegimeSentimentIntegrator.RISK_ON_THRESHOLD
                > RegimeSentimentIntegrator.RISK_OFF_THRESHOLD
                > RegimeSentimentIntegrator.EXTREME_RISK_OFF_THRESHOLD)

    def test_cb_thresholds_ordered_descending(self):
        assert (RegimeSentimentIntegrator.CB_GREEN_THRESHOLD
                > RegimeSentimentIntegrator.CB_YELLOW_THRESHOLD
                > RegimeSentimentIntegrator.CB_ORANGE_THRESHOLD)

    def test_threshold_types_are_float(self):
        thresholds = [
            RegimeSentimentIntegrator.EXTREME_RISK_ON_THRESHOLD,
            RegimeSentimentIntegrator.RISK_ON_THRESHOLD,
            RegimeSentimentIntegrator.RISK_OFF_THRESHOLD,
            RegimeSentimentIntegrator.EXTREME_RISK_OFF_THRESHOLD,
            RegimeSentimentIntegrator.CB_GREEN_THRESHOLD,
            RegimeSentimentIntegrator.CB_YELLOW_THRESHOLD,
            RegimeSentimentIntegrator.CB_ORANGE_THRESHOLD,
        ]
        for t in thresholds:
            assert isinstance(t, float), f"Threshold {t} is not float"


# ---------------------------------------------------------------------------
# Computation edge cases — zero, boundary, extreme inputs
# ---------------------------------------------------------------------------

class TestComputationEdgeCases:
    """Edge cases in weight adjustment, classification, scaling, tilts."""

    def test_adjust_weights_both_zero(self):
        integrator = RegimeSentimentIntegrator()
        tech_w, sent_w = integrator.adjust_weights(0.0, 0.0)
        # 0.0 < 0.3 → falls to heavy-sent branch
        assert tech_w == 0.40
        assert sent_w == 0.60

    def test_adjust_weights_both_one(self):
        integrator = RegimeSentimentIntegrator()
        tech_w, sent_w = integrator.adjust_weights(1.0, 1.0)
        # Both high → default weights
        assert tech_w == 0.70
        assert sent_w == 0.30

    def test_adjust_weights_tech_low_sent_low(self):
        integrator = RegimeSentimentIntegrator()
        tech_w, sent_w = integrator.adjust_weights(0.2, 0.3)
        # tech < 0.3 → heavy-sent branch (does not check sent > 0.7)
        assert tech_w == 0.40
        assert sent_w == 0.60

    def test_adjust_weights_tech_04_sent_071(self):
        integrator = RegimeSentimentIntegrator()
        tech_w, sent_w = integrator.adjust_weights(0.4, 0.71)
        # 0.4 < 0.5 AND 0.71 > 0.7 → first branch 50/50
        assert tech_w == 0.50
        assert sent_w == 0.50

    def test_adjust_weights_tech_049_sent_070(self):
        integrator = RegimeSentimentIntegrator()
        tech_w, sent_w = integrator.adjust_weights(0.49, 0.70)
        # 0.49 < 0.5 but 0.70 is NOT > 0.7 → first condition False
        # 0.49 >= 0.3 → second condition False → default
        assert tech_w == 0.70
        assert sent_w == 0.30

    def test_adjust_weights_tech_03_exact(self):
        integrator = RegimeSentimentIntegrator()
        tech_w, sent_w = integrator.adjust_weights(0.3, 0.5)
        # 0.3 < 0.5 AND 0.5 > 0.7? 0.5 > 0.7 is False
        # 0.3 < 0.3? False (strict < ) → default
        assert tech_w == 0.70
        assert sent_w == 0.30

    def test_classify_regime_between_thresholds(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.classify_combined_regime(0.1) == "neutral"
        assert integrator.classify_combined_regime(-0.1) == "neutral"
        assert integrator.classify_combined_regime(0.29) == "neutral"
        assert integrator.classify_combined_regime(-0.29) == "neutral"

    def test_classify_regime_extreme_values(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.classify_combined_regime(1.0) == "extreme_risk_on"
        assert integrator.classify_combined_regime(-1.0) == "extreme_risk_off"

    def test_classify_regime_just_above_threshold(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.classify_combined_regime(0.601) == "extreme_risk_on"
        assert integrator.classify_combined_regime(0.301) == "risk_on"

    def test_classify_regime_just_below_threshold(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.classify_combined_regime(-0.601) == "extreme_risk_off"
        assert integrator.classify_combined_regime(-0.301) == "risk_off"

    def test_circuit_breaker_all_levels(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.determine_circuit_breaker(1.0) == "green"
        assert integrator.determine_circuit_breaker(0.0) == "yellow"
        assert integrator.determine_circuit_breaker(-0.3) == "orange"
        assert integrator.determine_circuit_breaker(-1.0) == "red"

    def test_circuit_breaker_boundaries(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.determine_circuit_breaker(0.2001) == "green"
        assert integrator.determine_circuit_breaker(0.1999) == "yellow"
        assert integrator.determine_circuit_breaker(-0.1999) == "yellow"
        assert integrator.determine_circuit_breaker(-0.2001) == "orange"
        assert integrator.determine_circuit_breaker(-0.4999) == "orange"
        assert integrator.determine_circuit_breaker(-0.5001) == "red"

    def test_position_scaling_all_regimes(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.calculate_position_scaling("extreme_risk_on") == 1.0
        assert integrator.calculate_position_scaling("risk_on") == 0.95
        assert integrator.calculate_position_scaling("neutral") == 0.85
        assert integrator.calculate_position_scaling("risk_off") == 0.70
        assert integrator.calculate_position_scaling("extreme_risk_off") == 0.50

    def test_position_scaling_monotonic(self):
        integrator = RegimeSentimentIntegrator()
        scales = [
            integrator.calculate_position_scaling("extreme_risk_on"),
            integrator.calculate_position_scaling("risk_on"),
            integrator.calculate_position_scaling("neutral"),
            integrator.calculate_position_scaling("risk_off"),
            integrator.calculate_position_scaling("extreme_risk_off"),
        ]
        for i in range(len(scales) - 1):
            assert scales[i] >= scales[i + 1], (
                f"Scales not monotonic at index {i}: {scales}"
            )

    def test_allocation_tilts_score_zero(self):
        integrator = RegimeSentimentIntegrator()
        eq, bond, gold = integrator.calculate_allocation_tilts(0.0, "neutral")
        assert eq == 0.0
        assert bond == 0.0
        assert gold == 0.0

    def test_allocation_tilts_extreme_positive(self):
        integrator = RegimeSentimentIntegrator()
        eq, bond, gold = integrator.calculate_allocation_tilts(1.0, "extreme_risk_on")
        assert eq == 1.0
        assert bond == 0.3
        assert gold == 0.0

    def test_allocation_tilts_extreme_negative(self):
        integrator = RegimeSentimentIntegrator()
        eq, bond, gold = integrator.calculate_allocation_tilts(-1.0, "extreme_risk_off")
        assert eq == -1.0
        assert bond == -0.5
        assert gold == 0.7

    def test_allocation_tilts_score_neg_01_regime_neutral(self):
        integrator = RegimeSentimentIntegrator()
        eq, bond, gold = integrator.calculate_allocation_tilts(-0.1, "neutral")
        assert eq == pytest.approx(-0.15)
        assert bond == 0.0
        assert gold == 0.0

    def test_allocation_tilts_mild_negative_score_not_risk_off(self):
        integrator = RegimeSentimentIntegrator()
        # score=-0.3, regime=neutral → gold from score < -0.2 branch
        eq, bond, gold = integrator.calculate_allocation_tilts(-0.3, "neutral")
        assert eq == pytest.approx(-0.45)
        assert bond == 0.0
        assert gold == 0.4

    def test_allocation_tilts_equity_bounds(self):
        integrator = RegimeSentimentIntegrator()
        eq1, _, _ = integrator.calculate_allocation_tilts(0.67, "risk_on")
        assert eq1 == 1.0  # 0.67 * 1.5 = 1.005 → clip to 1.0
        eq2, _, _ = integrator.calculate_allocation_tilts(-0.67, "risk_off")
        assert eq2 == -1.0  # clip to -1.0


# ---------------------------------------------------------------------------
# NaN / Inf handling
# ---------------------------------------------------------------------------

class TestNanInfHandling:
    """Numeric stability with NaN and Inf inputs."""

    def test_adjust_weights_tech_nan(self):
        integrator = RegimeSentimentIntegrator()
        tech_w, sent_w = integrator.adjust_weights(float('nan'), 0.6)
        assert tech_w == 0.70
        assert sent_w == 0.30

    def test_adjust_weights_sent_nan(self):
        integrator = RegimeSentimentIntegrator()
        tech_w, sent_w = integrator.adjust_weights(0.4, float('nan'))
        # 0.4 < 0.5 is True, but nan > 0.7 is False → falls through
        assert tech_w == 0.70
        assert sent_w == 0.30

    def test_classify_regime_nan(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.classify_combined_regime(float('nan')) == "neutral"

    def test_classify_regime_inf(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.classify_combined_regime(float('inf')) == "extreme_risk_on"
        assert integrator.classify_combined_regime(float('-inf')) == "extreme_risk_off"

    def test_circuit_breaker_nan(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.determine_circuit_breaker(float('nan')) == "red"

    def test_circuit_breaker_inf(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.determine_circuit_breaker(float('inf')) == "green"
        assert integrator.determine_circuit_breaker(float('-inf')) == "red"

    def test_allocation_tilts_nan_score(self):
        integrator = RegimeSentimentIntegrator()
        import numpy as np
        eq, bond, gold = integrator.calculate_allocation_tilts(float('nan'), "neutral")
        assert np.isnan(eq)
        assert bond == 0.0
        assert gold == 0.0

    def test_map_sentiment_nan_key(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.map_sentiment_to_score(float('nan')) == 0.0

    def test_map_technical_nan_key(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.map_technical_to_score(float('nan')) == 0.0


# ---------------------------------------------------------------------------
# Wrong types / missing keys boundary conditions
# ---------------------------------------------------------------------------

class TestWrongTypesBoundary:
    """Methods called with unexpected types, missing keys."""

    def test_adjust_weights_strings_raises(self):
        integrator = RegimeSentimentIntegrator()
        with pytest.raises(TypeError):
            integrator.adjust_weights("low", "high")

    def test_map_sentiment_int_key(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.map_sentiment_to_score(42) == 0.0

    def test_map_technical_none_key(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.map_technical_to_score(None) == 0.0

    def test_position_scaling_none(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.calculate_position_scaling(None) == 0.85

    def test_position_scaling_empty_string(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.calculate_position_scaling("") == 0.85

    def test_position_scaling_case_sensitive(self):
        integrator = RegimeSentimentIntegrator()
        # Case mismatch → not in dict → defaults to 0.85
        assert integrator.calculate_position_scaling("RISK_ON") == 0.85
        assert integrator.calculate_position_scaling("Neutral") == 0.85

    def test_map_sentiment_case_sensitive(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.map_sentiment_to_score("Risk_On") == 0.0
        assert integrator.map_sentiment_to_score("NEUTRAL") == 0.0

    def test_map_technical_extra_regime(self):
        integrator = RegimeSentimentIntegrator()
        assert integrator.map_technical_to_score("unknown_label") == 0.0

    def test_combine_signals_empty_regime_labels(self):
        integrator = RegimeSentimentIntegrator()
        sentiment = _make_mock_sentiment("", 0.5)
        signal = integrator.combine_signals("", 0.5, sentiment)
        assert isinstance(signal, CombinedRegimeSignal)
        assert signal.technical_regime == ""
        assert signal.sentiment_regime == ""
        # Empty strings not in mapping → 0.0 scores → combined = 0.0
        assert signal.combined_score == 0.0
        assert signal.combined_regime == "neutral"

    def test_combine_signals_negative_confidence(self):
        integrator = RegimeSentimentIntegrator()
        sentiment = _make_mock_sentiment("neutral", -0.5)
        signal = integrator.combine_signals("bullish_momentum", -0.2, sentiment)
        assert isinstance(signal, CombinedRegimeSignal)
        # tech=-0.2 < 0.3 → heavy-sent branch: 0.4/0.6
        assert signal.technical_weight == 0.40
        assert signal.sentiment_weight == 0.60

    def test_combine_signals_confidence_gt_one(self):
        integrator = RegimeSentimentIntegrator()
        sentiment = _make_mock_sentiment("risk_on", 1.5)
        signal = integrator.combine_signals("bullish_momentum", 2.0, sentiment)
        assert isinstance(signal, CombinedRegimeSignal)
        assert signal.technical_weight == 0.70

    def test_combine_signals_sentiment_none(self):
        integrator = RegimeSentimentIntegrator()
        sentiment = _make_mock_sentiment(None, 0.5)
        signal = integrator.combine_signals("bullish_momentum", 0.8, sentiment)
        assert isinstance(signal, CombinedRegimeSignal)
        # None not in sentiment mapping → 0.0 score
        assert signal.combined_score == pytest.approx(0.7 * 0.7, abs=0.01)


# ---------------------------------------------------------------------------
# Combine signals — cross-product edge cases
# ---------------------------------------------------------------------------

class TestCombineSignalsCrossProduct:
    """Combine signals with various regime pairs and confidence levels."""

    def test_all_technical_regimes_produce_signal(self):
        integrator = RegimeSentimentIntegrator()
        sentiment = _make_mock_sentiment("neutral", 0.5)
        regimes = [
            "bullish_momentum", "neutral_trending", "volatile_chop",
            "bearish_momentum", "crisis", "recovery", "expansion", "contraction",
        ]
        for regime in regimes:
            signal = integrator.combine_signals(regime, 0.7, sentiment)
            assert isinstance(signal, CombinedRegimeSignal)
            assert -1 <= signal.combined_score <= 1

    def test_most_bullish_combination(self):
        integrator = RegimeSentimentIntegrator()
        sentiment = _make_mock_sentiment("risk_on", 1.0)
        signal = integrator.combine_signals("bullish_momentum", 1.0, sentiment)
        expected = 0.7 * 0.7 + 0.3 * 0.5  # 0.49 + 0.15 = 0.64
        assert signal.combined_score == pytest.approx(expected, abs=0.01)

    def test_most_bearish_combination(self):
        integrator = RegimeSentimentIntegrator()
        sentiment = _make_mock_sentiment("extreme_risk_off", 1.0)
        signal = integrator.combine_signals("crisis", 1.0, sentiment)
        expected = 0.7 * (-0.8) + 0.3 * (-0.8)  # -0.80
        assert signal.combined_score == pytest.approx(expected, abs=0.01)

    def test_opposing_signals_neutralize(self):
        integrator = RegimeSentimentIntegrator()
        sentiment = _make_mock_sentiment("risk_on", 0.8)
        signal = integrator.combine_signals("bearish_momentum", 0.4, sentiment)
        # tech=-0.5, sent=0.5, weights 0.5/0.5 (tech low boost)
        expected = 0.5 * (-0.5) + 0.5 * 0.5  # 0.0
        assert signal.combined_score == pytest.approx(expected, abs=0.01)
        assert signal.combined_regime == "neutral"

    def test_sentiment_weight_boosted_50_50(self):
        integrator = RegimeSentimentIntegrator()
        sentiment = _make_mock_sentiment("risk_on", 0.8)
        signal = integrator.combine_signals("volatile_chop", 0.4, sentiment)
        # tech=0.4 < 0.5, sent=0.8 > 0.7 → 50/50 split
        assert signal.technical_weight == 0.50
        assert signal.sentiment_weight == 0.50

    def test_sentiment_weight_boosted_40_60(self):
        integrator = RegimeSentimentIntegrator()
        sentiment = _make_mock_sentiment("risk_on", 0.5)
        signal = integrator.combine_signals("volatile_chop", 0.2, sentiment)
        # tech=0.2 < 0.3 → 40/60 split
        assert signal.technical_weight == 0.40
        assert signal.sentiment_weight == 0.60

    def test_weights_normalized_in_init(self):
        integrator = RegimeSentimentIntegrator(technical_weight=0.6, sentiment_weight=0.6)
        total = integrator.technical_weight + integrator.sentiment_weight
        assert total == pytest.approx(1.0)

    def test_normalized_weights_used_in_combine(self):
        integrator = RegimeSentimentIntegrator(technical_weight=0.6, sentiment_weight=0.6)
        sentiment = _make_mock_sentiment("neutral", 0.5)
        signal = integrator.combine_signals("bullish_momentum", 0.8, sentiment)
        assert signal.technical_weight == pytest.approx(0.5)
        assert signal.sentiment_weight == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Numeric stability — rounding, precision
# ---------------------------------------------------------------------------

class TestNumericStability:
    """Rounding and precision in combine_signals output."""

    def test_combined_score_rounded_four_decimals(self):
        integrator = RegimeSentimentIntegrator()
        sentiment = _make_mock_sentiment("risk_on", 0.8)
        signal = integrator.combine_signals("expansion", 0.7, sentiment)
        score_str = f"{signal.combined_score:.10f}"
        # Verify precision by checking roundtrip
        assert signal.combined_score == round(signal.combined_score, 4)

    def test_weights_sum_to_one_after_rounding(self):
        integrator = RegimeSentimentIntegrator()
        sentiment = _make_mock_sentiment("neutral", 0.5)
        signal = integrator.combine_signals("volatile_chop", 0.5, sentiment)
        assert signal.technical_weight + signal.sentiment_weight == pytest.approx(1.0, abs=0.001)

    def test_allocation_weights_rounded_four(self):
        pipeline = RegimeSentimentPipeline.__new__(RegimeSentimentPipeline)
        pipeline.integrator = RegimeSentimentIntegrator()
        signal = _make_signal(equity_tilt=0.3, gold_tilt=0.2, bond_duration_tilt=-0.1)
        weights = pipeline.get_current_allocation_weights(signal)
        for symbol, weight in weights.items():
            assert weight == round(weight, 4), f"{symbol} weight not rounded: {weight}"


# ---------------------------------------------------------------------------
# Pipeline edge cases
# ---------------------------------------------------------------------------

class TestPipelineEdgeCases:
    """Edge cases for RegimeSentimentPipeline methods."""

    def test_allocation_zero_base_clamped_to_minimums(self):
        pipeline = RegimeSentimentPipeline.__new__(RegimeSentimentPipeline)
        pipeline.integrator = RegimeSentimentIntegrator()
        signal = _make_signal(equity_tilt=0.0, gold_tilt=0.0, bond_duration_tilt=0.0)
        base = {"SPY": 0.0, "GLD": 0.0, "TLT": 0.0}
        weights = pipeline.get_current_allocation_weights(signal, base_allocation=base)
        # np.clip clamps to [0.20, 0.70], [0.20, 0.50], [0.05, 0.25]
        # So SPY=0.2, GLD=0.2, TLT=0.05 → total=0.45 → normalized
        assert weights["SPY"] == pytest.approx(0.2 / 0.45, abs=0.01)
        assert weights["GLD"] == pytest.approx(0.2 / 0.45, abs=0.01)
        assert weights["TLT"] == pytest.approx(0.05 / 0.45, abs=0.01)
        assert sum(weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_allocation_negative_base_still_normalizes(self):
        pipeline = RegimeSentimentPipeline.__new__(RegimeSentimentPipeline)
        pipeline.integrator = RegimeSentimentIntegrator()
        signal = _make_signal(equity_tilt=0.0, gold_tilt=0.0, bond_duration_tilt=0.0)
        base = {"SPY": -0.1, "GLD": 0.5, "TLT": 0.6}
        weights = pipeline.get_current_allocation_weights(signal, base_allocation=base)
        assert sum(weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_allocation_missing_keys_default_to_zero(self):
        pipeline = RegimeSentimentPipeline.__new__(RegimeSentimentPipeline)
        pipeline.integrator = RegimeSentimentIntegrator()
        signal = _make_signal(equity_tilt=0.0, gold_tilt=0.0, bond_duration_tilt=0.0)
        base = {"SPY": 0.5}
        weights = pipeline.get_current_allocation_weights(signal, base_allocation=base)
        # GLD: clip(0 + 0, 0.2, 0.5) = 0.2
        # TLT: clip(0 + 0, 0.05, 0.25) = 0.05
        assert weights["SPY"] == pytest.approx(0.5 / 0.75, abs=0.01)
        assert weights["GLD"] == pytest.approx(0.2 / 0.75, abs=0.01)
        assert weights["TLT"] == pytest.approx(0.05 / 0.75, abs=0.01)

    def test_allocation_negative_tilt_clamped_lower(self):
        pipeline = RegimeSentimentPipeline.__new__(RegimeSentimentPipeline)
        pipeline.integrator = RegimeSentimentIntegrator()
        signal = _make_signal(equity_tilt=-1.0, gold_tilt=-1.0, bond_duration_tilt=-1.0)
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        weights = pipeline.get_current_allocation_weights(signal, base_allocation=base)
        assert weights["SPY"] >= 0.20
        assert weights["GLD"] >= 0.20
        assert weights["TLT"] >= 0.05

    def test_save_signal_custom_filename(self, tmp_path):
        pipeline = RegimeSentimentPipeline(data_dir=tmp_path)
        signal = _make_signal()
        filepath = pipeline.save_signal(signal, filename="custom.json")
        assert filepath.name == "custom.json"
        assert filepath.exists()
        with open(filepath) as f:
            data = json.load(f)
        assert data["combined_regime"] == "risk_on"

    def test_save_signal_overwrites(self, tmp_path):
        pipeline = RegimeSentimentPipeline(data_dir=tmp_path)
        pipeline.save_signal(_make_signal(combined_score=0.5), filename="test.json")
        pipeline.save_signal(_make_signal(combined_score=-0.5), filename="test.json")
        filepath = tmp_path / "test.json"
        with open(filepath) as f:
            data = json.load(f)
        assert data["combined_score"] == -0.5

    def test_save_signal_default_filename(self, tmp_path):
        pipeline = RegimeSentimentPipeline(data_dir=tmp_path)
        signal = _make_signal()
        filepath = pipeline.save_signal(signal)
        assert filepath.suffix == ".json"
        assert filepath.name.startswith("regime_signal_")

    def test_init_data_dir_created(self, tmp_path):
        new_dir = tmp_path / "nested" / "subdir"
        pipeline = RegimeSentimentPipeline(data_dir=new_dir)
        assert new_dir.exists()
        assert pipeline.data_dir == new_dir


# ---------------------------------------------------------------------------
# Pipeline.get_combined_signal — with mocked sentiment pipeline
# ---------------------------------------------------------------------------

class TestGetCombinedSignal:
    """Test pipeline.get_combined_signal with mocked SentimentAnalyzerPipeline."""

    def test_get_combined_signal_no_texts(self):
        pipeline = RegimeSentimentPipeline.__new__(RegimeSentimentPipeline)
        pipeline.integrator = RegimeSentimentIntegrator()
        pipeline.sentiment_pipeline = MagicMock()
        pipeline.sentiment_pipeline.get_current_sentiment.return_value = (
            _make_mock_sentiment("neutral", 0.5)
        )
        signal = pipeline.get_combined_signal("bullish_momentum", 0.8)
        assert isinstance(signal, CombinedRegimeSignal)
        pipeline.sentiment_pipeline.get_current_sentiment.assert_called_once_with(
            news_texts=None, earnings_texts=None, macro_texts=None
        )

    def test_get_combined_signal_with_all_texts(self):
        pipeline = RegimeSentimentPipeline.__new__(RegimeSentimentPipeline)
        pipeline.integrator = RegimeSentimentIntegrator()
        pipeline.sentiment_pipeline = MagicMock()
        pipeline.sentiment_pipeline.get_current_sentiment.return_value = (
            _make_mock_sentiment("risk_on", 0.8)
        )
        signal = pipeline.get_combined_signal(
            "bullish_momentum", 0.8,
            news_texts=["good news"],
            earnings_texts=["great earnings"],
            macro_texts=["strong economy"],
        )
        assert isinstance(signal, CombinedRegimeSignal)
        pipeline.sentiment_pipeline.get_current_sentiment.assert_called_once_with(
            news_texts=["good news"],
            earnings_texts=["great earnings"],
            macro_texts=["strong economy"],
        )

    def test_get_combined_signal_crisis_regime(self):
        pipeline = RegimeSentimentPipeline.__new__(RegimeSentimentPipeline)
        pipeline.integrator = RegimeSentimentIntegrator()
        pipeline.sentiment_pipeline = MagicMock()
        pipeline.sentiment_pipeline.get_current_sentiment.return_value = (
            _make_mock_sentiment("extreme_risk_off", 0.9)
        )
        signal = pipeline.get_combined_signal("crisis", 0.9)
        assert signal.combined_score < 0
        assert signal.combined_regime in ("risk_off", "extreme_risk_off")

    def test_get_combined_signal_both_agree_crisis(self):
        pipeline = RegimeSentimentPipeline.__new__(RegimeSentimentPipeline)
        pipeline.integrator = RegimeSentimentIntegrator()
        pipeline.sentiment_pipeline = MagicMock()
        pipeline.sentiment_pipeline.get_current_sentiment.return_value = (
            _make_mock_sentiment("extreme_risk_off", 0.9)
        )
        signal = pipeline.get_combined_signal("crisis", 0.9)
        expected = 0.7 * (-0.8) + 0.3 * (-0.8)
        assert signal.combined_score == pytest.approx(expected, abs=0.01)
        assert signal.combined_regime == "extreme_risk_off"
        assert signal.circuit_breaker_level == "red"
        assert signal.position_scaling_factor == 0.5
        assert signal.equity_tilt < 0
        assert signal.bond_duration_tilt == -0.5
        assert signal.gold_tilt == 0.7

    def test_get_combined_signal_partial_texts(self):
        pipeline = RegimeSentimentPipeline.__new__(RegimeSentimentPipeline)
        pipeline.integrator = RegimeSentimentIntegrator()
        pipeline.sentiment_pipeline = MagicMock()
        pipeline.sentiment_pipeline.get_current_sentiment.return_value = (
            _make_mock_sentiment("risk_off", 0.6)
        )
        signal = pipeline.get_combined_signal(
            "bearish_momentum", 0.7,
            news_texts=["bad news"],
        )
        assert isinstance(signal, CombinedRegimeSignal)
        pipeline.sentiment_pipeline.get_current_sentiment.assert_called_once_with(
            news_texts=["bad news"], earnings_texts=None, macro_texts=None
        )


# ---------------------------------------------------------------------------
# CLI / __main__ guard
# ---------------------------------------------------------------------------

class TestCliGuard:
    """Test __main__ guard and demo function entry point."""

    def test_main_guard_present_in_source(self):
        import inspect
        from src.strategy import regime_sentiment as mod
        source = inspect.getsource(mod)
        assert 'if __name__ == "__main__":' in source
        assert "demo()" in source

    def test_demo_returns_signal_with_mocks(self):
        from src.strategy.regime_sentiment import demo
        with patch(
            "src.strategy.regime_sentiment.RegimeSentimentPipeline.get_combined_signal"
        ) as mock_get:
            mock_get.return_value = _make_signal()
            with patch(
                "src.strategy.regime_sentiment.RegimeSentimentPipeline.save_signal"
            ) as mock_save:
                mock_save.return_value = Path("/tmp/dummy.json")
                result = demo()
                assert isinstance(result, CombinedRegimeSignal)

    def test_demo_logs_with_caplog(self, caplog):
        from src.strategy.regime_sentiment import demo
        with patch(
            "src.strategy.regime_sentiment.RegimeSentimentPipeline.get_combined_signal"
        ) as mock_get:
            mock_get.return_value = _make_signal()
            with patch(
                "src.strategy.regime_sentiment.RegimeSentimentPipeline.save_signal"
            ) as mock_save:
                mock_save.return_value = Path("/tmp/dummy.json")
                with caplog.at_level("INFO"):
                    result = demo()
                    assert isinstance(result, CombinedRegimeSignal)
                    assert len(caplog.records) > 0
                    info_messages = [r.message for r in caplog.records]
                    assert any("Portfolio-Lab" in msg for msg in info_messages)

    def test_main_guard_does_not_fire_on_import(self):
        """When imported normally, __main__ guard must not call demo()."""
        import src.strategy.regime_sentiment as mod
        with patch.object(mod, "demo") as mock_demo:
            # Re-import does not re-execute the guard because the module is cached
            import importlib
            importlib.reload(mod)
            mock_demo.assert_not_called()


# ---------------------------------------------------------------------------
# Export completeness — __all__ coverage
# ---------------------------------------------------------------------------

class TestExportCompleteness:
    """Verify __all__ exports are complete and accessible."""

    def test_all_exports_present(self):
        import src.strategy.regime_sentiment as mod
        for name in mod.__all__:
            assert hasattr(mod, name), f"Missing export: {name}"

    def test_all_count_is_four(self):
        import src.strategy.regime_sentiment as mod
        assert len(mod.__all__) == 4

    def test_all_contains_enum(self):
        import src.strategy.regime_sentiment as mod
        assert "RegimeSentiment" in mod.__all__

    def test_all_contains_dataclass(self):
        import src.strategy.regime_sentiment as mod
        assert "CombinedRegimeSignal" in mod.__all__

    def test_all_contains_integrator(self):
        import src.strategy.regime_sentiment as mod
        assert "RegimeSentimentIntegrator" in mod.__all__

    def test_all_contains_pipeline(self):
        import src.strategy.regime_sentiment as mod
        assert "RegimeSentimentPipeline" in mod.__all__

    def test_demo_not_in_all(self):
        import src.strategy.regime_sentiment as mod
        assert "demo" not in mod.__all__

    def test_logger_not_in_all(self):
        import src.strategy.regime_sentiment as mod
        assert "logger" not in mod.__all__

    def test_constants_not_in_all(self):
        """Internal constants are not part of the public API."""
        import src.strategy.regime_sentiment as mod
        for name in ("DEFAULT_TECHNICAL_WEIGHT", "DEFAULT_SENTIMENT_WEIGHT",
                      "CB_GREEN_THRESHOLD", "CB_YELLOW_THRESHOLD", "CB_ORANGE_THRESHOLD",
                      "EXTREME_RISK_ON_THRESHOLD", "RISK_ON_THRESHOLD",
                      "RISK_OFF_THRESHOLD", "EXTREME_RISK_OFF_THRESHOLD"):
            assert name not in mod.__all__
