"""
Tests for src/strategy/regime_router.py — RegimeRouter, RouterDecision, route_regime.
Mocks KurtosisRegimeSignalGenerator to isolate routing logic.
"""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import asdict

from src.strategy.regime_router import (
    RegimeRouter,
    RouterDecision,
    route_regime,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_signal(
    regime="low_kurtosis",
    ker_ratio=0.8,
    is_transitioning=False,
    strategy_preference="trend_following",
    tsom_weight=0.7,
    mr_weight=0.3,
    recommended_exposure=1.0,
    fat_tail_risk=0.1,
    confidence=85.0,
    explanation="Low kurtosis regime detected",
):
    """Create a mock KurtosisRegimeSignal with sensible defaults."""
    signal = MagicMock()
    signal.regime = regime
    signal.ker_ratio = ker_ratio
    signal.is_transitioning = is_transitioning
    signal.strategy_preference = strategy_preference
    signal.tsom_weight = tsom_weight
    signal.mr_weight = mr_weight
    signal.recommended_exposure = recommended_exposure
    signal.fat_tail_risk = fat_tail_risk
    signal.confidence = confidence
    signal.explanation = explanation
    return signal


def _patch_generate_signal(signal):
    """Return a patcher that makes generate_signal return `signal`."""
    return patch(
        "src.strategy.regime_router.KurtosisRegimeSignalGenerator.generate_signal",
        return_value=signal,
    )


# ── RouterDecision ───────────────────────────────────────────────────


class TestRouterDecision:
    def test_to_dict_returns_all_fields(self):
        d = RouterDecision(
            timestamp="2026-01-01T00:00:00",
            kurtosis_regime="low_kurtosis",
            ker_ratio=0.8,
            is_transitioning=False,
            tsom_weight=0.7,
            mr_weight=0.3,
            cash_weight=0.0,
            strategy_preference="trend_following",
            recommended_exposure=1.0,
            fat_tail_risk=0.1,
            confidence=85.0,
            explanation="test",
            is_actionable=True,
        )
        result = d.to_dict()
        assert result == asdict(d)
        assert set(result.keys()) == {
            "timestamp", "kurtosis_regime", "ker_ratio", "is_transitioning",
            "tsom_weight", "mr_weight", "cash_weight", "strategy_preference",
            "recommended_exposure", "fat_tail_risk", "confidence",
            "explanation", "is_actionable",
        }

    def test_to_dict_is_plain_dict(self):
        d = RouterDecision(
            timestamp="t", kurtosis_regime="r", ker_ratio=1.0,
            is_transitioning=False, tsom_weight=0.5, mr_weight=0.5,
            cash_weight=0.0, strategy_preference="balanced",
            recommended_exposure=1.0, fat_tail_risk=0.0,
            confidence=50.0, explanation="e", is_actionable=False,
        )
        assert isinstance(d.to_dict(), dict)


# ── RegimeRouter.route() — cash allocation ───────────────────────────


class TestCashAllocation:
    """Cash weight depends on strategy_preference and is_transitioning."""

    def test_defensive_gives_30pct_cash(self):
        signal = _make_signal(strategy_preference="defensive", tsom_weight=0.5, mr_weight=0.5)
        with _patch_generate_signal(signal):
            decision = RegimeRouter().route()
        assert decision.cash_weight == 0.3

    def test_transitioning_gives_10pct_cash(self):
        signal = _make_signal(
            strategy_preference="trend_following",
            is_transitioning=True,
            tsom_weight=0.7,
            mr_weight=0.3,
        )
        with _patch_generate_signal(signal):
            decision = RegimeRouter().route()
        assert decision.cash_weight == 0.1

    def test_normal_gives_zero_cash(self):
        signal = _make_signal(
            strategy_preference="trend_following",
            is_transitioning=False,
        )
        with _patch_generate_signal(signal):
            decision = RegimeRouter().route()
        assert decision.cash_weight == 0.0

    def test_mean_reversion_normal_gives_zero_cash(self):
        signal = _make_signal(
            strategy_preference="mean_reversion",
            is_transitioning=False,
        )
        with _patch_generate_signal(signal):
            decision = RegimeRouter().route()
        assert decision.cash_weight == 0.0

    def test_defensive_transitions_gives_30pct_cash(self):
        """Defensive takes precedence over transitioning."""
        signal = _make_signal(
            strategy_preference="defensive",
            is_transitioning=True,
            tsom_weight=0.5,
            mr_weight=0.5,
        )
        with _patch_generate_signal(signal):
            decision = RegimeRouter().route()
        assert decision.cash_weight == 0.3


# ── RegimeRouter.route() — TSMOM/MR weight scaling ──────────────────


class TestWeightScaling:
    """When cash > 0, TSMOM and MR weights are scaled down by (1 - cash)."""

    def test_no_scaling_when_zero_cash(self):
        signal = _make_signal(
            strategy_preference="trend_following",
            is_transitioning=False,
            tsom_weight=0.7,
            mr_weight=0.3,
        )
        with _patch_generate_signal(signal):
            decision = RegimeRouter().route()
        assert decision.tsom_weight == 0.7
        assert decision.mr_weight == 0.3

    def test_scaled_when_transitioning(self):
        signal = _make_signal(
            strategy_preference="trend_following",
            is_transitioning=True,
            tsom_weight=0.7,
            mr_weight=0.3,
        )
        with _patch_generate_signal(signal):
            decision = RegimeRouter().route()
        # cash = 0.1, scale = 0.9
        assert decision.tsom_weight == pytest.approx(0.7 * 0.9, abs=0.001)
        assert decision.mr_weight == pytest.approx(0.3 * 0.9, abs=0.001)

    def test_scaled_when_defensive(self):
        signal = _make_signal(
            strategy_preference="defensive",
            tsom_weight=0.5,
            mr_weight=0.5,
        )
        with _patch_generate_signal(signal):
            decision = RegimeRouter().route()
        # cash = 0.3, scale = 0.7
        assert decision.tsom_weight == pytest.approx(0.5 * 0.7, abs=0.001)
        assert decision.mr_weight == pytest.approx(0.5 * 0.7, abs=0.001)

    def test_weights_sum_to_less_than_one_with_cash(self):
        signal = _make_signal(
            strategy_preference="defensive",
            tsom_weight=0.6,
            mr_weight=0.4,
        )
        with _patch_generate_signal(signal):
            decision = RegimeRouter().route()
        total = decision.tsom_weight + decision.mr_weight + decision.cash_weight
        assert total == pytest.approx(1.0, abs=0.01)


# ── RegimeRouter.route() — actionable flag ───────────────────────────


class TestActionableFlag:
    """is_actionable = transitioning OR strategy != trend_following."""

    def test_trend_following_stable_not_actionable(self):
        signal = _make_signal(
            strategy_preference="trend_following",
            is_transitioning=False,
        )
        with _patch_generate_signal(signal):
            decision = RegimeRouter().route()
        assert decision.is_actionable is False

    def test_trend_following_transitioning_is_actionable(self):
        signal = _make_signal(
            strategy_preference="trend_following",
            is_transitioning=True,
        )
        with _patch_generate_signal(signal):
            decision = RegimeRouter().route()
        assert decision.is_actionable is True

    def test_mean_reversion_is_actionable(self):
        signal = _make_signal(
            strategy_preference="mean_reversion",
            is_transitioning=False,
        )
        with _patch_generate_signal(signal):
            decision = RegimeRouter().route()
        assert decision.is_actionable is True

    def test_defensive_is_actionable(self):
        signal = _make_signal(
            strategy_preference="defensive",
            is_transitioning=False,
            tsom_weight=0.5,
            mr_weight=0.5,
        )
        with _patch_generate_signal(signal):
            decision = RegimeRouter().route()
        assert decision.is_actionable is True

    def test_balanced_is_actionable(self):
        signal = _make_signal(
            strategy_preference="balanced",
            is_transitioning=False,
        )
        with _patch_generate_signal(signal):
            decision = RegimeRouter().route()
        assert decision.is_actionable is True


# ── RegimeRouter.route() — field pass-through ────────────────────────


class TestFieldPassthrough:
    """Router passes signal fields through to RouterDecision."""

    def test_kurtosis_regime_passed(self):
        signal = _make_signal(regime="high_kurtosis")
        with _patch_generate_signal(signal):
            decision = RegimeRouter().route()
        assert decision.kurtosis_regime == "high_kurtosis"

    def test_ker_ratio_passed(self):
        signal = _make_signal(ker_ratio=2.5)
        with _patch_generate_signal(signal):
            decision = RegimeRouter().route()
        assert decision.ker_ratio == 2.5

    def test_recommended_exposure_passed(self):
        signal = _make_signal(recommended_exposure=0.6)
        with _patch_generate_signal(signal):
            decision = RegimeRouter().route()
        assert decision.recommended_exposure == 0.6

    def test_fat_tail_risk_passed(self):
        signal = _make_signal(fat_tail_risk=0.8)
        with _patch_generate_signal(signal):
            decision = RegimeRouter().route()
        assert decision.fat_tail_risk == 0.8

    def test_confidence_passed(self):
        signal = _make_signal(confidence=92.0)
        with _patch_generate_signal(signal):
            decision = RegimeRouter().route()
        assert decision.confidence == 92.0

    def test_explanation_passed(self):
        signal = _make_signal(explanation="Extreme kurtosis detected")
        with _patch_generate_signal(signal):
            decision = RegimeRouter().route()
        assert decision.explanation == "Extreme kurtosis detected"

    def test_is_transitioning_passed(self):
        signal = _make_signal(is_transitioning=True)
        with _patch_generate_signal(signal):
            decision = RegimeRouter().route()
        assert decision.is_transitioning is True

    def test_timestamp_is_iso_format(self):
        signal = _make_signal()
        with _patch_generate_signal(signal):
            decision = RegimeRouter().route()
        assert "T" in decision.timestamp  # ISO format contains T


# ── RegimeRouter.route() — returns argument passthrough ──────────────


class TestReturnsPassthrough:
    """route() passes returns argument to generate_signal."""

    def test_none_returns_passed(self):
        signal = _make_signal()
        with _patch_generate_signal(signal) as mock:
            RegimeRouter().route(None)
        mock.assert_called_once_with(None)

    def test_list_returns_passed(self):
        signal = _make_signal()
        returns = [0.01, -0.02, 0.03]
        with _patch_generate_signal(signal) as mock:
            RegimeRouter().route(returns)
        mock.assert_called_once_with(returns)


# ── RegimeRouter.get_ensemble_signal() ───────────────────────────────


class TestGetEnsembleSignal:
    """Ensemble signal mapping from strategy_preference to value."""

    def test_trend_following_signal(self):
        signal = _make_signal(strategy_preference="trend_following")
        with _patch_generate_signal(signal):
            result = RegimeRouter().get_ensemble_signal()
        assert result["signal"] == 0.3
        assert result["source"] == "regime_router"
        assert result["weight"] == 0.05

    def test_mean_reversion_signal(self):
        signal = _make_signal(strategy_preference="mean_reversion")
        with _patch_generate_signal(signal):
            result = RegimeRouter().get_ensemble_signal()
        assert result["signal"] == -0.2

    def test_defensive_signal(self):
        signal = _make_signal(strategy_preference="defensive", tsom_weight=0.5, mr_weight=0.5)
        with _patch_generate_signal(signal):
            result = RegimeRouter().get_ensemble_signal()
        assert result["signal"] == -0.5

    def test_balanced_signal_is_zero(self):
        signal = _make_signal(strategy_preference="balanced")
        with _patch_generate_signal(signal):
            result = RegimeRouter().get_ensemble_signal()
        assert result["signal"] == 0.0

    def test_confidence_normalized(self):
        signal = _make_signal(confidence=80.0)
        with _patch_generate_signal(signal):
            result = RegimeRouter().get_ensemble_signal()
        assert result["confidence"] == pytest.approx(0.8)

    def test_tsom_mr_weights_included(self):
        signal = _make_signal(strategy_preference="trend_following", tsom_weight=0.7, mr_weight=0.3)
        with _patch_generate_signal(signal):
            result = RegimeRouter().get_ensemble_signal()
        assert result["tsom_weight"] == pytest.approx(0.7, abs=0.001)
        assert result["mr_weight"] == pytest.approx(0.3, abs=0.001)

    def test_recommendation_is_explanation(self):
        signal = _make_signal(explanation="Low vol regime")
        with _patch_generate_signal(signal):
            result = RegimeRouter().get_ensemble_signal()
        assert result["recommendation"] == "Low vol regime"

    def test_returns_passthrough(self):
        signal = _make_signal()
        returns = [0.01, -0.01]
        with _patch_generate_signal(signal):
            RegimeRouter().get_ensemble_signal(returns)
        # If it didn't crash, returns were passed through

    def test_ensemble_weight_constant(self):
        signal = _make_signal()
        with _patch_generate_signal(signal):
            result = RegimeRouter().get_ensemble_signal()
        assert result["weight"] == RegimeRouter.ENSEMBLE_WEIGHT


# ── route_regime() convenience function ──────────────────────────────


class TestRouteRegimeConvenience:
    def test_returns_router_decision(self):
        signal = _make_signal()
        with _patch_generate_signal(signal):
            result = route_regime()
        assert isinstance(result, RouterDecision)

    def test_passes_returns(self):
        signal = _make_signal()
        returns = [0.01, 0.02]
        with _patch_generate_signal(signal) as mock:
            route_regime(returns)
        mock.assert_called_once_with(returns)


# ── ENSEMBLE_WEIGHT constant ─────────────────────────────────────────


class TestEnsembleWeight:
    def test_weight_is_5pct(self):
        assert RegimeRouter.ENSEMBLE_WEIGHT == 0.05


# ── Weight rounding ──────────────────────────────────────────────────


class TestWeightRounding:
    def test_tsom_rounded_to_3_decimals(self):
        signal = _make_signal(tsom_weight=0.66666, mr_weight=0.33334, strategy_preference="trend_following")
        with _patch_generate_signal(signal):
            decision = RegimeRouter().route()
        # tsom_weight is rounded to 3 decimal places
        assert decision.tsom_weight == round(0.66666, 3)

    def test_cash_rounded_to_2_decimals(self):
        signal = _make_signal(strategy_preference="defensive", tsom_weight=0.5, mr_weight=0.5)
        with _patch_generate_signal(signal):
            decision = RegimeRouter().route()
        # cash_weight is rounded to 2 decimal places
        assert decision.cash_weight == round(0.3, 2)
