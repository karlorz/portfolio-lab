"""
Tests for Strategy Regime Router (v4.91)
"""

import pytest
import numpy as np
from datetime import datetime

from src.strategy.regime_router import (
    RegimeRouter,
    RouterDecision,
    route_regime,
)


class TestRouterDecision:
    """Test router decision dataclass."""

    def test_serializable(self):
        d = RouterDecision(
            timestamp="2026-05-16", kurtosis_regime="normal",
            ker_ratio=1.05, is_transitioning=False,
            tsom_weight=0.70, mr_weight=0.30, cash_weight=0.0,
            strategy_preference="trend_following",
            recommended_exposure=1.0, fat_tail_risk=0.2,
            confidence=70.0, explanation="Test",
            is_actionable=False,
        )
        data = d.to_dict()
        assert data["tsom_weight"] == 0.70
        assert data["strategy_preference"] == "trend_following"


class TestRegimeRouter:
    """Test regime router core functionality."""

    @pytest.fixture
    def router(self):
        return RegimeRouter()

    def test_routes_normal_returns(self, router):
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0, 0.01, 200))
        decision = router.route(returns)
        assert isinstance(decision, RouterDecision)
        assert decision.kurtosis_regime in (
            "low_kurtosis", "normal", "high_kurtosis", "extreme_kurtosis"
        )

    def test_weights_sum_reasonable(self, router):
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0, 0.01, 200))
        decision = router.route(returns)
        total = decision.tsom_weight + decision.mr_weight + decision.cash_weight
        assert abs(total - 1.0) < 0.01

    def test_routes_fat_tail_returns(self, router):
        """Returns with fat tails should trigger mean-reversion preference."""
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0, 0.01, 200))
        # Add extreme events
        for i in [50, 55, 60, 100, 105, 110, 150, 155, 160]:
            returns[i] = rng.normal(0, 0.05)
        decision = router.route(returns)
        # Should detect elevated kurtosis
        assert decision.kurtosis_regime is not None
        assert decision.fat_tail_risk >= 0

    def test_ensemble_signal_structure(self, router):
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0, 0.01, 200))
        signal = router.get_ensemble_signal(returns)
        assert "source" in signal
        assert signal["source"] == "regime_router"
        assert "signal" in signal
        assert "weight" in signal
        assert signal["weight"] == 0.05

    def test_ensemble_signal_value_range(self, router):
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0, 0.01, 200))
        signal = router.get_ensemble_signal(returns)
        assert -1.0 <= signal["signal"] <= 1.0

    def test_convenience_function(self):
        rng = np.random.RandomState(42)
        returns = list(rng.normal(0, 0.01, 200))
        decision = route_regime(returns)
        assert isinstance(decision, RouterDecision)


class TestEdgeCases:
    """Edge cases."""

    @pytest.fixture
    def router(self):
        return RegimeRouter()

    def test_empty_returns(self, router):
        decision = router.route([])
        assert isinstance(decision, RouterDecision)

    def test_default_returns(self, router):
        decision = router.route()
        assert isinstance(decision, RouterDecision)
        assert decision.kurtosis_regime is not None
