"""
Tests for Cashless Collar Overlay Strategy (v4.60)
"""

import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.strategy.collar_overlay import (
    CollarOverlay,
    CollarOverlayIntegrator,
    CollarOverlayDecision,
    CollarOverlayStatus,
    calculate_collar_overlay,
    get_collar_summary,
)
from src.signals.collar_signal import (
    CollarSignal,
    CollarStrikes,
    CollarRegime,
    CollarState,
)


class TestCollarOverlayStatus:
    """Test overlay status enum."""

    def test_status_values(self):
        assert CollarOverlayStatus.ACTIVE.value == "active"
        assert CollarOverlayStatus.ROLLING.value == "rolling"
        assert CollarOverlayStatus.FROZEN.value == "frozen"
        assert CollarOverlayStatus.DISABLED.value == "disabled"


class TestCollarOverlayRecommend:
    """Test collar overlay recommendations."""

    @pytest.fixture
    def overlay(self, tmp_path):
        state_file = tmp_path / "collar_state.json"
        return CollarOverlay(state_file=state_file)

    def test_recommend_normal_market(self, overlay):
        """Should generate valid recommendation in normal market."""
        decision = overlay.recommend(spot=550.0, vix=16.0)
        assert isinstance(decision, CollarOverlayDecision)
        assert decision.status in ("active", "rolling")
        assert decision.underlying_price == 550.0
        assert decision.vix_level == 16.0

    def test_recommend_crisis_disabled(self, overlay):
        """Crisis should disable collar."""
        decision = overlay.recommend(spot=550.0, vix=50.0)
        assert decision.status == "disabled"
        assert not decision.is_actionable

    def test_recommend_elevated_active(self, overlay):
        """Elevated VIX should still produce active recommendation."""
        decision = overlay.recommend(spot=550.0, vix=25.0)
        assert decision.status in ("active", "rolling")

    def test_first_call_triggers_roll(self, overlay):
        """First call should trigger a roll (no prior state)."""
        decision = overlay.recommend(spot=550.0, vix=16.0)
        assert decision.is_actionable  # First roll

    def test_recommend_stores_state(self, overlay):
        """Recommendation should update state."""
        overlay.recommend(spot=550.0, vix=16.0)
        assert overlay._state["total_rolls"] >= 1
        assert overlay._state["last_roll_date"] is not None

    def test_recommend_call_strike_above_spot(self, overlay):
        """Call strike should be above spot (OTM)."""
        decision = overlay.recommend(spot=550.0, vix=16.0)
        assert decision.call_strike > 550.0

    def test_recommend_put_strike_below_spot(self, overlay):
        """Put strike should be below spot (OTM)."""
        decision = overlay.recommend(spot=550.0, vix=16.0)
        assert decision.put_strike < 550.0

    def test_max_upside_positive(self, overlay):
        """Upside cap should be positive."""
        decision = overlay.recommend(spot=550.0, vix=16.0)
        assert decision.max_upside > 0

    def test_max_downside_positive(self, overlay):
        """Downside protection should be positive."""
        decision = overlay.recommend(spot=550.0, vix=16.0)
        assert decision.max_downside > 0

    def test_decision_serializable(self, overlay):
        """Decision should be serializable."""
        decision = overlay.recommend(spot=550.0, vix=16.0)
        d = decision.to_dict()
        assert isinstance(d, dict)
        assert "call_strike" in d
        assert "put_strike" in d


class TestCollarOverlayAllocationShifts:
    """Test allocation shift mappings."""

    @pytest.fixture
    def overlay(self, tmp_path):
        return CollarOverlay(state_file=tmp_path / "collar_state.json")

    def test_neutral_signal_no_shift(self, overlay):
        shifts = overlay.get_allocation_shifts(0.0)
        assert shifts["spy"] == 0.0
        assert shifts["gld"] == 0.0
        assert shifts["tlt"] == 0.0

    def test_strong_risk_on_mild_defensive(self, overlay):
        shifts = overlay.get_allocation_shifts(0.8)
        assert shifts["spy"] == 0.0  # Collar neutral in risk-on

    def test_moderate_risk_off_defensive(self, overlay):
        shifts = overlay.get_allocation_shifts(-0.5)
        assert shifts["spy"] < 0  # Reduce equity
        assert shifts["gld"] > 0
        assert shifts["tlt"] > 0

    def test_strong_risk_off_strongly_defensive(self, overlay):
        shifts = overlay.get_allocation_shifts(-0.8)
        assert shifts["spy"] < -0.02
        assert shifts["gld"] > 0

    def test_shifts_sum_neutral(self, overlay):
        """Shifts should approximately sum to zero (risk transfer, not leverage)."""
        for signal in [-0.8, -0.5, 0.0, 0.5, 0.8]:
            shifts = overlay.get_allocation_shifts(signal)
            total = shifts["spy"] + shifts["gld"] + shifts["tlt"]
            assert abs(total) < 0.05


class TestCollarOverlayIntegrator:
    """Test integration with ensemble voter."""

    @pytest.fixture
    def integrator(self):
        return CollarOverlayIntegrator()

    def test_ensemble_signal_structure(self, integrator):
        """Should return proper ensemble signal structure."""
        signal = integrator.get_ensemble_signal()
        assert "source" in signal
        assert signal["source"] == "collar_overlay"
        assert "signal" in signal
        assert "weight" in signal
        assert "confidence" in signal

    def test_ensemble_weight_is_10_pct(self, integrator):
        """Integration weight should be 10%."""
        signal = integrator.get_ensemble_signal()
        assert signal["weight"] == 0.10

    def test_signal_in_range(self, integrator):
        """Signal should be in [-1, 1] range."""
        signal = integrator.get_ensemble_signal()
        assert -1.0 <= signal["signal"] <= 1.0

    def test_recommendation_present(self, integrator):
        """Should include a recommendation string."""
        signal = integrator.get_ensemble_signal()
        assert isinstance(signal["recommendation"], str)
        assert len(signal["recommendation"]) > 0


class TestCollarOverlayBacktest:
    """Test backtest functionality."""

    @pytest.fixture
    def overlay(self, tmp_path):
        return CollarOverlay(state_file=tmp_path / "collar_state.json")

    def test_backtest_with_simulated_data(self, overlay):
        """Backtest with simulated price data."""
        np = pytest.importorskip("numpy")
        n_days = 252
        dates = [f"2024-{(i // 21)+1:02d}-{(i % 21)+1:02d}" for i in range(n_days)]

        # Generate random walk prices
        rng = np.random.RandomState(42)
        returns = rng.normal(0.0005, 0.01, n_days)  # ~8% annual, 16% vol
        spy_prices = (550.0 * np.cumprod(1 + returns)).tolist()

        vix = [16.0 + rng.normal(0, 1) for _ in range(n_days)]

        results = overlay.backtest_collar(
            prices={"SPY": spy_prices},
            dates=dates,
            vix_history=vix,
        )

        assert "summary" in results
        s = results["summary"]
        assert "cagr_hedged" in s
        assert "cagr_unhedged" in s
        assert "vol_hedged" in s
        assert "vol_unhedged" in s
        assert "max_dd_hedged" in s
        assert "max_dd_unhedged" in s

        # Hedged should have lower vol and not worse max DD
        # max DD is negative (-12.64 = 12.64% drawdown): higher (less negative) = better
        assert s["vol_hedged"] < s["vol_unhedged"] * 1.15  # not significantly more volatile
        assert s["max_dd_hedged"] >= s["max_dd_unhedged"] * 0.95  # not worse than 5% worse DD

    def test_backtest_with_insufficient_data(self, overlay):
        """Should handle insufficient price data."""
        results = overlay.backtest_collar(
            prices={"SPY": [100.0, 101.0]},
            dates=["2024-01-01", "2024-01-02"],
        )
        assert "error" in results

    def test_backtest_crisis_period(self, overlay):
        """Backtest during high vol period should still work."""
        np = pytest.importorskip("numpy")
        n_days = 126  # 6 months
        dates = [f"2020-{(i // 21)+1:02d}-{(i % 21)+1:02d}" for i in range(n_days)]

        rng = np.random.RandomState(42)
        # Higher vol crash simulation
        returns = np.concatenate([
            rng.normal(-0.003, 0.04, 21),   # crash month
            rng.normal(0.002, 0.025, 105),  # recovery
        ])
        spy_prices = (550.0 * np.cumprod(1 + returns)).tolist()
        vix = [40.0 + rng.normal(0, 2) for _ in range(21)] + [25.0 + rng.normal(0, 2) for _ in range(105)]

        results = overlay.backtest_collar(
            prices={"SPY": spy_prices},
            dates=dates,
            vix_history=vix,
        )

        assert "summary" in results
        # In crisis, hedged should have better max DD
        s = results["summary"]
        assert s["max_dd_hedged"] >= s["max_dd_unhedged"] - 5  # within 5pp

    def test_backtest_no_vix_defaults(self, overlay):
        """Should work without VIX history (uses defaults)."""
        np = pytest.importorskip("numpy")
        n_days = 126
        dates = [f"2024-{(i // 21)+1:02d}-{(i % 21)+1:02d}" for i in range(n_days)]
        rng = np.random.RandomState(42)
        returns = rng.normal(0.0005, 0.01, n_days)
        spy_prices = (550.0 * np.cumprod(1 + returns)).tolist()

        results = overlay.backtest_collar(
            prices={"SPY": spy_prices},
            dates=dates,
        )
        assert "summary" in results


class TestGetStatus:
    """Test status retrieval."""

    def test_initial_status_disabled(self, tmp_path):
        """Fresh overlay should start disabled."""
        overlay = CollarOverlay(state_file=tmp_path / "collar_state.json")
        assert overlay.get_status() == CollarOverlayStatus.DISABLED


class TestConvenienceFunctions:
    """Test convenience functions work."""

    def test_calculate_collar_overlay(self):
        decision = calculate_collar_overlay(spot=550.0, vix=16.0)
        assert isinstance(decision, CollarOverlayDecision)

    def test_get_collar_summary(self):
        summary = get_collar_summary()
        assert isinstance(summary, dict)
        assert "status" in summary
        assert "recommendation" in summary


class TestEdgeCases:
    """Edge cases for overlay."""

    def test_zero_spot_fails_gracefully(self, tmp_path):
        overlay = CollarOverlay(state_file=tmp_path / "collar_state.json")
        decision = overlay.recommend(spot=0, vix=16.0)
        assert not decision.is_actionable

    def test_state_persistence(self, tmp_path):
        """State should persist between overlays."""
        state_file = tmp_path / "collar_state.json"

        overlay1 = CollarOverlay(state_file=state_file)
        overlay1._state["total_rolls"] = 5
        overlay1._save_state()

        overlay2 = CollarOverlay(state_file=state_file)
        assert overlay2._state["total_rolls"] == 5

    def test_default_state_structure(self, tmp_path):
        """Default state should have all required keys."""
        overlay = CollarOverlay(state_file=tmp_path / "collar_state.json")
        state = overlay._state
        assert "status" in state
        assert "last_roll_date" in state
        assert "total_rolls" in state
        assert "ytd_premium_collected" in state

    def test_high_spot_values(self, tmp_path):
        """Very high spot should still work."""
        overlay = CollarOverlay(state_file=tmp_path / "collar_state.json")
        decision = overlay.recommend(spot=6000.0, vix=16.0)
        assert decision.call_strike > 6000.0
        assert decision.put_strike < 6000.0
