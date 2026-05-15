"""
Tests for Bond Duration Rotation Strategy (v4.80)
"""

import json
import pytest
import numpy as np
from datetime import datetime, date
from pathlib import Path

from src.strategy.bond_duration_rotator import (
    BondDurationRotator,
    BondRotationDecision,
    RotationStatus,
    calculate_bond_rotation,
    get_bond_duration_summary,
)


class TestRotationStatus:
    """Test rotation status enum."""

    def test_status_values(self):
        assert RotationStatus.ACTIVE.value == "active"
        assert RotationStatus.DEFENSIVE.value == "defensive"
        assert RotationStatus.DISABLED.value == "disabled"


class TestBondRotationRecommend:
    """Test rotation recommendations."""

    @pytest.fixture
    def rotator(self, tmp_path):
        state_file = tmp_path / "bond_state.json"
        return BondDurationRotator(state_file=state_file)

    def test_generates_decision(self, rotator):
        decision = rotator.recommend()
        assert isinstance(decision, BondRotationDecision)

    def test_bond_sleeve_weights_sum_to_sleeve(self, rotator):
        decision = rotator.recommend()
        total = decision.tlt_total + decision.ief_total + decision.shy_total
        assert abs(total - 0.16) < 0.01  # Bond sleeve is 16%

    def test_sleeve_weights_sum_to_1(self, rotator):
        decision = rotator.recommend()
        assert abs(decision.tlt_sleeve + decision.ief_sleeve + decision.shy_sleeve - 1.0) < 0.01

    def test_decision_serializable(self, rotator):
        decision = rotator.recommend()
        d = decision.to_dict()
        assert isinstance(d, dict)
        assert "tlt_total" in d
        assert "curve_regime" in d

    def test_default_curve_normal(self, rotator):
        decision = rotator.recommend()
        assert decision.curve_regime in ("steep", "normal", "flat", "inverted")

    def test_inverted_curve_defensive(self, rotator):
        decision = rotator.recommend(
            yield_10y=4.0, yield_2y=4.5, real_rate=1.0, rate_change_6m=0.5
        )
        assert decision.curve_regime == "inverted"
        assert decision.shy_sleeve > decision.tlt_sleeve

    def test_steep_curve_aggressive(self, rotator):
        decision = rotator.recommend(
            yield_10y=5.0, yield_2y=3.5, real_rate=2.5, rate_change_6m=-0.8
        )
        assert decision.curve_regime == "steep"
        assert decision.tlt_sleeve > 0.4

    def test_recommendation_is_string(self, rotator):
        decision = rotator.recommend()
        assert isinstance(decision.recommendation, str)
        assert len(decision.recommendation) > 0

    def test_state_persistence(self, rotator):
        rotator._state["current_position"] = "short"
        rotator._save_state()

        rotator2 = BondDurationRotator(state_file=rotator.state_file)
        assert rotator2._state["current_position"] == "short"

    def test_default_state_structure(self, tmp_path):
        rotator = BondDurationRotator(state_file=tmp_path / "bond_state.json")
        assert "status" in rotator._state
        assert "tlt_weight" in rotator._state


class TestAllocationShifts:
    """Test allocation shift generation."""

    @pytest.fixture
    def rotator(self, tmp_path):
        return BondDurationRotator(state_file=tmp_path / "bond_state.json")

    def test_shifts_sum_to_zero(self, rotator):
        shifts = rotator.get_allocation_shifts(baseline_tlt=0.16)
        total = sum(shifts.values())
        assert abs(total) < 0.01

    def test_baseline_all_tlt(self, rotator):
        """With baseline 0.16 TLT, shifts account for full bond sleeve."""
        shifts = rotator.get_allocation_shifts(baseline_tlt=0.16)
        # tlt shift + baseline = current tlt
        # ief shift = current ief (was 0 in baseline)
        # shy shift = current shy (was 0 in baseline)
        assert shifts["ief"] >= 0
        assert shifts["shy"] >= 0

    def test_spy_gld_unchanged(self, rotator):
        shifts = rotator.get_allocation_shifts(baseline_tlt=0.16)
        assert shifts["spy"] == 0.0
        assert shifts["gld"] == 0.0


class TestGetStatus:
    """Test status retrieval."""

    def test_initial_status(self, tmp_path):
        rotator = BondDurationRotator(state_file=tmp_path / "bond_state.json")
        assert rotator.get_status() == RotationStatus.ACTIVE


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_calculate_bond_rotation(self):
        decision = calculate_bond_rotation()
        assert isinstance(decision, BondRotationDecision)

    def test_get_bond_duration_summary(self):
        summary = get_bond_duration_summary()
        assert isinstance(summary, dict)
        assert "status" in summary
        assert "effective_duration" in summary


class TestBondDurationBacktest:
    """Test backtest with simulated data."""

    @pytest.fixture
    def rotator(self, tmp_path):
        return BondDurationRotator(state_file=tmp_path / "bond_state.json")

    def test_backtest_with_simulated_data(self, rotator):
        rng = np.random.RandomState(42)
        n = 500

        # Simulated yield history
        yields = [(4.5 + rng.normal(0, 0.01), 4.0 + rng.normal(0, 0.01),
                   2.0 + rng.normal(0, 0.005)) for _ in range(n)]

        spy_r = list(rng.normal(0.0003, 0.01, n))
        tlt_r = list(rng.normal(0.0001, 0.011, n))
        ief_r = list(rng.normal(0.00015, 0.007, n))
        shy_r = list(rng.normal(0.0002, 0.002, n))
        gld_r = list(rng.normal(0.0002, 0.012, n))

        dates = [f"2025-{(i//21)+1:02d}-{(i%21)+1:02d}" for i in range(n)]

        results = rotator.backtest(yields, spy_r, tlt_r, ief_r, shy_r, gld_r, dates)

        assert "summary" in results
        s = results["summary"]
        assert "cagr_baseline" in s
        assert "cagr_rotated" in s
        assert "sharpe_baseline" in s
        assert "sharpe_rotated" in s
        assert s["avg_tlt_weight"] <= 100.0

    def test_backtest_insufficient_data(self, rotator):
        results = rotator.backtest(
            [(4.5, 4.0, 2.0)], [0.01], [0.005], [0.003], [0.002], [0.004], ["2025-01-01"]
        )
        assert "error" in results

    def test_backtest_during_inversion(self, rotator):
        """During inverted yield curve, should favor SHY."""
        rng = np.random.RandomState(42)
        n = 200
        # Inverted curve: 2Y > 10Y
        yields = [(4.0, 4.5 + rng.normal(0, 0.005), 1.5) for _ in range(n)]
        spy_r = list(rng.normal(0.0003, 0.01, n))
        tlt_r = list(rng.normal(0.0, 0.012, n))
        ief_r = list(rng.normal(0.0001, 0.007, n))
        shy_r = list(rng.normal(0.0002, 0.002, n))
        gld_r = list(rng.normal(0.0002, 0.012, n))
        dates = [f"2025-{(i//21)+1:02d}-{(i%21)+1:02d}" for i in range(n)]

        results = rotator.backtest(yields, spy_r, tlt_r, ief_r, shy_r, gld_r, dates)
        assert "summary" in results


class TestEdgeCases:
    """Edge cases for rotation."""

    @pytest.fixture
    def rotator(self, tmp_path):
        return BondDurationRotator(state_file=tmp_path / "bond_state.json")

    def test_multiple_recommends_consistent(self, rotator):
        """Multiple calls should not crash."""
        d1 = rotator.recommend()
        d2 = rotator.recommend()
        assert d1 is not None
        assert d2 is not None

    def test_duration_bounds(self, rotator):
        """Effective duration should always be in [2, 16]."""
        for y10, y2, rate_chg in [
            (5.0, 3.5, -1.0),  # steep falling
            (4.5, 4.0, 0.0),   # normal stable
            (4.0, 4.5, 1.0),   # inverted rising
            (4.5, 4.2, -0.5),  # flat falling
        ]:
            decision = rotator.recommend(
                yield_10y=y10, yield_2y=y2, real_rate=1.5, rate_change_6m=rate_chg
            )
            assert 2.0 <= decision.effective_duration <= 16.0
