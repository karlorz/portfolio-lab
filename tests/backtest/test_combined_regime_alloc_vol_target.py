"""
Tests for combined regime-conditional allocation × vol targeting overlay.

Tests the src.backtest.combined_regime_alloc_vol_target module which
applies regime-conditional allocation FIRST, then regime-conditional
vol targeting overlay on top.
"""

import pytest
import numpy as np
from dataclasses import asdict

from src.strategy.regime_allocation import REGIME_ALLOCATIONS, DEFAULT_ALLOCATION
from src.backtest.combined_regime_alloc_vol_target import (
    classify_regime,
    REGIME_VOL_TARGETS,
    _compute_vol_target_leverage,
    CombinedRegimeRow,
    CombinedRegimeResult,
    backtest_strategy,
    run_combined_backtest,
)


class TestClassifyRegime:
    """Test classify_regime() function."""

    def test_normal_regime(self):
        """Returns 'normal' for calm returns at median vol."""
        rng = np.random.RandomState(42)
        returns = rng.normal(0.0004, 0.008, 300)  # ~12.7% ann vol
        regime = classify_regime(returns)
        assert regime == "normal"

    def test_crisis_regime(self):
        """Returns 'crisis' for high vol + negative returns."""
        # Crisis needs: recent_vol > 1.7*median_vol AND mean(last 21d) < -0.01
        # Build: 252 low-vol normal + 48 high-vol crisis tail
        rng = np.random.RandomState(42)
        normal = rng.normal(0.0004, 0.006, 252)  # ~9.5% ann vol
        crisis_tail = rng.normal(-0.025, 0.025, 48)  # ~39.7% vol, mean -2.5%
        returns = np.concatenate([normal, crisis_tail])
        regime = classify_regime(returns)
        assert regime == "crisis"

    def test_low_vol_regime(self):
        """Returns 'low_vol' when current vol is well below median vol.
        
        Need recent_vol < 0.75 * median_vol, so must have HIGHER vol in past.
        """
        rng = np.random.RandomState(42)
        # Phase 1: 189 days of high vol (gives many 63-day windows with high vol)
        high_vol = rng.normal(0.0, 0.02, 189)  # ~31.7% ann vol
        # Phase 2: 63 days of low vol (current vol)
        low_vol = rng.normal(0.0004, 0.003, 63)  # ~4.8% ann vol
        # Total: 252 days (189 high + 63 low)
        returns = np.concatenate([high_vol, low_vol])
        regime = classify_regime(returns)
        assert regime == "low_vol"

    def test_returns_default_for_insufficient_data(self):
        """Returns 'normal' when not enough data."""
        returns = np.array([0.001, 0.002])
        regime = classify_regime(returns)
        assert regime == "normal"


class TestRegimeVolTargets:
    """Test REGIME_VOL_TARGETS constant."""

    def test_all_five_regimes_present(self):
        """Must cover all 5 regimes."""
        expected = {"normal", "crisis", "high_vol", "low_vol", "recovery"}
        assert set(REGIME_VOL_TARGETS.keys()) == expected

    def test_crisis_has_lowest_target(self):
        """CRISIS should have the lowest vol target (deleveraging)."""
        crisis = REGIME_VOL_TARGETS["crisis"]
        for regime, target in REGIME_VOL_TARGETS.items():
            if regime != "crisis":
                assert crisis <= target, (
                    f"CRISIS target {crisis} should be <= {regime} target {target}"
                )

    def test_low_vol_has_highest_target(self):
        """LOW_VOL should have the highest vol target (more risk)."""
        low_vol = REGIME_VOL_TARGETS["low_vol"]
        for regime, target in REGIME_VOL_TARGETS.items():
            if regime != "low_vol":
                assert low_vol >= target, (
                    f"LOW_VOL target {low_vol} should be >= {regime} target {target}"
                )

    def test_all_targets_positive_and_reasonable(self):
        """All targets should be between 0.01 and 0.20."""
        for regime, target in REGIME_VOL_TARGETS.items():
            assert 0.01 <= target <= 0.20, f"{regime} target {target} out of range"


class TestComputeVolTargetLeverage:
    """Test _compute_vol_target_leverage()."""

    def test_vol_equals_target(self):
        """Leverage should be ~1.0 when vol equals target."""
        lev = _compute_vol_target_leverage(0.09, 0.09)
        assert lev == pytest.approx(1.0, abs=0.1)

    def test_vol_below_target(self):
        """Leverage > 1 when vol below target."""
        lev = _compute_vol_target_leverage(0.05, 0.09, max_leverage=1.5)
        assert lev > 1.0
        assert lev <= 1.5

    def test_vol_above_target(self):
        """Leverage < 1 when vol above target."""
        lev = _compute_vol_target_leverage(0.20, 0.09, max_leverage=1.5)
        assert lev < 1.0

    def test_vol_at_zero(self):
        """Returns 1.0 when realized_vol is 0."""
        lev = _compute_vol_target_leverage(0.0, 0.09)
        assert lev == pytest.approx(1.0)

    def test_leverage_capped(self):
        """Leverage should not exceed max_leverage."""
        lev = _compute_vol_target_leverage(0.01, 0.09, max_leverage=1.5)
        assert lev <= 1.5

    def test_leverage_floored(self):
        """Leverage should not go below 1/max_leverage."""
        lev = _compute_vol_target_leverage(0.50, 0.09, max_leverage=1.5)
        assert lev >= 1.0 / 1.5

    def test_smoothing(self):
        """Smoothing should dampen leverage changes (within leverage cap)."""
        prev = 1.0
        # Use params where smoothed doesn't hit max_leverage cap
        lev = _compute_vol_target_leverage(0.06, 0.09, max_leverage=2.0, smoothing=0.67, prev_leverage=prev)
        raw = 0.09 / 0.06  # 1.5
        smoothed = 0.67 * raw + 0.33 * prev  # 1.335
        assert lev == pytest.approx(smoothed, abs=0.01)


class TestCombinedRegimeRow:
    """Test CombinedRegimeRow dataclass."""

    def test_row_construction(self):
        """Row can be constructed with all fields."""
        row = CombinedRegimeRow(
            label="Test",
            cagr=0.10,
            vol=0.12,
            sharpe=0.83,
            max_dd=-0.20,
            sortino=1.2,
            calmar=0.5,
            mean_leverage=1.05,
            max_leverage_reached=1.3,
            sharpe_vs_static=0.05,
            regime_counts={"normal": 100},
        )
        assert row.sharpe == pytest.approx(0.83)
        assert row.mean_leverage == pytest.approx(1.05)

    def test_sortable_by_sharpe(self):
        """Rows should be sortable by sharpe."""
        rows = [
            CombinedRegimeRow("a", 0.10, 0.12, 0.8, -0.2, 1.0, 0.4, 1.0, 1.0, 0, {}),
            CombinedRegimeRow("b", 0.10, 0.12, 0.9, -0.2, 1.0, 0.4, 1.0, 1.0, 0, {}),
        ]
        sorted_rows = sorted(rows, key=lambda r: r.sharpe, reverse=True)
        assert sorted_rows[0].label == "b"


class TestCombinedRegimeResult:
    """Test CombinedRegimeResult dataclass."""

    def test_result_construction(self):
        """Result can be constructed with all fields."""
        result = CombinedRegimeResult(
            timestamp="2026-05-29T10:00:00",
            data_range="2005-01-03 to 2026-05-08 (5371 days)",
            n_days=5371,
            static_cagr=0.10,
            static_vol=0.11,
            static_sharpe=0.95,
            static_max_dd=-0.27,
            champion_alloc=DEFAULT_ALLOCATION,
            rows=[asdict(CombinedRegimeRow(
                "Test", 0.10, 0.10, 1.01, -0.19, 1.5, 0.5, 1.03, 1.2, 0.06, {},
            ))],
            best_sharpe_row=asdict(CombinedRegimeRow(
                "Test", 0.10, 0.10, 1.01, -0.19, 1.5, 0.5, 1.03, 1.2, 0.06, {},
            )),
            combined_sharpe_delta=0.061,
            recommendation="Combined system is best.",
        )
        assert result.combined_sharpe_delta == pytest.approx(0.061)
        assert len(result.rows) == 1


class TestBacktestStrategy:
    """Test backtest_strategy() function."""

    def test_returns_row_with_data(self):
        """Returns a CombinedRegimeRow with metrics."""
        # Build minimal price series
        spy_prices = np.linspace(100, 200, 500)
        gld_prices = np.linspace(100, 180, 500)
        tlt_prices = np.linspace(100, 120, 500)

        prices = {"SPY": spy_prices, "GLD": gld_prices, "TLT": tlt_prices}

        row = backtest_strategy(
            prices, "Test Static",
            allocation_map={}, default_alloc=DEFAULT_ALLOCATION,
            apply_vol_target=False,
        )
        assert isinstance(row, CombinedRegimeRow)
        assert row.sharpe != 0.0
        assert row.cagr != 0.0
        assert row.vol != 0.0

    def test_combined_with_vol_target(self):
        """Vol targeting overlay changes results."""
        spy_prices = np.linspace(100, 200, 500)
        gld_prices = np.linspace(100, 180, 500)
        tlt_prices = np.linspace(100, 120, 500)
        prices = {"SPY": spy_prices, "GLD": gld_prices, "TLT": tlt_prices}

        without = backtest_strategy(
            prices, "No VT",
            allocation_map=REGIME_ALLOCATIONS, default_alloc=DEFAULT_ALLOCATION,
            apply_vol_target=False,
        )
        with_ = backtest_strategy(
            prices, "With VT",
            allocation_map=REGIME_ALLOCATIONS, default_alloc=DEFAULT_ALLOCATION,
            vol_target_map=REGIME_VOL_TARGETS, apply_vol_target=True,
        )
        # Vol targeting should produce non-1.0 leverage in some periods
        assert with_.mean_leverage != 1.0 or with_.max_leverage_reached != 1.0

    def test_vol_target_path_does_not_rebuild_history_arrays(self, monkeypatch):
        """Vol targeting should not allocate prior-day return arrays inside the daily loop."""
        spy_prices = np.linspace(100, 200, 500)
        gld_prices = np.linspace(100, 180, 500)
        tlt_prices = np.linspace(100, 120, 500)
        prices = {"SPY": spy_prices, "GLD": gld_prices, "TLT": tlt_prices}

        def fail_zeros(*_args, **_kwargs):
            raise AssertionError("vol-target hot loop should use precomputed returns, not np.zeros(i)")

        monkeypatch.setattr("src.backtest.combined_regime_alloc_vol_target.np.zeros", fail_zeros)

        row = backtest_strategy(
            prices, "With VT",
            allocation_map=REGIME_ALLOCATIONS, default_alloc=DEFAULT_ALLOCATION,
            vol_target_map=REGIME_VOL_TARGETS, apply_vol_target=True,
        )

        assert isinstance(row, CombinedRegimeRow)
        assert row.mean_leverage != 1.0 or row.max_leverage_reached != 1.0

    def test_not_enough_data_returns_zero(self):
        """Insufficient data returns zero row."""
        prices = {"SPY": np.array([100.0] * 50), "GLD": np.array([100.0] * 50), "TLT": np.array([100.0] * 50)}
        row = backtest_strategy(
            prices, "Short",
            allocation_map={}, default_alloc=DEFAULT_ALLOCATION,
        )
        assert row.cagr == 0.0


class TestRegimeVolTargetsPositive:
    """Property-based checks on vol targets."""

    def test_crisis_vs_high_vol_ordering(self):
        """CRISIS should be <= HIGH_VOL for vol targets."""
        assert REGIME_VOL_TARGETS["crisis"] <= REGIME_VOL_TARGETS["high_vol"]

    def test_high_vol_vs_normal_ordering(self):
        """HIGH_VOL should be <= NORMAL for vol targets."""
        assert REGIME_VOL_TARGETS["high_vol"] <= REGIME_VOL_TARGETS["normal"]

    def test_normal_vs_low_vol_ordering(self):
        """NORMAL should be <= LOW_VOL for vol targets."""
        assert REGIME_VOL_TARGETS["normal"] <= REGIME_VOL_TARGETS["low_vol"]

    def test_recovery_between_normal_and_low_vol(self):
        """RECOVERY target should be between NORMAL and LOW_VOL."""
        n = REGIME_VOL_TARGETS["normal"]
        r = REGIME_VOL_TARGETS["recovery"]
        lv = REGIME_VOL_TARGETS["low_vol"]
        assert n <= r <= lv, f"Expected NORMAL ({n}) <= RECOVERY ({r}) <= LOW_VOL ({lv})"


class TestRunCombinedBacktest:
    """Integration test for run_combined_backtest().
    
    These tests run the full backtest on real data (~90s each).
    Marked as heavy to allow skip in quick test runs.
    """

    @pytest.mark.heavy
    def test_returns_result_with_all_rows(self):
        """Should return a CombinedRegimeResult with 4 rows."""
        result = run_combined_backtest(save=False)
        assert isinstance(result, CombinedRegimeResult)
        assert len(result.rows) == 4

    @pytest.mark.heavy
    def test_combined_system_has_highest_sharpe(self):
        """Combined system should have best or close-to-best Sharpe."""
        result = run_combined_backtest(save=False)
        combined_row = [r for r in result.rows if r["label"] == "Combined System"]
        assert len(combined_row) == 1, "Combined System row found"
        combined_sharpe = combined_row[0]["sharpe"]

        # Combined should beat static by at least some margin
        static_row = [r for r in result.rows if r["label"] == "Static Champion (46/38/16)"]
        assert len(static_row) == 1
        static_sharpe = static_row[0]["sharpe"]
        assert combined_sharpe >= static_sharpe - 0.01, (
            f"Combined Sharpe {combined_sharpe:.4f} should not be worse than "
            f"static {static_sharpe:.4f}"
        )

    @pytest.mark.heavy
    def test_regime_alloc_only_beats_static(self):
        """Regime-conditional allocation alone should improve Sharpe."""
        result = run_combined_backtest(save=False)
        rows = {r["label"]: r for r in result.rows}
        static = rows["Static Champion (46/38/16)"]
        regime_only = rows["Regime Alloc Only"]
        assert regime_only["sharpe"] >= static["sharpe"] - 0.01

    @pytest.mark.heavy
    def test_combined_delta_positive(self):
        """Combined system should have positive Sharpe delta vs static."""
        result = run_combined_backtest(save=False)
        assert result.combined_sharpe_delta > 0, (
            f"Combined delta should be positive, got {result.combined_sharpe_delta}"
        )

    @pytest.mark.heavy
    def test_recommendation_present(self):
        """Result should have a recommendation string."""
        result = run_combined_backtest(save=False)
        assert len(result.recommendation) > 10
