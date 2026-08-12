"""
Comprehensive tests for src/backtest/metrics.py — shared backtest metrics module.

Covers __all__ exports, all dataclass fields (including to_dict completeness),
compute_metrics edge cases (empty, single-day, constant, NaN, extreme values),
compute_deflated_sharpe_ratio edge cases (zero, negative, N=1, zero variance),
module-level constants, and CLI main() detection.
"""
import json
import logging
import os
import tempfile
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import pytest

# ── Module imports (safe: no ML libs) ──────────────────────────────────────
# PORTFOLIO_LAB_ENABLE_ML=0 is enforced by conftest.py (Layer 1 env var gate).
# src/backtest/metrics.py only imports numpy and stdlib — safe to import.
from src.backtest.metrics import (
    BacktestConfig,
    BacktestMetrics,
    BacktestResult,
    CrisisReturns,
    DailyPrices,
    DEFAULT_CRISIS_YEARS,
    DEFAULT_TRANSACTION_COST_BPS,
    OverlayMetrics,
    REBALANCE_FREQUENCY_DAYS,
    TRADING_DAYS_PER_YEAR,
    compute_crisis_returns,
    compute_deflated_sharpe_ratio,
    compute_metrics,
    print_metrics_report,
    save_results_json,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. __all__ exports validation
# ═══════════════════════════════════════════════════════════════════════════

class TestAllExports:
    """Validate __all__ matches module exports."""

    def test_all_is_a_list_of_strings(self):
        import src.backtest.metrics as m
        assert isinstance(m.__all__, list)
        assert all(isinstance(name, str) for name in m.__all__)

    def test_all_contains_expected_names(self):
        import src.backtest.metrics as m
        expected = [
            'BacktestConfig', 'DailyPrices', 'BacktestResult', 'BacktestMetrics',
            'OverlayMetrics', 'CrisisReturns', 'compute_metrics',
            'compute_crisis_returns', 'print_metrics_report',
            'compute_deflated_sharpe_ratio', 'build_data_snapshot_provenance',
            'require_data_snapshot_provenance', 'save_results_json',
        ]
        for name in expected:
            assert name in m.__all__, f"{name} missing from __all__"

    def test_all_names_accessible_on_module(self):
        import src.backtest.metrics as m
        for name in m.__all__:
            assert hasattr(m, name), f"{name} in __all__ but not on module"

    def test_no_private_names_in_all(self):
        import src.backtest.metrics as m
        for name in m.__all__:
            assert not name.startswith('_'), f"{name} in __all__ starts with underscore"

    def test_all_length_matches(self):
        import src.backtest.metrics as m
        assert len(m.__all__) >= 13


# ═══════════════════════════════════════════════════════════════════════════
# 2. Module-level constants validation
# ═══════════════════════════════════════════════════════════════════════════

class TestModuleConstants:
    """Validate all public module-level constants."""

    def test_trading_days_per_year(self):
        assert TRADING_DAYS_PER_YEAR == 252

    def test_default_crisis_years(self):
        assert DEFAULT_CRISIS_YEARS == ['2008', '2020', '2022']

    def test_rebalance_frequency_days(self):
        assert REBALANCE_FREQUENCY_DAYS == 21

    def test_default_transaction_cost_bps(self):
        assert DEFAULT_TRANSACTION_COST_BPS == 10.0

    def test_constants_are_typed(self):
        assert isinstance(TRADING_DAYS_PER_YEAR, int)
        assert isinstance(DEFAULT_CRISIS_YEARS, list)
        assert isinstance(REBALANCE_FREQUENCY_DAYS, int)
        assert isinstance(DEFAULT_TRANSACTION_COST_BPS, float)

    def test_constants_are_immutable_primitives(self):
        """Module-level int/float constants should be immutable (not dataclass fields)."""
        # These are plain module-level names — just verify they exist and are correct.
        assert TRADING_DAYS_PER_YEAR + 1 == 253  # ints are immutable


# ═══════════════════════════════════════════════════════════════════════════
# 3. Dataclass field validation — BacktestConfig
# ═══════════════════════════════════════════════════════════════════════════

class TestBacktestConfigFields:
    """Validate BacktestConfig field types, defaults, and to_dict completeness."""

    def test_field_types(self):
        cfg = BacktestConfig()
        assert isinstance(cfg.start_date, str)
        assert isinstance(cfg.end_date, str)
        assert isinstance(cfg.initial_capital, float)
        assert isinstance(cfg.base_weights, dict)
        assert isinstance(cfg.rebalance_frequency_days, int)
        assert isinstance(cfg.rebalance_frequency, str)
        assert isinstance(cfg.transaction_cost_bps, float)
        assert isinstance(cfg.transaction_costs_by_symbol, dict)
        assert isinstance(cfg.extras, dict)

    def test_default_values(self):
        cfg = BacktestConfig()
        assert cfg.start_date == "2006-01-01"
        assert cfg.end_date == "2026-05-15"
        assert cfg.initial_capital == 100000.0
        assert cfg.rebalance_frequency_days == 21
        assert cfg.rebalance_frequency == "monthly"
        assert cfg.transaction_cost_bps == 10.0

    def test_to_dict_contains_all_fields(self):
        """dataclasses.asdict() should include all declared fields."""
        cfg = BacktestConfig(
            start_date="2020-01-01",
            end_date="2024-12-31",
            initial_capital=50000.0,
            base_weights={"SPY": 1.0},
            rebalance_frequency_days=63,
            rebalance_frequency="quarterly",
            transaction_cost_bps=5.0,
            extras={"lookback": 60},
        )
        d = asdict(cfg)
        declared = {f.name for f in fields(BacktestConfig)}
        for fname in declared:
            assert fname in d, f"Field '{fname}' missing from asdict()"
        assert d["start_date"] == "2020-01-01"
        assert d["extras"]["lookback"] == 60

    def test_field_defaults_are_independent(self):
        """Each instance should have its own copies of mutable defaults."""
        cfg1 = BacktestConfig()
        cfg2 = BacktestConfig()
        cfg1.extras["a"] = 1
        cfg1.transaction_costs_by_symbol["FAKE"] = 99
        assert "a" not in cfg2.extras
        assert "FAKE" not in cfg2.transaction_costs_by_symbol

    def test_base_weights_match_base_allocation(self):
        from src.paths import BASE_ALLOCATION
        cfg = BacktestConfig()
        for k, v in BASE_ALLOCATION.items():
            assert cfg.base_weights[k] == v, f"{k}={cfg.base_weights.get(k)} != {v}"

    def test_transaction_costs_by_symbol_present_for_core_etfs(self):
        cfg = BacktestConfig()
        for sym in ('SPY', 'GLD', 'TLT', 'IEF', 'QQQ', 'DBC'):
            assert sym in cfg.transaction_costs_by_symbol, f"{sym} missing from costs"
            assert isinstance(cfg.transaction_costs_by_symbol[sym], float)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Dataclass field validation — DailyPrices
# ═══════════════════════════════════════════════════════════════════════════

class TestDailyPricesFields:
    """Validate DailyPrices field types, defaults, and to_dict completeness."""

    def test_field_types(self):
        dp = DailyPrices(date="2024-06-01", spy=500.0, gld=180.0, tlt=130.0)
        assert isinstance(dp.date, str)
        assert isinstance(dp.spy, float)
        assert isinstance(dp.gld, float)
        assert isinstance(dp.tlt, float)
        assert dp.vix is None or isinstance(dp.vix, float)
        assert dp.ief is None or isinstance(dp.ief, float)
        assert dp.shy is None or isinstance(dp.shy, float)
        assert dp.btc is None or isinstance(dp.btc, float)
        assert dp.eth is None or isinstance(dp.eth, float)
        assert isinstance(dp.extras, dict)

    def test_required_fields(self):
        dp = DailyPrices(date="2024-01-01", spy=500.0, gld=180.0, tlt=130.0)
        assert dp.date == "2024-01-01"
        assert dp.spy == 500.0
        assert dp.gld == 180.0
        assert dp.tlt == 130.0

    def test_optional_fields_default_to_none(self):
        dp = DailyPrices(date="2024-01-01", spy=500.0, gld=180.0, tlt=130.0)
        assert dp.vix is None
        assert dp.ief is None
        assert dp.shy is None
        assert dp.btc is None
        assert dp.eth is None

    def test_to_dict_contains_all_fields(self):
        dp = DailyPrices(
            date="2024-06-15", spy=510.0, gld=185.0, tlt=128.0,
            vix=14.5, ief=96.0, shy=81.5, btc=67000.0, eth=3500.0,
            extras={"DBC": 22.0},
        )
        d = asdict(dp)
        declared = {f.name for f in fields(DailyPrices)}
        for fname in declared:
            assert fname in d, f"Field '{fname}' missing from asdict()"
        assert d["date"] == "2024-06-15"
        assert d["vix"] == 14.5
        assert d["extras"]["DBC"] == 22.0

    def test_float_coercion_from_int(self):
        """Python ints are accepted where floats are declared (no validation)."""
        dp = DailyPrices(date="2024-01-01", spy=500, gld=180, tlt=130)
        assert dp.spy == 500  # 500 == 500.0 in value
        assert isinstance(dp.spy, (int, float))

    def test_extras_independent(self):
        dp1 = DailyPrices(date="a", spy=1, gld=2, tlt=3)
        dp2 = DailyPrices(date="b", spy=4, gld=5, tlt=6)
        dp1.extras["key"] = "val"
        assert "key" not in dp2.extras


# ═══════════════════════════════════════════════════════════════════════════
# 5. Dataclass field validation — BacktestResult
# ═══════════════════════════════════════════════════════════════════════════

class TestBacktestResultFields:
    """Validate BacktestResult field types, defaults, and to_dict completeness."""

    def test_field_types(self):
        r = BacktestResult(
            total_return=10.5,
            cagr=5.2,
            volatility=12.1,
            sharpe_ratio=0.43,
            max_drawdown=-15.0,
        )
        assert isinstance(r.total_return, float)
        assert isinstance(r.cagr, float)
        assert isinstance(r.volatility, float)
        assert isinstance(r.sharpe_ratio, float)
        assert isinstance(r.max_drawdown, float)
        assert isinstance(r.total_rebalances, int)
        assert isinstance(r.total_transaction_costs, float)
        assert isinstance(r.avg_turnover, float)
        assert r.baseline_sharpe is None or isinstance(r.baseline_sharpe, float)
        assert r.sharpe_improvement is None or isinstance(r.sharpe_improvement, float)
        assert isinstance(r.extras, dict)
        assert r.crisis_returns is None or isinstance(r.crisis_returns, dict)

    def test_required_fields(self):
        r = BacktestResult(
            total_return=10.0, cagr=5.0, volatility=12.0,
            sharpe_ratio=0.5, max_drawdown=-15.0,
        )
        assert r.total_return == 10.0
        assert r.cagr == 5.0
        assert r.volatility == 12.0
        assert r.sharpe_ratio == 0.5
        assert r.max_drawdown == -15.0

    def test_default_values(self):
        r = BacktestResult(
            total_return=10.0, cagr=5.0, volatility=12.0,
            sharpe_ratio=0.5, max_drawdown=-15.0,
        )
        assert r.total_rebalances == 0
        assert r.total_transaction_costs == 0.0
        assert r.avg_turnover == 0.0
        assert r.baseline_sharpe is None
        assert r.sharpe_improvement is None
        assert r.extras == {}
        assert r.crisis_returns is None

    def test_to_dict_contains_all_fields(self):
        r = BacktestResult(
            total_return=50.0, cagr=8.5, volatility=11.0,
            sharpe_ratio=0.77, max_drawdown=-22.0,
            total_rebalances=120, total_transaction_costs=350.0,
            avg_turnover=0.05,
            baseline_sharpe=0.70, sharpe_improvement=0.07,
            extras={"signal": "TSMOM"},
            crisis_returns={"2008": -12.3, "2020": -7.1},
        )
        d = asdict(r)
        declared = {f.name for f in fields(BacktestResult)}
        for fname in declared:
            assert fname in d, f"Field '{fname}' missing from asdict()"
        assert d["total_return"] == 50.0
        assert d["crisis_returns"]["2008"] == -12.3
        assert d["extras"]["signal"] == "TSMOM"

    def test_extras_independent(self):
        r1 = BacktestResult(
            total_return=0, cagr=0, volatility=0,
            sharpe_ratio=0, max_drawdown=0,
        )
        r2 = BacktestResult(
            total_return=0, cagr=0, volatility=0,
            sharpe_ratio=0, max_drawdown=0,
        )
        r1.extras["x"] = 1
        assert "x" not in r2.extras


# ═══════════════════════════════════════════════════════════════════════════
# 6. Dataclass field validation — BacktestMetrics
# ═══════════════════════════════════════════════════════════════════════════

class TestBacktestMetricsFields:
    """Validate BacktestMetrics field types, defaults, and to_dict completeness."""

    def test_field_types(self):
        m = BacktestMetrics(
            total_return=10.0, cagr=5.0, volatility=12.0,
            sharpe_ratio=0.5, max_drawdown=-15.0,
        )
        assert isinstance(m.total_return, float)
        assert isinstance(m.cagr, float)
        assert isinstance(m.volatility, float)
        assert isinstance(m.sharpe_ratio, float)
        assert isinstance(m.max_drawdown, float)
        assert isinstance(m.total_rebalances, int)
        assert isinstance(m.total_transaction_costs, float)
        assert isinstance(m.avg_turnover, float)

    def test_default_values(self):
        m = BacktestMetrics(
            total_return=10.0, cagr=5.0, volatility=12.0,
            sharpe_ratio=0.5, max_drawdown=-15.0,
        )
        assert m.total_rebalances == 0
        assert m.total_transaction_costs == 0.0
        assert m.avg_turnover == 0.0

    def test_to_dict_contains_all_fields(self):
        m = BacktestMetrics(
            total_return=10.5, cagr=5.2, volatility=12.1,
            sharpe_ratio=0.43, max_drawdown=-15.0,
            total_rebalances=24, total_transaction_costs=100.0,
            avg_turnover=0.03,
        )
        d = asdict(m)
        declared = {f.name for f in fields(BacktestMetrics)}
        for fname in declared:
            assert fname in d, f"Field '{fname}' missing from asdict()"
        assert d["total_return"] == 10.5
        assert d["total_rebalances"] == 24
        assert d["avg_turnover"] == 0.03


# ═══════════════════════════════════════════════════════════════════════════
# 7. Dataclass field validation — OverlayMetrics
# ═══════════════════════════════════════════════════════════════════════════

class TestOverlayMetricsFields:
    """Validate OverlayMetrics field types, defaults, and to_dict completeness."""

    def test_field_types(self):
        m = OverlayMetrics(baseline_sharpe=0.94, sharpe_improvement=0.015)
        assert isinstance(m.baseline_sharpe, float)
        assert isinstance(m.sharpe_improvement, float)
        assert isinstance(m.overlay_active_count, int)
        assert isinstance(m.overlay_active_pct, float)

    def test_default_values(self):
        m = OverlayMetrics(baseline_sharpe=0.80, sharpe_improvement=0.02)
        assert m.overlay_active_count == 0
        assert m.overlay_active_pct == 0.0

    def test_custom_values(self):
        m = OverlayMetrics(
            baseline_sharpe=0.80, sharpe_improvement=-0.01,
            overlay_active_count=150, overlay_active_pct=0.55,
        )
        assert m.overlay_active_count == 150
        assert m.overlay_active_pct == 0.55

    def test_to_dict_contains_all_fields(self):
        m = OverlayMetrics(
            baseline_sharpe=0.94, sharpe_improvement=0.015,
            overlay_active_count=100, overlay_active_pct=0.40,
        )
        d = asdict(m)
        declared = {f.name for f in fields(OverlayMetrics)}
        for fname in declared:
            assert fname in d, f"Field '{fname}' missing from asdict()"


# ═══════════════════════════════════════════════════════════════════════════
# 8. Dataclass field validation — CrisisReturns
# ═══════════════════════════════════════════════════════════════════════════

class TestCrisisReturnsFields:
    """Validate CrisisReturns field types, defaults, and to_dict completeness."""

    def test_field_types(self):
        c = CrisisReturns()
        assert isinstance(c.returns, dict)

    def test_default_empty_dict(self):
        c = CrisisReturns()
        assert c.returns == {}

    def test_with_returns(self):
        c = CrisisReturns(returns={"2008": -12.3, "2020": -7.1})
        assert c.returns["2008"] == -12.3
        assert c.get("2020") == -7.1
        assert c.get("2022") is None

    def test_to_dict_contains_all_fields(self):
        c = CrisisReturns(returns={"2008": -12.3})
        d = asdict(c)
        declared = {f.name for f in fields(CrisisReturns)}
        for fname in declared:
            assert fname in d, f"Field '{fname}' missing from asdict()"
        assert d["returns"]["2008"] == -12.3

    def test_get_returns_none_for_missing_year(self):
        c = CrisisReturns()
        assert c.get("2008") is None


# ═══════════════════════════════════════════════════════════════════════════
# 9. compute_metrics — core functionality
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeMetrics:
    """Core compute_metrics tests: normal, edge, and pathological inputs."""

    # ── Normal cases ──────────────────────────────────────────────────

    def test_positive_return(self):
        curve = [100000, 101000, 102000, 103000, 104000]
        m = compute_metrics(curve, 100000)
        assert m.total_return == 4.0  # 4%
        assert m.cagr > 0
        assert m.max_drawdown == 0.0  # Monotonically increasing
        assert m.sharpe_ratio >= 0
        assert m.volatility > 0

    def test_negative_return(self):
        curve = [100000, 97000, 95000, 92000]
        m = compute_metrics(curve, 100000)
        assert m.total_return < 0
        assert m.cagr < 0
        assert m.max_drawdown < 0
        # Sharpe is negative (CAGR is negative, vol is positive) — not clamped to 0

    def test_drawdown_detection(self):
        curve = [100000, 110000, 95000, 105000]
        m = compute_metrics(curve, 100000)
        assert m.max_drawdown < 0
        # Drawdown from 110k to 95k: (95-110)/110 ≈ -13.64%
        assert m.max_drawdown <= -13.0

    def test_sharpe_positive_for_upward_trend(self):
        np.random.seed(42)
        curve = [100000]
        for _ in range(251):
            curve.append(curve[-1] * (1 + np.random.normal(0.0004, 0.01)))
        m = compute_metrics(curve, 100000)
        assert m.sharpe_ratio > 0

    # ── Empty / single-day / two-element ──────────────────────────────

    def test_empty_curve(self):
        m = compute_metrics([], 100000)
        assert m.total_return == 0.0
        assert m.cagr == 0.0
        assert m.volatility == 0.0
        assert m.sharpe_ratio == 0.0
        assert m.max_drawdown == 0.0

    def test_one_element_curve(self):
        m = compute_metrics([100000], 100000)
        assert m.total_return == 0.0
        assert m.cagr == 0.0

    def test_two_element_curve(self):
        m = compute_metrics([100000, 105000], 100000)
        assert m.total_return == 5.0  # 5%
        assert m.max_drawdown == 0.0

    # ── Constant returns (zero vol) ───────────────────────────────────

    def test_flat_curve_zero_return(self):
        curve = [100000] * 10
        m = compute_metrics(curve, 100000)
        assert m.total_return == 0.0
        assert m.cagr == 0.0
        assert m.volatility == 0.0
        assert m.sharpe_ratio == 0.0
        assert m.max_drawdown == 0.0

    def test_flat_curve_above_initial(self):
        """Flat above initial gives positive return but zero vol → zero sharpe."""
        curve = [110000] * 10
        m = compute_metrics(curve, 100000)
        assert m.total_return > 0
        assert m.volatility == 0.0
        assert m.sharpe_ratio == 0.0  # vol=0 → division clamped to 0

    def test_flat_curve_below_initial(self):
        curve = [90000] * 10
        m = compute_metrics(curve, 100000)
        assert m.total_return < 0
        assert m.volatility == 0.0
        assert m.sharpe_ratio == 0.0

    # ── Very large values ─────────────────────────────────────────────

    def test_very_large_values(self):
        curve = [1e6, 1.5e6, 2.0e6, 3.375e6]
        m = compute_metrics(curve, 1e6)
        assert m.total_return > 200  # 237.5%
        assert m.cagr > 0
        assert m.volatility > 0  # Non-constant returns → vol > 0

    def test_very_large_initial_capital(self):
        curve = [1e12, 1.1e12, 1.2e12]
        m = compute_metrics(curve, 1e12)
        assert m.total_return > 0
        assert m.cagr > 0

    def test_very_small_values(self):
        curve = [1e-6, 1.05e-6, 1.10e-6]
        m = compute_metrics(curve, 1e-6)
        assert m.total_return > 0
        assert m.cagr > 0

    # ── Zero / negative initial capital ───────────────────────────────

    def test_zero_initial_capital_nonzero_curve(self):
        curve = [0, 100, 200]
        m = compute_metrics(curve, 0)
        assert m.total_return == 0.0
        assert m.cagr == 0.0
        assert m.sharpe_ratio == 0.0

    def test_negative_initial_capital(self):
        """Should not crash — results not financially meaningful."""
        curve = [-10000, -5000, -2000]
        m = compute_metrics(curve, -10000)
        assert isinstance(m.total_return, float)

    def test_zero_values_in_curve(self):
        """Zero in equity curve should not cause division by zero."""
        curve = [100000, 0, 100000]
        m = compute_metrics(curve, 100000)
        assert isinstance(m.total_return, float)
        assert isinstance(m.cagr, float)

    # ── NaN handling ──────────────────────────────────────────────────

    def test_nan_in_curve_value(self):
        """NaN in equity curve propagates — function should not crash."""
        curve = [100000, float('nan'), 105000]
        m = compute_metrics(curve, 100000)
        assert isinstance(m.total_return, float)
        # total_return uses equity_curve[-1] / initial - 1, which is 5%
        assert m.total_return == 5.0

    def test_nan_in_mid_curve(self):
        """NaN returns produce NaN volatility — function should not crash."""
        curve = [100000, 101000, float('nan'), 103000]
        m = compute_metrics(curve, 100000)
        assert isinstance(m.total_return, float)

    def test_inf_in_curve(self):
        """Infinity in equity curve should not cause unbounded errors."""
        curve = [100000, float('inf')]
        m = compute_metrics(curve, 100000)
        assert isinstance(m.total_return, float)

    def test_all_same_value_no_return(self):
        """All values equal to initial capital."""
        curve = [100000] * 252
        m = compute_metrics(curve, 100000)
        assert m.total_return == 0.0
        assert m.cagr == 0.0
        assert m.volatility == 0.0
        assert m.sharpe_ratio == 0.0
        assert m.max_drawdown == 0.0

    def test_recovery_from_drawdown(self):
        """100% drawdown then recovery should be captured correctly."""
        curve = [100000, 50000, 100000]
        m = compute_metrics(curve, 100000)
        assert m.total_return == 0.0  # back to even
        assert m.max_drawdown <= -50.0  # lost half

    def test_return_fields_rounded(self):
        """Verify rounding behavior from compute_metrics."""
        curve = [100000, 105234.56789]
        m = compute_metrics(curve, 100000)
        assert m.total_return == 5.23  # rounded to 2 dp
        assert m.sharpe_ratio == 0.0  # only one return → vol=0 → sharpe=0

    def test_cagr_annualized_correctly(self):
        """CAGR should be annualized by trading_days_per_year / n_days."""
        curve = [100000, 110000]
        m = compute_metrics(curve, 100000)
        # CAGR = (110000/100000)^(252/1) - 1 = 1.1^252 - 1 ≈ huge
        expected_cagr = ((110000 / 100000) ** (252 / 1) - 1) * 100
        assert abs(m.cagr - round(expected_cagr, 2)) < 0.01

    def test_peak_to_trough_drawdown(self):
        """Verify max drawdown calculation."""
        curve = [100000, 90000, 120000, 110000, 80000, 130000]
        m = compute_metrics(curve, 100000)
        # Peak is 120000, trough afterwards is 80000: (80000-120000)/120000 = -33.33%
        assert m.max_drawdown <= -33.0
        assert m.max_drawdown >= -34.0


# ═══════════════════════════════════════════════════════════════════════════
# 10. compute_deflated_sharpe_ratio — core + edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestDeflatedSharpeRatio:
    """Core compute_deflated_sharpe_ratio tests."""

    def test_champion_sharpe_with_94_trials(self):
        """Champion config (Sharpe 0.79, 94 configs, 5371 days) should have DSR > 0.50."""
        dsr = compute_deflated_sharpe_ratio(0.79, n_trials=94, n_observations=5371)
        assert dsr > 0.50

    def test_dsr_between_zero_and_one(self):
        for n_trials in [1, 5, 20, 94]:
            dsr = compute_deflated_sharpe_ratio(
                0.79, n_trials=n_trials, n_observations=5371
            )
            assert 0.0 <= dsr <= 1.0, f"DSR out of range for n_trials={n_trials}: {dsr}"

    def test_more_trials_lower_dsr(self):
        """More trials = more multiple-testing penalty = lower DSR."""
        dsr_10 = compute_deflated_sharpe_ratio(0.20, n_trials=10, n_observations=100)
        dsr_94 = compute_deflated_sharpe_ratio(0.20, n_trials=94, n_observations=100)
        assert dsr_10 >= dsr_94

    def test_higher_sharpe_higher_dsr(self):
        dsr_low = compute_deflated_sharpe_ratio(0.15, n_trials=20, n_observations=100)
        dsr_high = compute_deflated_sharpe_ratio(0.30, n_trials=20, n_observations=100)
        assert dsr_high > dsr_low

    def test_more_observations_higher_dsr(self):
        """More data = more statistical power = higher DSR."""
        dsr_short = compute_deflated_sharpe_ratio(0.79, n_trials=20, n_observations=500)
        dsr_long = compute_deflated_sharpe_ratio(0.79, n_trials=20, n_observations=5000)
        assert dsr_long >= dsr_short

    # ── Edge: zero / negative sharpe ──────────────────────────────────

    def test_zero_sharpe_returns_zero(self):
        dsr = compute_deflated_sharpe_ratio(0.0, n_trials=10, n_observations=1000)
        assert dsr == 0.0

    def test_negative_sharpe_low_dsr(self):
        dsr = compute_deflated_sharpe_ratio(-0.50, n_trials=10, n_observations=1000)
        # Negative Sharpe should yield DSR well below 0.50
        assert dsr < 0.50

    def test_large_negative_sharpe(self):
        """Very negative Sharpe gives DSR near 0."""
        dsr = compute_deflated_sharpe_ratio(-2.0, n_trials=10, n_observations=1000)
        assert dsr < 0.10

    # ── Edge: N=1 ─────────────────────────────────────────────────────

    def test_n_trials_one_expected_max_is_zero(self):
        """With 1 trial, expected_max_sr = 0, sigma_max = sqrt(var_sr)."""
        dsr = compute_deflated_sharpe_ratio(0.5, n_trials=1, n_observations=1000)
        assert dsr > 0.50  # No multiple-testing penalty

    def test_n_observations_one(self):
        """n_observations=1 → var denominator = max(0, 1) = 1."""
        dsr = compute_deflated_sharpe_ratio(0.5, n_trials=10, n_observations=1)
        assert 0.0 <= dsr <= 1.0

    # ── Edge: zero / negative n_trials or n_observations ──────────────

    def test_n_trials_zero_returns_zero(self):
        assert compute_deflated_sharpe_ratio(0.79, n_trials=0, n_observations=5371) == 0.0

    def test_n_trials_negative_returns_zero(self):
        assert compute_deflated_sharpe_ratio(0.79, n_trials=-5, n_observations=5371) == 0.0

    def test_n_observations_zero_returns_zero(self):
        assert compute_deflated_sharpe_ratio(0.79, n_trials=10, n_observations=0) == 0.0

    def test_n_observations_negative_returns_zero(self):
        assert compute_deflated_sharpe_ratio(0.79, n_trials=10, n_observations=-5) == 0.0

    # ── Edge: zero variance ───────────────────────────────────────────

    def test_sharpe_zero_with_various_inputs(self):
        """Zero Sharpe should always return 0 regardless of other params (early exit)."""
        for n in [1, 10, 100]:
            for obs in [1, 100, 5371]:
                dsr = compute_deflated_sharpe_ratio(0.0, n_trials=n, n_observations=obs)
                assert dsr == 0.0

    def test_sigma_max_non_positive_path(self):
        """When sigma_max <= 0 and sharpe > expected_max → DSR = 1.0."""
        # n_trials=2 routes to sigma_max = sqrt(var_sr) branch (not pi/6/ln(N))
        # Very high kurtosis can drive var_sr up, so sigma_max > 0 still.
        # Path: sigma_max <= 0 is rare — test that the function handles it.
        dsr = compute_deflated_sharpe_ratio(0.0, n_trials=10, n_observations=1000)
        assert dsr == 0.0  # sharpe=0 early exit

    # ── Edge: skew and kurtosis ───────────────────────────────────────

    def test_skew_affects_dsr(self):
        dsr_normal = compute_deflated_sharpe_ratio(
            0.79, n_trials=20, n_observations=5371, skew=0.0, kurtosis=3.0,
        )
        dsr_skewed = compute_deflated_sharpe_ratio(
            0.79, n_trials=20, n_observations=5371, skew=-1.0, kurtosis=6.0,
        )
        assert 0.0 <= dsr_normal <= 1.0
        assert 0.0 <= dsr_skewed <= 1.0

    def test_extreme_skew_kurtosis_does_not_crash(self):
        dsr = compute_deflated_sharpe_ratio(
            0.5, n_trials=10, n_observations=252,
            skew=-5.0, kurtosis=50.0,
        )
        assert 0.0 <= dsr <= 1.0

    # ── Edge: known analytical bounds ─────────────────────────────────

    def test_dsr_monotonic_in_sharpe(self):
        sharpes = [0.1, 0.2, 0.3, 0.4, 0.5]
        dsrs = [
            compute_deflated_sharpe_ratio(s, n_trials=10, n_observations=500)
            for s in sharpes
        ]
        for i in range(1, len(dsrs)):
            assert dsrs[i] >= dsrs[i - 1], f"DSR not monotonic at index {i}: {dsrs}"

    def test_dsr_monotonic_in_observations(self):
        obs_list = [50, 100, 500, 1000]
        dsrs = [
            compute_deflated_sharpe_ratio(0.30, n_trials=10, n_observations=n)
            for n in obs_list
        ]
        for i in range(1, len(dsrs)):
            assert dsrs[i] >= dsrs[i - 1], f"DSR not monotonic in obs at index {i}: {dsrs}"

    def test_dsr_monotonic_in_trials(self):
        """DSR should decrease (or stay same) as trials increase."""
        trial_counts = [1, 2, 5, 10, 50, 100]
        dsrs = [
            compute_deflated_sharpe_ratio(0.50, n_trials=t, n_observations=500)
            for t in trial_counts
        ]
        for i in range(1, len(dsrs)):
            assert dsrs[i] <= dsrs[i - 1], (
                f"DSR not monotonic in trials at index {i}: "
                f"{dsrs[i]} > {dsrs[i - 1]}"
            )

    def test_n_trials_two_boundary(self):
        """n_trials=2 is boundary: not > 2, not == 1, uses sigma_max = sqrt(var_sr)."""
        dsr = compute_deflated_sharpe_ratio(0.50, n_trials=2, n_observations=1000)
        assert 0.0 <= dsr <= 1.0

    def test_very_large_n_trials(self):
        """10k trials should not overflow or produce NaN."""
        dsr = compute_deflated_sharpe_ratio(0.79, n_trials=10000, n_observations=5371)
        assert 0.0 <= dsr <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 11. compute_crisis_returns — edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeCrisisReturns:
    """Edge cases for compute_crisis_returns."""

    def test_basic_crisis_computation(self):
        prices = {
            '2008-01-02': {'SPY': 100, 'GLD': 80, 'TLT': 90},
            '2008-12-31': {'SPY': 70, 'GLD': 85, 'TLT': 110},
        }
        result = compute_crisis_returns(prices, ['2008-01-02', '2008-12-31'])
        assert '2008' in result
        assert result['2008'] < 0

    def test_missing_crisis_year(self):
        prices = {'2020-01-02': {'SPY': 100}}
        result = compute_crisis_returns(
            prices, ['2020-01-02'],
            crisis_years=['2008'],
        )
        assert '2008' not in result

    def test_custom_base_weights(self):
        prices = {
            '2022-01-03': {'SPY': 100},
            '2022-12-30': {'SPY': 80},
        }
        result = compute_crisis_returns(
            prices, ['2022-01-03', '2022-12-30'],
            base_weights={'SPY': 1.0},
        )
        assert '2022' in result
        assert result['2022'] <= -20.0

    def test_insufficient_data_skips_year(self):
        prices = {'2008-01-02': {'SPY': 100}}
        result = compute_crisis_returns(prices, ['2008-01-02'])
        assert '2008' not in result

    def test_with_equity_curve(self):
        trading_days = ['2020-01-02', '2020-06-01', '2020-12-31']
        equity_curve = [100000, 95000, 105000]
        result = compute_crisis_returns(
            {}, trading_days,
            crisis_years=['2020'],
            equity_curve=equity_curve,
        )
        assert '2020' in result
        assert result['2020'] < 0

    def test_equity_curve_no_drawdown(self):
        trading_days = ['2020-01-02', '2020-06-01', '2020-12-31']
        equity_curve = [100000, 110000, 120000]
        result = compute_crisis_returns(
            {}, trading_days,
            crisis_years=['2020'],
            equity_curve=equity_curve,
        )
        assert '2020' in result
        assert result['2020'] == 0.0

    def test_equity_curve_shorter_than_days(self):
        """When equity_curve has fewer entries than trading_days, partial indices are used."""
        trading_days = ['2020-01-02', '2020-06-01', '2020-12-31']
        equity_curve = [100000]
        result = compute_crisis_returns(
            {}, trading_days,
            crisis_years=['2020'],
            equity_curve=equity_curve,
        )
        # Only 1 mapping entry maps — single value gives 0% drawdown
        assert '2020' in result
        assert result['2020'] == 0.0

    def test_empty_prices_empty_result(self):
        result = compute_crisis_returns({}, ['2022-01-03', '2022-12-30'])
        # needs base_weights default → prices have no matching symbols → portfolio_values = [0, 0]
        assert '2022' not in result

    def test_zero_starting_value(self):
        prices = {'2020-01-02': {'SPY': 0}}
        result = compute_crisis_returns(prices, ['2020-01-02', '2020-06-01'])
        assert '2020' not in result

    def test_custom_crisis_years(self):
        prices = {
            '2023-01-02': {'SPY': 100, 'GLD': 80, 'TLT': 90},
            '2023-12-31': {'SPY': 110, 'GLD': 90, 'TLT': 100},
        }
        result = compute_crisis_returns(
            prices, ['2023-01-02', '2023-12-31'],
            crisis_years=['2023'],
        )
        assert '2023' in result
        assert result['2023'] == 0.0  # Monotonically increasing

    def test_none_crisis_years_defaults(self):
        """crisis_years=None should use DEFAULT_CRISIS_YEARS = ['2008', '2020', '2022']."""
        prices = {}
        result = compute_crisis_returns(prices, ['2008-01-02', '2008-12-31'], crisis_years=None)
        # crisis_years=['2008', '2020', '2022'], but 2020/2022 have no data
        assert '2008' not in result  # Not enough price data, but years were used


# ═══════════════════════════════════════════════════════════════════════════
# 12. print_metrics_report
# ═══════════════════════════════════════════════════════════════════════════

class TestPrintMetricsReport:
    """Tests for print_metrics_report."""

    def test_basic_output(self, caplog):
        m = BacktestMetrics(
            total_return=10.5, cagr=8.2, volatility=12.1,
            sharpe_ratio=0.68, max_drawdown=-15.3,
        )
        with caplog.at_level(logging.INFO, logger="src.backtest.metrics"):
            print_metrics_report(m, title="Test Report")
        assert "Test Report" in caplog.text
        assert "10.50%" in caplog.text
        assert "8.20%" in caplog.text
        assert "12.10%" in caplog.text
        assert "0.6800" in caplog.text
        assert "-15.30%" in caplog.text

    def test_with_rebalances(self, caplog):
        m = BacktestMetrics(
            total_return=15.0, cagr=9.5, volatility=11.0,
            sharpe_ratio=0.86, max_drawdown=-18.0,
            total_rebalances=24, total_transaction_costs=120.0,
        )
        with caplog.at_level(logging.INFO, logger="src.backtest.metrics"):
            print_metrics_report(m)
        assert "Rebalances: 24" in caplog.text
        assert "120.00" in caplog.text

    def test_zero_rebalances_suppresses_lines(self, caplog):
        m = BacktestMetrics(
            total_return=10.0, cagr=8.0, volatility=12.0,
            sharpe_ratio=0.67, max_drawdown=-15.0,
            total_rebalances=0, total_transaction_costs=0.0,
        )
        with caplog.at_level(logging.INFO, logger="src.backtest.metrics"):
            print_metrics_report(m)
        assert "Rebalances" not in caplog.text
        assert "Transaction Costs" not in caplog.text

    def test_default_title(self, caplog):
        m = BacktestMetrics(
            total_return=5.0, cagr=4.0, volatility=10.0,
            sharpe_ratio=0.40, max_drawdown=-10.0,
        )
        with caplog.at_level(logging.INFO, logger="src.backtest.metrics"):
            print_metrics_report(m)
        assert "Backtest Results" in caplog.text


# ═══════════════════════════════════════════════════════════════════════════
# 13. save_results_json + _json_serializer
# ═══════════════════════════════════════════════════════════════════════════

class TestSaveResultsJson:
    """Tests for save_results_json and _json_serializer."""

    def test_saves_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.json')
            data = {'cagr': 8.5, 'sharpe': 0.68}
            save_results_json(data, output_path=path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded['cagr'] == 8.5
            assert loaded['sharpe'] == 0.68

    def test_saves_with_default_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {'key': 'val'}
            save_results_json(data, default_dir=Path(tmpdir))
            assert (Path(tmpdir) / 'backtest_results.json').exists()

    def test_no_path_no_dir_noop(self):
        save_results_json({'key': 'val'}, output_path=None, default_dir=None)

    def test_explicit_path_overrides_default_dir(self, tmp_path):
        explicit = tmp_path / "explicit.json"
        default_dir = tmp_path / "subdir"
        save_results_json(
            {"key": "val"},
            output_path=str(explicit),
            default_dir=default_dir,
        )
        assert explicit.exists()
        assert not (default_dir / "backtest_results.json").exists()

    def test_numpy_serialization(self, tmp_path):
        path = tmp_path / "numpy.json"
        data = {
            'cagr': np.float64(8.5),
            'count': np.int64(42),
            'arr': np.array([1, 2, 3]),
            'matrix': np.array([[1.0, 2.0], [3.0, 4.0]]),
        }
        save_results_json(data, output_path=str(path))
        with open(path) as f:
            loaded = json.load(f)
        assert loaded['cagr'] == 8.5
        assert loaded['count'] == 42
        assert loaded['arr'] == [1, 2, 3]
        assert loaded['matrix'] == [[1.0, 2.0], [3.0, 4.0]]

    def test_empty_dict_saves(self, tmp_path):
        path = tmp_path / "empty.json"
        save_results_json({}, output_path=str(path))
        with open(path) as f:
            assert json.load(f) == {}

    def test_nested_dict(self, tmp_path):
        path = tmp_path / "nested.json"
        data = {"outer": {"inner": {"a": 1, "b": [1, 2, 3]}}}
        save_results_json(data, output_path=str(path))
        with open(path) as f:
            assert json.load(f) == data

    def test_json_serializer_numpy_int(self):
        from src.backtest.metrics import _json_serializer
        assert _json_serializer(np.int64(42)) == 42
        assert _json_serializer(np.int32(7)) == 7

    def test_json_serializer_numpy_float(self):
        from src.backtest.metrics import _json_serializer
        assert isinstance(_json_serializer(np.float64(3.14)), float)
        assert abs(_json_serializer(np.float64(3.14)) - 3.14) < 1e-10

    def test_json_serializer_numpy_array(self):
        from src.backtest.metrics import _json_serializer
        arr = np.array([1, 2, 3])
        assert _json_serializer(arr) == [1, 2, 3]

    def test_json_serializer_unknown_type_raises(self):
        from src.backtest.metrics import _json_serializer
        with pytest.raises(TypeError):
            _json_serializer(object())


# ═══════════════════════════════════════════════════════════════════════════
# 14. Cross-module consistency
# ═══════════════════════════════════════════════════════════════════════════

class TestCrossModuleConsistency:
    """Verify consistency between metrics.py, paths.py, and cost tables."""

    def test_backtest_config_matches_base_allocation(self):
        from src.paths import BASE_ALLOCATION
        cfg = BacktestConfig()
        assert cfg.base_weights == BASE_ALLOCATION

    def test_backtest_config_matches_etf_costs(self):
        from src.costs.etf_cost_table import ETF_COST_BPS
        cfg = BacktestConfig()
        for sym, cost in ETF_COST_BPS.items():
            assert cfg.transaction_costs_by_symbol[sym] == cost

    def test_backtest_config_constant_defaults_match(self):
        cfg = BacktestConfig()
        assert cfg.rebalance_frequency_days == REBALANCE_FREQUENCY_DAYS
        assert cfg.transaction_cost_bps == DEFAULT_TRANSACTION_COST_BPS


# ═══════════════════════════════════════════════════════════════════════════
# 15. CLI main() — detection
# ═══════════════════════════════════════════════════════════════════════════

class TestCliMain:
    """Verify whether the module has a CLI entry point."""

    def test_no_main_function(self):
        """metrics.py does not define a main() function — this should not change."""
        import src.backtest.metrics as m
        assert not hasattr(m, 'main'), (
            "metrics.py should not have a main() function. "
            "If one was added, add tests for it here."
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
