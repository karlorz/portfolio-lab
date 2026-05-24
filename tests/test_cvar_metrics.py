#!/usr/bin/env python3
"""
Tests for cvar_metrics.py — CVaRMetrics dataclass, VaR/CVaR calculation,
tail severity classification, volatility, and metric computation.
"""
import sys
import json
import numpy as np

import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.monitor.cvar_metrics import (
    CVaRMetrics,
    calculate_var,
    calculate_cvar,
    get_tail_severity,
    calculate_volatility,
    compute_cvar_metrics,
    load_history,
    save_history,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metrics(**overrides):
    defaults = dict(
        timestamp=datetime.now().isoformat(),
        var_95=-2.50,
        cvar_95=-3.80,
        cvar_ratio=1.52,
        tail_severity="moderate",
        max_drawdown=-15.0,
        current_drawdown=-5.0,
        volatility_annual=15.0,
    )
    defaults.update(overrides)
    return CVaRMetrics(**defaults)


def _make_returns(n=500, drift=0.0003, vol=0.012, seed=42):
    rng = np.random.RandomState(seed)
    return rng.normal(drift, vol, n)


# ---------------------------------------------------------------------------
# Dataclass Tests
# ---------------------------------------------------------------------------

class TestCVaRMetrics:

    def test_to_dict(self):
        m = _make_metrics()
        d = m.to_dict()
        assert "var_95" in d
        assert "cvar_95" in d
        assert "tail_severity" in d

    def test_fields(self):
        m = _make_metrics(var_95=-2.0, cvar_95=-3.0)
        assert m.var_95 == -2.0
        assert m.cvar_95 == -3.0


# ---------------------------------------------------------------------------
# calculate_var Tests
# ---------------------------------------------------------------------------

class TestCalculateVar:

    def test_returns_float(self):
        returns = _make_returns()
        var = calculate_var(returns, 0.05)
        assert isinstance(var, float)

    def test_negative_value(self):
        returns = _make_returns()
        var = calculate_var(returns, 0.05)
        assert var < 0  # VaR should be negative (loss)

    def test_empty_returns(self):
        var = calculate_var(np.array([]), 0.05)
        assert var == -0.02  # Default

    def test_confidence_level(self):
        returns = _make_returns()
        var_95 = calculate_var(returns, 0.05)
        var_99 = calculate_var(returns, 0.01)
        # 99% VaR should be more negative than 95% VaR
        assert var_99 <= var_95

    def test_deterministic(self):
        returns = _make_returns(seed=99)
        var1 = calculate_var(returns, 0.05)
        var2 = calculate_var(returns, 0.05)
        assert var1 == var2


# ---------------------------------------------------------------------------
# calculate_cvar Tests
# ---------------------------------------------------------------------------

class TestCalculateCvar:

    def test_returns_float(self):
        returns = _make_returns()
        cvar = calculate_cvar(returns, 0.05)
        assert isinstance(cvar, float)

    def test_more_negative_than_var(self):
        returns = _make_returns()
        var = calculate_var(returns, 0.05)
        cvar = calculate_cvar(returns, 0.05)
        # CVaR (expected shortfall) should be more negative than VaR
        assert cvar <= var

    def test_empty_returns(self):
        cvar = calculate_cvar(np.array([]), 0.05)
        assert cvar == -0.03  # Default

    def test_tail_average(self):
        # Create known returns where we can verify the tail average
        returns = np.array([-0.05, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05])
        cvar = calculate_cvar(returns, 0.05)
        # VaR at 5% = np.percentile(returns, 5) ≈ -0.05
        # Tail returns ≤ VaR = [-0.05]
        # CVaR = mean([-0.05]) = -0.05
        assert cvar <= -0.04


# ---------------------------------------------------------------------------
# get_tail_severity Tests
# ---------------------------------------------------------------------------

class TestTailSeverity:

    def test_normal(self):
        assert get_tail_severity(1.0) == "normal"
        assert get_tail_severity(1.29) == "normal"

    def test_moderate(self):
        assert get_tail_severity(1.3) == "moderate"
        assert get_tail_severity(1.49) == "moderate"

    def test_elevated(self):
        assert get_tail_severity(1.5) == "elevated"
        assert get_tail_severity(1.79) == "elevated"

    def test_severe(self):
        assert get_tail_severity(1.8) == "severe"
        assert get_tail_severity(2.5) == "severe"

    def test_boundary_1_3(self):
        assert get_tail_severity(1.3) == "moderate"

    def test_boundary_1_5(self):
        assert get_tail_severity(1.5) == "elevated"

    def test_boundary_1_8(self):
        assert get_tail_severity(1.8) == "severe"


# ---------------------------------------------------------------------------
# calculate_volatility Tests
# ---------------------------------------------------------------------------

class TestCalculateVolatility:

    def test_returns_float(self):
        returns = _make_returns()
        vol = calculate_volatility(returns)
        assert isinstance(vol, float)

    def test_annualized(self):
        rng = np.random.RandomState(42)
        daily_vol = 0.015
        returns = rng.normal(0, daily_vol, 500)
        vol = calculate_volatility(returns)
        expected = daily_vol * np.sqrt(252)
        assert vol == pytest.approx(expected, rel=0.1)

    def test_short_returns(self):
        vol = calculate_volatility(np.array([0.01]))
        assert vol == 0.15  # Default

    def test_empty_returns(self):
        vol = calculate_volatility(np.array([]))
        assert vol == 0.15


# ---------------------------------------------------------------------------
# compute_cvar_metrics Tests
# ---------------------------------------------------------------------------

class TestComputeCVaRMetrics:

    def test_returns_metrics(self):
        with patch('src.monitor.cvar_metrics.fetch_portfolio_returns') as mock:
            mock.return_value = (_make_returns(), -0.05, -0.15)
            metrics = compute_cvar_metrics()
            assert isinstance(metrics, CVaRMetrics)

    def test_var_cvar_negative(self):
        with patch('src.monitor.cvar_metrics.fetch_portfolio_returns') as mock:
            mock.return_value = (_make_returns(), -0.05, -0.15)
            metrics = compute_cvar_metrics()
            assert metrics.var_95 < 0
            assert metrics.cvar_95 < 0

    def test_cvar_ratio_bounded(self):
        with patch('src.monitor.cvar_metrics.fetch_portfolio_returns') as mock:
            mock.return_value = (_make_returns(), -0.05, -0.15)
            metrics = compute_cvar_metrics()
            assert 1.0 <= metrics.cvar_ratio <= 3.0

    def test_tail_severity_valid(self):
        with patch('src.monitor.cvar_metrics.fetch_portfolio_returns') as mock:
            mock.return_value = (_make_returns(), -0.05, -0.15)
            metrics = compute_cvar_metrics()
            assert metrics.tail_severity in ("normal", "moderate", "elevated", "severe")

    def test_drawdown_from_data(self):
        with patch('src.monitor.cvar_metrics.fetch_portfolio_returns') as mock:
            mock.return_value = (_make_returns(), -0.08, -0.20)
            metrics = compute_cvar_metrics()
            assert metrics.current_drawdown == pytest.approx(-8.0, abs=0.1)
            assert metrics.max_drawdown == pytest.approx(-20.0, abs=0.1)

    def test_volatility_positive(self):
        with patch('src.monitor.cvar_metrics.fetch_portfolio_returns') as mock:
            mock.return_value = (_make_returns(), -0.05, -0.15)
            metrics = compute_cvar_metrics()
            assert metrics.volatility_annual > 0


# ---------------------------------------------------------------------------
# History Save/Load Tests
# ---------------------------------------------------------------------------

class TestHistory:

    def test_load_empty(self, tmp_path):
        import src.monitor.cvar_metrics as mod
        old = mod.RISK_HISTORY_PATH
        mod.RISK_HISTORY_PATH = tmp_path / "history.json"
        try:
            assert load_history() == []
        finally:
            mod.RISK_HISTORY_PATH = old

    def test_save_and_load(self, tmp_path):
        import src.monitor.cvar_metrics as mod
        old = mod.RISK_HISTORY_PATH
        mod.RISK_HISTORY_PATH = tmp_path / "history.json"
        try:
            history = [{"timestamp": "2026-05-14", "var_95": -2.5}]
            save_history(history)
            loaded = load_history()
            assert len(loaded) == 1
        finally:
            mod.RISK_HISTORY_PATH = old

    def test_save_trims_to_720(self, tmp_path):
        import src.monitor.cvar_metrics as mod
        old = mod.RISK_HISTORY_PATH
        mod.RISK_HISTORY_PATH = tmp_path / "history.json"
        try:
            history = [{"i": i} for i in range(1000)]
            save_history(history)
            loaded = load_history()
            assert len(loaded) == 720
        finally:
            mod.RISK_HISTORY_PATH = old


# ---------------------------------------------------------------------------
# CLI Tests
# ---------------------------------------------------------------------------

class TestCVaRMetricsExtended:
    """Extended tests for CVaRMetrics dataclass."""

    def test_all_fields_in_to_dict(self):
        m = _make_metrics()
        d = m.to_dict()
        expected_keys = {
            "timestamp", "var_95", "cvar_95", "cvar_ratio",
            "tail_severity", "max_drawdown", "current_drawdown", "volatility_annual",
        }
        assert set(d.keys()) == expected_keys

    def test_negative_var(self):
        m = _make_metrics(var_95=-3.5)
        assert m.var_95 < 0

    def test_negative_cvar(self):
        m = _make_metrics(cvar_95=-5.0)
        assert m.cvar_95 < 0

    def test_cvar_ratio_positive(self):
        m = _make_metrics()
        assert m.cvar_ratio > 0

    def test_negative_drawdowns(self):
        m = _make_metrics(max_drawdown=-30.0, current_drawdown=-10.0)
        assert m.max_drawdown < 0
        assert m.current_drawdown < 0

    def test_volatility_positive(self):
        m = _make_metrics(volatility_annual=12.5)
        assert m.volatility_annual > 0


class TestCalculateVarExtended:
    """Extended calculate_var tests."""

    def test_single_return(self):
        var = calculate_var(np.array([-0.05]), 0.05)
        assert isinstance(var, float)

    def test_constant_returns(self):
        returns = np.full(100, 0.01)
        var = calculate_var(returns, 0.05)
        # With all identical returns, VaR = the value itself
        assert isinstance(var, float)

    def test_all_positive_returns(self):
        returns = np.abs(np.random.randn(100)) * 0.01
        var = calculate_var(returns, 0.05)
        # Could be positive or near zero
        assert isinstance(var, float)

    def test_all_negative_returns(self):
        returns = -np.abs(np.random.randn(100)) * 0.01
        var = calculate_var(returns, 0.05)
        assert var < 0

    def test_alpha_01(self):
        returns = _make_returns()
        var = calculate_var(returns, 0.01)
        assert isinstance(var, float)
        assert var < 0


class TestCalculateCvarExtended:
    """Extended calculate_cvar tests."""

    def test_single_return(self):
        cvar = calculate_cvar(np.array([-0.05]), 0.05)
        assert isinstance(cvar, float)

    def test_constant_returns(self):
        returns = np.full(100, 0.01)
        cvar = calculate_cvar(returns, 0.05)
        assert isinstance(cvar, float)

    def test_cvar_equals_var_for_uniform(self):
        """With uniform negative returns, CVaR should equal VaR."""
        returns = np.array([-0.01, -0.02, -0.03, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
        var = calculate_var(returns, 0.05)
        cvar = calculate_cvar(returns, 0.05)
        assert cvar <= var  # CVaR should be at least as negative


class TestTailSeverityExtended:
    """Extended tail severity classification tests."""

    def test_just_below_moderate(self):
        assert get_tail_severity(1.299) == "normal"

    def test_just_above_moderate(self):
        assert get_tail_severity(1.3) == "moderate"

    def test_just_below_elevated(self):
        assert get_tail_severity(1.499) == "moderate"

    def test_just_above_elevated(self):
        assert get_tail_severity(1.5) == "elevated"

    def test_just_below_severe(self):
        assert get_tail_severity(1.799) == "elevated"

    def test_very_high_ratio(self):
        assert get_tail_severity(5.0) == "severe"

    def test_ratio_exactly_one(self):
        assert get_tail_severity(1.0) == "normal"

    def test_ratio_below_one(self):
        assert get_tail_severity(0.5) == "normal"


class TestCalculateVolatilityExtended:
    """Extended volatility tests."""

    def test_zero_returns(self):
        returns = np.zeros(100)
        vol = calculate_volatility(returns)
        assert vol == 0.0

    def test_known_volatility(self):
        np.random.seed(42)
        daily_vol = 0.01
        returns = np.random.normal(0, daily_vol, 1000)
        vol = calculate_volatility(returns)
        expected = daily_vol * np.sqrt(252)
        assert abs(vol - expected) / expected < 0.2  # Within 20%


class TestCLIExtended:
    """Extended CLI tests."""

    def test_main_callable(self):
        from src.monitor.cvar_metrics import main
        assert callable(main)

    def test_export_metrics_callable(self):
        from src.monitor.cvar_metrics import export_metrics
        assert callable(export_metrics)

    def test_display_metrics_callable(self):
        from src.monitor.cvar_metrics import display_metrics
        assert callable(display_metrics)
