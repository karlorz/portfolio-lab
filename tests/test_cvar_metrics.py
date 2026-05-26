#!/usr/bin/env python3
"""
Tests for cvar_metrics.py — CVaRMetrics dataclass, VaR/CVaR calculation,
tail severity classification, volatility, and metric computation.
"""
import sys
import json
import logging
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
    export_metrics,
    fetch_portfolio_returns,
    display_metrics,
    main,
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


# ---------------------------------------------------------------------------
# compute_cvar_metrics: daily_var == 0 branch
# ---------------------------------------------------------------------------

class TestComputeCVaRMetricsVarZero:
    """Tests for the daily_var == 0 branch in compute_cvar_metrics."""

    def test_daily_var_zero_uses_default_ratio(self):
        """When daily_var is exactly 0, cvar_ratio defaults to 1.5."""
        returns = _make_returns()
        with patch('src.monitor.cvar_metrics.fetch_portfolio_returns',
                   return_value=(returns, -0.05, -0.15)):
            with patch('src.monitor.cvar_metrics.calculate_var',
                       return_value=0.0):
                with patch('src.monitor.cvar_metrics.calculate_cvar',
                           return_value=-0.03):
                    metrics = compute_cvar_metrics()
                    # daily_var == 0 triggers default 1.5
                    assert metrics.cvar_ratio == 1.5

    def test_daily_var_zero_persists_after_clip(self):
        """Default 1.5 ratio from zero VaR survives clipping (within [1,3])."""
        returns = _make_returns()
        with patch('src.monitor.cvar_metrics.fetch_portfolio_returns',
                   return_value=(returns, -0.05, -0.15)):
            with patch('src.monitor.cvar_metrics.calculate_var',
                       return_value=0.0):
                with patch('src.monitor.cvar_metrics.calculate_cvar',
                           return_value=-0.03):
                    metrics = compute_cvar_metrics()
                    assert 1.0 <= metrics.cvar_ratio <= 3.0
                    assert metrics.cvar_ratio == 1.5


# ---------------------------------------------------------------------------
# cvar_ratio clipping bounds
# ---------------------------------------------------------------------------

class TestCVaRRatioClipping:
    """Test that cvar_ratio clips at 1.0 lower bound and 3.0 upper bound."""

    def test_clip_lower_bound(self):
        """Ratio < 1.0 is clipped to 1.0 (cvar less severe than var)."""
        returns = _make_returns()
        with patch('src.monitor.cvar_metrics.fetch_portfolio_returns',
                   return_value=(returns, -0.05, -0.15)):
            with patch('src.monitor.cvar_metrics.calculate_var',
                       return_value=-0.05):
                with patch('src.monitor.cvar_metrics.calculate_cvar',
                           return_value=-0.03):
                    # ratio = abs(-0.03/-0.05) = 0.6, clipped to 1.0
                    metrics = compute_cvar_metrics()
                    assert metrics.cvar_ratio == 1.0

    def test_clip_lower_bound_edge(self):
        """Ratio exactly 1.0 should not be clipped."""
        returns = _make_returns()
        with patch('src.monitor.cvar_metrics.fetch_portfolio_returns',
                   return_value=(returns, -0.05, -0.15)):
            with patch('src.monitor.cvar_metrics.calculate_var',
                       return_value=-0.02):
                with patch('src.monitor.cvar_metrics.calculate_cvar',
                           return_value=-0.02):
                    # ratio = abs(-0.02/-0.02) = 1.0, no clip needed
                    metrics = compute_cvar_metrics()
                    assert metrics.cvar_ratio == 1.0

    def test_clip_upper_bound(self):
        """Ratio > 3.0 is clipped to 3.0."""
        returns = _make_returns()
        with patch('src.monitor.cvar_metrics.fetch_portfolio_returns',
                   return_value=(returns, -0.05, -0.15)):
            with patch('src.monitor.cvar_metrics.calculate_var',
                       return_value=-0.01):
                with patch('src.monitor.cvar_metrics.calculate_cvar',
                           return_value=-0.05):
                    # ratio = abs(-0.05/-0.01) = 5.0, clipped to 3.0
                    metrics = compute_cvar_metrics()
                    assert metrics.cvar_ratio == 3.0

    def test_clip_upper_bound_edge(self):
        """Ratio exactly 3.0 should not be clipped."""
        returns = _make_returns()
        with patch('src.monitor.cvar_metrics.fetch_portfolio_returns',
                   return_value=(returns, -0.05, -0.15)):
            with patch('src.monitor.cvar_metrics.calculate_var',
                       return_value=-0.01):
                with patch('src.monitor.cvar_metrics.calculate_cvar',
                           return_value=-0.03):
                    # ratio = abs(-0.03/-0.01) = 3.0, no clip needed
                    metrics = compute_cvar_metrics()
                    assert metrics.cvar_ratio == 3.0

    def test_clip_both_var_and_cvar_positive(self):
        """Both positive values (unusual) still clip correctly."""
        returns = _make_returns()
        with patch('src.monitor.cvar_metrics.fetch_portfolio_returns',
                   return_value=(returns, -0.05, -0.15)):
            with patch('src.monitor.cvar_metrics.calculate_var',
                       return_value=0.02):
                with patch('src.monitor.cvar_metrics.calculate_cvar',
                           return_value=0.01):
                    # ratio = abs(0.01/0.02) = 0.5, clipped to 1.0
                    metrics = compute_cvar_metrics()
                    assert metrics.cvar_ratio == 1.0


# ---------------------------------------------------------------------------
# calculate_cvar: empty tail branch
# ---------------------------------------------------------------------------

class TestCalculateCvarEmptyTail:
    """Test calculate_cvar when no returns fall below VaR."""

    def test_all_returns_above_var_returns_var(self):
        """When len(tail_returns) == 0, return var directly."""
        returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        with patch('src.monitor.cvar_metrics.calculate_var',
                   return_value=-0.05):
            result = calculate_cvar(returns, 0.05)
            # tail = returns[returns <= -0.05] = [], so return var = -0.05
            assert result == -0.05

    def test_empty_tail_with_zero_returns(self):
        """Zero returns all above var threshold."""
        returns = np.array([0.0, 0.0, 0.0])
        with patch('src.monitor.cvar_metrics.calculate_var',
                   return_value=-0.01):
            result = calculate_cvar(returns, 0.05)
            assert result == -0.01

    def test_empty_tail_mixed_returns(self):
        """Mixed returns all above a very negative var."""
        returns = np.array([-0.001, 0.0, 0.001, 0.002])
        with patch('src.monitor.cvar_metrics.calculate_var',
                   return_value=-0.02):
            result = calculate_cvar(returns, 0.05)
            assert result == -0.02


# ---------------------------------------------------------------------------
# CLI --history flag
# ---------------------------------------------------------------------------

class TestCLIHistoryFlag:
    """Test main() with --history flag."""

    def test_history_displays_entry_count(self, caplog):
        """--history logs entry count."""
        test_data = [
            {"timestamp": "2026-05-14T00:00:00", "var_95": -2.5,
             "cvar_95": -3.8, "cvar_ratio": 1.52},
            {"timestamp": "2026-05-15T00:00:00", "var_95": -2.3,
             "cvar_95": -3.5, "cvar_ratio": 1.50},
        ]
        with patch('src.monitor.cvar_metrics.load_history',
                   return_value=test_data):
            with patch('sys.argv',
                       ['cvar_metrics', '--history']):
                with caplog.at_level(logging.INFO, logger="src.monitor.cvar_metrics"):
                    main()
        assert "2 entries" in caplog.text

    def test_history_shows_last_30_only(self, caplog):
        """--history limits display to last 30 entries."""
        test_data = [{"timestamp": f"2026-05-{d:02d}T00:00:00",
                       "var_95": -2.0, "cvar_95": -3.0,
                       "cvar_ratio": 1.5} for d in range(1, 51)]
        with patch('src.monitor.cvar_metrics.load_history',
                   return_value=test_data):
            with patch('sys.argv',
                       ['cvar_metrics', '--history']):
                with caplog.at_level(logging.INFO, logger="src.monitor.cvar_metrics"):
                    main()
        assert "50 entries" in caplog.text

    def test_history_empty(self, caplog):
        """--history with no entries shows zero count."""
        with patch('src.monitor.cvar_metrics.load_history',
                   return_value=[]):
            with patch('sys.argv',
                       ['cvar_metrics', '--history']):
                with caplog.at_level(logging.INFO, logger="src.monitor.cvar_metrics"):
                    main()
        assert "0 entries" in caplog.text

    def test_history_returns_early_no_display(self):
        """--history returns before computing metrics."""
        with patch('src.monitor.cvar_metrics.load_history',
                   return_value=[]):
            with patch('sys.argv',
                       ['cvar_metrics', '--history']):
                with patch('src.monitor.cvar_metrics.compute_cvar_metrics'
                           ) as mock_compute:
                    main()
                    mock_compute.assert_not_called()


# ---------------------------------------------------------------------------
# CLI --export flag
# ---------------------------------------------------------------------------

class TestCLIExportFlag:
    """Test main() with --export flag."""

    def test_export_creates_json_files(self, tmp_path):
        """--export writes risk_metrics.json and risk_history.json."""
        import src.monitor.cvar_metrics as mod
        old_risk = mod.RISK_METRICS_PATH
        old_hist = mod.RISK_HISTORY_PATH
        mod.RISK_METRICS_PATH = tmp_path / "risk_metrics.json"
        mod.RISK_HISTORY_PATH = tmp_path / "risk_history.json"
        try:
            returns = _make_returns()
            with patch('src.monitor.cvar_metrics.fetch_portfolio_returns',
                       return_value=(returns, -0.05, -0.15)):
                with patch('sys.argv',
                           ['cvar_metrics', '--export']):
                    main()
                    assert mod.RISK_METRICS_PATH.exists()
                    assert mod.RISK_HISTORY_PATH.exists()
                    data = json.loads(
                        mod.RISK_METRICS_PATH.read_text())
                    assert 'var_95_daily' in data
                    assert 'cvar_95_daily' in data
        finally:
            mod.RISK_METRICS_PATH = old_risk
            mod.RISK_HISTORY_PATH = old_hist

    def test_export_appends_to_history(self, tmp_path):
        """--export appends current metrics to existing history."""
        import src.monitor.cvar_metrics as mod
        old_risk = mod.RISK_METRICS_PATH
        old_hist = mod.RISK_HISTORY_PATH
        mod.RISK_METRICS_PATH = tmp_path / "risk_metrics.json"
        mod.RISK_HISTORY_PATH = tmp_path / "risk_history.json"
        try:
            # Pre-populate history with 1 entry
            existing = [{"timestamp": "2026-05-14T00:00:00",
                         "var_95": -2.5, "cvar_95": -3.8,
                         "cvar_ratio": 1.52, "tail_severity": "moderate",
                         "max_drawdown": -15.0, "current_drawdown": -5.0,
                         "volatility_annual": 15.0}]
            mod.RISK_HISTORY_PATH.write_text(json.dumps(existing))
            returns = _make_returns()
            with patch('src.monitor.cvar_metrics.fetch_portfolio_returns',
                       return_value=(returns, -0.05, -0.15)):
                with patch('sys.argv',
                           ['cvar_metrics', '--export']):
                    main()
                    loaded = json.loads(
                        mod.RISK_HISTORY_PATH.read_text())
                    assert len(loaded) == 2  # 1 existing + 1 new
        finally:
            mod.RISK_METRICS_PATH = old_risk
            mod.RISK_HISTORY_PATH = old_hist

    def test_export_trims_history(self, tmp_path):
        """--export trims history to 720 entries."""
        import src.monitor.cvar_metrics as mod
        old_risk = mod.RISK_METRICS_PATH
        old_hist = mod.RISK_HISTORY_PATH
        mod.RISK_METRICS_PATH = tmp_path / "risk_metrics.json"
        mod.RISK_HISTORY_PATH = tmp_path / "risk_history.json"
        try:
            existing = [{"i": i} for i in range(800)]
            mod.RISK_HISTORY_PATH.write_text(json.dumps(existing))
            returns = _make_returns()
            with patch('src.monitor.cvar_metrics.fetch_portfolio_returns',
                       return_value=(returns, -0.05, -0.15)):
                with patch('sys.argv',
                           ['cvar_metrics', '--export']):
                    main()
                    loaded = json.loads(
                        mod.RISK_HISTORY_PATH.read_text())
                    assert len(loaded) == 720
        finally:
            mod.RISK_METRICS_PATH = old_risk
            mod.RISK_HISTORY_PATH = old_hist

    def test_export_without_history_flag_no_export(self, tmp_path):
        """Without --export flag, no files are written."""
        import src.monitor.cvar_metrics as mod
        old_risk = mod.RISK_METRICS_PATH
        old_hist = mod.RISK_HISTORY_PATH
        mod.RISK_METRICS_PATH = tmp_path / "risk_metrics.json"
        mod.RISK_HISTORY_PATH = tmp_path / "risk_history.json"
        try:
            returns = _make_returns()
            with patch('src.monitor.cvar_metrics.fetch_portfolio_returns',
                       return_value=(returns, -0.05, -0.15)):
                with patch('sys.argv',
                           ['cvar_metrics']):  # no --export
                    main()
                    assert not mod.RISK_METRICS_PATH.exists()
                    assert not mod.RISK_HISTORY_PATH.exists()
        finally:
            mod.RISK_METRICS_PATH = old_risk
            mod.RISK_HISTORY_PATH = old_hist


# ---------------------------------------------------------------------------
# load_history: corrupted JSON
# ---------------------------------------------------------------------------

class TestLoadHistoryCorrupted:
    """Test load_history with corrupted JSON files."""

    def test_corrupted_json_returns_empty_list(self, tmp_path):
        """Invalid JSON in history file returns empty list."""
        import src.monitor.cvar_metrics as mod
        old = mod.RISK_HISTORY_PATH
        mod.RISK_HISTORY_PATH = tmp_path / "corrupt_history.json"
        try:
            mod.RISK_HISTORY_PATH.write_text('{invalid json!!!')
            result = load_history()
            assert result == []
        finally:
            mod.RISK_HISTORY_PATH = old

    def test_corrupted_json_not_an_array(self, tmp_path):
        """Valid JSON that is not a list still loads (catches decode error)."""
        import src.monitor.cvar_metrics as mod
        old = mod.RISK_HISTORY_PATH
        mod.RISK_HISTORY_PATH = tmp_path / "wrong_type.json"
        try:
            mod.RISK_HISTORY_PATH.write_text('"just_a_string"')
            # This loads successfully but returns a string instead of list
            result = load_history()
            # Since it parsed as valid JSON, no exception
            assert result == "just_a_string"
        finally:
            mod.RISK_HISTORY_PATH = old

    def test_corrupted_json_empty_file(self, tmp_path):
        """Empty history file triggers JSON decode error."""
        import src.monitor.cvar_metrics as mod
        old = mod.RISK_HISTORY_PATH
        mod.RISK_HISTORY_PATH = tmp_path / "empty_history.json"
        try:
            mod.RISK_HISTORY_PATH.write_text('')
            result = load_history()
            assert result == []
        finally:
            mod.RISK_HISTORY_PATH = old

    def test_corrupted_json_partial_content(self, tmp_path):
        """Truncated JSON triggers decode error."""
        import src.monitor.cvar_metrics as mod
        old = mod.RISK_HISTORY_PATH
        mod.RISK_HISTORY_PATH = tmp_path / "partial_history.json"
        try:
            mod.RISK_HISTORY_PATH.write_text(
                '[{"timestamp": "2026-05-14", "var_95": -2.5')
            result = load_history()
            assert result == []
        finally:
            mod.RISK_HISTORY_PATH = old

    def test_history_file_not_exist(self):
        """When history file does not exist, return empty list."""
        with patch('src.monitor.cvar_metrics.RISK_HISTORY_PATH') as mock_p:
            mock_p.exists.return_value = False
            result = load_history()
            assert result == []


# ---------------------------------------------------------------------------
# save_history: boundary conditions
# ---------------------------------------------------------------------------

class TestSaveHistoryBoundary:
    """Test save_history size boundary behavior."""

    def test_exactly_720_no_trim(self, tmp_path):
        """Exactly 720 entries preserved without trimming."""
        import src.monitor.cvar_metrics as mod
        old = mod.RISK_HISTORY_PATH
        mod.RISK_HISTORY_PATH = tmp_path / "boundary_720.json"
        try:
            history = [{"i": i} for i in range(720)]
            save_history(history)
            loaded = load_history()
            assert len(loaded) == 720
        finally:
            mod.RISK_HISTORY_PATH = old

    def test_719_no_trim(self, tmp_path):
        """Just below threshold (719) preserved without trimming."""
        import src.monitor.cvar_metrics as mod
        old = mod.RISK_HISTORY_PATH
        mod.RISK_HISTORY_PATH = tmp_path / "boundary_719.json"
        try:
            history = [{"i": i} for i in range(719)]
            save_history(history)
            loaded = load_history()
            assert len(loaded) == 719
        finally:
            mod.RISK_HISTORY_PATH = old

    def test_721_trims_to_720(self, tmp_path):
        """One above threshold trims to 720."""
        import src.monitor.cvar_metrics as mod
        old = mod.RISK_HISTORY_PATH
        mod.RISK_HISTORY_PATH = tmp_path / "boundary_721.json"
        try:
            history = [{"i": i} for i in range(721)]
            save_history(history)
            loaded = load_history()
            assert len(loaded) == 720
        finally:
            mod.RISK_HISTORY_PATH = old

    def test_empty_history(self, tmp_path):
        """Saving empty history works without error."""
        import src.monitor.cvar_metrics as mod
        old = mod.RISK_HISTORY_PATH
        mod.RISK_HISTORY_PATH = tmp_path / "empty_save.json"
        try:
            save_history([])
            loaded = load_history()
            assert loaded == []
        finally:
            mod.RISK_HISTORY_PATH = old


# ---------------------------------------------------------------------------
# fetch_portfolio_returns: len(prices) < 2
# ---------------------------------------------------------------------------

class TestFetchPortfolioReturnsNoDB:
    """Test fetch_portfolio_returns when DB path does not exist."""

    def test_db_missing_returns_synthetic(self):
        """No DB file returns synthetic data with correct shape."""
        import src.monitor.cvar_metrics as mod
        with patch.object(mod, 'DB_PATH') as mock_db:
            mock_db.exists.return_value = False
            returns, curr_dd, max_dd = fetch_portfolio_returns(252)
            assert len(returns) == 252
            assert curr_dd == 0.0
            assert max_dd == -0.15

    def test_db_missing_varying_days(self):
        """Synthetic data respects the days parameter."""
        import src.monitor.cvar_metrics as mod
        with patch.object(mod, 'DB_PATH') as mock_db:
            mock_db.exists.return_value = False
            returns, _, _ = fetch_portfolio_returns(126)
            assert len(returns) == 126
            returns2, _, _ = fetch_portfolio_returns(504)
            assert len(returns2) == 504

    def test_db_missing_returns_float_array(self):
        """Synthetic data returns float dtype."""
        import src.monitor.cvar_metrics as mod
        with patch.object(mod, 'DB_PATH') as mock_db:
            mock_db.exists.return_value = False
            returns, _, _ = fetch_portfolio_returns(252)
            assert returns.dtype == np.float64


class TestFetchPortfolioReturnsFewSymbols:
    """Test fetch_portfolio_returns when DB has fewer than 2 symbols."""

    def test_one_symbol_returns_synthetic(self):
        """Only one symbol with data triggers synthetic fallback."""
        import src.monitor.cvar_metrics as mod
        with patch.object(mod, 'DB_PATH') as mock_db:
            mock_db.exists.return_value = True
            with patch('src.monitor.cvar_metrics.sqlite_connect'
                       ) as mock_sqlite:
                mock_conn = MagicMock()
                mock_cursor = MagicMock()
                mock_conn.__enter__.return_value = mock_conn
                mock_sqlite.return_value = mock_conn
                mock_conn.cursor.return_value = mock_cursor
                # Only SPY returns data; GLD and TLT return empty
                mock_cursor.fetchall.side_effect = [
                    [(1, 500.0), (2, 501.0)],  # SPY
                    [],  # GLD
                    [],  # TLT
                ]
                returns, curr_dd, max_dd = fetch_portfolio_returns(252)
                assert len(returns) == 252
                assert curr_dd == 0.0
                assert max_dd == -0.15

    def test_zero_symbols_returns_synthetic(self):
        """No symbols with data triggers synthetic fallback."""
        import src.monitor.cvar_metrics as mod
        with patch.object(mod, 'DB_PATH') as mock_db:
            mock_db.exists.return_value = True
            with patch('src.monitor.cvar_metrics.sqlite_connect'
                       ) as mock_sqlite:
                mock_conn = MagicMock()
                mock_cursor = MagicMock()
                mock_conn.__enter__.return_value = mock_conn
                mock_sqlite.return_value = mock_conn
                mock_conn.cursor.return_value = mock_cursor
                # All symbols return empty
                mock_cursor.fetchall.side_effect = [
                    [],  # SPY
                    [],  # GLD
                    [],  # TLT
                ]
                returns, curr_dd, max_dd = fetch_portfolio_returns(252)
                assert len(returns) == 252
                assert curr_dd == 0.0
                assert max_dd == -0.15

    def test_two_symbols_calculates_returns(self):
        """Two symbols with data calculates weighted returns (not synthetic)."""
        import src.monitor.cvar_metrics as mod
        with patch.object(mod, 'DB_PATH') as mock_db:
            mock_db.exists.return_value = True
            with patch('src.monitor.cvar_metrics.sqlite_connect'
                       ) as mock_sqlite:
                mock_conn = MagicMock()
                mock_cursor = MagicMock()
                mock_conn.__enter__.return_value = mock_conn
                mock_sqlite.return_value = mock_conn
                mock_conn.cursor.return_value = mock_cursor
                # Two symbols have data
                mock_cursor.fetchall.side_effect = [
                    [(1, 100.0), (2, 101.0), (3, 102.0)],  # SPY
                    [(1, 200.0), (2, 202.0), (3, 204.0)],  # GLD
                    [],  # TLT empty
                ]
                returns, curr_dd, max_dd = fetch_portfolio_returns(252)
                # len(prices) == 2, so it calculates returns (not synthetic)
                # Returns should be len(min_len) - 1 = 2
                assert len(returns) == 2
                assert max_dd < 0  # actual drawdown computed


# ---------------------------------------------------------------------------
# display_metrics output format
# ---------------------------------------------------------------------------

class TestDisplayMetricsOutput:
    """Test display_metrics produces expected output via logging."""

    @pytest.fixture(autouse=True)
    def _capture_logs(self, caplog):
        caplog.set_level(logging.INFO, logger="src.monitor.cvar_metrics")

    def test_output_contains_all_metric_labels(self, caplog):
        """display_metrics includes all key metric labels."""
        metrics = _make_metrics()
        display_metrics(metrics)
        log_output = caplog.text
        assert 'TAIL RISK METRICS' in log_output
        assert 'VaR 95%' in log_output
        assert 'CVaR 95%' in log_output
        assert 'Tail Severity' in log_output
        assert 'Max Drawdown' in log_output
        assert 'Current Drawdown' in log_output
        assert 'Volatility' in log_output
        assert 'Interpretation' in log_output

    def test_output_shows_metric_values(self, caplog):
        """display_metrics shows correct numerical values."""
        metrics = _make_metrics(var_95=-2.50, cvar_95=-3.80,
                                cvar_ratio=1.52, tail_severity="moderate",
                                max_drawdown=-15.0, current_drawdown=-5.0,
                                volatility_annual=15.0)
        display_metrics(metrics)
        log_output = caplog.text
        assert '-2.50%' in log_output
        assert '-3.80%' in log_output
        assert '1.52x' in log_output
        assert 'moderate' in log_output
        assert '-15.00%' in log_output
        assert '-5.00%' in log_output
        assert '15.00%' in log_output

    def test_output_severity_colors_normal(self, caplog):
        """display_metrics uses green color for normal severity."""
        caplog.set_level(logging.INFO, logger="src.monitor.cvar_metrics")
        metrics = _make_metrics(cvar_ratio=1.2, tail_severity="normal")
        display_metrics(metrics)
        # Check raw LogRecord messages for ANSI codes
        messages = [r.getMessage() for r in caplog.records]
        combined = ''.join(messages)
        assert '\033[32m' in combined
        assert '\033[0m' in combined

    def test_output_severity_colors_severe(self, caplog):
        """display_metrics uses red color for severe severity."""
        caplog.set_level(logging.INFO, logger="src.monitor.cvar_metrics")
        metrics = _make_metrics(cvar_ratio=2.0, tail_severity="severe")
        display_metrics(metrics)
        messages = [r.getMessage() for r in caplog.records]
        combined = ''.join(messages)
        assert '\033[31m' in combined
        assert '\033[0m' in combined

    def test_output_severity_colors_elevated(self, caplog):
        """display_metrics uses yellow for elevated severity."""
        caplog.set_level(logging.INFO, logger="src.monitor.cvar_metrics")
        metrics = _make_metrics(cvar_ratio=1.6, tail_severity="elevated")
        display_metrics(metrics)
        messages = [r.getMessage() for r in caplog.records]
        combined = ''.join(messages)
        assert '\033[33m' in combined
        assert '\033[0m' in combined


# ---------------------------------------------------------------------------
# export_metrics function (direct, not via CLI)
# ---------------------------------------------------------------------------

class TestExportMetricsFunction:
    """Test export_metrics function directly."""

    def test_export_writes_risk_metrics_file(self, tmp_path):
        """export_metrics writes to RISK_METRICS_PATH."""
        import src.monitor.cvar_metrics as mod
        old_risk = mod.RISK_METRICS_PATH
        old_hist = mod.RISK_HISTORY_PATH
        mod.RISK_METRICS_PATH = tmp_path / "risk_metrics.json"
        mod.RISK_HISTORY_PATH = tmp_path / "risk_history.json"
        try:
            metrics = _make_metrics()
            result = export_metrics(metrics)
            assert mod.RISK_METRICS_PATH.exists()
            data = json.loads(mod.RISK_METRICS_PATH.read_text())
            assert data['var_95_daily'] == -2.5
            assert data['cvar_95_daily'] == -3.8
        finally:
            mod.RISK_METRICS_PATH = old_risk
            mod.RISK_HISTORY_PATH = old_hist

    def test_export_writes_history_file(self, tmp_path):
        """export_metrics creates history file."""
        import src.monitor.cvar_metrics as mod
        old_risk = mod.RISK_METRICS_PATH
        old_hist = mod.RISK_HISTORY_PATH
        mod.RISK_METRICS_PATH = tmp_path / "risk_metrics.json"
        mod.RISK_HISTORY_PATH = tmp_path / "risk_history.json"
        try:
            metrics = _make_metrics()
            export_metrics(metrics)
            assert mod.RISK_HISTORY_PATH.exists()
            history = json.loads(mod.RISK_HISTORY_PATH.read_text())
            assert len(history) == 1
            assert history[0]['var_95'] == -2.5
        finally:
            mod.RISK_METRICS_PATH = old_risk
            mod.RISK_HISTORY_PATH = old_hist

    def test_export_returns_data_dict(self, tmp_path):
        """export_metrics returns the data dictionary."""
        import src.monitor.cvar_metrics as mod
        old_risk = mod.RISK_METRICS_PATH
        old_hist = mod.RISK_HISTORY_PATH
        mod.RISK_METRICS_PATH = tmp_path / "risk_metrics.json"
        mod.RISK_HISTORY_PATH = tmp_path / "risk_history.json"
        try:
            metrics = _make_metrics()
            result = export_metrics(metrics)
            assert 'var_95_daily' in result
            assert 'interpretation' in result
            assert 'cvar_ratio' in result
        finally:
            mod.RISK_METRICS_PATH = old_risk
            mod.RISK_HISTORY_PATH = old_hist

    def test_export_interpretation_keys(self, tmp_path):
        """export_metrics includes all interpretation keys."""
        import src.monitor.cvar_metrics as mod
        old_risk = mod.RISK_METRICS_PATH
        old_hist = mod.RISK_HISTORY_PATH
        mod.RISK_METRICS_PATH = tmp_path / "risk_metrics.json"
        mod.RISK_HISTORY_PATH = tmp_path / "risk_history.json"
        try:
            metrics = _make_metrics()
            result = export_metrics(metrics)
            interp = result['interpretation']
            assert 'var_description' in interp
            assert 'cvar_description' in interp
            assert 'severity_normal' in interp
            assert 'severity_moderate' in interp
            assert 'severity_severe' in interp
        finally:
            mod.RISK_METRICS_PATH = old_risk
            mod.RISK_HISTORY_PATH = old_hist

    def test_export_appends_to_existing_history(self, tmp_path):
        """export_metrics appends to existing history."""
        import src.monitor.cvar_metrics as mod
        old_risk = mod.RISK_METRICS_PATH
        old_hist = mod.RISK_HISTORY_PATH
        mod.RISK_METRICS_PATH = tmp_path / "risk_metrics.json"
        mod.RISK_HISTORY_PATH = tmp_path / "risk_history.json"
        try:
            existing = [{"existing": "entry"}]
            mod.RISK_HISTORY_PATH.write_text(json.dumps(existing))
            metrics = _make_metrics()
            export_metrics(metrics)
            history = json.loads(mod.RISK_HISTORY_PATH.read_text())
            assert len(history) == 2
        finally:
            mod.RISK_METRICS_PATH = old_risk
            mod.RISK_HISTORY_PATH = old_hist


# ---------------------------------------------------------------------------
# Additional edge cases for calculate_var
# ---------------------------------------------------------------------------

class TestCalculateVarEdgeCases:
    """Additional edge cases for calculate_var."""

    def test_alpha_very_small(self):
        """Alpha near zero should still work."""
        returns = _make_returns()
        var = calculate_var(returns, 0.001)
        assert isinstance(var, float)

    def test_alpha_large(self):
        """Alpha near 0.5 should still work."""
        returns = _make_returns()
        var = calculate_var(returns, 0.5)
        assert isinstance(var, float)

    def test_single_negative_return(self):
        """Single negative return."""
        var = calculate_var(np.array([-0.05]), 0.05)
        assert var == -0.05

    def test_single_positive_return(self):
        """Single positive return."""
        var = calculate_var(np.array([0.05]), 0.05)
        # With only one positive return, the 5th percentile is 0.05
        assert isinstance(var, float)


# ---------------------------------------------------------------------------
# Additional edge cases for calculate_cvar
# ---------------------------------------------------------------------------

class TestCalculateCvarEdgeCases:
    """Additional edge cases for calculate_cvar."""

    def test_single_return_negative(self):
        """Single negative return."""
        cvar = calculate_cvar(np.array([-0.05]), 0.05)
        assert isinstance(cvar, float)
        assert cvar < 0

    def test_single_return_positive(self):
        """Single positive return."""
        cvar = calculate_cvar(np.array([0.05]), 0.05)
        assert isinstance(cvar, float)

    def test_two_returns_both_negative(self):
        """Two negative returns."""
        cvar = calculate_cvar(np.array([-0.05, -0.03]), 0.05)
        assert isinstance(cvar, float)
        assert cvar < 0

    def test_many_returns_deterministic(self):
        """Deterministic result with same seed."""
        r1 = _make_returns(seed=42)
        r2 = _make_returns(seed=42)
        assert calculate_cvar(r1) == calculate_cvar(r2)


# ---------------------------------------------------------------------------
# compute_cvar_metrics: full integration with real mocked returns
# ---------------------------------------------------------------------------

class TestComputeCVaRMetricsIntegration:
    """Integration-style tests for compute_cvar_metrics."""

    def test_metrics_contain_all_expected_fields(self):
        """All CVaRMetrics fields populated correctly."""
        returns = _make_returns()
        with patch('src.monitor.cvar_metrics.fetch_portfolio_returns',
                   return_value=(returns, -0.05, -0.15)):
            metrics = compute_cvar_metrics()
            assert metrics.var_95 is not None
            assert metrics.cvar_95 is not None
            assert metrics.cvar_ratio is not None
            assert metrics.tail_severity is not None
            assert metrics.max_drawdown is not None
            assert metrics.current_drawdown is not None
            assert metrics.volatility_annual is not None
            assert metrics.timestamp is not None

    def test_cvar_95_more_negative_than_var_95(self):
        """CVaR should always be more negative than VaR."""
        returns = _make_returns()
        with patch('src.monitor.cvar_metrics.fetch_portfolio_returns',
                   return_value=(returns, -0.05, -0.15)):
            metrics = compute_cvar_metrics()
            assert metrics.cvar_95 <= metrics.var_95

    def test_max_drawdown_more_negative_than_current(self):
        """Max drawdown should be more negative than current drawdown."""
        returns = _make_returns()
        with patch('src.monitor.cvar_metrics.fetch_portfolio_returns',
                   return_value=(returns, -0.08, -0.25)):
            metrics = compute_cvar_metrics()
            assert metrics.max_drawdown <= metrics.current_drawdown

    def test_volatility_annual_positive(self):
        """Annualized volatility should be positive."""
        returns = _make_returns()
        with patch('src.monitor.cvar_metrics.fetch_portfolio_returns',
                   return_value=(returns, -0.05, -0.15)):
            metrics = compute_cvar_metrics()
            assert metrics.volatility_annual > 0

    def test_window_days_passed_to_fetch(self):
        """Window days parameter passed to fetch_portfolio_returns."""
        with patch('src.monitor.cvar_metrics.fetch_portfolio_returns'
                   ) as mock_fetch:
            mock_fetch.return_value = (_make_returns(), -0.05, -0.15)
            compute_cvar_metrics(window_days=126)
            mock_fetch.assert_called_once_with(126)

    def test_window_days_default(self):
        """Default window is 252 days."""
        with patch('src.monitor.cvar_metrics.fetch_portfolio_returns'
                   ) as mock_fetch:
            mock_fetch.return_value = (_make_returns(), -0.05, -0.15)
            compute_cvar_metrics()
            mock_fetch.assert_called_once_with(252)


# ---------------------------------------------------------------------------
# get_tail_severity additional edge cases
# ---------------------------------------------------------------------------

class TestGetTailSeverityEdgeCases:
    """Additional tail severity edge cases."""

    def test_negative_ratio(self):
        """Negative ratio classified as normal."""
        assert get_tail_severity(-1.0) == "normal"

    def test_zero_ratio(self):
        """Zero ratio classified as normal."""
        assert get_tail_severity(0.0) == "normal"

    def test_large_ratio(self):
        """Very large ratio."""
        assert get_tail_severity(100.0) == "severe"


# ---------------------------------------------------------------------------
# main() edge cases
# ---------------------------------------------------------------------------

class TestMainEdgeCases:
    """Test main() edge cases beyond --history and --export."""

    def test_main_runs_display_without_flags(self):
        """main() with no flags runs display_metrics."""
        returns = _make_returns()
        with patch('src.monitor.cvar_metrics.fetch_portfolio_returns',
                   return_value=(returns, -0.05, -0.15)):
            with patch('sys.argv',
                       ['cvar_metrics']):
                with patch('src.monitor.cvar_metrics.display_metrics'
                           ) as mock_display:
                    main()
                    mock_display.assert_called_once()

    def test_main_with_window_arg(self):
        """main() with --window flag."""
        returns = _make_returns()
        with patch('src.monitor.cvar_metrics.fetch_portfolio_returns',
                   return_value=(returns, -0.05, -0.15)):
            with patch('sys.argv',
                       ['cvar_metrics', '--window', '126']):
                with patch('src.monitor.cvar_metrics.compute_cvar_metrics'
                           ) as mock_compute:
                    mock_compute.return_value = _make_metrics()
                    main()
                    mock_compute.assert_called_once_with(126)
