#!/usr/bin/env python3
"""
Tests for ensemble backtest engine — data class, returns calculation,
max drawdown, crisis alpha, allocation deltas, and target validation.
"""
import numpy as np

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.backtest.metrics import BacktestResult
from src.backtest.ensemble_backtest import (
    EnsembleBacktestEngine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result():
    return BacktestResult(
        total_return=3.5, cagr=0.10,
        volatility=0.11, sharpe_ratio=0.90,
        max_drawdown=-0.25,
        total_rebalances=250,
        extras={
            "start_date": "2005-01-01",
            "end_date": "2026-05-13",
            "portfolio": "46/38/16",
            "sortino_ratio": 1.2,
            "max_dd_duration": 180,
            "calmar_ratio": 0.40,
            "var_95": -0.018,
            "cvar_95": -0.025,
            "avg_signal_confidence": 0.65,
            "regime_distribution": {"normal": 0.6, "high_vol": 0.25, "crisis": 0.1, "recovery": 0.05},
            "crisis_alpha_2008": 0.05,
            "crisis_alpha_2020": 0.03,
            "crisis_alpha_2022": 0.02,
            "source_contributions": {
                "tsfm": {"hits": 150, "avg_confidence": 0.72, "return": 0.02, "sharpe": 0.15},
                "cta": {"hits": 120, "avg_confidence": 0.65, "return": 0.01, "sharpe": 0.10},
            },
            "rolling_sharpe_1y": [("2025-01-01", 0.85), ("2025-06-01", 0.92)],
        },
    )


def _make_engine(tmp_path):
    engine = EnsembleBacktestEngine.__new__(EnsembleBacktestEngine)
    engine.db_path = tmp_path / "market.db"
    engine.integrator = MagicMock()
    engine._price_cache = {}
    engine._signal_cache = {}
    return engine


# ---------------------------------------------------------------------------
# EnsembleBacktestResult tests
# ---------------------------------------------------------------------------

class TestEnsembleBacktestResult:
    def test_creation(self):
        r = _make_result()
        assert r.extras["portfolio"] == '46/38/16'
        assert r.sharpe_ratio == 0.90

    def test_to_dict(self):
        from dataclasses import asdict
        r = _make_result()
        d = asdict(r)
        assert d['sharpe_ratio'] == 0.90
        assert d['max_drawdown'] == -0.25
        assert 'crisis_alpha_2008' in d['extras']
        assert 'source_contributions' in d['extras']

    def test_regime_distribution(self):
        r = _make_result()
        assert abs(sum(r.extras["regime_distribution"].values()) - 1.0) < 0.01


# ---------------------------------------------------------------------------
# EnsembleBacktestEngine tests
# ---------------------------------------------------------------------------

class TestEnsembleBacktestEngine:
    def test_crisis_periods_defined(self):
        assert '2008' in EnsembleBacktestEngine.CRISIS_PERIODS
        assert '2020' in EnsembleBacktestEngine.CRISIS_PERIODS
        assert '2022' in EnsembleBacktestEngine.CRISIS_PERIODS

    def test_tx_cost(self):
        assert EnsembleBacktestEngine.TX_COST_BPS == 5.0

    def test_calculate_returns(self, tmp_path):
        engine = _make_engine(tmp_path)
        prices = [100.0, 101.0, 102.0, 100.0, 103.0]
        returns = engine._calculate_returns(prices)
        assert len(returns) == 4
        assert isinstance(returns, np.ndarray)

    def test_calculate_returns_log(self, tmp_path):
        engine = _make_engine(tmp_path)
        prices = [100.0, 110.0]
        returns = engine._calculate_returns(prices)
        expected = np.log(110.0 / 100.0)
        assert abs(returns[0] - expected) < 0.001

    def test_max_drawdown_no_dd(self, tmp_path):
        engine = _make_engine(tmp_path)
        curve = np.array([100, 110, 120, 130])
        dd, duration = engine._calculate_max_drawdown(curve)
        assert dd == 0.0
        assert duration == 0

    def test_max_drawdown_simple(self, tmp_path):
        engine = _make_engine(tmp_path)
        curve = np.array([100, 110, 90, 95, 120])
        dd, duration = engine._calculate_max_drawdown(curve)
        assert dd < 0
        assert abs(dd - (90 - 110) / 110) < 0.01

    def test_max_drawdown_duration(self, tmp_path):
        engine = _make_engine(tmp_path)
        # Peak at 100 (index 0), then below peak for 4 days (indices 1-4), then recovery
        curve = np.array([100, 95, 90, 85, 95, 105])
        dd, duration = engine._calculate_max_drawdown(curve)
        assert duration == 4  # 4 days below peak

    def test_max_drawdown_all_time_high(self, tmp_path):
        engine = _make_engine(tmp_path)
        curve = np.array([100, 90, 80, 70, 60])
        dd, duration = engine._calculate_max_drawdown(curve)
        assert dd == (60 - 100) / 100
        assert duration == 4

    def test_crisis_alpha(self, tmp_path):
        engine = _make_engine(tmp_path)
        portfolio = {'2020-02-20': -0.02, '2020-02-21': -0.03, '2020-02-24': 0.01}
        benchmark = {'2020-02-20': -0.03, '2020-02-21': -0.04, '2020-02-24': -0.01}
        alpha = engine._calculate_crisis_alpha(
            portfolio, benchmark, ('2020-02-20', '2020-02-24')
        )
        assert alpha > 0  # Portfolio outperformed

    def test_crisis_alpha_empty(self, tmp_path):
        engine = _make_engine(tmp_path)
        alpha = engine._calculate_crisis_alpha({}, {}, ('2020-02-20', '2020-02-24'))
        assert alpha == 0.0

    def test_crisis_alpha_no_overlap(self, tmp_path):
        engine = _make_engine(tmp_path)
        portfolio = {'2019-01-01': 0.01}
        benchmark = {'2019-01-01': 0.01}
        alpha = engine._calculate_crisis_alpha(
            portfolio, benchmark, ('2020-02-20', '2020-02-24')
        )
        assert alpha == 0.0

    def test_allocation_deltas_neutral(self, tmp_path):
        engine = _make_engine(tmp_path)
        current = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16}
        signals = {
            'SPY': {'score': 0.0, 'confidence': 0.5, 'regime': 'neutral', 'sources': []},
            'GLD': {'score': 0.0, 'confidence': 0.5, 'regime': 'neutral', 'sources': []},
            'TLT': {'score': 0.0, 'confidence': 0.5, 'regime': 'neutral', 'sources': []},
        }
        target = engine._calculate_allocation_deltas(current, signals)
        assert abs(sum(target.values()) - 1.0) < 0.01

    def test_allocation_deltas_positive_signal(self, tmp_path):
        engine = _make_engine(tmp_path)
        current = {'SPY': 0.5, 'GLD': 0.5}
        signals = {
            'SPY': {'score': 0.8, 'confidence': 0.9, 'regime': 'bull', 'sources': []},
            'GLD': {'score': -0.2, 'confidence': 0.5, 'regime': 'neutral', 'sources': []},
        }
        target = engine._calculate_allocation_deltas(current, signals)
        assert target['SPY'] > current['SPY']

    def test_allocation_deltas_sums_to_one(self, tmp_path):
        engine = _make_engine(tmp_path)
        current = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16}
        signals = {
            'SPY': {'score': 0.5, 'confidence': 0.8, 'regime': 'bull', 'sources': []},
            'GLD': {'score': -0.3, 'confidence': 0.6, 'regime': 'neutral', 'sources': []},
            'TLT': {'score': 0.1, 'confidence': 0.4, 'regime': 'neutral', 'sources': []},
        }
        target = engine._calculate_allocation_deltas(current, signals)
        assert abs(sum(target.values()) - 1.0) < 0.01

    def test_fetch_prices_no_db(self, tmp_path):
        engine = _make_engine(tmp_path)
        engine.db_path = tmp_path / "nonexistent.db"
        result = engine._fetch_historical_prices('SPY', '2020-01-01', '2020-12-31')
        assert result == []

    def test_fetch_prices_with_db(self, tmp_path):
        engine = _make_engine(tmp_path)
        import sqlite3
        conn = sqlite3.connect(str(engine.db_path))
        conn.execute("CREATE TABLE prices (symbol TEXT, date TEXT, close REAL, open REAL, high REAL, low REAL, volume INTEGER)")
        conn.execute("INSERT INTO prices VALUES ('SPY', '2020-01-02', 323.0, 322.0, 324.0, 321.0, 1000000)")
        conn.execute("INSERT INTO prices VALUES ('SPY', '2020-01-03', 324.0, 323.0, 325.0, 322.0, 1100000)")
        conn.commit()
        conn.close()
        result = engine._fetch_historical_prices('SPY', '2020-01-01', '2020-12-31')
        assert len(result) == 2
        assert result[0]['close'] == 323.0

    def test_fetch_prices_caching(self, tmp_path):
        engine = _make_engine(tmp_path)
        engine._price_cache['SPY:2020-01-01:2020-12-31'] = [{'date': '2020-01-02', 'close': 323.0}]
        result = engine._fetch_historical_prices('SPY', '2020-01-01', '2020-12-31')
        assert len(result) == 1

    def test_validate_target_pass(self, tmp_path):
        engine = _make_engine(tmp_path)
        result = _make_result()
        result.sharpe_ratio = 1.0
        assert engine.validate_target(result, target_sharpe=0.95) is True

    def test_validate_target_fail(self, tmp_path):
        engine = _make_engine(tmp_path)
        result = _make_result()
        result.sharpe_ratio = 0.80
        assert engine.validate_target(result, target_sharpe=0.95) is False


class TestEnsembleBacktestEngineExtended:
    """Extended tests for EnsembleBacktestEngine."""

    def test_crisis_periods_keys(self):
        """Crisis periods should cover 2008, 2020, 2022."""
        periods = EnsembleBacktestEngine.CRISIS_PERIODS
        assert '2008' in periods
        assert '2020' in periods
        assert '2022' in periods

    def test_crisis_periods_are_tuples(self):
        """Each crisis period should be a tuple of (start, end) strings."""
        for key, period in EnsembleBacktestEngine.CRISIS_PERIODS.items():
            assert isinstance(period, tuple), f"{key} is not a tuple"
            assert len(period) == 2, f"{key} should have start/end"

    def test_tx_cost_bps(self):
        """TX_COST_BPS should be a reasonable positive value."""
        assert EnsembleBacktestEngine.TX_COST_BPS > 0
        assert EnsembleBacktestEngine.TX_COST_BPS < 100  # Less than 1%

    def test_calculate_returns_single_price(self, tmp_path):
        """Single price should return empty array."""
        engine = _make_engine(tmp_path)
        returns = engine._calculate_returns([100.0])
        assert len(returns) == 0

    def test_calculate_returns_constant_price(self, tmp_path):
        """Constant prices should give near-zero returns."""
        engine = _make_engine(tmp_path)
        returns = engine._calculate_returns([100.0] * 10)
        assert all(abs(r) < 1e-10 for r in returns)

    def test_calculate_returns_two_prices(self, tmp_path):
        """Two prices should give one return."""
        engine = _make_engine(tmp_path)
        returns = engine._calculate_returns([100.0, 105.0])
        assert len(returns) == 1
        expected = np.log(105.0 / 100.0)
        assert abs(returns[0] - expected) < 0.001

    def test_max_drawdown_mountain_shape(self, tmp_path):
        """Mountain-shaped curve: up then down then up."""
        engine = _make_engine(tmp_path)
        curve = np.array([100, 120, 110, 130, 115, 140])
        dd, duration = engine._calculate_max_drawdown(curve)
        assert dd < 0
        assert duration > 0

    def test_max_drawdown_flat_curve(self, tmp_path):
        """Flat curve should have zero drawdown."""
        engine = _make_engine(tmp_path)
        curve = np.array([100.0] * 10)
        dd, duration = engine._calculate_max_drawdown(curve)
        assert dd == 0.0
        assert duration == 0

    def test_max_drawdown_single_point(self, tmp_path):
        """Single point should have zero drawdown."""
        engine = _make_engine(tmp_path)
        curve = np.array([100.0])
        dd, duration = engine._calculate_max_drawdown(curve)
        assert dd == 0.0
        assert duration == 0

    def test_crisis_alpha_portfolio_underperforms(self, tmp_path):
        """When portfolio underperforms, alpha should be negative."""
        engine = _make_engine(tmp_path)
        portfolio = {'2020-02-20': -0.05, '2020-02-21': -0.03}
        benchmark = {'2020-02-20': -0.01, '2020-02-21': 0.01}
        alpha = engine._calculate_crisis_alpha(
            portfolio, benchmark, ('2020-02-20', '2020-02-21')
        )
        assert alpha < 0

    def test_crisis_alpha_exact_match(self, tmp_path):
        """When portfolio matches benchmark, alpha should be near zero."""
        engine = _make_engine(tmp_path)
        portfolio = {'2020-02-20': 0.01, '2020-02-21': -0.02}
        benchmark = {'2020-02-20': 0.01, '2020-02-21': -0.02}
        alpha = engine._calculate_crisis_alpha(
            portfolio, benchmark, ('2020-02-20', '2020-02-21')
        )
        assert abs(alpha) < 0.001

    def test_allocation_deltas_negative_signal(self, tmp_path):
        """Negative signal should reduce allocation."""
        engine = _make_engine(tmp_path)
        current = {'SPY': 0.5, 'GLD': 0.5}
        signals = {
            'SPY': {'score': -0.8, 'confidence': 0.9, 'regime': 'bear', 'sources': []},
            'GLD': {'score': 0.2, 'confidence': 0.5, 'regime': 'neutral', 'sources': []},
        }
        target = engine._calculate_allocation_deltas(current, signals)
        assert target['SPY'] < current['SPY']

    def test_allocation_deltas_zero_confidence(self, tmp_path):
        """Zero confidence signals should produce minimal change."""
        engine = _make_engine(tmp_path)
        current = {'SPY': 0.5, 'GLD': 0.5}
        signals = {
            'SPY': {'score': 0.8, 'confidence': 0.0, 'regime': 'neutral', 'sources': []},
            'GLD': {'score': 0.0, 'confidence': 0.0, 'regime': 'neutral', 'sources': []},
        }
        target = engine._calculate_allocation_deltas(current, signals)
        # With zero confidence, deltas should be minimal
        for asset in current:
            assert abs(target[asset] - current[asset]) < 0.05

    def test_validate_target_exact_threshold(self, tmp_path):
        """Result at exactly target Sharpe should pass."""
        engine = _make_engine(tmp_path)
        result = _make_result()
        result.sharpe_ratio = 0.95
        assert engine.validate_target(result, target_sharpe=0.95) is True

    def test_validate_target_custom_threshold(self, tmp_path):
        """Custom target threshold should be respected."""
        engine = _make_engine(tmp_path)
        result = _make_result()
        result.sharpe_ratio = 0.50
        assert engine.validate_target(result, target_sharpe=0.30) is True
        assert engine.validate_target(result, target_sharpe=0.80) is False

    def test_fetch_prices_empty_db(self, tmp_path):
        """Database with no matching rows should return empty list."""
        engine = _make_engine(tmp_path)
        import sqlite3
        conn = sqlite3.connect(str(engine.db_path))
        conn.execute("CREATE TABLE prices (symbol TEXT, date TEXT, close REAL, open REAL, high REAL, low REAL, volume INTEGER)")
        conn.commit()
        conn.close()
        result = engine._fetch_historical_prices('SPY', '2020-01-01', '2020-12-31')
        assert result == []


class TestEnsembleBacktestResultExtended:
    """Extended tests for ensemble backtest results."""

    def test_result_all_extras_keys(self):
        """Result should contain expected extras keys."""
        r = _make_result()
        expected_keys = [
            "start_date", "end_date", "portfolio", "sortino_ratio",
            "max_dd_duration", "calmar_ratio", "var_95", "cvar_95",
            "avg_signal_confidence", "regime_distribution",
        ]
        for key in expected_keys:
            assert key in r.extras, f"Missing key: {key}"

    def test_result_crisis_alpha_keys(self):
        """Result should contain crisis alpha for each crisis year."""
        r = _make_result()
        assert "crisis_alpha_2008" in r.extras
        assert "crisis_alpha_2020" in r.extras
        assert "crisis_alpha_2022" in r.extras

    def test_result_source_contributions(self):
        """Result should contain source contributions."""
        r = _make_result()
        assert "source_contributions" in r.extras
        for name, contrib in r.extras["source_contributions"].items():
            assert "hits" in contrib
            assert "avg_confidence" in contrib

    def test_result_rolling_sharpe(self):
        """Result should contain rolling Sharpe data."""
        r = _make_result()
        assert "rolling_sharpe_1y" in r.extras
        assert len(r.extras["rolling_sharpe_1y"]) > 0

    def test_result_negative_drawdown(self):
        """Max drawdown should be negative."""
        r = _make_result()
        assert r.max_drawdown < 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
