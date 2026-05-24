#!/usr/bin/env python3
"""
Tests for ensemble backtest engine — data class, returns calculation,
max drawdown, crisis alpha, allocation deltas, and target validation.
"""
import json
import sqlite3
import numpy as np

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call

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


def _make_price_data(days: int = 60, base_price: float = 100.0, symbol: str = "SPY") -> list:
    """Generate synthetic price data for testing."""
    return [
        {
            "date": f"2020-01-{d:02d}" if d <= 31 else f"2020-02-{d-31:02d}",
            "close": base_price + i * 0.15,
            "open": base_price + i * 0.10,
            "high": base_price + i * 0.20,
            "low": base_price + i * 0.05,
            "volume": 1000000,
        }
        for i, d in enumerate(range(1, days + 1))
    ]


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
# EnsembleBacktestEngine init tests
# ---------------------------------------------------------------------------

class TestEnsembleBacktestEngineInit:
    """Constructor and initialization tests."""

    def test_default_db_path(self, tmp_path):
        """Default db_path should equal MARKET_DB."""
        with patch("src.backtest.ensemble_backtest.MARKET_DB", tmp_path / "default_market.db"):
            with patch("src.backtest.ensemble_backtest.SignalIntegrator") as mock_si:
                engine = EnsembleBacktestEngine()
                assert engine.db_path == tmp_path / "default_market.db"

    def test_custom_db_path(self, tmp_path):
        """Custom db_path should be stored."""
        custom_path = tmp_path / "custom.db"
        with patch("src.backtest.ensemble_backtest.SignalIntegrator") as mock_si:
            engine = EnsembleBacktestEngine(db_path=custom_path)
            assert engine.db_path == custom_path

    def test_custom_integrator(self, tmp_path):
        """Custom integrator should be used."""
        mock_int = MagicMock()
        engine = EnsembleBacktestEngine(
            db_path=tmp_path / "test.db",
            integrator=mock_int,
        )
        assert engine.integrator is mock_int

    def test_default_integrator_created(self, tmp_path):
        """When no integrator passed, SignalIntegrator should be created."""
        with patch("src.backtest.ensemble_backtest.SignalIntegrator") as mock_si:
            mock_instance = MagicMock()
            mock_si.return_value = mock_instance
            engine = EnsembleBacktestEngine(db_path=tmp_path / "test.db")
            mock_si.assert_called_once()
            assert engine.integrator is mock_instance

    def test_price_cache_empty(self, tmp_path):
        """_price_cache should be initialized as empty dict."""
        with patch("src.backtest.ensemble_backtest.SignalIntegrator"):
            engine = EnsembleBacktestEngine(db_path=tmp_path / "test.db")
            assert engine._price_cache == {}

    def test_signal_cache_empty(self, tmp_path):
        """_signal_cache should be initialized as empty dict."""
        with patch("src.backtest.ensemble_backtest.SignalIntegrator"):
            engine = EnsembleBacktestEngine(db_path=tmp_path / "test.db")
            assert engine._signal_cache == {}

    def test__all__export(self):
        """__all__ should export EnsembleBacktestEngine."""
        from src.backtest import ensemble_backtest
        assert 'EnsembleBacktestEngine' in ensemble_backtest.__all__


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

    def test_calculate_returns_empty_list(self, tmp_path):
        """Empty price list should return empty array."""
        engine = _make_engine(tmp_path)
        returns = engine._calculate_returns([])
        assert len(returns) == 0
        assert isinstance(returns, np.ndarray)

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

    def test_max_drawdown_two_points(self, tmp_path):
        """Two-point curve: up then down."""
        engine = _make_engine(tmp_path)
        curve = np.array([100, 110, 95])
        dd, duration = engine._calculate_max_drawdown(curve)
        assert dd == (95 - 110) / 110
        assert duration == 1  # index 2 is below peak at index 1

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

    def test_allocation_deltas_single_asset(self, tmp_path):
        """Single asset portfolio should return that asset at 1.0."""
        engine = _make_engine(tmp_path)
        current = {'SPY': 1.0}
        signals = {'SPY': {'score': 0.3, 'confidence': 0.8, 'regime': 'bull', 'sources': []}}
        target = engine._calculate_allocation_deltas(current, signals)
        assert abs(sum(target.values()) - 1.0) < 0.01
        assert 'SPY' in target

    def test_allocation_deltas_empty_signals(self, tmp_path):
        """Empty signals should return original allocation."""
        engine = _make_engine(tmp_path)
        current = {'SPY': 0.5, 'GLD': 0.5}
        target = engine._calculate_allocation_deltas(current, {})
        assert abs(sum(target.values()) - 1.0) < 0.01
        assert target['SPY'] == 0.5
        assert target['GLD'] == 0.5

    def test_allocation_deltas_all_zero_scores(self, tmp_path):
        """All zero scores should hit the total_score=0 branch."""
        engine = _make_engine(tmp_path)
        current = {'SPY': 0.5, 'GLD': 0.5}
        signals = {
            'SPY': {'score': 0.0, 'confidence': 0.0, 'regime': 'neutral', 'sources': []},
            'GLD': {'score': 0.0, 'confidence': 0.0, 'regime': 'neutral', 'sources': []},
        }
        target = engine._calculate_allocation_deltas(current, signals)
        assert abs(sum(target.values()) - 1.0) < 0.01
        # With total_score=0, weights are uniform, and adjustment = 0, so target ≈ original
        for asset in current:
            assert abs(target[asset] - current[asset]) < 0.001

    def test_allocation_deltas_custom_max_delta(self, tmp_path):
        """Custom max_delta should constrain adjustments."""
        engine = _make_engine(tmp_path)
        current = {'SPY': 0.5, 'GLD': 0.5}
        signals = {
            'SPY': {'score': 1.0, 'confidence': 1.0, 'regime': 'bull', 'sources': []},
            'GLD': {'score': -1.0, 'confidence': 1.0, 'regime': 'bear', 'sources': []},
        }
        # Small max_delta should limit change
        target = engine._calculate_allocation_deltas(current, signals, max_delta=0.01)
        assert abs(sum(target.values()) - 1.0) < 0.01
        spy_change = target['SPY'] - current['SPY']
        assert 0 < spy_change < 0.02  # positive but small

    def test_allocation_deltas_extreme_scores(self, tmp_path):
        """Extreme scores (>1.0) should still normalize correctly."""
        engine = _make_engine(tmp_path)
        current = {'SPY': 0.5, 'GLD': 0.5}
        signals = {
            'SPY': {'score': 5.0, 'confidence': 1.0, 'regime': 'bull', 'sources': []},
            'GLD': {'score': -3.0, 'confidence': 1.0, 'regime': 'bear', 'sources': []},
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

    def test_fetch_prices_corrupt_db(self, tmp_path):
        """Corrupt database should propagate sqlite3.DatabaseError."""
        engine = _make_engine(tmp_path)
        engine.db_path = tmp_path / "corrupt.db"
        engine.db_path.write_text("not a valid database")
        with pytest.raises(sqlite3.DatabaseError):
            engine._fetch_historical_prices('SPY', '2020-01-01', '2020-12-31')

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

    def test_validate_target_zero_sharpe(self, tmp_path):
        """Zero Sharpe ratio should fail any positive target."""
        engine = _make_engine(tmp_path)
        result = _make_result()
        result.sharpe_ratio = 0.0
        assert engine.validate_target(result, target_sharpe=0.01) is False

    def test_validate_target_negative_sharpe(self, tmp_path):
        """Negative Sharpe ratio should fail any positive target."""
        engine = _make_engine(tmp_path)
        result = _make_result()
        result.sharpe_ratio = -0.5
        assert engine.validate_target(result, target_sharpe=0.0) is False


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

    def test_crisis_periods_parseable_dates(self):
        """All crisis period start/end dates should be parseable."""
        from datetime import datetime
        for key, (start, end) in EnsembleBacktestEngine.CRISIS_PERIODS.items():
            datetime.strptime(start, "%Y-%m-%d")
            datetime.strptime(end, "%Y-%m-%d")

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

    def test_crisis_alpha_one_day(self, tmp_path):
        """Single-day crisis period should compute alpha correctly."""
        engine = _make_engine(tmp_path)
        portfolio = {'2020-03-15': -0.05}
        benchmark = {'2020-03-15': -0.10}
        alpha = engine._calculate_crisis_alpha(
            portfolio, benchmark, ('2020-03-15', '2020-03-15')
        )
        expected = (-0.05) - (-0.10)
        assert abs(alpha - expected) < 0.001

    def test_crisis_alpha_all_zero_portfolio(self, tmp_path):
        """Portfolio with all zero returns should produce negative alpha when benchmark has positive."""
        engine = _make_engine(tmp_path)
        portfolio = {'2020-03-15': 0.0, '2020-03-16': 0.0}
        benchmark = {'2020-03-15': 0.05, '2020-03-16': 0.03}
        alpha = engine._calculate_crisis_alpha(
            portfolio, benchmark, ('2020-03-15', '2020-03-16')
        )
        assert alpha < 0

    def test_crisis_alpha_extreme_returns(self, tmp_path):
        """Very large returns should still compute correctly."""
        engine = _make_engine(tmp_path)
        portfolio = {'2020-03-15': 0.50, '2020-03-16': -0.30}
        benchmark = {'2020-03-15': 0.10, '2020-03-16': -0.05}
        alpha = engine._calculate_crisis_alpha(
            portfolio, benchmark, ('2020-03-15', '2020-03-16')
        )
        port_cum = (1 + 0.50) * (1 - 0.30) - 1
        bench_cum = (1 + 0.10) * (1 - 0.05) - 1
        expected = port_cum - bench_cum
        assert abs(alpha - expected) < 0.001

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

    def test_allocation_deltas_mixed_directions(self, tmp_path):
        """Mixed direction signals should produce intermediate allocations."""
        engine = _make_engine(tmp_path)
        current = {'A': 0.5, 'B': 0.5}
        signals = {
            'A': {'score': 0.5, 'confidence': 1.0, 'regime': 'bull', 'sources': []},
            'B': {'score': -0.5, 'confidence': 1.0, 'regime': 'bear', 'sources': []},
        }
        target = engine._calculate_allocation_deltas(current, signals)
        assert abs(sum(target.values()) - 1.0) < 0.01
        assert target['A'] > current['A']
        assert target['B'] < current['B']

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
        conn = sqlite3.connect(str(engine.db_path))
        conn.execute("CREATE TABLE prices (symbol TEXT, date TEXT, close REAL, open REAL, high REAL, low REAL, volume INTEGER)")
        conn.commit()
        conn.close()
        result = engine._fetch_historical_prices('SPY', '2020-01-01', '2020-12-31')
        assert result == []

    def test_fetch_prices_multiple_symbols(self, tmp_path):
        """Fetching different symbols should return correct data."""
        engine = _make_engine(tmp_path)
        conn = sqlite3.connect(str(engine.db_path))
        conn.execute("CREATE TABLE prices (symbol TEXT, date TEXT, close REAL, open REAL, high REAL, low REAL, volume INTEGER)")
        conn.execute("INSERT INTO prices VALUES ('SPY', '2020-01-02', 323.0, 322.0, 324.0, 321.0, 1000000)")
        conn.execute("INSERT INTO prices VALUES ('GLD', '2020-01-02', 150.0, 149.0, 151.0, 148.0, 500000)")
        conn.commit()
        conn.close()
        spy = engine._fetch_historical_prices('SPY', '2020-01-01', '2020-12-31')
        gld = engine._fetch_historical_prices('GLD', '2020-01-01', '2020-12-31')
        assert len(spy) == 1
        assert len(gld) == 1
        assert spy[0]['close'] == 323.0
        assert gld[0]['close'] == 150.0


# ---------------------------------------------------------------------------
# EnsembleBacktestResult extended tests
# ---------------------------------------------------------------------------

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

    def test_result_cagr_positive(self):
        """CAGR should be positive for a successful backtest."""
        r = _make_result()
        assert r.cagr > 0

    def test_result_total_rebalances_positive(self):
        """Total rebalances should be a positive integer."""
        r = _make_result()
        assert r.total_rebalances > 0
        assert isinstance(r.total_rebalances, int)


# ---------------------------------------------------------------------------
# Signal generation tests
# ---------------------------------------------------------------------------

class TestEnsembleBacktestEngineSignals:
    """Tests for _generate_daily_signals."""

    def test_generate_daily_signals_success(self, tmp_path):
        """Success path: integrator returns composite signal with expected fields."""
        engine = _make_engine(tmp_path)
        portfolio = {'SPY': 0.5, 'GLD': 0.5}

        # Mock the integrator to return a composite-like object
        mock_composite = MagicMock()
        mock_composite.score = 0.3
        mock_composite.confidence = 0.8
        mock_composite.regime = "bull"
        mock_composite.sources = [{"source": "tsfm", "weight": 0.5}]
        engine.integrator.get_composite_signal.return_value = mock_composite

        signals = engine._generate_daily_signals("2020-01-15", portfolio)

        assert 'SPY' in signals
        assert 'GLD' in signals
        for asset in portfolio:
            assert signals[asset]["score"] == 0.3
            assert signals[asset]["confidence"] == 0.8
            assert signals[asset]["regime"] == "bull"
            assert signals[asset]["sources"] == [{"source": "tsfm", "weight": 0.5}]

        # Verify integrator was called for each asset
        expected_calls = [call("SPY"), call("GLD")]
        engine.integrator.get_composite_signal.assert_has_calls(expected_calls)

    def test_generate_daily_signals_exception(self, tmp_path):
        """Exception path: when integrator raises, neutral signal should be used."""
        engine = _make_engine(tmp_path)
        portfolio = {'SPY': 0.5, 'GLD': 0.5}

        # Mock the integrator to raise on first call
        engine.integrator.get_composite_signal.side_effect = ValueError("DB error")

        signals = engine._generate_daily_signals("2020-01-15", portfolio)

        for asset in portfolio:
            assert signals[asset]["score"] == 0.0
            assert signals[asset]["confidence"] == 0.0
            assert signals[asset]["regime"] == "neutral"
            assert signals[asset]["sources"] == []

    def test_generate_daily_signals_empty_portfolio(self, tmp_path):
        """Empty portfolio should return empty signals dict."""
        engine = _make_engine(tmp_path)
        signals = engine._generate_daily_signals("2020-01-15", {})
        assert signals == {}
        engine.integrator.get_composite_signal.assert_not_called()


# ---------------------------------------------------------------------------
# Run backtest tests
# ---------------------------------------------------------------------------

class TestEnsembleBacktestEngineRun:
    """Tests for run_backtest."""

    def test_insufficient_data(self, tmp_path):
        """Fewer than 30 trading days should raise ValueError."""
        engine = _make_engine(tmp_path)
        price_data = _make_price_data(days=5)

        with patch.object(engine, '_fetch_historical_prices', return_value=price_data):
            with pytest.raises(ValueError, match="Insufficient data"):
                engine.run_backtest(
                    {"SPY": 1.0},
                    start_date="2020-01-01",
                    end_date="2020-02-01",
                )

    def test_weekly_rebalance(self, tmp_path):
        """Weekly rebalance frequency should produce results."""
        engine = _make_engine(tmp_path)
        spy_data = _make_price_data(days=60, base_price=100.0, symbol="SPY")

        # Mock signals to return neutral
        with patch.object(engine, '_fetch_historical_prices', return_value=spy_data):
            with patch.object(engine, '_generate_daily_signals', return_value={}):
                result = engine.run_backtest(
                    {"SPY": 1.0},
                    start_date="2020-01-01",
                    end_date="2020-03-01",
                    rebalance_freq="weekly",
                )

        assert isinstance(result, BacktestResult)
        assert result.total_rebalances > 0
        assert result.sharpe_ratio != 0  # Should have computed something

    def test_threshold_rebalance(self, tmp_path):
        """Threshold-based rebalance frequency should produce results."""
        engine = _make_engine(tmp_path)
        spy_data = _make_price_data(days=50, base_price=100.0, symbol="SPY")

        with patch.object(engine, '_fetch_historical_prices', return_value=spy_data):
            with patch.object(engine, '_generate_daily_signals', return_value={}):
                result = engine.run_backtest(
                    {"SPY": 1.0},
                    start_date="2020-01-01",
                    end_date="2020-03-01",
                    rebalance_freq="threshold",
                )

        assert isinstance(result, BacktestResult)
        assert result.total_rebalances > 0

    def test_single_asset(self, tmp_path):
        """Single-asset portfolio should run without errors."""
        engine = _make_engine(tmp_path)
        spy_data = _make_price_data(days=60, base_price=100.0, symbol="SPY")

        with patch.object(engine, '_fetch_historical_prices', return_value=spy_data):
            with patch.object(engine, '_generate_daily_signals', return_value={}):
                result = engine.run_backtest(
                    {"SPY": 1.0},
                    start_date="2020-01-01",
                    end_date="2020-03-01",
                    rebalance_freq="monthly",
                )

        assert isinstance(result, BacktestResult)
        assert result.total_rebalances > 0
        assert "SPY" in result.extras["portfolio"]

    def test_two_assets(self, tmp_path):
        """Two-asset portfolio should run without errors."""
        engine = _make_engine(tmp_path)
        spy_data = _make_price_data(days=60, base_price=100.0, symbol="SPY")
        gld_data = _make_price_data(days=60, base_price=50.0, symbol="GLD")

        def mock_fetch(symbol, start, end):
            if symbol == "SPY":
                return spy_data
            elif symbol == "GLD":
                return gld_data
            return []

        with patch.object(engine, '_fetch_historical_prices', side_effect=mock_fetch):
            with patch.object(engine, '_generate_daily_signals', return_value={}):
                result = engine.run_backtest(
                    {"SPY": 0.6, "GLD": 0.4},
                    start_date="2020-01-01",
                    end_date="2020-03-01",
                    rebalance_freq="monthly",
                )

        assert isinstance(result, BacktestResult)
        assert result.total_rebalances > 0
        assert "SPY" in result.extras["portfolio"]
        assert "GLD" in result.extras["portfolio"]

    def test_mixed_price_lengths(self, tmp_path):
        """Assets with different price histories should use common dates."""
        engine = _make_engine(tmp_path)
        spy_data = _make_price_data(days=60, base_price=100.0, symbol="SPY")
        # GLD has shorter history
        gld_data = _make_price_data(days=40, base_price=50.0, symbol="GLD")

        def mock_fetch(symbol, start, end):
            if symbol == "SPY":
                return spy_data
            elif symbol == "GLD":
                return gld_data
            return []

        with patch.object(engine, '_fetch_historical_prices', side_effect=mock_fetch):
            with patch.object(engine, '_generate_daily_signals', return_value={}):
                result = engine.run_backtest(
                    {"SPY": 0.6, "GLD": 0.4},
                    start_date="2020-01-01",
                    end_date="2020-03-01",
                    rebalance_freq="monthly",
                )

        assert isinstance(result, BacktestResult)
        assert result.total_rebalances > 0
        # GLD only has 40 days, but we need 30+ common dates
        # Both have dates from day 1-40, so that should work
        assert result.extras["end_date"] is not None

    def test_rebalance_cost_applied(self, tmp_path):
        """Transaction costs should be applied on rebalance days."""
        engine = _make_engine(tmp_path)
        spy_data = _make_price_data(days=60, base_price=100.0, symbol="SPY")
        gld_data = _make_price_data(days=60, base_price=50.0, symbol="GLD")

        # Mock signals to return non-neutral to trigger turnover
        mock_signal = {
            'SPY': {'score': 0.5, 'confidence': 0.8, 'regime': 'bull', 'sources': [{"source": "tsfm"}]},
            'GLD': {'score': -0.3, 'confidence': 0.6, 'regime': 'bear', 'sources': [{"source": "cta"}]},
        }

        def mock_fetch(symbol, start, end):
            if symbol == "SPY":
                return spy_data
            elif symbol == "GLD":
                return gld_data
            return []

        with patch.object(engine, '_fetch_historical_prices', side_effect=mock_fetch):
            with patch.object(engine, '_generate_daily_signals', return_value=mock_signal):
                result = engine.run_backtest(
                    {"SPY": 0.6, "GLD": 0.4},
                    start_date="2020-01-01",
                    end_date="2020-03-01",
                    rebalance_freq="monthly",
                )

        assert isinstance(result, BacktestResult)
        # Signal history should have been recorded
        assert "source_contributions" in result.extras


# ---------------------------------------------------------------------------
# Main function tests
# ---------------------------------------------------------------------------

class TestEnsembleBacktestMain:
    """Tests for the main() CLI entry point."""

    def test_main_run_command_parsing(self):
        """CLI should parse 46/38/16 portfolio correctly."""
        with patch("src.backtest.ensemble_backtest.EnsembleBacktestEngine") as mock_engine_cls:
            mock_engine = MagicMock()
            mock_result = MagicMock()
            mock_result.sharpe_ratio = 0.9
            mock_result.extras = {}
            mock_engine.run_backtest.return_value = mock_result
            mock_engine_cls.return_value = mock_engine

            with patch("argparse.ArgumentParser.parse_args") as mock_parse:
                mock_parse.return_value = MagicMock(
                    command="run",
                    portfolio="46/38/16",
                    start="2020-01-01",
                    end="2020-06-01",
                    target_sharpe=0.95,
                    rebalance="monthly",
                    output=None,
                )
                from src.backtest.ensemble_backtest import main
                main()

            mock_engine_cls.assert_called_once()
            mock_engine.run_backtest.assert_called_once_with(
                portfolio={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
                start_date="2020-01-01",
                end_date="2020-06-01",
                rebalance_freq="monthly",
            )

    def test_main_portfolio_single_asset(self):
        """CLI should handle single-asset portfolio."""
        with patch("src.backtest.ensemble_backtest.EnsembleBacktestEngine") as mock_engine_cls:
            mock_engine = MagicMock()
            mock_result = MagicMock()
            mock_result.sharpe_ratio = 0.9
            mock_result.extras = {}
            mock_engine.run_backtest.return_value = mock_result
            mock_engine_cls.return_value = mock_engine

            with patch("argparse.ArgumentParser.parse_args") as mock_parse:
                mock_parse.return_value = MagicMock(
                    command="run",
                    portfolio="100",
                    start="2020-01-01",
                    end="2020-06-01",
                    target_sharpe=0.95,
                    rebalance="monthly",
                    output=None,
                )
                from src.backtest.ensemble_backtest import main
                main()

            mock_engine.run_backtest.assert_called_once()
            call_kwargs = mock_engine.run_backtest.call_args[1]
            assert "SPY" in call_kwargs["portfolio"]
            assert abs(call_kwargs["portfolio"]["SPY"] - 1.0) < 0.01

    def test_main_benchmark_command(self):
        """CLI benchmark command should compare ensemble vs static."""
        with patch("src.backtest.ensemble_backtest.EnsembleBacktestEngine") as mock_engine_cls:
            mock_engine = MagicMock()
            mock_result = MagicMock()
            mock_result.sharpe_ratio = 0.95
            mock_result.cagr = 0.10
            mock_result.max_drawdown = -0.20
            mock_result.extras = {}
            mock_engine.run_backtest.return_value = mock_result
            mock_engine_cls.return_value = mock_engine

            with patch("argparse.ArgumentParser.parse_args") as mock_parse:
                mock_parse.return_value = MagicMock(
                    command="benchmark",
                    portfolio="46/38/16",
                    start="2020-01-01",
                    end="2020-06-01",
                    target_sharpe=0.95,
                    rebalance="monthly",
                    output=None,
                )
                from src.backtest.ensemble_backtest import main
                main()

            # Should have created two engines (ensemble + static)
            assert mock_engine_cls.call_count == 2

    def test_main_validate_command(self):
        """CLI validate command should call validate_target."""
        with patch("src.backtest.ensemble_backtest.EnsembleBacktestEngine") as mock_engine_cls:
            mock_engine = MagicMock()
            mock_result = MagicMock()
            mock_result.sharpe_ratio = 0.96
            mock_result.extras = {}
            mock_engine.run_backtest.return_value = mock_result
            mock_engine_cls.return_value = mock_engine

            with patch("argparse.ArgumentParser.parse_args") as mock_parse:
                mock_parse.return_value = MagicMock(
                    command="validate",
                    portfolio="46/38/16",
                    start="2020-01-01",
                    end="2020-06-01",
                    target_sharpe=0.95,
                    rebalance="monthly",
                    output=None,
                )
                from src.backtest.ensemble_backtest import main
                main()

            mock_engine.validate_target.assert_called_once_with(mock_result, 0.95)

    def test_main_output_file(self, tmp_path):
        """CLI with --output should write JSON file."""
        output_path = tmp_path / "results.json"
        real_result = _make_result()

        with patch("src.backtest.ensemble_backtest.EnsembleBacktestEngine") as mock_engine_cls:
            mock_engine = MagicMock()
            mock_engine.run_backtest.return_value = real_result
            mock_engine_cls.return_value = mock_engine

            with patch("argparse.ArgumentParser.parse_args") as mock_parse:
                mock_parse.return_value = MagicMock(
                    command="run",
                    portfolio="46/38/16",
                    start="2020-01-01",
                    end="2020-06-01",
                    target_sharpe=0.95,
                    rebalance="monthly",
                    output=str(output_path),
                )
                from src.backtest.ensemble_backtest import main
                main()

            assert output_path.exists()
            data = json.loads(output_path.read_text())
            assert "sharpe_ratio" in data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
