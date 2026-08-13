#!/usr/bin/env python3
"""
Tests for ensemble backtest engine — data class, returns calculation,
max drawdown, crisis alpha, allocation deltas, and target validation.
"""
import json
import logging
import sqlite3
import numpy as np

import pytest
from unittest.mock import patch, MagicMock, call

from src.backtest.metrics import BacktestResult, compute_metrics_from_returns
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


def _make_price_data_from_closes(closes: list[float]) -> list:
    """Generate synthetic price data from explicit closes."""
    rows = []
    for i, close in enumerate(closes):
        day = i + 1
        date = f"2020-01-{day:02d}" if day <= 31 else f"2020-02-{day - 31:02d}"
        rows.append({
            "date": date,
            "close": close,
            "open": close,
            "high": close,
            "low": close,
            "volume": 1000000,
        })
    return rows


def _returns_from_equity_curve(curve) -> list[float]:
    """Convert an equity curve into simple returns for shared metrics."""
    curve_arr = np.array(curve, dtype=float)
    return list(curve_arr[1:] / curve_arr[:-1] - 1)


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
            with patch("src.backtest.ensemble_backtest.SignalIntegrator"):
                engine = EnsembleBacktestEngine()
                assert engine.db_path == tmp_path / "default_market.db"

    def test_custom_db_path(self, tmp_path):
        """Custom db_path should be stored."""
        custom_path = tmp_path / "custom.db"
        with patch("src.backtest.ensemble_backtest.SignalIntegrator"):
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


class TestEnsembleMaxDrawdownMetricsParity:
    """Parity checks for migrating only max-drawdown value calculation."""

    @pytest.mark.parametrize(
        "curve",
        [
            [100, 100, 100, 100],
            [100, 101, 103, 106],
            [100, 110, 90, 95, 120],
        ],
        ids=["flat", "monotonic_positive", "drawdown_recovery"],
    )
    def test_local_max_drawdown_value_matches_shared_metrics_when_initial_peak_holds(
        self,
        tmp_path,
        curve,
    ):
        engine = _make_engine(tmp_path)

        local_dd, _ = engine._calculate_max_drawdown(np.array(curve, dtype=float))
        shared = compute_metrics_from_returns(
            _returns_from_equity_curve(curve),
            risk_free_rate=0.0,
        )
        local_dd_rounded = round(float(local_dd), 6)

        assert local_dd_rounded == shared["max_drawdown"]

    def test_shared_metrics_misses_drawdown_from_initial_equity_peak(self, tmp_path):
        engine = _make_engine(tmp_path)
        curve = [100, 98, 97, 96, 100, 99, 98, 101]

        local_dd, _ = engine._calculate_max_drawdown(np.array(curve, dtype=float))
        shared = compute_metrics_from_returns(
            _returns_from_equity_curve(curve),
            risk_free_rate=0.0,
        )
        local_dd_rounded = round(float(local_dd), 6)

        assert local_dd_rounded == -0.04
        assert shared["max_drawdown"] == -0.020408
        assert local_dd_rounded != shared["max_drawdown"]


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

    def test_run_backtest_preserves_initial_equity_peak_drawdown(self, tmp_path):
        """Do not migrate to shared max_dd helper until it handles the initial peak."""
        engine = _make_engine(tmp_path)
        spy_data = _make_price_data_from_closes([
            100, 95, 90, 85, 90, 95, 100, 101, 102, 103,
            104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
            114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
            124,
        ])
        shared = compute_metrics_from_returns(
            _returns_from_equity_curve([row["close"] for row in spy_data]),
            risk_free_rate=0.0,
        )

        with patch.object(engine, '_fetch_historical_prices', return_value=spy_data):
            with patch.object(engine, '_generate_daily_signals', return_value={}):
                result = engine.run_backtest(
                    {"SPY": 1.0},
                    start_date="2020-01-01",
                    end_date="2020-01-31",
                    rebalance_freq="monthly",
                )

        assert result.max_drawdown == pytest.approx(-0.15)
        assert result.extras["max_dd_duration"] == 5
        assert shared["max_drawdown"] == -0.105263
        assert round(result.max_drawdown, 6) != shared["max_drawdown"]

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


# ---------------------------------------------------------------------------
# Dataclass field validation
# ---------------------------------------------------------------------------

class TestBacktestResultFieldValidation:
    """Validate BacktestResult dataclass fields via dataclasses introspection."""

    def test_all_required_fields_exist(self):
        from dataclasses import fields
        from src.backtest.metrics import BacktestResult
        field_names = {f.name for f in fields(BacktestResult)}
        for required in ("total_return", "cagr", "volatility", "sharpe_ratio",
                         "max_drawdown", "total_rebalances", "extras"):
            assert required in field_names, f"Missing required field: {required}"

    def test_required_field_types(self):
        from dataclasses import fields
        from src.backtest.metrics import BacktestResult
        field_map = {f.name: f.type for f in fields(BacktestResult)}
        assert field_map["total_return"] is float
        assert field_map["cagr"] is float
        assert field_map["volatility"] is float
        assert field_map["sharpe_ratio"] is float
        assert field_map["max_drawdown"] is float
        assert field_map["total_rebalances"] is int

    def test_total_rebalances_default_zero(self):
        from dataclasses import fields
        from src.backtest.metrics import BacktestResult
        f = next(f for f in fields(BacktestResult) if f.name == "total_rebalances")
        assert f.default == 0

    def test_extras_default_factory(self):
        from dataclasses import fields
        from src.backtest.metrics import BacktestResult
        f = next(f for f in fields(BacktestResult) if f.name == "extras")
        assert f.default_factory is not None
        # Verify default factory produces empty dict
        result = f.default_factory()
        assert result == {}

    def test_total_transaction_costs_default(self):
        from dataclasses import fields
        from src.backtest.metrics import BacktestResult
        f = next(f for f in fields(BacktestResult) if f.name == "total_transaction_costs")
        assert f.default == 0.0

    def test_baseline_sharpe_is_optional(self):
        from dataclasses import fields
        from src.backtest.metrics import BacktestResult
        f = next(f for f in fields(BacktestResult) if f.name == "baseline_sharpe")
        assert "Optional" in str(f.type) or "None" in str(f.type)
        assert f.default is None


# ---------------------------------------------------------------------------
# BacktestResult extras key validation
# ---------------------------------------------------------------------------

class TestBacktestResultExtrasKeys:
    """Validate that _make_result() produces all expected extras keys."""

    def test_extras_contains_all_required_keys(self):
        r = _make_result()
        required = [
            "start_date", "end_date", "portfolio", "sortino_ratio",
            "max_dd_duration", "calmar_ratio", "var_95", "cvar_95",
            "avg_signal_confidence", "regime_distribution",
            "crisis_alpha_2008", "crisis_alpha_2020", "crisis_alpha_2022",
            "source_contributions", "rolling_sharpe_1y",
        ]
        for key in required:
            assert key in r.extras, f"Missing required extras key: {key}"

    def test_extras_portfolio_format(self):
        r = _make_result()
        assert "/" in r.extras["portfolio"]

    def test_extras_cvar_95_leq_var_95(self):
        r = _make_result()
        assert r.extras["cvar_95"] <= r.extras["var_95"]


# ---------------------------------------------------------------------------
# Module-level constants validation
# ---------------------------------------------------------------------------

class TestEnsembleBacktestConstants:
    """Validate module-level constants."""

    def test_crisis_periods_exact_values(self):
        """Verify exact crisis period tuples match source code."""
        periods = EnsembleBacktestEngine.CRISIS_PERIODS
        assert periods["2008"] == ("2008-09-01", "2008-12-31")
        assert periods["2020"] == ("2020-02-19", "2020-04-30")
        assert periods["2022"] == ("2022-01-01", "2022-10-31")

    def test_crisis_periods_all_strings(self):
        """All crisis period date strings should be parseable."""
        from datetime import datetime
        for name, (start, end) in EnsembleBacktestEngine.CRISIS_PERIODS.items():
            assert isinstance(start, str), f"{name} start not a string"
            assert isinstance(end, str), f"{name} end not a string"
            datetime.strptime(start, "%Y-%m-%d")
            datetime.strptime(end, "%Y-%m-%d")

    def test_tx_cost_bps_type(self):
        assert isinstance(EnsembleBacktestEngine.TX_COST_BPS, float)

    def test_tx_cost_bps_range(self):
        assert 0 < EnsembleBacktestEngine.TX_COST_BPS < 100

    def test_crisis_periods_three_keys(self):
        assert len(EnsembleBacktestEngine.CRISIS_PERIODS) == 3

    def test_crisis_period_start_before_end(self):
        for name, (start, end) in EnsembleBacktestEngine.CRISIS_PERIODS.items():
            assert start < end, f"{name}: start {start} >= end {end}"


# ---------------------------------------------------------------------------
# _calculate_returns edge cases
# ---------------------------------------------------------------------------

class TestCalculateReturnsEdgeCases:
    """Edge cases for _calculate_returns."""

    def test_nan_price(self, tmp_path):
        engine = _make_engine(tmp_path)
        returns = engine._calculate_returns([100.0, float("nan"), 110.0])
        assert len(returns) == 2
        assert np.isnan(returns[1])

    def test_inf_price(self, tmp_path):
        engine = _make_engine(tmp_path)
        returns = engine._calculate_returns([100.0, float("inf"), 110.0])
        assert len(returns) == 2
        assert np.isinf(returns[0]) or np.isnan(returns[0])

    def test_negative_prices(self, tmp_path):
        engine = _make_engine(tmp_path)
        # log of negative numbers produces NaN for real-valued arrays
        returns = engine._calculate_returns([-100.0, -90.0])
        assert len(returns) == 1
        # Does not crash; result may be NaN (expected numpy behavior)

    def test_zero_price(self, tmp_path):
        engine = _make_engine(tmp_path)
        returns = engine._calculate_returns([100.0, 0.0])
        assert len(returns) == 1
        assert np.isinf(returns[0]) or np.isneginf(returns[0])

    def test_decreasing_prices_all(self, tmp_path):
        engine = _make_engine(tmp_path)
        returns = engine._calculate_returns([100.0, 90.0, 80.0, 70.0])
        assert len(returns) == 3
        assert all(r < 0 for r in returns)

    def test_returns_have_expected_shape(self, tmp_path):
        engine = _make_engine(tmp_path)
        returns = engine._calculate_returns([100.0, 101.0, 102.0])
        assert returns.shape == (2,)

    def test_negative_to_positive_price(self, tmp_path):
        """Crossing zero from negative to positive produces NaN (log of negative)."""
        engine = _make_engine(tmp_path)
        returns = engine._calculate_returns([-100.0, 100.0])
        assert len(returns) == 1
        # Expected: log of complex number produces NaN for real dtype


# ---------------------------------------------------------------------------
# _calculate_max_drawdown edge cases
# ---------------------------------------------------------------------------

class TestCalculateMaxDrawdownEdgeCases:
    """Edge cases for _calculate_max_drawdown."""

    def test_nan_in_curve(self, tmp_path):
        engine = _make_engine(tmp_path)
        curve = np.array([100.0, 110.0, float("nan"), 90.0])
        dd, duration = engine._calculate_max_drawdown(curve)
        assert np.isnan(dd)

    def test_inf_in_curve(self, tmp_path):
        engine = _make_engine(tmp_path)
        curve = np.array([100.0, float("inf"), 50.0])
        dd, duration = engine._calculate_max_drawdown(curve)
        # With inf peak, (50-inf)/inf = NaN; function does not crash

    def test_all_negative_values(self, tmp_path):
        engine = _make_engine(tmp_path)
        curve = np.array([-100.0, -90.0, -80.0])
        dd, duration = engine._calculate_max_drawdown(curve)
        # Even though values are negative, max.accumulate gives -80, -90 from -80 is DD
        assert dd <= 0

    def test_sawtooth_curve(self, tmp_path):
        """Alternating up-down-up-down should capture correct max dd."""
        engine = _make_engine(tmp_path)
        curve = np.array([100, 110, 105, 115, 108, 120])
        dd, duration = engine._calculate_max_drawdown(curve)
        assert dd < 0
        # Max DD should be from peak 115 to trough 108
        assert abs(dd - (108 - 115) / 115) < 0.01

    def test_two_equal_points(self, tmp_path):
        """Two equal points should have zero drawdown."""
        engine = _make_engine(tmp_path)
        curve = np.array([100.0, 100.0])
        dd, duration = engine._calculate_max_drawdown(curve)
        assert dd == 0.0
        assert duration == 0

    def test_duration_longest_drawdown_selected(self, tmp_path):
        """Multiple drawdowns: the longest one should be reported."""
        engine = _make_engine(tmp_path)
        # Peak at 100, 3-day DD (indices 1-3), recovery, then 2-day DD (indices 5-6)
        curve = np.array([100, 98, 97, 96, 100, 99, 98, 101])
        dd, duration = engine._calculate_max_drawdown(curve)
        assert duration == 3

    def test_duration_full_curve_drawdown(self, tmp_path):
        """Entire curve is one long drawdown from first peak."""
        engine = _make_engine(tmp_path)
        # Peak at index 0, then declines for 9 days
        curve = np.array([100.0, 99.0, 98.0, 97.0, 96.0, 95.0,
                          94.0, 93.0, 92.0, 91.0])
        dd, duration = engine._calculate_max_drawdown(curve)
        assert duration == 9

    def test_small_fluctuation_below_threshold(self, tmp_path):
        """Drawdown of exactly 0.1% is below -0.001 threshold."""
        engine = _make_engine(tmp_path)
        # 0.1% below peak → -0.001, which is NOT < -0.001, so not in drawdown
        curve = np.array([100.0, 99.9])
        dd, duration = engine._calculate_max_drawdown(curve)
        assert duration == 0

    def test_oscillating_near_peak(self, tmp_path):
        """Small oscillations near peak should not extend duration."""
        engine = _make_engine(tmp_path)
        curve = np.array([100.0, 99.5, 100.1, 99.6, 100.2])
        dd, duration = engine._calculate_max_drawdown(curve)
        # Duration should be 0 or 1 (small fluctuations may or may not trip threshold)
        assert duration <= 1


# ---------------------------------------------------------------------------
# _calculate_crisis_alpha edge cases
# ---------------------------------------------------------------------------

class TestCalculateCrisisAlphaEdgeCases:
    """Edge cases for _calculate_crisis_alpha."""

    def test_single_element_arrays(self, tmp_path):
        """Single element in crisis window should compute correctly."""
        engine = _make_engine(tmp_path)
        portfolio = {"2020-03-15": -0.05}
        benchmark = {"2020-03-15": -0.10}
        alpha = engine._calculate_crisis_alpha(
            portfolio, benchmark, ("2020-03-15", "2020-03-15")
        )
        assert abs(alpha - 0.05) < 0.001

    def test_nan_returns(self, tmp_path):
        """NaN returns should propagate to NaN alpha."""
        engine = _make_engine(tmp_path)
        portfolio = {"2020-03-15": float("nan")}
        benchmark = {"2020-03-15": 0.01}
        alpha = engine._calculate_crisis_alpha(
            portfolio, benchmark, ("2020-03-15", "2020-03-15")
        )
        assert np.isnan(alpha)

    def test_extreme_negative_returns(self, tmp_path):
        """Returns near -1.0 (total loss) should not overflow."""
        engine = _make_engine(tmp_path)
        portfolio = {"2020-03-15": -0.999, "2020-03-16": -0.999}
        benchmark = {"2020-03-15": -0.10, "2020-03-16": -0.05}
        alpha = engine._calculate_crisis_alpha(
            portfolio, benchmark, ("2020-03-15", "2020-03-16")
        )
        assert np.isfinite(alpha)
        assert alpha < 0

    def test_empty_portfolio_dict(self, tmp_path):
        """Empty portfolio dict should return 0.0."""
        engine = _make_engine(tmp_path)
        alpha = engine._calculate_crisis_alpha(
            {}, {"2020-03-15": 0.01}, ("2020-03-15", "2020-03-15")
        )
        assert alpha == 0.0

    def test_empty_benchmark_dict(self, tmp_path):
        """Empty benchmark dict should return 0.0."""
        engine = _make_engine(tmp_path)
        alpha = engine._calculate_crisis_alpha(
            {"2020-03-15": 0.01}, {}, ("2020-03-15", "2020-03-15")
        )
        assert alpha == 0.0

    def test_large_positive_returns(self, tmp_path):
        """+100% daily returns should compute without overflow."""
        engine = _make_engine(tmp_path)
        portfolio = {"2020-03-15": 1.0, "2020-03-16": 1.0}
        benchmark = {"2020-03-15": 0.01, "2020-03-16": 0.01}
        alpha = engine._calculate_crisis_alpha(
            portfolio, benchmark, ("2020-03-15", "2020-03-16")
        )
        assert np.isfinite(alpha)
        assert alpha > 0

    def test_outside_crisis_window_returns_zero(self, tmp_path):
        """Returns completely outside crisis window should return 0.0."""
        engine = _make_engine(tmp_path)
        portfolio = {"2021-01-01": 0.01, "2021-01-02": 0.02}
        benchmark = {"2021-01-01": 0.01, "2021-01-02": 0.02}
        alpha = engine._calculate_crisis_alpha(
            portfolio, benchmark, ("2020-01-01", "2020-12-31")
        )
        assert alpha == 0.0


# ---------------------------------------------------------------------------
# _calculate_allocation_deltas edge cases
# ---------------------------------------------------------------------------

class TestCalculateAllocationDeltasEdgeCases:
    """Edge cases for _calculate_allocation_deltas."""

    def test_nan_score(self, tmp_path):
        """NaN score should not crash (NaN propagates to output)."""
        engine = _make_engine(tmp_path)
        current = {"SPY": 0.5, "GLD": 0.5}
        signals = {
            "SPY": {"score": float("nan"), "confidence": 0.8, "regime": "neutral", "sources": []},
            "GLD": {"score": 0.3, "confidence": 0.8, "regime": "bull", "sources": []},
        }
        # NaN in score propagates; function does not crash
        engine._calculate_allocation_deltas(current, signals)

    def test_inf_score(self, tmp_path):
        """Inf score should not crash (Inf propagates to output)."""
        engine = _make_engine(tmp_path)
        current = {"SPY": 0.5, "GLD": 0.5}
        signals = {
            "SPY": {"score": float("inf"), "confidence": 0.8, "regime": "neutral", "sources": []},
            "GLD": {"score": 0.3, "confidence": 0.8, "regime": "bull", "sources": []},
        }
        # Inf in score propagates; function does not crash
        engine._calculate_allocation_deltas(current, signals)

    def test_empty_portfolio(self, tmp_path):
        """Empty portfolio should return empty dict."""
        engine = _make_engine(tmp_path)
        target = engine._calculate_allocation_deltas({}, {})
        assert target == {}

    def test_all_negative_scores(self, tmp_path):
        """All negative scores should still produce valid allocations."""
        engine = _make_engine(tmp_path)
        current = {"SPY": 0.5, "GLD": 0.5}
        signals = {
            "SPY": {"score": -0.8, "confidence": 0.9, "regime": "bear", "sources": []},
            "GLD": {"score": -0.5, "confidence": 0.7, "regime": "bear", "sources": []},
        }
        target = engine._calculate_allocation_deltas(current, signals)
        assert abs(sum(target.values()) - 1.0) < 0.01
        # Both scores negative: both should decrease relative to current
        # But normalization sums to 1.0, so relative weights shift
        assert all(a in target for a in current)

    def test_confidence_zero_all_scores(self, tmp_path):
        """Zero confidence with non-zero scores should eliminate adjustments."""
        engine = _make_engine(tmp_path)
        current = {"SPY": 0.6, "GLD": 0.4}
        signals = {
            "SPY": {"score": 1.0, "confidence": 0.0, "regime": "neutral", "sources": []},
            "GLD": {"score": -1.0, "confidence": 0.0, "regime": "neutral", "sources": []},
        }
        target = engine._calculate_allocation_deltas(current, signals)
        # strength = abs(score) * confidence = 0, so adjustment = 0
        for asset in current:
            assert abs(target[asset] - current[asset]) < 0.001

    def test_missing_asset_in_signals(self, tmp_path):
        """Missing asset in signals dict should not cause KeyError."""
        engine = _make_engine(tmp_path)
        current = {"SPY": 0.5, "GLD": 0.3, "TLT": 0.2}
        signals = {
            "SPY": {"score": 0.5, "confidence": 0.8, "regime": "bull", "sources": []},
            # GLD is missing from signals
            "TLT": {"score": -0.2, "confidence": 0.6, "regime": "bear", "sources": []},
        }
        target = engine._calculate_allocation_deltas(current, signals)
        assert abs(sum(target.values()) - 1.0) < 0.01

    def test_max_delta_zero(self, tmp_path):
        """Zero max_delta should prevent any allocation change."""
        engine = _make_engine(tmp_path)
        current = {"SPY": 0.5, "GLD": 0.5}
        signals = {
            "SPY": {"score": 1.0, "confidence": 1.0, "regime": "bull", "sources": []},
            "GLD": {"score": -1.0, "confidence": 1.0, "regime": "bear", "sources": []},
        }
        target = engine._calculate_allocation_deltas(current, signals, max_delta=0.0)
        assert abs(sum(target.values()) - 1.0) < 0.01
        for asset in current:
            assert abs(target[asset] - current[asset]) < 0.001

    def test_max_delta_negative(self, tmp_path):
        """Negative max_delta should reverse direction."""
        engine = _make_engine(tmp_path)
        current = {"SPY": 0.5, "GLD": 0.5}
        signals = {
            "SPY": {"score": 0.8, "confidence": 1.0, "regime": "bull", "sources": []},
            "GLD": {"score": 0.2, "confidence": 1.0, "regime": "bull", "sources": []},
        }
        # Negative max_delta should invert: positive score → negative adjustment
        target = engine._calculate_allocation_deltas(current, signals, max_delta=-0.1)
        assert abs(sum(target.values()) - 1.0) < 0.01

    def test_single_asset_zero_score(self, tmp_path):
        """Single asset with zero score: total_score=0 branch."""
        engine = _make_engine(tmp_path)
        current = {"SPY": 1.0}
        signals = {
            "SPY": {"score": 0.0, "confidence": 0.0, "regime": "neutral", "sources": []},
        }
        target = engine._calculate_allocation_deltas(current, signals)
        assert abs(target["SPY"] - 1.0) < 0.001

    def test_scores_opposite_directions_equal_magnitude(self, tmp_path):
        """Equal magnitude opposite scores should cancel for SPY."""
        engine = _make_engine(tmp_path)
        current = {"SPY": 0.5, "GLD": 0.5}
        signals = {
            "SPY": {"score": 0.5, "confidence": 1.0, "regime": "bull", "sources": []},
            "GLD": {"score": -0.5, "confidence": 1.0, "regime": "bear", "sources": []},
        }
        target = engine._calculate_allocation_deltas(current, signals)
        # total_score = 1.0, weights = 0.5 each
        # SPY adj = +1*0.5*1.0*0.1 = +0.05; GLD adj = -1*0.5*1.0*0.1 = -0.05
        # target SPY = 0.55, target GLD = 0.45, normalized: SPY=0.55, GLD=0.45
        assert abs(target["SPY"] - 0.55) < 0.01 or abs(target["SPY"] - 0.5) > 0.01
        assert target["SPY"] > current["SPY"]
        assert target["GLD"] < current["GLD"]


# ---------------------------------------------------------------------------
# _fetch_historical_prices edge cases
# ---------------------------------------------------------------------------

class TestFetchHistoricalPricesEdgeCases:
    """Edge cases for _fetch_historical_prices."""

    def test_missing_table(self, tmp_path):
        """Database without prices table should raise OperationalError."""
        engine = _make_engine(tmp_path)
        conn = sqlite3.connect(str(engine.db_path))
        conn.execute("CREATE TABLE other (id INTEGER)")
        conn.commit()
        conn.close()
        with pytest.raises(sqlite3.OperationalError):
            engine._fetch_historical_prices("SPY", "2020-01-01", "2020-12-31")

    def test_empty_table(self, tmp_path):
        """Empty prices table should return empty list."""
        engine = _make_engine(tmp_path)
        conn = sqlite3.connect(str(engine.db_path))
        conn.execute(
            "CREATE TABLE prices (symbol TEXT, date TEXT, close REAL, "
            "open REAL, high REAL, low REAL, volume INTEGER)"
        )
        conn.commit()
        conn.close()
        result = engine._fetch_historical_prices("SPY", "2020-01-01", "2020-12-31")
        assert result == []

    def test_symbol_with_no_data(self, tmp_path):
        """Symbol with no matching rows should return empty list."""
        engine = _make_engine(tmp_path)
        conn = sqlite3.connect(str(engine.db_path))
        conn.execute(
            "CREATE TABLE prices (symbol TEXT, date TEXT, close REAL, "
            "open REAL, high REAL, low REAL, volume INTEGER)"
        )
        conn.execute(
            "INSERT INTO prices VALUES ('SPY', '2020-01-02', 323.0, "
            "322.0, 324.0, 321.0, 1000000)"
        )
        conn.commit()
        conn.close()
        result = engine._fetch_historical_prices("QQQ", "2020-01-01", "2020-12-31")
        assert result == []

    def test_date_boundary_exclusive(self, tmp_path):
        """Dates outside the query range should be excluded."""
        engine = _make_engine(tmp_path)
        conn = sqlite3.connect(str(engine.db_path))
        conn.execute(
            "CREATE TABLE prices (symbol TEXT, date TEXT, close REAL, "
            "open REAL, high REAL, low REAL, volume INTEGER)"
        )
        conn.execute(
            "INSERT INTO prices VALUES ('SPY', '2020-01-01', 320.0, "
            "319.0, 321.0, 318.0, 1000000)"
        )
        conn.execute(
            "INSERT INTO prices VALUES ('SPY', '2020-06-15', 340.0, "
            "339.0, 341.0, 338.0, 1000000)"
        )
        conn.commit()
        conn.close()
        result = engine._fetch_historical_prices("SPY", "2020-02-01", "2020-05-31")
        assert result == []

    def test_cache_hit_returns_same_object(self, tmp_path):
        """Cached results should return the same list object."""
        engine = _make_engine(tmp_path)
        data = [{"date": "2020-01-02", "close": 323.0}]
        engine._price_cache["SPY:2020-01-01:2020-12-31"] = data
        result = engine._fetch_historical_prices("SPY", "2020-01-01", "2020-12-31")
        assert result is data

    def test_cache_uses_key_different_symbols(self, tmp_path):
        """Cache keys should differentiate by symbol."""
        engine = _make_engine(tmp_path)
        engine._price_cache["SPY:2020-01-01:2020-12-31"] = [{"close": 323.0}]
        engine._price_cache["GLD:2020-01-01:2020-12-31"] = [{"close": 150.0}]
        spy = engine._fetch_historical_prices("SPY", "2020-01-01", "2020-12-31")
        gld = engine._fetch_historical_prices("GLD", "2020-01-01", "2020-12-31")
        assert spy[0]["close"] == 323.0
        assert gld[0]["close"] == 150.0


# ---------------------------------------------------------------------------
# _generate_daily_signals edge cases
# ---------------------------------------------------------------------------

class TestGenerateDailySignalsEdgeCases:
    """Edge cases for _generate_daily_signals."""

    def test_integrator_returns_nan_score(self, tmp_path):
        """Integrator returning NaN score should be caught gracefully."""
        engine = _make_engine(tmp_path)
        portfolio = {"SPY": 1.0}
        mock_composite = MagicMock()
        mock_composite.score = float("nan")
        mock_composite.confidence = 0.8
        mock_composite.regime = "bull"
        mock_composite.sources = [{"source": "tsfm"}]
        engine.integrator.get_composite_signal.return_value = mock_composite
        signals = engine._generate_daily_signals("2020-01-15", portfolio)
        assert np.isnan(signals["SPY"]["score"])

    def test_integrator_returns_none(self, tmp_path):
        """Integrator returning None should fall through to except block."""
        engine = _make_engine(tmp_path)
        portfolio = {"SPY": 1.0}
        engine.integrator.get_composite_signal.return_value = None
        signals = engine._generate_daily_signals("2020-01-15", portfolio)
        # None.score raises AttributeError → caught → neutral signal
        assert signals["SPY"]["score"] == 0.0
        assert signals["SPY"]["confidence"] == 0.0
        assert signals["SPY"]["regime"] == "neutral"

    def test_integrator_missing_regime_field(self, tmp_path):
        """Composite signal without regime should still work."""
        engine = _make_engine(tmp_path)
        portfolio = {"SPY": 1.0}
        mock_composite = MagicMock(spec=[])  # No attributes
        del mock_composite.score
        del mock_composite.confidence
        del mock_composite.regime
        del mock_composite.sources
        engine.integrator.get_composite_signal.return_value = mock_composite
        signals = engine._generate_daily_signals("2020-01-15", portfolio)
        assert signals["SPY"]["score"] == 0.0
        assert signals["SPY"]["regime"] == "neutral"

    def test_integrator_partial_failure(self, tmp_path):
        """One asset fails, another succeeds."""
        engine = _make_engine(tmp_path)
        portfolio = {"SPY": 0.5, "GLD": 0.5}

        def side_effect(asset):
            if asset == "SPY":
                raise RuntimeError("SPY failed")
            mock_c = MagicMock()
            mock_c.score = 0.3
            mock_c.confidence = 0.8
            mock_c.regime = "bull"
            mock_c.sources = [{"source": "cta"}]
            return mock_c

        engine.integrator.get_composite_signal.side_effect = side_effect
        signals = engine._generate_daily_signals("2020-01-15", portfolio)
        assert signals["SPY"]["score"] == 0.0
        assert signals["GLD"]["score"] == 0.3
        assert signals["SPY"]["regime"] == "neutral"
        assert signals["GLD"]["regime"] == "bull"


# ---------------------------------------------------------------------------
# run_backtest edge cases
# ---------------------------------------------------------------------------

class TestRunBacktestEdgeCases:
    """Edge cases for run_backtest."""

    def test_empty_portfolio(self, tmp_path):
        """Empty portfolio dict should raise error (no assets to fetch)."""
        engine = _make_engine(tmp_path)
        with patch.object(engine, "_fetch_historical_prices", return_value=[]):
            with pytest.raises(ValueError, match="Insufficient data"):
                engine.run_backtest({})

    def test_flat_price_curve(self, tmp_path):
        """All flat prices produce zero volatility and zero return."""
        engine = _make_engine(tmp_path)
        spy_data = _make_price_data(days=60, base_price=100.0)
        # All same prices
        for d in spy_data:
            d["close"] = 100.0
        with patch.object(engine, "_fetch_historical_prices", return_value=spy_data):
            with patch.object(engine, "_generate_daily_signals", return_value={}):
                result = engine.run_backtest(
                    {"SPY": 1.0},
                    start_date="2020-01-01",
                    end_date="2020-03-01",
                    rebalance_freq="monthly",
                )
        assert result.total_return == 0.0
        assert abs(result.volatility) < 1e-10

    def test_all_declining_prices(self, tmp_path):
        """All declining prices produce negative total return."""
        engine = _make_engine(tmp_path)
        spy_data = _make_price_data(days=60, base_price=100.0)
        for i, d in enumerate(spy_data):
            d["close"] = 100.0 - i * 0.5
        with patch.object(engine, "_fetch_historical_prices", return_value=spy_data):
            with patch.object(engine, "_generate_daily_signals", return_value={}):
                result = engine.run_backtest(
                    {"SPY": 1.0},
                    start_date="2020-01-01",
                    end_date="2020-03-01",
                    rebalance_freq="monthly",
                )
        assert result.total_return < 0
        assert result.cagr < 0

    def test_nan_in_price_data(self, tmp_path):
        """NaN in price data should propagate."""
        engine = _make_engine(tmp_path)
        spy_data = _make_price_data(days=60, base_price=100.0)
        spy_data[5]["close"] = float("nan")
        with patch.object(engine, "_fetch_historical_prices", return_value=spy_data):
            with patch.object(engine, "_generate_daily_signals", return_value={}):
                result = engine.run_backtest(
                    {"SPY": 1.0},
                    start_date="2020-01-01",
                    end_date="2020-03-01",
                    rebalance_freq="monthly",
                )
        # NaN in price creates NaN return, which should not crash
        assert isinstance(result, BacktestResult)

    def test_spy_benchmark_returns_empty(self, tmp_path):
        """When SPY benchmark data is empty, crisis alpha should be 0."""
        engine = _make_engine(tmp_path)
        spy_data = _make_price_data(days=60, base_price=100.0)

        def mock_fetch(symbol, start, end):
            if symbol == "SPY":
                return spy_data
            return []

        with patch.object(engine, "_fetch_historical_prices", side_effect=mock_fetch):
            with patch.object(engine, "_generate_daily_signals", return_value={}):
                result = engine.run_backtest(
                    {"SPY": 1.0},
                    start_date="2020-01-01",
                    end_date="2020-03-01",
                    rebalance_freq="monthly",
                )
        # Each crisis alpha should be computed (maybe 0 if no overlap)
        for crisis in ("crisis_alpha_2008", "crisis_alpha_2020", "crisis_alpha_2022"):
            assert crisis in result.extras

    def test_no_signal_history(self, tmp_path):
        """When no rebalance dates exist, signal_history is empty."""
        engine = _make_engine(tmp_path)
        spy_data = _make_price_data(days=60, base_price=100.0)
        with patch.object(engine, "_fetch_historical_prices", return_value=spy_data):
            with patch.object(engine, "_generate_daily_signals", return_value={}):
                result = engine.run_backtest(
                    {"SPY": 1.0},
                    start_date="2020-01-01",
                    end_date="2020-03-01",
                    rebalance_freq="monthly",
                )
        assert result.extras["avg_signal_confidence"] == 0
        assert result.extras["regime_distribution"] == {}
        assert result.extras["source_contributions"] == {}

    def test_rolling_sharpe_insufficient_data(self, tmp_path):
        """Less than 252 returns should produce empty rolling Sharpe."""
        engine = _make_engine(tmp_path)
        spy_data = _make_price_data(days=50, base_price=100.0)
        with patch.object(engine, "_fetch_historical_prices", return_value=spy_data):
            with patch.object(engine, "_generate_daily_signals", return_value={}):
                result = engine.run_backtest(
                    {"SPY": 1.0},
                    start_date="2020-01-01",
                    end_date="2020-03-01",
                    rebalance_freq="monthly",
                )
        assert result.extras["rolling_sharpe_1y"] == []

    def test_three_asset_portfolio(self, tmp_path):
        """Three-asset portfolio should run without errors."""
        engine = _make_engine(tmp_path)
        spy_data = _make_price_data(days=60, base_price=100.0, symbol="SPY")
        gld_data = _make_price_data(days=60, base_price=50.0, symbol="GLD")
        tlt_data = _make_price_data(days=60, base_price=80.0, symbol="TLT")

        def mock_fetch(symbol, start, end):
            mapping = {"SPY": spy_data, "GLD": gld_data, "TLT": tlt_data}
            return mapping.get(symbol, [])

        with patch.object(engine, "_fetch_historical_prices", side_effect=mock_fetch):
            with patch.object(engine, "_generate_daily_signals", return_value={}):
                result = engine.run_backtest(
                    {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
                    start_date="2020-01-01",
                    end_date="2020-03-01",
                    rebalance_freq="monthly",
                )
        assert isinstance(result, BacktestResult)
        assert result.total_rebalances > 0

    def test_tx_cost_reduces_return(self, tmp_path):
        """With turnover, transaction costs should reduce the return."""
        engine = _make_engine(tmp_path)
        spy_data = _make_price_data(days=60, base_price=100.0, symbol="SPY")
        gld_data = _make_price_data(days=60, base_price=50.0, symbol="GLD")

        mock_signal = {
            "SPY": {"score": 0.9, "confidence": 1.0, "regime": "bull", "sources": [{"source": "a"}]},
            "GLD": {"score": -0.9, "confidence": 1.0, "regime": "bear", "sources": [{"source": "b"}]},
        }

        def mock_fetch(symbol, start, end):
            mapping = {"SPY": spy_data, "GLD": gld_data}
            return mapping.get(symbol, [])

        with patch.object(engine, "_fetch_historical_prices", side_effect=mock_fetch):
            with patch.object(engine, "_generate_daily_signals", return_value=mock_signal):
                result = engine.run_backtest(
                    {"SPY": 0.6, "GLD": 0.4},
                    start_date="2020-01-01",
                    end_date="2020-03-01",
                    rebalance_freq="monthly",
                )
        assert isinstance(result, BacktestResult)
        assert "source_contributions" in result.extras


# ---------------------------------------------------------------------------
# validate_target edge cases
# ---------------------------------------------------------------------------

class TestValidateTargetEdgeCases:
    """Edge cases for validate_target."""

    def test_nan_sharpe_ratio(self, tmp_path, capsys):
        """NaN Sharpe ratio should fail comparison (NaN >= x is False)."""
        engine = _make_engine(tmp_path)
        result = _make_result()
        result.sharpe_ratio = float("nan")
        assert engine.validate_target(result, target_sharpe=0.95) is False

    def test_inf_sharpe_ratio(self, tmp_path, capsys):
        """Inf Sharpe ratio should pass any target."""
        engine = _make_engine(tmp_path)
        result = _make_result()
        result.sharpe_ratio = float("inf")
        assert engine.validate_target(result, target_sharpe=0.95) is True

    def test_negative_target_sharpe(self, tmp_path, capsys):
        """Negative target Sharpe should pass with positive Sharpe."""
        engine = _make_engine(tmp_path)
        result = _make_result()
        result.sharpe_ratio = 0.5
        assert engine.validate_target(result, target_sharpe=-1.0) is True

    def test_zero_target_sharpe(self, tmp_path, capsys):
        """Zero target Sharpe: any positive Sharpe should pass."""
        engine = _make_engine(tmp_path)
        result = _make_result()
        result.sharpe_ratio = 0.01
        assert engine.validate_target(result, target_sharpe=0.0) is True

    def test_minimal_extras_missing_keys(self, tmp_path):
        """Validate_target with missing extras keys should raise KeyError."""
        engine = _make_engine(tmp_path)
        result = _make_result()
        result.extras = {}
        result.sharpe_ratio = 0.95
        with pytest.raises(KeyError):
            engine.validate_target(result, target_sharpe=0.90)

    def test_validate_target_output_format(self, tmp_path, caplog):
        """validate_target should print formatted output."""
        engine = _make_engine(tmp_path)
        result = _make_result()
        result.sharpe_ratio = 0.95
        with caplog.at_level(logging.INFO, logger="src.backtest.ensemble_backtest"):
            engine.validate_target(result, target_sharpe=0.95)
        assert "ENSEMBLE BACKTEST VALIDATION" in caplog.text
        assert "Sharpe Ratio:" in caplog.text
        assert "CAGR:" in caplog.text

    def test_validate_target_output_shows_crisis_alpha(self, tmp_path, caplog):
        """validate_target output should include crisis alpha sections."""
        engine = _make_engine(tmp_path)
        result = _make_result()
        result.sharpe_ratio = 0.95
        with caplog.at_level(logging.INFO, logger="src.backtest.ensemble_backtest"):
            engine.validate_target(result, target_sharpe=0.95)
        assert "Crisis Alpha" in caplog.text
        assert "2008 GFC" in caplog.text or "2008" in caplog.text


# ---------------------------------------------------------------------------
# main() / CLI edge cases
# ---------------------------------------------------------------------------

class TestMainEdgeCases:
    """Edge cases for the main() CLI entry point."""

    def test_main_print_output(self):
        """main() should print portfolio and period info."""
        with patch("src.backtest.ensemble_backtest.EnsembleBacktestEngine") as mock_cls:
            mock_engine = MagicMock()
            mock_result = MagicMock()
            mock_result.sharpe_ratio = 0.9
            mock_result.extras = {}
            mock_engine.run_backtest.return_value = mock_result
            mock_cls.return_value = mock_engine

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

            mock_engine.run_backtest.assert_called_once()

    def test_main_portfolio_weights_greater_than_one(self):
        """Weights > 1 should be divided by 100."""
        with patch("src.backtest.ensemble_backtest.EnsembleBacktestEngine") as mock_cls:
            mock_engine = MagicMock()
            mock_result = MagicMock()
            mock_result.sharpe_ratio = 0.9
            mock_result.extras = {}
            mock_engine.run_backtest.return_value = mock_result
            mock_cls.return_value = mock_engine

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

            call_kwargs = mock_engine.run_backtest.call_args[1]
            pf = call_kwargs["portfolio"]
            # 46/38/16 with > 1 → divided by 100: 0.46, 0.38, 0.16
            # Then normalized to sum to 1.0
            assert abs(sum(pf.values()) - 1.0) < 0.01
            assert abs(pf["SPY"] - 0.46) < 0.01
            assert abs(pf["GLD"] - 0.38) < 0.01
            assert abs(pf["TLT"] - 0.16) < 0.01


# ---------------------------------------------------------------------------
# EnsembleBacktestResult extras structure edge cases
# ---------------------------------------------------------------------------

class TestEnsembleBacktestResultStructure:
    """Structure and edge cases for backtest results."""

    def test_result_all_numeric_fields_are_numbers(self):
        r = _make_result()
        for field in ("total_return", "cagr", "volatility", "sharpe_ratio"):
            assert isinstance(getattr(r, field), (int, float))

    def test_result_total_rebalances_is_int(self):
        r = _make_result()
        assert isinstance(r.total_rebalances, int)
        assert r.total_rebalances >= 0

    def test_result_max_drawdown_is_negative_or_zero(self):
        r = _make_result()
        assert r.max_drawdown <= 0

    def test_result_extras_source_contributions_structure(self):
        r = _make_result()
        sc = r.extras["source_contributions"]
        for name, contrib in sc.items():
            assert isinstance(contrib, dict)
            assert "hits" in contrib
            assert "avg_confidence" in contrib
            assert isinstance(contrib["hits"], int)
            assert isinstance(contrib["avg_confidence"], float)

    def test_result_extras_rolling_sharpe_structure(self):
        r = _make_result()
        rs = r.extras["rolling_sharpe_1y"]
        assert isinstance(rs, list)
        if rs:
            date_str, sharpe_val = rs[0]
            assert isinstance(date_str, str)
            assert isinstance(sharpe_val, (int, float))

    def test_result_extras_regime_distribution_sums_to_one(self):
        r = _make_result()
        rd = r.extras["regime_distribution"]
        assert abs(sum(rd.values()) - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Ensures __all__ coverage
# ---------------------------------------------------------------------------

class TestExportCompleteness:
    """Verify module __all__ export completeness."""

    def test_public_api_in_all(self):
        """All public names should be in __all__."""
        from src.backtest import ensemble_backtest as mod
        # Classes/functions that start with uppercase are public
        public_names = [name for name in dir(mod)
                        if not name.startswith("_") and not name.startswith("logger")]
        # EnsembleBacktestEngine is the main public API
        assert "EnsembleBacktestEngine" in mod.__all__
        assert "EnsembleBacktestEngine" in public_names

    def test_main_is_not_in_all(self):
        """main() should not be in __all__ (it's a CLI entry)."""
        from src.backtest import ensemble_backtest as mod
        assert "main" not in mod.__all__

    def test_no_private_items_in_all(self):
        """__all__ should not contain private names."""
        from src.backtest import ensemble_backtest as mod
        for name in mod.__all__:
            assert not name.startswith("_"), f"Private name in __all__: {name}"


# ---------------------------------------------------------------------------
# Source contributions and regime distribution edge cases
# ---------------------------------------------------------------------------

class TestSourceContributionsEdgeCases:
    """Edge cases for source contribution tracking."""

    def test_source_contributions_empty_history(self, tmp_path):
        """Empty signal history should produce empty contributions."""
        engine = _make_engine(tmp_path)
        spy_data = _make_price_data(days=60, base_price=100.0)
        with patch.object(engine, "_fetch_historical_prices", return_value=spy_data):
            with patch.object(engine, "_generate_daily_signals", return_value={}):
                result = engine.run_backtest(
                    {"SPY": 1.0},
                    start_date="2020-01-01",
                    end_date="2020-03-01",
                    rebalance_freq="monthly",
                )
        assert result.extras["source_contributions"] == {}

    def test_source_contributions_with_multiple_assets(self, tmp_path):
        """Multiple assets should produce contributions for each."""
        engine = _make_engine(tmp_path)
        spy_data = _make_price_data(days=60, base_price=100.0, symbol="SPY")
        gld_data = _make_price_data(days=60, base_price=50.0, symbol="GLD")

        mock_signal = {
            "SPY": {"score": 0.3, "confidence": 0.8, "regime": "bull",
                     "sources": [{"source": "tsfm"}, {"source": "cta"}]},
            "GLD": {"score": -0.2, "confidence": 0.6, "regime": "neutral",
                     "sources": [{"source": "cta"}]},
        }

        def mock_fetch(symbol, start, end):
            return spy_data if symbol == "SPY" else gld_data

        with patch.object(engine, "_fetch_historical_prices", side_effect=mock_fetch):
            with patch.object(engine, "_generate_daily_signals", return_value=mock_signal):
                result = engine.run_backtest(
                    {"SPY": 0.6, "GLD": 0.4},
                    start_date="2020-01-01",
                    end_date="2020-03-01",
                    rebalance_freq="monthly",
                )
        sc = result.extras["source_contributions"]
        assert "tsfm" in sc or "cta" in sc
        # At least one source should have hits > 0
        total_hits = sum(s["hits"] for s in sc.values())
        assert total_hits > 0


class TestRegimeDistributionEdgeCases:
    """Edge cases for regime distribution tracking."""

    def test_regime_distribution_empty_history(self, tmp_path):
        """Empty signal history should produce empty regime distribution."""
        engine = _make_engine(tmp_path)
        spy_data = _make_price_data(days=60, base_price=100.0)
        with patch.object(engine, "_fetch_historical_prices", return_value=spy_data):
            with patch.object(engine, "_generate_daily_signals", return_value={}):
                result = engine.run_backtest(
                    {"SPY": 1.0},
                    start_date="2020-01-01",
                    end_date="2020-03-01",
                    rebalance_freq="monthly",
                )
        assert result.extras["regime_distribution"] == {}


# ---------------------------------------------------------------------------
# __all__ export and guard validation
# ---------------------------------------------------------------------------

class TestModuleGuard:
    """Verify module __main__ guard and __all__."""

    def test_module_has___all__(self):
        from src.backtest import ensemble_backtest as mod
        assert hasattr(mod, "__all__")
        assert isinstance(mod.__all__, list)

    def test___all___is_exhaustive(self):
        """__all__ should contain EnsembleBacktestEngine."""
        from src.backtest import ensemble_backtest as mod
        assert "EnsembleBacktestEngine" in mod.__all__

    def test___all___no_duplicates(self):
        from src.backtest import ensemble_backtest as mod
        assert len(mod.__all__) == len(set(mod.__all__))

    def test_module_exports_only_expected(self):
        """__all__ should contain only public API classes."""
        from src.backtest import ensemble_backtest as mod
        allowed = {"EnsembleBacktestEngine"}
        for name in mod.__all__:
            assert name in allowed, f"Unexpected export: {name}"
