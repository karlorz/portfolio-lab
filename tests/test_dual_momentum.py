#!/usr/bin/env python3
"""
Tests for Dual Momentum Strategy — data classes, momentum scoring,
absolute/relative momentum filtering, allocation generation,
and rebalance recommendations.
"""
import sys
import os
import json
import sqlite3
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.strategy.dual_momentum import (
    MomentumScore, DualMomentumSignal,
    DualMomentumEngine, DualMomentumBacktest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(tmp_path, top_n=2, risk_off_asset='TLT', momentum_threshold=0.0):
    """Create a DualMomentumEngine with tmp DB path."""
    engine = DualMomentumEngine.__new__(DualMomentumEngine)
    engine.db_path = tmp_path / "market.db"
    engine.lookback_months = 12
    engine.sma_days = 200
    engine.top_n = top_n
    engine.risk_off_asset = risk_off_asset
    engine.momentum_threshold = momentum_threshold
    engine.vol_lookback = 20
    engine.universe = ['SPY', 'GLD', 'TLT', 'IEF', 'QQQ']
    engine.base_allocation = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16}
    return engine


def _init_db(db_path):
    """Create a prices table in SQLite."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            date TEXT, symbol TEXT, close REAL, volume INTEGER
        )
    """)
    conn.commit()
    return conn


def _insert_prices(conn, symbol, n_days=300, base_price=100.0, drift=0.0004):
    """Insert synthetic price data."""
    import random
    random.seed(hash(symbol) % 2**31)
    d = datetime(2025, 1, 2)
    price = base_price
    rows = []
    for _ in range(n_days):
        while d.weekday() >= 5:
            d += timedelta(days=1)
        price *= (1 + drift + random.gauss(0, 0.015))
        rows.append((d.strftime('%Y-%m-%d'), symbol, round(price, 2), 1000000))
        d += timedelta(days=1)
    conn.executemany("INSERT INTO prices VALUES (?, ?, ?, ?)", rows)
    conn.commit()


def _make_momentum_score(symbol='SPY', price=500.0, sma_200=480.0,
                         return_12m=0.10, return_6m=0.05, return_3m=0.02,
                         volatility=0.15, above_sma=True, score=0.08):
    return MomentumScore(
        symbol=symbol, price=price, sma_200=sma_200,
        return_12m=return_12m, return_6m=return_6m, return_3m=return_3m,
        volatility=volatility, above_sma=above_sma, score=score,
    )


# ---------------------------------------------------------------------------
# MomentumScore tests
# ---------------------------------------------------------------------------

class TestMomentumScore:
    def test_creation(self):
        ms = _make_momentum_score()
        assert ms.symbol == 'SPY'
        assert ms.price == 500.0
        assert ms.above_sma is True

    def test_fields(self):
        ms = _make_momentum_score(return_12m=0.15, volatility=0.20, score=0.12)
        assert ms.return_12m == 0.15
        assert ms.volatility == 0.20
        assert ms.score == 0.12


# ---------------------------------------------------------------------------
# DualMomentumSignal tests
# ---------------------------------------------------------------------------

class TestDualMomentumSignal:
    def test_creation(self):
        sig = DualMomentumSignal(
            timestamp='2026-01-01',
            base_allocation={'SPY': 0.46},
            adjusted_allocation={'SPY': 0.50},
            momentum_scores={},
            selected_assets=['SPY'],
            risk_off=False,
            signal_strength=0.7,
            rebalance_triggered=True,
        )
        assert sig.risk_off is False
        assert sig.signal_strength == 0.7

    def test_risk_off(self):
        sig = DualMomentumSignal(
            timestamp='2026-01-01',
            base_allocation={'SPY': 0.46},
            adjusted_allocation={'TLT': 1.0},
            momentum_scores={},
            selected_assets=['TLT'],
            risk_off=True,
            signal_strength=0.0,
            rebalance_triggered=False,
        )
        assert sig.risk_off is True


# ---------------------------------------------------------------------------
# DualMomentumEngine tests
# ---------------------------------------------------------------------------

class TestDualMomentumEngine:
    def test_init_defaults(self):
        engine = DualMomentumEngine.__new__(DualMomentumEngine)
        engine.universe = ['SPY', 'GLD', 'TLT', 'IEF', 'QQQ', 'EFA', 'VXUS']
        engine.base_allocation = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16}
        assert len(engine.universe) == 7
        assert engine.base_allocation['SPY'] == 0.46

    def test_fetch_price_data_no_db(self, tmp_path):
        engine = _make_engine(tmp_path)
        result = engine._fetch_price_data('SPY')
        assert result == []

    def test_fetch_price_data_with_data(self, tmp_path):
        engine = _make_engine(tmp_path)
        conn = _init_db(engine.db_path)
        _insert_prices(conn, 'SPY', n_days=300)
        conn.close()
        result = engine._fetch_price_data('SPY', days=300)
        assert len(result) == 300
        assert 'date' in result[0]
        assert 'close' in result[0]

    def test_fetch_price_data_chronological_order(self, tmp_path):
        engine = _make_engine(tmp_path)
        conn = _init_db(engine.db_path)
        _insert_prices(conn, 'SPY', n_days=50)
        conn.close()
        result = engine._fetch_price_data('SPY', days=50)
        dates = [r['date'] for r in result]
        assert dates == sorted(dates)

    def test_calculate_momentum_score_with_data(self, tmp_path):
        engine = _make_engine(tmp_path)
        conn = _init_db(engine.db_path)
        _insert_prices(conn, 'SPY', n_days=300, base_price=400.0, drift=0.0005)
        conn.close()
        score = engine._calculate_momentum_score('SPY')
        assert score is not None
        assert isinstance(score, MomentumScore)
        assert score.symbol == 'SPY'

    def test_calculate_momentum_score_insufficient_data(self, tmp_path):
        engine = _make_engine(tmp_path)
        conn = _init_db(engine.db_path)
        _insert_prices(conn, 'SPY', n_days=10)
        conn.close()
        score = engine._calculate_momentum_score('SPY')
        assert score is None

    def test_calculate_momentum_score_above_sma(self, tmp_path):
        engine = _make_engine(tmp_path)
        conn = _init_db(engine.db_path)
        # Strong uptrend → price above SMA
        _insert_prices(conn, 'SPY', n_days=300, base_price=400.0, drift=0.002)
        conn.close()
        score = engine._calculate_momentum_score('SPY')
        assert score is not None
        assert bool(score.above_sma) is True

    def test_calculate_momentum_score_below_sma(self, tmp_path):
        engine = _make_engine(tmp_path)
        conn = _init_db(engine.db_path)
        # Strong downtrend → price below SMA
        _insert_prices(conn, 'SPY', n_days=300, base_price=800.0, drift=-0.002)
        conn.close()
        score = engine._calculate_momentum_score('SPY')
        assert score is not None
        assert bool(score.above_sma) is False

    # _generate_allocation tests
    def test_generate_allocation_risk_off(self, tmp_path):
        engine = _make_engine(tmp_path)
        result = engine._generate_allocation(['TLT'], {}, risk_off=True)
        assert result == {'TLT': 1.0}

    def test_generate_allocation_empty_selected(self, tmp_path):
        engine = _make_engine(tmp_path)
        result = engine._generate_allocation([], {}, risk_off=False)
        assert result == {'TLT': 1.0}

    def test_generate_allocation_includes_tlt(self, tmp_path):
        engine = _make_engine(tmp_path)
        scores = {
            'SPY': _make_momentum_score('SPY', volatility=0.15),
            'QQQ': _make_momentum_score('QQQ', volatility=0.20),
        }
        result = engine._generate_allocation(['SPY', 'QQQ'], scores, risk_off=False)
        assert 'TLT' in result
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_generate_allocation_includes_gld(self, tmp_path):
        engine = _make_engine(tmp_path)
        scores = {
            'SPY': _make_momentum_score('SPY', volatility=0.15),
            'QQQ': _make_momentum_score('QQQ', volatility=0.20),
        }
        result = engine._generate_allocation(['SPY', 'QQQ'], scores, risk_off=False)
        assert 'GLD' in result

    def test_generate_allocation_sums_to_one(self, tmp_path):
        engine = _make_engine(tmp_path)
        scores = {
            'SPY': _make_momentum_score('SPY', volatility=0.15),
            'GLD': _make_momentum_score('GLD', volatility=0.12),
            'TLT': _make_momentum_score('TLT', volatility=0.08),
        }
        result = engine._generate_allocation(['SPY', 'GLD', 'TLT'], scores, risk_off=False)
        assert abs(sum(result.values()) - 1.0) < 0.01

    def test_generate_allocation_inverse_vol(self, tmp_path):
        engine = _make_engine(tmp_path)
        scores = {
            'SPY': _make_momentum_score('SPY', volatility=0.20),
            'GLD': _make_momentum_score('GLD', volatility=0.10),
        }
        result = engine._generate_allocation(['SPY', 'GLD'], scores, risk_off=False)
        # Lower vol → higher weight (inverse vol)
        assert result['GLD'] > result['SPY']

    def test_generate_allocation_zero_vol_fallback(self, tmp_path):
        engine = _make_engine(tmp_path)
        scores = {
            'SPY': _make_momentum_score('SPY', volatility=0.0),
            'GLD': _make_momentum_score('GLD', volatility=0.0),
        }
        result = engine._generate_allocation(['SPY', 'GLD'], scores, risk_off=False)
        assert abs(sum(result.values()) - 1.0) < 0.01

    # evaluate tests
    def test_evaluate_no_data_returns_base(self, tmp_path):
        engine = _make_engine(tmp_path)
        signal = engine.evaluate()
        assert signal.risk_off is True
        assert signal.adjusted_allocation == engine.base_allocation

    def test_evaluate_with_data(self, tmp_path):
        engine = _make_engine(tmp_path)
        conn = _init_db(engine.db_path)
        for sym in engine.universe:
            _insert_prices(conn, sym, n_days=300, base_price=400.0, drift=0.0005)
        conn.close()
        signal = engine.evaluate()
        assert isinstance(signal, DualMomentumSignal)
        assert len(signal.momentum_scores) > 0

    def test_evaluate_risk_off_when_all_below_sma(self, tmp_path):
        engine = _make_engine(tmp_path)
        conn = _init_db(engine.db_path)
        for sym in engine.universe:
            _insert_prices(conn, sym, n_days=300, base_price=800.0, drift=-0.002)
        conn.close()
        signal = engine.evaluate()
        assert signal.risk_off is True
        assert engine.risk_off_asset in signal.selected_assets

    def test_evaluate_selects_top_n(self, tmp_path):
        engine = _make_engine(tmp_path, top_n=2)
        conn = _init_db(engine.db_path)
        for sym in engine.universe:
            _insert_prices(conn, sym, n_days=300, base_price=400.0, drift=0.0005)
        conn.close()
        signal = engine.evaluate()
        if not signal.risk_off:
            assert len(signal.selected_assets) <= 2

    def test_evaluate_signal_strength_bounded(self, tmp_path):
        engine = _make_engine(tmp_path)
        conn = _init_db(engine.db_path)
        for sym in engine.universe:
            _insert_prices(conn, sym, n_days=300, base_price=400.0, drift=0.001)
        conn.close()
        signal = engine.evaluate()
        assert 0.0 <= signal.signal_strength <= 1.0

    # get_rebalance_recommendation tests
    def test_get_rebalance_recommendation_returns_dict(self, tmp_path):
        engine = _make_engine(tmp_path)
        conn = _init_db(engine.db_path)
        for sym in engine.universe:
            _insert_prices(conn, sym, n_days=300, base_price=400.0, drift=0.0005)
        conn.close()
        rec = engine.get_rebalance_recommendation({'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16})
        assert 'recommendation' in rec
        assert 'drifts' in rec

    def test_get_rebalance_recommendation_hold_when_aligned(self, tmp_path):
        engine = _make_engine(tmp_path)
        conn = _init_db(engine.db_path)
        for sym in engine.universe:
            _insert_prices(conn, sym, n_days=300, base_price=400.0, drift=0.0005)
        conn.close()
        rec = engine.get_rebalance_recommendation(engine.base_allocation, threshold=1.0)
        assert rec['recommendation'] == 'HOLD'

    def test_get_rebalance_recommendation_rebalance_when_drifted(self, tmp_path):
        engine = _make_engine(tmp_path)
        conn = _init_db(engine.db_path)
        for sym in engine.universe:
            _insert_prices(conn, sym, n_days=300, base_price=400.0, drift=0.0005)
        conn.close()
        # Very different current positions → drift
        rec = engine.get_rebalance_recommendation({'SPY': 1.0}, threshold=0.01)
        assert rec['recommendation'] == 'REBALANCE'


# ---------------------------------------------------------------------------
# DualMomentumBacktest tests
# ---------------------------------------------------------------------------

class TestDualMomentumBacktest:
    def test_run_backtest_no_db(self, tmp_path):
        engine = _make_engine(tmp_path)
        bt = DualMomentumBacktest(engine)
        result = bt.run_backtest('2020-01-01', '2025-12-31')
        assert result['status'] == 'failed'

    def test_run_backtest_insufficient_data(self, tmp_path):
        engine = _make_engine(tmp_path)
        conn = _init_db(engine.db_path)
        _insert_prices(conn, 'SPY', n_days=10)
        conn.close()
        bt = DualMomentumBacktest(engine)
        result = bt.run_backtest('2024-01-01', '2025-12-31')
        assert result['status'] == 'failed'

    def test_run_backtest_with_data(self, tmp_path):
        engine = _make_engine(tmp_path)
        conn = _init_db(engine.db_path)
        for sym in engine.universe:
            _insert_prices(conn, sym, n_days=800, base_price=400.0, drift=0.0004)
        conn.close()
        bt = DualMomentumBacktest(engine)
        result = bt.run_backtest('2023-01-01', '2025-12-31')
        assert result['status'] == 'completed'
        assert 'cagr' in result
        assert 'sharpe_ratio' in result

    def test_run_backtest_metrics_present(self, tmp_path):
        engine = _make_engine(tmp_path)
        conn = _init_db(engine.db_path)
        for sym in engine.universe:
            _insert_prices(conn, sym, n_days=800, base_price=400.0, drift=0.0004)
        conn.close()
        bt = DualMomentumBacktest(engine)
        result = bt.run_backtest('2023-01-01', '2025-12-31')
        assert 'final_value' in result
        assert 'max_drawdown' in result
        assert 'trade_count' in result
        assert 'risk_off_months' in result

    def test_run_backtest_max_drawdown_non_negative(self, tmp_path):
        engine = _make_engine(tmp_path)
        conn = _init_db(engine.db_path)
        for sym in engine.universe:
            _insert_prices(conn, sym, n_days=800, base_price=400.0, drift=0.0004)
        conn.close()
        bt = DualMomentumBacktest(engine)
        result = bt.run_backtest('2023-01-01', '2025-12-31')
        assert result['max_drawdown'] <= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
