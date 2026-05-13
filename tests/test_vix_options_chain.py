#!/usr/bin/env python3
"""
Tests for VIX Options Data Pipeline — data classes, delta approximation,
insurance candidate selection, DB operations, and historical context.
"""
import sys
import os
import json
import sqlite3
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

# Mock yfinance and aiohttp before importing (not installed in test env)
_orig_yf = sys.modules.get('yfinance')
_orig_aio = sys.modules.get('aiohttp')
sys.modules['yfinance'] = MagicMock()
sys.modules['aiohttp'] = MagicMock()

from src.data.vix_options_chain import (
    VIXOption, VIXOptionsChain, VIXDataPipeline,
)

# Restore original modules to prevent pollution
if _orig_yf is None:
    sys.modules.pop('yfinance', None)
else:
    sys.modules['yfinance'] = _orig_yf
if _orig_aio is None:
    sys.modules.pop('aiohttp', None)
else:
    sys.modules['aiohttp'] = _orig_aio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline(tmp_path):
    """Create a VIXDataPipeline with tmp paths."""
    pipeline = VIXDataPipeline.__new__(VIXDataPipeline)
    pipeline.DB_PATH = tmp_path / "vix_options.db"
    pipeline.DATA_DIR = tmp_path / "signals"
    pipeline.DATA_DIR.mkdir(parents=True, exist_ok=True)
    pipeline._init_db()
    return pipeline


def _make_option(strike=22.0, expiration=None, option_type='call',
                 bid=1.50, ask=2.00, last_price=1.75, volume=100,
                 open_interest=500, implied_vol=25.0, delta=0.30,
                 gamma=None, theta=None):
    if expiration is None:
        expiration = (datetime.now() + timedelta(days=60)).strftime('%Y-%m-%d')
    return VIXOption(
        strike=strike, expiration=expiration, option_type=option_type,
        bid=bid, ask=ask, last_price=last_price, volume=volume,
        open_interest=open_interest, implied_vol=implied_vol, delta=delta,
        gamma=gamma, theta=theta,
    )


def _make_chain(calls=None, puts=None, vix_spot=18.0):
    return VIXOptionsChain(
        timestamp=datetime.now().isoformat(),
        vix_spot=vix_spot,
        vix_9day=17.5,
        vix_3m=19.0,
        term_structure={'2026-06-20': 22.0, '2026-07-18': 23.5},
        calls=calls or [_make_option()],
        puts=puts or [],
    )


# ---------------------------------------------------------------------------
# VIXOption tests
# ---------------------------------------------------------------------------

class TestVIXOption:
    def test_creation(self):
        o = _make_option()
        assert o.strike == 22.0
        assert o.option_type == 'call'

    def test_mid_price(self):
        o = _make_option(bid=1.50, ask=2.00)
        assert o.mid_price == 1.75

    def test_premium(self):
        o = _make_option(bid=1.50, ask=2.00)
        assert o.premium == 175.0  # 1.75 * 100

    def test_mid_price_zero_spread(self):
        o = _make_option(bid=2.00, ask=2.00)
        assert o.mid_price == 2.00

    def test_optional_greeks(self):
        o = _make_option(delta=0.30, gamma=0.05, theta=-0.02)
        assert o.delta == 0.30
        assert o.gamma == 0.05
        assert o.theta == -0.02


# ---------------------------------------------------------------------------
# VIXOptionsChain tests
# ---------------------------------------------------------------------------

class TestVIXOptionsChain:
    def test_creation(self):
        c = _make_chain()
        assert c.vix_spot == 18.0
        assert len(c.calls) == 1

    def test_to_dict(self):
        c = _make_chain()
        d = c.to_dict()
        assert d['vix_spot'] == 18.0
        assert 'calls' in d
        assert 'puts' in d
        assert isinstance(d['calls'], list)

    def test_to_dict_calls_are_dicts(self):
        c = _make_chain()
        d = c.to_dict()
        assert isinstance(d['calls'][0], dict)
        assert 'strike' in d['calls'][0]

    def test_term_structure(self):
        c = _make_chain()
        assert '2026-06-20' in c.term_structure


# ---------------------------------------------------------------------------
# VIXDataPipeline tests
# ---------------------------------------------------------------------------

class TestVIXDataPipeline:
    def test_init_creates_db(self, tmp_path):
        p = _make_pipeline(tmp_path)
        assert p.DB_PATH.exists()

    def test_init_creates_tables(self, tmp_path):
        p = _make_pipeline(tmp_path)
        conn = sqlite3.connect(str(p.DB_PATH))
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()
        assert 'vix_history' in tables
        assert 'options_chain' in tables
        assert 'insurance_candidates' in tables
        assert 'historical_vix_events' in tables

    def test_init_creates_data_dir(self, tmp_path):
        p = _make_pipeline(tmp_path)
        assert p.DATA_DIR.exists()

    # calculate_delta_approx
    def test_delta_approx_atm(self, tmp_path):
        p = _make_pipeline(tmp_path)
        delta = p.calculate_delta_approx(18.0, 18.0, 60, 25.0)
        assert 0.45 <= delta <= 0.55  # ATM ≈ 0.5

    def test_delta_approx_itm(self, tmp_path):
        p = _make_pipeline(tmp_path)
        delta = p.calculate_delta_approx(15.0, 18.0, 60, 25.0)
        assert delta > 0.5  # ITM call

    def test_delta_approx_otm(self, tmp_path):
        p = _make_pipeline(tmp_path)
        delta = p.calculate_delta_approx(22.0, 18.0, 60, 25.0)
        assert delta < 0.5  # OTM call

    def test_delta_approx_bounded(self, tmp_path):
        p = _make_pipeline(tmp_path)
        delta = p.calculate_delta_approx(10.0, 18.0, 60, 25.0)
        assert 0.0 <= delta <= 1.0

    def test_delta_approx_zero_dte(self, tmp_path):
        p = _make_pipeline(tmp_path)
        delta = p.calculate_delta_approx(18.0, 18.0, 0, 25.0)
        assert delta == 0.0

    def test_delta_approx_zero_iv(self, tmp_path):
        p = _make_pipeline(tmp_path)
        delta = p.calculate_delta_approx(18.0, 18.0, 60, 0.0)
        assert delta == 0.0

    def test_delta_approx_deep_itm(self, tmp_path):
        p = _make_pipeline(tmp_path)
        delta = p.calculate_delta_approx(10.0, 18.0, 60, 25.0)
        assert delta > 0.8

    def test_delta_approx_deep_otm(self, tmp_path):
        p = _make_pipeline(tmp_path)
        delta = p.calculate_delta_approx(30.0, 18.0, 60, 25.0)
        assert delta < 0.2

    # select_insurance_candidates (async)
    def test_select_candidates_returns_list(self, tmp_path):
        import asyncio
        p = _make_pipeline(tmp_path)
        exp = (datetime.now() + timedelta(days=60)).strftime('%Y-%m-%d')
        call = _make_option(strike=22.0, expiration=exp, delta=0.30)
        chain = _make_chain(calls=[call])
        candidates = asyncio.run(p.select_insurance_candidates(chain))
        assert isinstance(candidates, list)

    def test_select_candidates_filters_dte(self, tmp_path):
        import asyncio
        p = _make_pipeline(tmp_path)
        short_exp = (datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d')
        good_exp = (datetime.now() + timedelta(days=60)).strftime('%Y-%m-%d')
        calls = [
            _make_option(strike=22.0, expiration=short_exp, delta=0.30),
            _make_option(strike=22.0, expiration=good_exp, delta=0.30),
        ]
        chain = _make_chain(calls=calls)
        candidates = asyncio.run(p.select_insurance_candidates(chain))
        for c in candidates:
            assert c['days_to_expiration'] >= 45

    def test_select_candidates_filters_delta(self, tmp_path):
        import asyncio
        p = _make_pipeline(tmp_path)
        exp = (datetime.now() + timedelta(days=60)).strftime('%Y-%m-%d')
        calls = [
            _make_option(strike=22.0, expiration=exp, delta=0.10),
            _make_option(strike=21.0, expiration=exp, delta=0.30),
            _make_option(strike=20.0, expiration=exp, delta=0.60),
        ]
        chain = _make_chain(calls=calls)
        candidates = asyncio.run(p.select_insurance_candidates(chain))
        for c in candidates:
            assert 0.20 <= c['delta'] <= 0.40

    def test_select_candidates_sorted_by_delta_distance(self, tmp_path):
        import asyncio
        p = _make_pipeline(tmp_path)
        exp = (datetime.now() + timedelta(days=60)).strftime('%Y-%m-%d')
        calls = [
            _make_option(strike=22.0, expiration=exp, delta=0.25),
            _make_option(strike=21.0, expiration=exp, delta=0.32),
        ]
        chain = _make_chain(calls=calls)
        candidates = asyncio.run(p.select_insurance_candidates(chain))
        if len(candidates) > 1:
            assert candidates[0]['delta_distance'] <= candidates[1]['delta_distance']

    def test_select_candidates_max_5(self, tmp_path):
        import asyncio
        p = _make_pipeline(tmp_path)
        exp = (datetime.now() + timedelta(days=60)).strftime('%Y-%m-%d')
        calls = [_make_option(strike=20+i, expiration=exp, delta=0.28+i*0.01)
                 for i in range(10)]
        chain = _make_chain(calls=calls)
        candidates = asyncio.run(p.select_insurance_candidates(chain))
        assert len(candidates) <= 5

    def test_select_candidates_has_breakeven(self, tmp_path):
        import asyncio
        p = _make_pipeline(tmp_path)
        exp = (datetime.now() + timedelta(days=60)).strftime('%Y-%m-%d')
        call = _make_option(strike=22.0, expiration=exp, delta=0.30, bid=1.50, ask=2.00)
        chain = _make_chain(calls=[call])
        candidates = asyncio.run(p.select_insurance_candidates(chain))
        if candidates:
            expected_be = 22.0 + 1.75
            assert abs(candidates[0]['breakeven_vix'] - expected_be) < 0.1

    def test_select_candidates_gain_scenarios(self, tmp_path):
        import asyncio
        p = _make_pipeline(tmp_path)
        exp = (datetime.now() + timedelta(days=60)).strftime('%Y-%m-%d')
        call = _make_option(strike=22.0, expiration=exp, delta=0.30, bid=1.50, ask=2.00)
        chain = _make_chain(calls=[call])
        candidates = asyncio.run(p.select_insurance_candidates(chain))
        if candidates:
            c = candidates[0]
            assert c['max_gain_scenario_40'] > 0
            assert c['max_gain_scenario_60'] > c['max_gain_scenario_40']

    # get_latest_candidates
    def test_get_latest_candidates_empty(self, tmp_path):
        p = _make_pipeline(tmp_path)
        assert p.get_latest_candidates() == []

    def test_get_latest_candidates_with_data(self, tmp_path):
        p = _make_pipeline(tmp_path)
        conn = sqlite3.connect(str(p.DB_PATH))
        conn.execute("""
            INSERT INTO insurance_candidates
            (timestamp, expiration_date, days_to_expiration, strike, delta,
             premium, breakeven_vix, max_gain_scenario_40, max_gain_scenario_60)
            VALUES ('2026-01-01', '2026-03-01', 60, 22.0, 0.30, 175.0, 23.75, 16.25, 36.25)
        """)
        conn.commit()
        conn.close()
        candidates = p.get_latest_candidates()
        assert len(candidates) == 1
        assert candidates[0]['strike'] == 22.0

    # get_historical_context
    def test_get_historical_context_empty(self, tmp_path):
        p = _make_pipeline(tmp_path)
        ctx = p.get_historical_context()
        assert ctx['available_history_days'] == 0

    def test_get_historical_context_with_data(self, tmp_path):
        p = _make_pipeline(tmp_path)
        conn = sqlite3.connect(str(p.DB_PATH))
        for i in range(5):
            conn.execute("""
                INSERT INTO vix_history (timestamp, vix_spot)
                VALUES (?, ?)
            """, (f'2026-01-{i+1:02d}', 18.0 + i))
        conn.commit()
        conn.close()
        ctx = p.get_historical_context(days=5)
        assert ctx['available_history_days'] == 5
        assert ctx['vix_current'] == 22.0  # Latest

    def test_get_historical_context_stats(self, tmp_path):
        p = _make_pipeline(tmp_path)
        conn = sqlite3.connect(str(p.DB_PATH))
        for i in range(10):
            conn.execute("""
                INSERT INTO vix_history (timestamp, vix_spot)
                VALUES (?, ?)
            """, (f'2026-01-{i+1:02d}', 15.0 + i))
        conn.commit()
        conn.close()
        ctx = p.get_historical_context(days=10)
        assert ctx['vix_30d_min'] == 15.0
        assert ctx['vix_30d_max'] == 24.0

    # VIX futures tickers
    def test_vix_futures_tickers(self):
        assert 'front_month' in VIXDataPipeline.VIX_FUTURES
        assert 'second_month' in VIXDataPipeline.VIX_FUTURES


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
