#!/usr/bin/env python3
"""
Tests for cta_overlay.py — TrendSignal/CTAPosition dataclasses, SMA/volatility
calculation, trend signal generation, ensemble scoring, position sizing, and
full CTA evaluation.
"""
import sys
import os
import sqlite3
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pathlib import Path
from datetime import datetime, date, timedelta
from unittest.mock import patch, MagicMock

from src.strategy.cta_overlay import (
    TrendSignal,
    CTAPosition,
    CTATrendEngine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trend_signal(**overrides):
    defaults = dict(
        timeframe=60,
        score=0.5,
        strength=0.7,
        price_vs_sma=2.5,
        regime="uptrend",
    )
    defaults.update(overrides)
    return TrendSignal(**defaults)


def _make_cta_position(**overrides):
    defaults = dict(
        symbol="SPY",
        asset_class="equity",
        base_weight=0.15,
        trend_score=0.4,
        trend_strength=0.6,
        realized_vol=0.18,
        target_vol=0.10,
        position_scalar=0.8,
        final_weight=0.12,
        signal="long",
        last_update=datetime.now().isoformat(),
    )
    defaults.update(overrides)
    return CTAPosition(**defaults)


def _make_prices(n=200, start=450.0, drift=0.0003, vol=0.012, seed=42):
    """Generate synthetic price series."""
    rng = np.random.RandomState(seed)
    prices = [start]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + rng.normal(drift, vol)))
    return np.array(prices)


def _setup_test_db(db_path, symbols=None, days=250):
    """Create a minimal prices table with synthetic data."""
    if symbols is None:
        symbols = ["SPY", "GLD", "TLT", "QQQ", "IWM", "EFA", "VXUS",
                    "IEF", "HYG", "LQD", "DBC", "VIX"]
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS prices (symbol TEXT, date TEXT, close REAL, volume INTEGER)")
    rng = np.random.RandomState(42)
    base_prices = {"SPY": 450, "GLD": 190, "TLT": 95, "QQQ": 380, "IWM": 200,
                   "EFA": 75, "VXUS": 55, "IEF": 105, "HYG": 78, "LQD": 110,
                   "DBC": 23, "VIX": 18}
    today = date(2026, 5, 14)
    for sym in symbols:
        price = float(base_prices.get(sym, 100))
        for i in range(days):
            d = today - timedelta(days=days - i)
            if d.weekday() >= 5:
                continue
            price *= (1 + rng.normal(0.0003, 0.012))
            c.execute("INSERT INTO prices VALUES (?, ?, ?, ?)",
                      (sym, d.isoformat(), price, int(rng.uniform(1e6, 1e8))))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# TrendSignal Tests
# ---------------------------------------------------------------------------

class TestTrendSignal:

    def test_fields(self):
        ts = _make_trend_signal(timeframe=20, score=-0.3, strength=0.5, regime="downtrend")
        assert ts.timeframe == 20
        assert ts.score == -0.3
        assert ts.strength == 0.5
        assert ts.regime == "downtrend"

    def test_uptrend_regime(self):
        ts = _make_trend_signal(score=0.5, regime="uptrend")
        assert ts.regime == "uptrend"

    def test_chop_regime(self):
        ts = _make_trend_signal(score=0.1, regime="chop")
        assert ts.regime == "chop"


# ---------------------------------------------------------------------------
# CTAPosition Tests
# ---------------------------------------------------------------------------

class TestCTAPosition:

    def test_fields(self):
        pos = _make_cta_position(symbol="GLD", asset_class="commodity")
        assert pos.symbol == "GLD"
        assert pos.asset_class == "commodity"

    def test_signal_types(self):
        for sig in ["long", "short", "neutral"]:
            pos = _make_cta_position(signal=sig)
            assert pos.signal == sig


# ---------------------------------------------------------------------------
# CTATrendEngine — constants
# ---------------------------------------------------------------------------

class TestEngineConstants:

    def test_timeframes(self):
        assert "short" in CTATrendEngine.TIMEFRAMES
        assert "medium" in CTATrendEngine.TIMEFRAMES
        assert "long" in CTATrendEngine.TIMEFRAMES
        assert CTATrendEngine.TIMEFRAMES["short"]["days"] == 20
        assert CTATrendEngine.TIMEFRAMES["medium"]["days"] == 60
        assert CTATrendEngine.TIMEFRAMES["long"]["days"] == 120

    def test_timeframe_weights_sum_to_one(self):
        total = sum(t["weight"] for t in CTATrendEngine.TIMEFRAMES.values())
        assert total == pytest.approx(1.0)

    def test_universe(self):
        assert "SPY" in CTATrendEngine.UNIVERSE
        assert "GLD" in CTATrendEngine.UNIVERSE
        assert "TLT" in CTATrendEngine.UNIVERSE

    def test_risk_params(self):
        assert CTATrendEngine.TARGET_VOL == 0.10
        assert CTATrendEngine.MAX_LEVERAGE == 2.0
        assert CTATrendEngine.MIN_LEVERAGE == 0.25

    def test_init_default(self, tmp_path):
        engine = CTATrendEngine()
        assert engine.vol_lookback == 20

    def test_init_custom(self, tmp_path):
        db = tmp_path / "custom.db"
        engine = CTATrendEngine(db_path=db)
        assert engine.db_path == db


# ---------------------------------------------------------------------------
# CTATrendEngine — _fetch_data
# ---------------------------------------------------------------------------

class TestFetchData:

    def test_fetch_returns_list(self, tmp_path):
        db = tmp_path / "test.db"
        _setup_test_db(db, ["SPY"])
        engine = CTATrendEngine(db_path=db)
        data = engine._fetch_data("SPY", 50)
        assert isinstance(data, list)
        assert len(data) > 0
        assert "close" in data[0]
        assert "date" in data[0]

    def test_fetch_nonexistent_symbol(self, tmp_path):
        db = tmp_path / "test.db"
        _setup_test_db(db, ["SPY"])
        engine = CTATrendEngine(db_path=db)
        data = engine._fetch_data("FAKE", 50)
        assert data == []

    def test_fetch_missing_db(self, tmp_path):
        engine = CTATrendEngine(db_path=tmp_path / "nonexistent.db")
        data = engine._fetch_data("SPY", 50)
        assert data == []

    def test_fetch_limit(self, tmp_path):
        db = tmp_path / "test.db"
        _setup_test_db(db, ["SPY"], days=300)
        engine = CTATrendEngine(db_path=db)
        data = engine._fetch_data("SPY", 100)
        assert len(data) <= 100


# ---------------------------------------------------------------------------
# CTATrendEngine — _calculate_sma
# ---------------------------------------------------------------------------

class TestCalculateSMA:

    def test_sma_basic(self):
        engine = CTATrendEngine()
        prices = np.array([100, 101, 102, 103, 104])
        sma = engine._calculate_sma(prices, 3)
        assert sma == pytest.approx(np.mean([102, 103, 104]))

    def test_sma_short_data(self):
        engine = CTATrendEngine()
        prices = np.array([100, 101])
        sma = engine._calculate_sma(prices, 10)
        assert sma == pytest.approx(101)  # Returns last price

    def test_sma_empty(self):
        engine = CTATrendEngine()
        prices = np.array([])
        sma = engine._calculate_sma(prices, 10)
        assert sma == 0

    def test_sma_exact_length(self):
        engine = CTATrendEngine()
        prices = np.array([100, 200, 300])
        sma = engine._calculate_sma(prices, 3)
        assert sma == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# CTATrendEngine — _calculate_volatility
# ---------------------------------------------------------------------------

class TestCalculateVolatility:

    def test_vol_basic(self):
        engine = CTATrendEngine()
        prices = _make_prices(100, vol=0.02, seed=42)
        vol = engine._calculate_volatility(prices, 20)
        assert 0.05 <= vol <= 1.0  # Floor at 5%

    def test_vol_floor(self):
        engine = CTATrendEngine()
        # Constant prices → near-zero vol, but floor is 5%
        prices = np.array([100.0] * 30)
        vol = engine._calculate_volatility(prices, 20)
        assert vol == pytest.approx(0.05)

    def test_vol_short_data(self):
        engine = CTATrendEngine()
        prices = np.array([100, 101])
        vol = engine._calculate_volatility(prices, 20)
        assert vol == 0.15  # Default when insufficient data

    def test_vol_annualization(self):
        engine = CTATrendEngine()
        rng = np.random.RandomState(99)
        daily_vol = 0.01
        prices = 100 * np.cumprod(1 + rng.normal(0, daily_vol, 100))
        vol = engine._calculate_volatility(prices, 50)
        expected_annual = daily_vol * np.sqrt(252)
        assert vol == pytest.approx(max(expected_annual, 0.05), rel=0.2)


# ---------------------------------------------------------------------------
# CTATrendEngine — _calculate_trend_signal
# ---------------------------------------------------------------------------

class TestCalculateTrendSignal:

    def test_returns_trend_signal(self):
        engine = CTATrendEngine()
        prices = _make_prices(200, drift=0.002, seed=10)
        sig = engine._calculate_trend_signal("SPY", "medium", prices)
        assert isinstance(sig, TrendSignal)
        assert sig.timeframe == 60

    def test_uptrend_detection(self):
        engine = CTATrendEngine()
        # Strong uptrend: steady positive drift
        prices = _make_prices(200, start=100, drift=0.005, vol=0.005, seed=11)
        sig = engine._calculate_trend_signal("SPY", "medium", prices)
        assert sig is not None
        assert sig.score > 0

    def test_downtrend_detection(self):
        engine = CTATrendEngine()
        # Strong downtrend: steady negative drift
        prices = _make_prices(200, start=500, drift=-0.005, vol=0.005, seed=12)
        sig = engine._calculate_trend_signal("SPY", "medium", prices)
        assert sig is not None
        assert sig.score < 0

    def test_short_data_returns_none(self):
        engine = CTATrendEngine()
        prices = np.array([100, 101, 102])
        sig = engine._calculate_trend_signal("SPY", "long", prices)
        assert sig is None

    def test_score_bounded(self):
        engine = CTATrendEngine()
        prices = _make_prices(200, drift=0.01, vol=0.003, seed=13)
        sig = engine._calculate_trend_signal("SPY", "short", prices)
        assert sig is not None
        assert -1 <= sig.score <= 1

    def test_strength_bounded(self):
        engine = CTATrendEngine()
        prices = _make_prices(200, seed=14)
        sig = engine._calculate_trend_signal("SPY", "medium", prices)
        assert sig is not None
        assert 0 <= sig.strength <= 1

    def test_regime_classification(self):
        engine = CTATrendEngine()
        # Flat-ish prices should give "chop"
        rng = np.random.RandomState(15)
        prices = 100 + rng.normal(0, 0.5, 200).cumsum()
        prices = np.maximum(prices, 50)  # Keep positive
        sig = engine._calculate_trend_signal("SPY", "medium", prices)
        assert sig is not None
        assert sig.regime in ("uptrend", "downtrend", "chop")

    def test_zero_std_handled(self):
        engine = CTATrendEngine()
        # Flat prices → price_std = 0
        prices = np.array([100.0] * 200)
        sig = engine._calculate_trend_signal("SPY", "medium", prices)
        # Should not crash; score should be ~0
        assert sig is not None
        assert abs(sig.score) < 0.01


# ---------------------------------------------------------------------------
# CTATrendEngine — _ensemble_trend_score
# ---------------------------------------------------------------------------

class TestEnsembleTrendScore:

    def test_empty_signals(self):
        engine = CTATrendEngine()
        score, strength, regime = engine._ensemble_trend_score([])
        assert score == 0.0
        assert strength == 0.0
        assert regime == "neutral"

    def test_single_signal(self):
        engine = CTATrendEngine()
        sig = _make_trend_signal(timeframe=60, score=0.5, strength=0.8, regime="uptrend")
        score, strength, regime = engine._ensemble_trend_score([sig])
        assert abs(score) > 0
        assert strength == 0.8

    def test_agreement_boosts_strength(self):
        engine = CTATrendEngine()
        signals = [
            _make_trend_signal(timeframe=20, score=0.5, strength=0.7, regime="uptrend"),
            _make_trend_signal(timeframe=60, score=0.6, strength=0.7, regime="uptrend"),
            _make_trend_signal(timeframe=120, score=0.4, strength=0.7, regime="uptrend"),
        ]
        score, strength, regime = engine._ensemble_trend_score(signals)
        assert regime == "uptrend"
        assert strength > 0

    def test_disagreement_reduces_strength(self):
        engine = CTATrendEngine()
        signals = [
            _make_trend_signal(timeframe=20, score=0.5, strength=0.7, regime="uptrend"),
            _make_trend_signal(timeframe=60, score=-0.5, strength=0.7, regime="downtrend"),
            _make_trend_signal(timeframe=120, score=0.3, strength=0.7, regime="uptrend"),
        ]
        score, strength, regime = engine._ensemble_trend_score(signals)
        assert strength < 0.7  # Lower than full agreement

    def test_mixed_regime(self):
        engine = CTATrendEngine()
        signals = [
            _make_trend_signal(timeframe=20, score=0.1, strength=0.3, regime="chop"),
            _make_trend_signal(timeframe=60, score=-0.1, strength=0.3, regime="chop"),
        ]
        score, strength, regime = engine._ensemble_trend_score(signals)
        assert regime == "mixed"

    def test_downtrend_consensus(self):
        engine = CTATrendEngine()
        signals = [
            _make_trend_signal(timeframe=20, score=-0.5, strength=0.6, regime="downtrend"),
            _make_trend_signal(timeframe=60, score=-0.4, strength=0.6, regime="downtrend"),
        ]
        score, strength, regime = engine._ensemble_trend_score(signals)
        assert regime == "downtrend"
        assert score < 0


# ---------------------------------------------------------------------------
# CTATrendEngine — _calculate_position_scalar
# ---------------------------------------------------------------------------

class TestCalculatePositionScalar:

    def test_vol_targeting(self):
        engine = CTATrendEngine()
        scalar = engine._calculate_position_scalar(0.20, 0.5)
        # target_vol / realized_vol = 0.10 / 0.20 = 0.5
        # conviction = 0.5 + 0.5 * 0.5 = 0.75
        # combined = 0.5 * 0.75 = 0.375
        assert scalar == pytest.approx(0.375, abs=0.01)

    def test_low_vol_increases_leverage(self):
        engine = CTATrendEngine()
        scalar = engine._calculate_position_scalar(0.05, 0.8)
        # vol_scalar = 0.10 / 0.05 = 2.0
        # conviction = 0.5 + 0.8 * 0.5 = 0.9
        # combined = 2.0 * 0.9 = 1.8 → within bounds
        assert scalar > 1.0

    def test_high_vol_reduces_leverage(self):
        engine = CTATrendEngine()
        scalar = engine._calculate_position_scalar(0.40, 0.5)
        # vol_scalar = 0.10 / 0.40 = 0.25
        assert scalar < 1.0

    def test_max_leverage_cap(self):
        engine = CTATrendEngine()
        scalar = engine._calculate_position_scalar(0.02, 1.0)
        assert scalar == engine.MAX_LEVERAGE

    def test_min_leverage_floor(self):
        engine = CTATrendEngine()
        scalar = engine._calculate_position_scalar(1.0, 0.0)
        assert scalar == engine.MIN_LEVERAGE

    def test_zero_vol_returns_one(self):
        engine = CTATrendEngine()
        scalar = engine._calculate_position_scalar(0, 0.5)
        assert scalar == 1.0

    def test_negative_vol_returns_one(self):
        engine = CTATrendEngine()
        scalar = engine._calculate_position_scalar(-0.05, 0.5)
        assert scalar == 1.0


# ---------------------------------------------------------------------------
# CTATrendEngine — analyze_symbol
# ---------------------------------------------------------------------------

class TestAnalyzeSymbol:

    def test_unknown_symbol_returns_none(self, tmp_path):
        db = tmp_path / "test.db"
        _setup_test_db(db)
        engine = CTATrendEngine(db_path=db)
        assert engine.analyze_symbol("UNKNOWN") is None

    def test_insufficient_data_returns_none(self, tmp_path):
        db = tmp_path / "test.db"
        _setup_test_db(db, ["SPY"], days=50)
        engine = CTATrendEngine(db_path=db)
        assert engine.analyze_symbol("SPY") is None

    def test_valid_symbol_returns_position(self, tmp_path):
        db = tmp_path / "test.db"
        _setup_test_db(db, ["SPY"])
        engine = CTATrendEngine(db_path=db)
        pos = engine.analyze_symbol("SPY")
        assert isinstance(pos, CTAPosition)
        assert pos.symbol == "SPY"
        assert pos.asset_class == "equity"

    def test_position_fields_populated(self, tmp_path):
        db = tmp_path / "test.db"
        _setup_test_db(db, ["GLD"])
        engine = CTATrendEngine(db_path=db)
        pos = engine.analyze_symbol("GLD")
        assert pos is not None
        assert pos.base_weight == CTATrendEngine.UNIVERSE["GLD"]["base_weight"]
        assert pos.target_vol == CTATrendEngine.TARGET_VOL
        assert -1 <= pos.trend_score <= 1
        assert 0 <= pos.trend_strength <= 1
        assert pos.realized_vol > 0

    def test_signal_classification(self, tmp_path):
        db = tmp_path / "test.db"
        _setup_test_db(db, ["SPY"])
        engine = CTATrendEngine(db_path=db)
        pos = engine.analyze_symbol("SPY")
        assert pos.signal in ("long", "short", "neutral")

    def test_final_weight_capped(self, tmp_path):
        db = tmp_path / "test.db"
        _setup_test_db(db, ["SPY"])
        engine = CTATrendEngine(db_path=db)
        pos = engine.analyze_symbol("SPY")
        assert pos.final_weight <= engine.MAX_POSITION_RISK


# ---------------------------------------------------------------------------
# CTATrendEngine — evaluate
# ---------------------------------------------------------------------------

class TestEvaluate:

    def test_evaluate_structure(self, tmp_path):
        db = tmp_path / "test.db"
        _setup_test_db(db)
        engine = CTATrendEngine(db_path=db)
        result = engine.evaluate()
        assert "timestamp" in result
        assert "positions" in result
        assert "allocation" in result
        assert "summary" in result

    def test_evaluate_with_positions(self, tmp_path):
        db = tmp_path / "test.db"
        _setup_test_db(db)
        engine = CTATrendEngine(db_path=db)
        result = engine.evaluate()
        assert len(result["positions"]) > 0
        assert len(result["allocation"]) > 0

    def test_allocation_sums_to_one(self, tmp_path):
        db = tmp_path / "test.db"
        _setup_test_db(db)
        engine = CTATrendEngine(db_path=db)
        result = engine.evaluate()
        total = sum(result["allocation"].values())
        if total > 0:
            assert total == pytest.approx(1.0, abs=0.01)

    def test_summary_fields(self, tmp_path):
        db = tmp_path / "test.db"
        _setup_test_db(db)
        engine = CTATrendEngine(db_path=db)
        result = engine.evaluate()
        summary = result["summary"]
        assert "total_positions" in summary
        assert "signal_counts" in summary
        assert "avg_trend_score" in summary
        assert "avg_trend_strength" in summary
        assert "avg_realized_vol" in summary
        assert "asset_class_distribution" in summary

    def test_signal_counts(self, tmp_path):
        db = tmp_path / "test.db"
        _setup_test_db(db)
        engine = CTATrendEngine(db_path=db)
        result = engine.evaluate()
        counts = result["summary"]["signal_counts"]
        assert "long" in counts
        assert "short" in counts
        assert "neutral" in counts

    def test_evaluate_empty_db(self, tmp_path):
        db = tmp_path / "empty.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE prices (symbol TEXT, date TEXT, close REAL, volume INTEGER)")
        conn.commit()
        conn.close()
        engine = CTATrendEngine(db_path=db)
        result = engine.evaluate()
        assert result.get("error") is not None or len(result["positions"]) == 0


# ---------------------------------------------------------------------------
# CTATrendEngine — get_crisis_alpha_signals
# ---------------------------------------------------------------------------

class TestCrisisAlpha:

    def test_returns_dict(self, tmp_path):
        db = tmp_path / "test.db"
        _setup_test_db(db)
        engine = CTATrendEngine(db_path=db)
        signals = engine.get_crisis_alpha_signals()
        assert isinstance(signals, dict)

    def test_insufficient_spy_data(self, tmp_path):
        db = tmp_path / "test.db"
        _setup_test_db(db, ["SPY"], days=20)
        engine = CTATrendEngine(db_path=db)
        signals = engine.get_crisis_alpha_signals()
        assert signals == {}

    def test_missing_db(self, tmp_path):
        engine = CTATrendEngine(db_path=tmp_path / "nonexistent.db")
        signals = engine.get_crisis_alpha_signals()
        assert signals == {}
