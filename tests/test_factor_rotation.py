#!/usr/bin/env python3
"""
Tests for factor rotation engine — momentum scoring, allocation, signal strength,
category diversity, recommendation generation.
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

from src.strategy.factor_rotation import (
    FactorScore, FactorMomentumEngine, FactorRotationBacktest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_market_db(db_path: Path, symbols=None, days=600, base_price=100.0):
    """Create a market.db with synthetic price data for testing."""
    if symbols is None:
        symbols = ["MTUM", "VLUE", "USMV", "QUAL", "SPY"]

    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            symbol TEXT, date TEXT, close REAL, volume INTEGER,
            PRIMARY KEY (symbol, date)
        )
    """)

    np.random.seed(42)
    dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
             for i in range(days)]

    for sym in symbols:
        # Different drift per symbol so rankings are deterministic
        drift = {"MTUM": 0.0005, "VLUE": 0.0003, "USMV": 0.0002,
                 "QUAL": 0.0004, "SPY": 0.0003, "VTV": 0.00025,
                 "SPLV": 0.00015, "IJR": 0.00035, "VBR": 0.0003,
                 "QQQ": 0.00045, "SPHQ": 0.00035}.get(sym, 0.0003)
        price = base_price
        for i, date in enumerate(dates):
            ret = np.random.normal(drift, 0.015)
            price *= (1 + ret)
            conn.execute(
                "INSERT OR REPLACE INTO prices VALUES (?, ?, ?, ?)",
                (sym, date, round(price, 2), 1000000)
            )

    conn.commit()
    conn.close()


def _make_engine(tmp_path, symbols=None, top_n=2, min_momentum=0.0):
    """Create a FactorMomentumEngine with a test DB."""
    db_path = tmp_path / "market.db"
    _create_market_db(db_path, symbols=symbols)
    return FactorMomentumEngine(
        db_path=db_path,
        lookback_months=12,
        top_n=top_n,
        min_momentum=min_momentum,
    )


# ---------------------------------------------------------------------------
# FactorScore tests
# ---------------------------------------------------------------------------

class TestFactorScore:
    """Test FactorScore dataclass."""

    def test_creation(self):
        score = FactorScore(
            symbol="MTUM", factor_name="Momentum", price=150.0,
            return_12m=0.25, return_6m=0.15, return_3m=0.08,
            volatility=0.18, sharpe_12m=1.39, momentum_score=0.12,
            rank=1,
        )
        assert score.symbol == "MTUM"
        assert score.return_12m == 0.25
        assert score.rank == 1

    def test_ml_defaults(self):
        score = FactorScore(
            symbol="VLUE", factor_name="Value", price=100.0,
            return_12m=0.10, return_6m=0.05, return_3m=0.02,
            volatility=0.15, sharpe_12m=0.67, momentum_score=0.05,
            rank=2,
        )
        assert score.value_momentum_synergy == 0.0
        assert score.tsfm_score == 0.0
        assert score.tsfm_allocation_scalar == 1.0


# ---------------------------------------------------------------------------
# FactorMomentumEngine tests
# ---------------------------------------------------------------------------

class TestFactorMomentumEngine:
    """Test the core engine."""

    def test_initialization(self, tmp_path):
        """Engine initializes with correct defaults."""
        engine = _make_engine(tmp_path)
        assert engine.top_n == 2
        assert engine.min_momentum == 0.0
        assert engine.lookback_months == 12
        assert len(engine.universe) > 0

    def test_factors_defined(self):
        """All expected factor ETFs are defined."""
        assert "MTUM" in FactorMomentumEngine.FACTORS
        assert "VLUE" in FactorMomentumEngine.FACTORS
        assert "USMV" in FactorMomentumEngine.FACTORS
        assert "QUAL" in FactorMomentumEngine.FACTORS
        assert "SPY" in FactorMomentumEngine.FACTORS

    def test_factor_categories(self):
        """Factor categories are correctly assigned."""
        assert FactorMomentumEngine.FACTORS["MTUM"]["category"] == "momentum"
        assert FactorMomentumEngine.FACTORS["VLUE"]["category"] == "value"
        assert FactorMomentumEngine.FACTORS["USMV"]["category"] == "low_vol"
        assert FactorMomentumEngine.FACTORS["SPY"]["category"] == "core"


class TestFetchPriceData:
    """Test _fetch_price_data."""

    def test_returns_data(self, tmp_path):
        """Returns price data for existing symbol."""
        engine = _make_engine(tmp_path, symbols=["MTUM"])
        data = engine._fetch_price_data("MTUM", days=100)
        assert len(data) > 0
        assert "close" in data[0]
        assert "date" in data[0]

    def test_empty_for_missing_symbol(self, tmp_path):
        """Returns empty list for symbol not in DB."""
        engine = _make_engine(tmp_path, symbols=["MTUM"])
        data = engine._fetch_price_data("NONEXISTENT", days=100)
        assert data == []

    def test_empty_for_missing_db(self, tmp_path):
        """Returns empty list when DB doesn't exist."""
        engine = FactorMomentumEngine(db_path=tmp_path / "nonexistent.db")
        data = engine._fetch_price_data("MTUM", days=100)
        assert data == []


class TestCalculateFactorScore:
    """Test _calculate_factor_score."""

    def test_returns_factor_score(self, tmp_path):
        """Returns FactorScore for symbol with enough data."""
        engine = _make_engine(tmp_path, symbols=["MTUM"])
        score = engine._calculate_factor_score("MTUM")
        assert score is not None
        assert isinstance(score, FactorScore)
        assert score.symbol == "MTUM"
        assert score.volatility > 0

    def test_returns_none_insufficient_data(self, tmp_path):
        """Returns None when less than 252 days of data."""
        db_path = tmp_path / "market.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE prices (symbol TEXT, date TEXT, close REAL, volume INTEGER,
            PRIMARY KEY (symbol, date))
        """)
        # Only 10 days of data
        for i in range(10):
            conn.execute("INSERT INTO prices VALUES (?, ?, ?, ?)",
                         ("MTUM", f"2026-01-{i+1:02d}", 100.0 + i, 1000))
        conn.commit()
        conn.close()

        engine = FactorMomentumEngine(db_path=db_path)
        score = engine._calculate_factor_score("MTUM")
        assert score is None

    def test_momentum_score_is_float(self, tmp_path):
        """Momentum score is a finite float."""
        engine = _make_engine(tmp_path, symbols=["MTUM"])
        score = engine._calculate_factor_score("MTUM")
        assert score is not None
        assert np.isfinite(score.momentum_score)

    def test_tsfm_score_capped(self, tmp_path):
        """TSFM score is capped at ±2."""
        engine = _make_engine(tmp_path, symbols=["MTUM"])
        score = engine._calculate_factor_score("MTUM")
        assert score is not None
        assert -2.0 <= score.tsfm_score <= 2.0


class TestEvaluate:
    """Test evaluate method."""

    def test_evaluate_returns_dict(self, tmp_path):
        """evaluate() returns a valid result dict."""
        engine = _make_engine(tmp_path)
        result = engine.evaluate()
        assert "timestamp" in result
        assert "selected_factors" in result
        assert "allocation" in result
        assert "signal_strength" in result
        assert "recommendation" in result

    def test_evaluate_selects_top_factors(self, tmp_path):
        """evaluate() selects top_n factors."""
        engine = _make_engine(tmp_path, top_n=2)
        result = engine.evaluate()
        assert len(result["selected_factors"]) <= 2

    def test_evaluate_allocation_sums_to_one(self, tmp_path):
        """Allocation weights sum to ~1.0."""
        engine = _make_engine(tmp_path)
        result = engine.evaluate()
        if result["allocation"]:
            total = sum(result["allocation"].values())
            assert abs(total - 1.0) < 0.01

    def test_evaluate_no_data_returns_error(self, tmp_path):
        """No data → error result."""
        engine = FactorMomentumEngine(db_path=tmp_path / "nonexistent.db")
        result = engine.evaluate()
        assert "error" in result
        assert result["selected_factors"] == []

    def test_category_diversity(self, tmp_path):
        """No more than max_per_category from same category."""
        engine = _make_engine(tmp_path, top_n=3)
        result = engine.evaluate()
        if len(result["selected_factors"]) >= 2:
            categories = [
                engine.FACTORS[s]["category"]
                for s in result["selected_factors"]
            ]
            from collections import Counter
            counts = Counter(categories)
            for cat, count in counts.items():
                assert count <= engine.max_per_category


class TestGenerateAllocation:
    """Test _generate_allocation."""

    def test_empty_returns_spy(self, tmp_path):
        """Empty selection → SPY 100%."""
        engine = _make_engine(tmp_path)
        alloc = engine._generate_allocation([])
        assert alloc == {"SPY": 1.0}

    def test_inverse_volatility_weighting(self, tmp_path):
        """Lower volatility → higher weight."""
        engine = _make_engine(tmp_path)
        low_vol = FactorScore(
            symbol="USMV", factor_name="Low Vol", price=100.0,
            return_12m=0.10, return_6m=0.05, return_3m=0.02,
            volatility=0.10, sharpe_12m=1.0, momentum_score=0.05,
            rank=1,
        )
        high_vol = FactorScore(
            symbol="MTUM", factor_name="Momentum", price=100.0,
            return_12m=0.20, return_6m=0.10, return_3m=0.05,
            volatility=0.25, sharpe_12m=0.8, momentum_score=0.10,
            rank=2,
        )
        alloc = engine._generate_allocation([("USMV", low_vol), ("MTUM", high_vol)])
        # USMV has lower vol → higher weight
        assert alloc["USMV"] > alloc["MTUM"]


class TestSignalStrength:
    """Test _calculate_signal_strength."""

    def test_empty_returns_zero(self, tmp_path):
        """Empty selection → 0.0 strength."""
        engine = _make_engine(tmp_path)
        assert engine._calculate_signal_strength([]) == 0.0

    def test_strong_momentum_returns_high(self, tmp_path):
        """All positive momentum, low vol → high strength."""
        engine = _make_engine(tmp_path)
        selected = [
            ("MTUM", FactorScore(
                symbol="MTUM", factor_name="Momentum", price=100.0,
                return_12m=0.25, return_6m=0.15, return_3m=0.08,
                volatility=0.15, sharpe_12m=1.67, momentum_score=0.15,
                rank=1,
            )),
            ("QUAL", FactorScore(
                symbol="QUAL", factor_name="Quality", price=100.0,
                return_12m=0.18, return_6m=0.10, return_3m=0.05,
                volatility=0.12, sharpe_12m=1.5, momentum_score=0.10,
                rank=2,
            )),
        ]
        strength = engine._calculate_signal_strength(selected)
        assert strength > 0.5

    def test_negative_momentum_returns_low(self, tmp_path):
        """Negative momentum → lower strength."""
        engine = _make_engine(tmp_path)
        selected = [
            ("MTUM", FactorScore(
                symbol="MTUM", factor_name="Momentum", price=100.0,
                return_12m=-0.10, return_6m=-0.05, return_3m=-0.02,
                volatility=0.20, sharpe_12m=-0.5, momentum_score=-0.05,
                rank=1,
            )),
        ]
        strength = engine._calculate_signal_strength(selected)
        assert strength < 0.5


class TestGenerateRecommendation:
    """Test _generate_recommendation."""

    def test_empty_selection(self, tmp_path):
        """No factors → hold SPY message."""
        engine = _make_engine(tmp_path)
        rec = engine._generate_recommendation([], {})
        assert "SPY" in rec

    def test_strong_momentum(self, tmp_path):
        """High 12m return → 'strong momentum'."""
        engine = _make_engine(tmp_path)
        selected = [
            ("MTUM", FactorScore(
                symbol="MTUM", factor_name="Momentum", price=100.0,
                return_12m=0.25, return_6m=0.15, return_3m=0.08,
                volatility=0.15, sharpe_12m=1.67, momentum_score=0.15,
                rank=1,
            )),
            ("QUAL", FactorScore(
                symbol="QUAL", factor_name="Quality", price=100.0,
                return_12m=0.22, return_6m=0.12, return_3m=0.06,
                volatility=0.12, sharpe_12m=1.83, momentum_score=0.12,
                rank=2,
            )),
        ]
        rec = engine._generate_recommendation(selected, {})
        assert "strong" in rec

    def test_weak_momentum(self, tmp_path):
        """Low 12m return → 'weak momentum'."""
        engine = _make_engine(tmp_path)
        selected = [
            ("MTUM", FactorScore(
                symbol="MTUM", factor_name="Momentum", price=100.0,
                return_12m=0.05, return_6m=0.02, return_3m=0.01,
                volatility=0.15, sharpe_12m=0.33, momentum_score=0.03,
                rank=1,
            )),
            ("QUAL", FactorScore(
                symbol="QUAL", factor_name="Quality", price=100.0,
                return_12m=0.08, return_6m=0.04, return_3m=0.02,
                volatility=0.12, sharpe_12m=0.67, momentum_score=0.04,
                rank=2,
            )),
        ]
        rec = engine._generate_recommendation(selected, {})
        assert "weak" in rec


class TestFactorRotationBacktest:
    """Test backtest runner."""

    def test_run_backtest_returns_metrics(self, tmp_path):
        """Backtest returns CAGR, Sharpe, max DD."""
        engine = _make_engine(tmp_path)
        backtest = FactorRotationBacktest(engine)
        # Need 252+ trading days in range — use 2 years
        result = backtest.run_backtest("2024-01-01", "2026-01-01")

        assert "cagr" in result
        assert "sharpe_ratio" in result
        assert "max_drawdown" in result
        assert "trading_days" in result
        assert isinstance(result["cagr"], float)
        assert isinstance(result["sharpe_ratio"], float)

    def test_backtest_with_no_data(self, tmp_path):
        """Backtest with empty DB returns error or zero metrics."""
        engine = FactorMomentumEngine(db_path=tmp_path / "nonexistent.db")
        backtest = FactorRotationBacktest(engine)
        result = backtest.run_backtest("2025-01-01", "2025-06-01")
        # Should not crash, may return error or zero values
        assert isinstance(result, dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
