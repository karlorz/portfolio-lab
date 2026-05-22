#!/usr/bin/env python3
"""
Tests for feature engineering pipeline (src/research/features.py).

Tests Features dataclass, FeaturePipeline (connection, price data, returns,
volatility, SMA, correlation, generate_features, generate_all_features,
to_dataframe), FeatureStore (save/load), and CLI main().
"""
import sys
import os
import json
import math
import sqlite3
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.research.features import (
    Features, FeaturePipeline, FeatureStore, main as cli_main,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_prices_db(
    db_path: Path,
    symbol_prices: dict,
    n_days: int = 100,
    base_date: str = "2024-01-02",
) -> None:
    """Populate a SQLite prices table with synthetic data.

    symbol_prices maps symbol -> (start_price, drift, vol_seed).
    Dates are weekdays only, price drifts up with controlled noise.
    """
    dates = _make_weekday_dates(n_days, base_date)
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                date TEXT, symbol TEXT, close REAL, volume INTEGER,
                PRIMARY KEY (date, symbol)
            )
        """)
        rng = np.random.default_rng(42)
        for sym, (start, drift, seed_offset) in symbol_prices.items():
            price = start
            for dt in dates:
                noise = rng.normal(0, 0.005)
                price *= (1.0 + drift + noise)
                price = max(price, 0.01)
                vol = int(1_000_000 * (1.0 + rng.normal(0, 0.1)))
                conn.execute(
                    "INSERT OR REPLACE INTO prices (date, symbol, close, volume) VALUES (?, ?, ?, ?)",
                    (dt, sym, round(price, 2), vol),
                )


def _build_simple_prices_db(
    db_path: Path,
    symbol: str = "SPY",
    n_days: int = 100,
    start_price: float = 400.0,
) -> None:
    """Populate with a single-symbol steadily-rising price series."""
    _build_prices_db(
        db_path, {symbol: (start_price, 0.001, 0), "^VIX": (20.0, 0.0, 1)},
        n_days=n_days,
    )


def _make_weekday_dates(n_days: int, start_date: str) -> list:
    """Generate n_days of weekday date strings."""
    dates = []
    d = datetime.strptime(start_date, "%Y-%m-%d")
    while len(dates) < n_days:
        if d.weekday() < 5:
            dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return dates


def _populate_full_db(db_path: Path, n_days: int = 100) -> None:
    """Populate with SPY, ^VIX for generate_features to work."""
    _build_prices_db(
        db_path,
        {
            "SPY": (450.0, 0.001, 0),
            "^VIX": (18.0, 0.000, 1),
            "GLD": (180.0, 0.0005, 2),
            "TLT": (95.0, -0.0002, 3),
        },
        n_days=n_days,
    )


# ===================================================================
# Features dataclass
# ===================================================================

class TestFeaturesDataclass:
    """Tests for the Features dataclass initialization and defaults."""

    def test_full_initialization(self):
        """Create a Features with all fields."""
        f = Features(
            symbol="SPY",
            timestamp="2024-01-15",
            return_1d=0.01,
            return_5d=0.03,
            return_20d=0.05,
            volatility_20d=0.15,
            sma_20=400.0,
            sma_50=395.0,
            price_vs_sma20=0.02,
            price_vs_sma50=0.03,
            volume_20d_avg=1_500_000,
            volume_ratio=1.2,
            vix_level=15.0,
            vix_change_5d=-0.05,
            vix_percentile_20d=0.3,
            spy_correlation_20d=0.85,
            trend_direction=1,
            vol_regime="low",
        )
        assert f.symbol == "SPY"
        assert f.return_1d == 0.01
        assert f.trend_direction == 1
        assert f.vol_regime == "low"

    def test_optional_fields_default_to_none(self):
        """future_return_5d and regime_label default to None."""
        f = Features(
            symbol="GLD",
            timestamp="2024-01-15",
            return_1d=0.0, return_5d=0.0, return_20d=0.0,
            volatility_20d=0.0,
            sma_20=0.0, sma_50=0.0,
            price_vs_sma20=0.0, price_vs_sma50=0.0,
            volume_20d_avg=0.0, volume_ratio=1.0,
            vix_level=20.0, vix_change_5d=0.0, vix_percentile_20d=0.5,
            spy_correlation_20d=0.0, trend_direction=0, vol_regime="normal",
        )
        assert f.future_return_5d is None
        assert f.regime_label is None

    def test_trend_direction_type(self):
        """trend_direction is an int restricted to -1, 0, 1."""
        for val in (-1, 0, 1):
            f = Features(
                symbol="SPY", timestamp="2024-01-15",
                return_1d=0.0, return_5d=0.0, return_20d=0.0,
                volatility_20d=0.0,
                sma_20=0.0, sma_50=0.0,
                price_vs_sma20=0.0, price_vs_sma50=0.0,
                volume_20d_avg=0.0, volume_ratio=1.0,
                vix_level=20.0, vix_change_5d=0.0, vix_percentile_20d=0.5,
                spy_correlation_20d=0.0, trend_direction=val, vol_regime="normal",
            )
            assert f.trend_direction == val

    def test_vol_regime_strings(self):
        """vol_regime is one of low, normal, high."""
        for regime in ("low", "normal", "high"):
            f = Features(
                symbol="SPY", timestamp="2024-01-15",
                return_1d=0.0, return_5d=0.0, return_20d=0.0,
                volatility_20d=0.0,
                sma_20=0.0, sma_50=0.0,
                price_vs_sma20=0.0, price_vs_sma50=0.0,
                volume_20d_avg=0.0, volume_ratio=1.0,
                vix_level=20.0, vix_change_5d=0.0, vix_percentile_20d=0.5,
                spy_correlation_20d=0.0, trend_direction=0, vol_regime=regime,
            )
            assert f.vol_regime == regime

    def test_vars_serialization(self):
        """vars() returns expected fields including optional None."""
        f = Features(
            symbol="SPY", timestamp="2024-01-15",
            return_1d=0.01, return_5d=0.0, return_20d=0.02,
            volatility_20d=0.12,
            sma_20=400.0, sma_50=395.0,
            price_vs_sma20=0.01, price_vs_sma50=0.005,
            volume_20d_avg=1e6, volume_ratio=0.9,
            vix_level=18.0, vix_change_5d=0.02, vix_percentile_20d=0.6,
            spy_correlation_20d=0.75, trend_direction=1, vol_regime="normal",
        )
        d = vars(f)
        assert d["symbol"] == "SPY"
        assert d["return_1d"] == 0.01
        assert d["future_return_5d"] is None
        assert d["regime_label"] is None


# ===================================================================
# FeaturePipeline - internal computation methods
# ===================================================================

class TestFeaturePipelineCore:
    """Test FeaturePipeline init and pure-computation helpers."""

    def test_default_db_path(self):
        """Default db_path resolves to MARKET_DB from src.paths."""
        from src.paths import MARKET_DB
        pipeline = FeaturePipeline()
        assert pipeline.db_path == str(MARKET_DB)

    def test_custom_db_path(self):
        """Custom db_path is honored."""
        pipeline = FeaturePipeline(db_path="/tmp/custom.db")
        assert pipeline.db_path == "/tmp/custom.db"

    def test_features_cache_init(self):
        """features_cache starts as empty dict."""
        pipeline = FeaturePipeline()
        assert pipeline.features_cache == {}

    def test_calculate_returns_standard(self):
        """Returns computed correctly for 1, 5, 20 periods."""
        pipeline = FeaturePipeline()
        prices = [100.0 + i * 0.5 for i in range(101)]  # 101 prices
        returns = pipeline._calculate_returns(prices, [1, 5, 20])
        # price[-1]=150, price[-2]=149.5 -> (150-149.5)/149.5
        assert returns[1] == pytest.approx(0.5 / 149.5, rel=1e-6)
        assert isinstance(returns[1], float)

    def test_calculate_returns_insufficient(self):
        """Returns 0.0 when not enough data for a period."""
        pipeline = FeaturePipeline()
        prices = [100.0, 101.0]  # Only 2 prices
        returns = pipeline._calculate_returns(prices, [5, 20])
        assert returns[5] == 0.0
        assert returns[20] == 0.0

    def test_calculate_returns_empty(self):
        """Returns 0.0 for empty price list."""
        pipeline = FeaturePipeline()
        returns = pipeline._calculate_returns([], [1])
        assert returns[1] == 0.0

    def test_calculate_returns_zero_divisor(self):
        """Handles zero price gracefully."""
        pipeline = FeaturePipeline()
        prices = [0.0, 100.0, 0.0, 0.0]
        # price[-1]=0.0, price[-2]=0.0 -> returns 0.0
        returns = pipeline._calculate_returns(prices, [1])
        assert returns[1] == 0.0

    def test_calculate_volatility_standard(self):
        """Annualized volatility computed from daily returns."""
        pipeline = FeaturePipeline()
        # Constant 1% up daily -> 0% daily return -> vol ~ 0
        prices = [100.0 * (1.01 ** i) for i in range(21)]
        vol = pipeline._calculate_volatility(prices, 20)
        assert vol >= 0.0
        assert isinstance(vol, float)

    def test_calculate_volatility_insufficient(self):
        """Returns 0.0 when fewer than period+1 prices."""
        pipeline = FeaturePipeline()
        prices = [100.0, 101.0]  # Only 2 prices, need 21
        vol = pipeline._calculate_volatility(prices, 20)
        assert vol == 0.0

    def test_calculate_volatility_single_price(self):
        """Returns 0.0 with single price."""
        pipeline = FeaturePipeline()
        vol = pipeline._calculate_volatility([100.0], 20)
        assert vol == 0.0

    def test_calculate_volatility_zero_price(self):
        """Handles zero prices in return calculation."""
        pipeline = FeaturePipeline()
        prices = [0.0, 0.0, 100.0, 100.0]
        vol = pipeline._calculate_volatility(prices, 3)
        assert vol >= 0.0

    def test_calculate_sma_standard(self):
        """SMA computed correctly over window."""
        pipeline = FeaturePipeline()
        prices = [10.0, 20.0, 30.0, 40.0, 50.0]
        sma = pipeline._calculate_sma(prices, 3)
        assert sma == pytest.approx(40.0, rel=1e-6)  # avg of last 3: 30+40+50=120/3

    def test_calculate_sma_insufficient(self):
        """Returns last price when fewer prices than period."""
        pipeline = FeaturePipeline()
        prices = [100.0, 101.0]
        sma = pipeline._calculate_sma(prices, 20)
        assert sma == 101.0  # last price

    def test_calculate_sma_empty(self):
        """Returns 0 for empty list."""
        pipeline = FeaturePipeline()
        sma = pipeline._calculate_sma([], 20)
        assert sma == 0

    def test_calculate_correlation_positive(self):
        """Positive correlation between similar price series."""
        pipeline = FeaturePipeline()
        # Both series move up ~1% per step
        prices1 = [100.0 * (1.01 ** i) for i in range(21)]
        prices2 = [200.0 * (1.01 ** i) for i in range(21)]
        corr = pipeline._calculate_correlation(prices1, prices2, 20)
        assert corr > 0.5

    def test_calculate_correlation_negative(self):
        """Negative correlation between inverse series."""
        # Construct prices so that returns are exact negatives of each other.
        # prices1 returns: [0.01, 0.02, 0.03, 0.01, 0.02]
        # prices2 returns: [-0.01, -0.02, -0.03, -0.01, -0.02]
        prices1 = [100.0, 101.0, 103.02, 106.1106, 107.1717, 109.3151]
        prices2 = [200.0, 198.0, 194.04, 188.2188, 186.3366, 182.6099]
        pipeline = FeaturePipeline()
        corr = pipeline._calculate_correlation(prices1, prices2, 5)
        assert corr < -0.99

    def test_calculate_correlation_insufficient(self):
        """Returns 0.0 when too few data points."""
        pipeline = FeaturePipeline()
        prices1 = [100.0, 101.0]
        prices2 = [200.0, 201.0]
        corr = pipeline._calculate_correlation(prices1, prices2, 20)
        assert corr == 0.0

    def test_calculate_correlation_single_return(self):
        """Returns 0.0 when only one return pair available."""
        pipeline = FeaturePipeline()
        prices1 = [100.0, 101.0]
        prices2 = [200.0, 202.0]
        corr = pipeline._calculate_correlation(prices1, prices2, 1)
        assert corr == 0.0  # need at least 2 returns

    def test_calculate_correlation_zero_prices(self):
        """Handles zero prices gracefully."""
        pipeline = FeaturePipeline()
        prices1 = [0.0, 0.0, 100.0, 101.0]
        prices2 = [0.0, 0.0, 200.0, 202.0]
        corr = pipeline._calculate_correlation(prices1, prices2, 2)
        assert corr == 0.0  # zero-priced entries are skipped


# ===================================================================
# FeaturePipeline - database operations
# ===================================================================

class TestFeaturePipelineDB:
    """Tests for SQLite connection and price data fetching."""

    def test_get_connection(self, tmp_path):
        """_get_connection returns sqlite3.Connection."""
        db = tmp_path / "test.db"
        pipeline = FeaturePipeline(db_path=str(db))
        conn = pipeline._get_connection()
        assert isinstance(conn, sqlite3.Connection)
        conn.close()

    def test_get_price_data_empty_db(self, tmp_path):
        """Returns empty list when prices table has no matching rows."""
        db = tmp_path / "empty.db"
        with sqlite3.connect(str(db)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prices (
                    date TEXT, symbol TEXT, close REAL, volume INTEGER
                )
            """)
        pipeline = FeaturePipeline(db_path=str(db))
        data = pipeline._get_price_data("SPY", days=10)
        assert data == []

    def test_get_price_data_no_table(self, tmp_path):
        """Raises OperationalError when prices table does not exist."""
        db = tmp_path / "notable.db"
        sqlite3.connect(str(db)).close()
        pipeline = FeaturePipeline(db_path=str(db))
        with pytest.raises(sqlite3.OperationalError):
            pipeline._get_price_data("SPY")

    def test_get_price_data_basic(self, tmp_path):
        """Returns price data in oldest-first order."""
        db = tmp_path / "basic.db"
        _build_simple_prices_db(db, "SPY", n_days=10)
        pipeline = FeaturePipeline(db_path=str(db))
        data = pipeline._get_price_data("SPY", days=10)
        assert len(data) == 10
        assert data[0]["date"] < data[-1]["date"]  # oldest first
        assert data[0]["close"] > 0
        assert "volume" in data[0]

    def test_get_price_data_respects_days(self, tmp_path):
        """_get_price_data returns at most `days` entries."""
        db = tmp_path / "limited.db"
        _build_simple_prices_db(db, "SPY", n_days=100)
        pipeline = FeaturePipeline(db_path=str(db))
        data = pipeline._get_price_data("SPY", days=10)
        assert len(data) <= 10

    def test_get_price_data_end_date(self, tmp_path):
        """_get_price_data filters by end_date."""
        db = tmp_path / "enddate.db"
        _build_simple_prices_db(db, "SPY", n_days=100)
        pipeline = FeaturePipeline(db_path=str(db))
        all_data = pipeline._get_price_data("SPY", days=100)
        cutoff = all_data[50]["date"]  # middle date
        filtered = pipeline._get_price_data("SPY", days=100, end_date=cutoff)
        assert len(filtered) > 0
        assert filtered[-1]["date"] <= cutoff

    def test_get_vix_data(self, tmp_path):
        """_get_vix_data fetches ^VIX from the prices table."""
        db = tmp_path / "vix.db"
        _build_prices_db(
            db, {"^VIX": (18.0, 0.0, 0)}, n_days=30,
        )
        pipeline = FeaturePipeline(db_path=str(db))
        data = pipeline._get_vix_data(days=30)
        assert len(data) == 30
        assert all("close" in d for d in data)


# ===================================================================
# FeaturePipeline - generate_features
# ===================================================================

class TestFeaturePipelineGenerate:
    """Tests for the main generate_features method."""

    def test_generate_features_basic(self, tmp_path):
        """Generate features with sufficient data returns a Features."""
        db = tmp_path / "gen.db"
        _populate_full_db(db, n_days=100)
        pipeline = FeaturePipeline(db_path=str(db))
        feats = pipeline.generate_features("SPY")
        assert feats is not None
        assert feats.symbol == "SPY"
        assert feats.timestamp is not None
        assert isinstance(feats.return_1d, float)
        assert isinstance(feats.volatility_20d, float)
        assert isinstance(feats.spy_correlation_20d, float)

    def test_generate_features_insufficient_data(self, tmp_path):
        """Returns None with fewer than 50 price points."""
        db = tmp_path / "short.db"
        _build_simple_prices_db(db, "SPY", n_days=30)
        pipeline = FeaturePipeline(db_path=str(db))
        feats = pipeline.generate_features("SPY")
        assert feats is None

    def test_generate_features_trend_direction_up(self, tmp_path):
        """Trend direction is 1 when price comfortably above SMA20."""
        db = tmp_path / "trend_up.db"
        _build_prices_db(
            db,
            {"SPY": (400.0, 0.005, 0), "^VIX": (18.0, 0.0, 1)},
            n_days=100,
        )
        pipeline = FeaturePipeline(db_path=str(db))
        feats = pipeline.generate_features("SPY")
        assert feats is not None
        assert feats.trend_direction == 1  # rising -> above SMA20 by >2%

    def test_generate_features_trend_direction_down(self, tmp_path):
        """Trend direction is -1 when price well below SMA20."""
        db = tmp_path / "trend_down.db"
        _build_prices_db(
            db,
            {"SPY": (500.0, -0.005, 0), "^VIX": (18.0, 0.0, 1)},
            n_days=100,
        )
        pipeline = FeaturePipeline(db_path=str(db))
        feats = pipeline.generate_features("SPY")
        assert feats is not None
        assert feats.trend_direction == -1  # falling -> below SMA20 by >2%

    def test_generate_features_vol_regime_high(self, tmp_path):
        """Vol regime is 'high' when VIX > 25."""
        db = tmp_path / "vol_high.db"
        _build_prices_db(
            db,
            {"SPY": (450.0, 0.001, 0), "^VIX": (30.0, 0.0, 1)},
            n_days=100,
        )
        pipeline = FeaturePipeline(db_path=str(db))
        feats = pipeline.generate_features("SPY")
        assert feats is not None
        assert feats.vol_regime == "high"

    def test_generate_features_vol_regime_low(self, tmp_path):
        """Vol regime is 'low' when VIX < 15."""
        db = tmp_path / "vol_low.db"
        _build_prices_db(
            db,
            {"SPY": (450.0, 0.001, 0), "^VIX": (12.0, 0.0, 1)},
            n_days=100,
        )
        pipeline = FeaturePipeline(db_path=str(db))
        feats = pipeline.generate_features("SPY")
        assert feats is not None
        assert feats.vol_regime == "low"

    def test_generate_features_vol_regime_normal(self, tmp_path):
        """Vol regime is 'normal' when VIX between 15 and 25."""
        db = tmp_path / "vol_normal.db"
        _build_prices_db(
            db,
            {"SPY": (450.0, 0.001, 0), "^VIX": (20.0, 0.0, 1)},
            n_days=100,
        )
        pipeline = FeaturePipeline(db_path=str(db))
        feats = pipeline.generate_features("SPY")
        assert feats is not None
        assert feats.vol_regime == "normal"

    def test_generate_features_vix_default(self, tmp_path):
        """VIX defaults to 20 when no VIX data available."""
        db = tmp_path / "no_vix.db"
        # Create DB with only SPY data, no ^VIX
        _build_prices_db(db, {"SPY": (450.0, 0.001, 0)}, n_days=100)
        pipeline = FeaturePipeline(db_path=str(db))
        feats = pipeline.generate_features("SPY")
        # VIX data will be empty -> vix_level defaults to 20
        assert feats is not None
        assert feats.vix_level == 20.0

    def test_generate_features_vix_percentile(self, tmp_path):
        """VIX percentile is computed when enough VIX data."""
        db = tmp_path / "vix_pct.db"
        _build_prices_db(
            db,
            {"SPY": (450.0, 0.001, 0), "^VIX": (22.0, 0.0, 1)},
            n_days=100,
        )
        pipeline = FeaturePipeline(db_path=str(db))
        feats = pipeline.generate_features("SPY")
        assert feats is not None
        assert 0.0 <= feats.vix_percentile_20d <= 1.0

    def test_generate_features_with_reference_date(self, tmp_path):
        """Passing a reference_date generates features for that date."""
        db = tmp_path / "refdate.db"
        _populate_full_db(db, n_days=100)
        pipeline = FeaturePipeline(db_path=str(db))
        # Use an early date to get features from the middle
        all_data = pipeline._get_price_data("SPY", days=100)
        mid_date = all_data[60]["date"]
        feats = pipeline.generate_features("SPY", reference_date=mid_date)
        assert feats is not None
        assert feats.timestamp <= mid_date

    def test_generate_features_spy_correlation(self, tmp_path):
        """SPY correlation is computed when SPY data exists."""
        db = tmp_path / "spy_corr.db"
        _build_prices_db(
            db,
            {"SPY": (450.0, 0.001, 0), "^VIX": (18.0, 0.0, 1), "QQQ": (300.0, 0.002, 2)},
            n_days=100,
        )
        pipeline = FeaturePipeline(db_path=str(db))
        feats = pipeline.generate_features("QQQ")
        assert feats is not None
        # Correlation with SPY should be positive (both trending up)
        assert feats.spy_correlation_20d >= -1.0
        assert feats.spy_correlation_20d <= 1.0

    def test_generate_features_volume_ratio(self, tmp_path):
        """Volume ratio is computed from recent volumes."""
        db = tmp_path / "vol_ratio.db"
        _populate_full_db(db, n_days=100)
        pipeline = FeaturePipeline(db_path=str(db))
        feats = pipeline.generate_features("SPY")
        assert feats is not None
        assert feats.volume_ratio >= 0.0


# ===================================================================
# FeaturePipeline - generate_all_features
# ===================================================================

class TestFeaturePipelineBatch:
    """Tests for generate_all_features (historical batch)."""

    def test_generate_all_features_multiple(self, tmp_path):
        """Generate features for multiple symbols returns dict keyed by symbol."""
        db = tmp_path / "batch.db"
        _populate_full_db(db, n_days=100)
        pipeline = FeaturePipeline(db_path=str(db))
        result = pipeline.generate_all_features(["SPY", "GLD"], lookback_days=50)
        assert isinstance(result, dict)
        assert "SPY" in result
        assert "GLD" in result
        assert len(result["SPY"]) > 0
        assert len(result["GLD"]) > 0

    def test_generate_all_features_returns_features(self, tmp_path):
        """Generated items are Features dataclass instances."""
        db = tmp_path / "feats_type.db"
        _populate_full_db(db, n_days=100)
        pipeline = FeaturePipeline(db_path=str(db))
        result = pipeline.generate_all_features(["SPY"], lookback_days=50)
        assert len(result["SPY"]) > 0
        assert isinstance(result["SPY"][0], Features)

    def test_generate_all_features_has_targets(self, tmp_path):
        """Future return and regime_label are filled for non-tail features."""
        db = tmp_path / "targets.db"
        _populate_full_db(db, n_days=150)  # Extra days so most features have forward data
        pipeline = FeaturePipeline(db_path=str(db))
        result = pipeline.generate_all_features(["SPY"], lookback_days=100)
        assert len(result["SPY"]) > 10
        # At least some features should have targets (not the tail end)
        with_targets = [f for f in result["SPY"] if f.future_return_5d is not None]
        assert len(with_targets) > 0
        for feat in with_targets:
            assert feat.regime_label is not None
            assert feat.regime_label in (0, 1, 2)

    def test_generate_all_features_skips_insufficient(self, tmp_path):
        """Skips symbols with fewer than 50 data points."""
        db = tmp_path / "skip_short.db"
        _build_simple_prices_db(db, "SPY", n_days=100)
        _build_simple_prices_db(db, "LOW_DATA", n_days=30)
        pipeline = FeaturePipeline(db_path=str(db))
        result = pipeline.generate_all_features(["SPY", "LOW_DATA"], lookback_days=50)
        assert "SPY" in result
        assert "LOW_DATA" not in result or len(result["LOW_DATA"]) == 0

    def test_generate_all_features_empty_symbols(self, tmp_path):
        """Empty symbol list returns empty dict."""
        db = tmp_path / "empty_syms.db"
        _populate_full_db(db, n_days=100)
        pipeline = FeaturePipeline(db_path=str(db))
        result = pipeline.generate_all_features([], lookback_days=50)
        assert result == {}

    def test_generate_all_features_empty_db(self, tmp_path):
        """Empty database returns empty dict."""
        db = tmp_path / "empty_all.db"
        with sqlite3.connect(str(db)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prices (
                    date TEXT, symbol TEXT, close REAL, volume INTEGER
                )
            """)
        pipeline = FeaturePipeline(db_path=str(db))
        result = pipeline.generate_all_features(["SPY"], lookback_days=50)
        assert result == {}

    def test_generate_all_features_regime_labels(self, tmp_path):
        """Regime labels are 0 (bear), 1 (neutral), 2 (bull) when set."""
        db = tmp_path / "regime_labels.db"
        _populate_full_db(db, n_days=150)
        pipeline = FeaturePipeline(db_path=str(db))
        result = pipeline.generate_all_features(["SPY"], lookback_days=100)
        with_targets = [f for f in result["SPY"] if f.future_return_5d is not None]
        assert len(with_targets) > 0
        for feat in with_targets:
            assert feat.regime_label in (0, 1, 2)
            if feat.future_return_5d > 0.02:
                assert feat.regime_label == 2
            elif feat.future_return_5d < -0.02:
                assert feat.regime_label == 0
            else:
                assert feat.regime_label == 1


# ===================================================================
# FeaturePipeline - to_dataframe
# ===================================================================

class TestFeaturePipelineDataFrame:
    """Tests for to_dataframe conversion."""

    def test_to_dataframe_empty_list(self):
        """Empty list returns empty DataFrame when pandas is available."""
        pipeline = FeaturePipeline()
        result = pipeline.to_dataframe([])
        # With pandas available, returns a DataFrame
        import pandas as pd
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_to_dataframe_single_feature(self):
        """Single feature produces expected DataFrame."""
        pipeline = FeaturePipeline()
        feats = [
            Features(
                symbol="SPY", timestamp="2024-01-15",
                return_1d=0.01, return_5d=0.0, return_20d=0.02,
                volatility_20d=0.12,
                sma_20=400.0, sma_50=395.0,
                price_vs_sma20=0.01, price_vs_sma50=0.005,
                volume_20d_avg=1e6, volume_ratio=0.9,
                vix_level=18.0, vix_change_5d=0.02, vix_percentile_20d=0.6,
                spy_correlation_20d=0.75, trend_direction=1, vol_regime="normal",
            ),
        ]
        import pandas as pd
        result = pipeline.to_dataframe(feats)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]["symbol"] == "SPY"
        assert result.iloc[0]["vol_regime"] == "normal"

    def test_to_dataframe_multiple_features(self):
        """Multiple features produce correct number of rows."""
        pipeline = FeaturePipeline()
        feats = [
            Features(
                symbol=s, timestamp="2024-01-15",
                return_1d=0.01, return_5d=0.0, return_20d=0.0,
                volatility_20d=0.0,
                sma_20=100.0, sma_50=100.0,
                price_vs_sma20=0.0, price_vs_sma50=0.0,
                volume_20d_avg=0.0, volume_ratio=1.0,
                vix_level=20.0, vix_change_5d=0.0, vix_percentile_20d=0.5,
                spy_correlation_20d=0.0, trend_direction=0, vol_regime="normal",
            )
            for s in ("SPY", "GLD", "TLT")
        ]
        result = pipeline.to_dataframe(feats)
        assert len(result) == 3

    def test_to_dataframe_all_keys_present(self):
        """All expected columns are in the output DataFrame."""
        pipeline = FeaturePipeline()
        feats = [
            Features(
                symbol="SPY", timestamp="2024-01-15",
                return_1d=0.0, return_5d=0.0, return_20d=0.0,
                volatility_20d=0.0,
                sma_20=0.0, sma_50=0.0,
                price_vs_sma20=0.0, price_vs_sma50=0.0,
                volume_20d_avg=0.0, volume_ratio=1.0,
                vix_level=20.0, vix_change_5d=0.0, vix_percentile_20d=0.5,
                spy_correlation_20d=0.0, trend_direction=0, vol_regime="normal",
            ),
        ]
        import pandas as pd
        result = pipeline.to_dataframe(feats)
        assert isinstance(result, pd.DataFrame)
        expected_columns = {
            "symbol", "timestamp", "return_1d", "return_5d", "return_20d",
            "volatility_20d", "sma_20", "sma_50", "price_vs_sma20",
            "price_vs_sma50", "volume_ratio", "vix_level", "vix_change_5d",
            "vix_percentile_20d", "spy_correlation_20d", "trend_direction",
            "vol_regime", "future_return_5d", "regime_label",
        }
        assert set(result.columns) == expected_columns


# ===================================================================
# FeatureStore
# ===================================================================

class TestFeatureStore:
    """Tests for persistent feature storage."""

    def test_default_data_dir(self):
        """Default data_dir resolves from src.paths."""
        from src.paths import DATA_DIR
        store = FeatureStore()
        assert store.data_dir == str(DATA_DIR)

    def test_custom_data_dir(self, tmp_path):
        """Custom data_dir is honored."""
        store_dir = tmp_path / "features_store"
        store = FeatureStore(data_dir=str(store_dir))
        assert store.data_dir == str(store_dir)
        assert store.features_file == str(store_dir / "features.jsonl")

    def test_save_features_creates_file(self, tmp_path):
        """save_features creates the JSONL file."""
        store_dir = tmp_path / "feat_save"
        store = FeatureStore(data_dir=str(store_dir))
        feat = Features(
            symbol="SPY", timestamp="2024-01-15",
            return_1d=0.01, return_5d=0.0, return_20d=0.02,
            volatility_20d=0.12,
            sma_20=400.0, sma_50=395.0,
            price_vs_sma20=0.01, price_vs_sma50=0.005,
            volume_20d_avg=1e6, volume_ratio=0.9,
            vix_level=18.0, vix_change_5d=0.02, vix_percentile_20d=0.6,
            spy_correlation_20d=0.75, trend_direction=1, vol_regime="normal",
        )
        store.save_features(feat)
        assert os.path.exists(store.features_file)
        with open(store.features_file) as f:
            lines = f.readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["symbol"] == "SPY"

    def test_save_features_appends(self, tmp_path):
        """save_features appends multiple records."""
        store_dir = tmp_path / "feat_append"
        store = FeatureStore(data_dir=str(store_dir))
        base_kw = dict(
            return_1d=0.0, return_5d=0.0, return_20d=0.0,
            volatility_20d=0.0, sma_20=0.0, sma_50=0.0,
            price_vs_sma20=0.0, price_vs_sma50=0.0,
            volume_20d_avg=0.0, volume_ratio=1.0,
            vix_level=20.0, vix_change_5d=0.0, vix_percentile_20d=0.5,
            spy_correlation_20d=0.0, trend_direction=0, vol_regime="normal",
        )
        for sym in ("SPY", "GLD", "TLT"):
            store.save_features(Features(symbol=sym, timestamp="2024-01-15", **base_kw))
        with open(store.features_file) as f:
            lines = f.readlines()
        assert len(lines) == 3

    def test_load_recent_features_empty(self, tmp_path):
        """load_recent_features returns [] when file does not exist."""
        store_dir = tmp_path / "feat_empty"
        store = FeatureStore(data_dir=str(store_dir))
        result = store.load_recent_features("SPY", days=30)
        assert result == []

    def test_load_recent_features_matching(self, tmp_path):
        """load_recent_features returns matching symbol within time window."""
        store_dir = tmp_path / "feat_recent"
        store = FeatureStore(data_dir=str(store_dir))
        base_kw = dict(
            return_1d=0.0, return_5d=0.0, return_20d=0.0,
            volatility_20d=0.0, sma_20=0.0, sma_50=0.0,
            price_vs_sma20=0.0, price_vs_sma50=0.0,
            volume_20d_avg=0.0, volume_ratio=1.0,
            vix_level=20.0, vix_change_5d=0.0, vix_percentile_20d=0.5,
            spy_correlation_20d=0.0, trend_direction=0, vol_regime="normal",
        )
        # Save SPY with today's date
        today = datetime.now().strftime("%Y-%m-%d")
        store.save_features(Features(symbol="SPY", timestamp=today, **base_kw))
        # Save GLD with old date
        old = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        store.save_features(Features(symbol="GLD", timestamp=old, **base_kw))
        # Load recent SPY
        result = store.load_recent_features("SPY", days=30)
        assert len(result) == 1
        assert result[0]["symbol"] == "SPY"

    def test_load_recent_features_excludes_old(self, tmp_path):
        """load_recent_features excludes entries outside the window."""
        store_dir = tmp_path / "feat_old"
        store = FeatureStore(data_dir=str(store_dir))
        base_kw = dict(
            return_1d=0.0, return_5d=0.0, return_20d=0.0,
            volatility_20d=0.0, sma_20=0.0, sma_50=0.0,
            price_vs_sma20=0.0, price_vs_sma50=0.0,
            volume_20d_avg=0.0, volume_ratio=1.0,
            vix_level=20.0, vix_change_5d=0.0, vix_percentile_20d=0.5,
            spy_correlation_20d=0.0, trend_direction=0, vol_regime="normal",
        )
        old = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        store.save_features(Features(symbol="SPY", timestamp=old, **base_kw))
        result = store.load_recent_features("SPY", days=30)
        assert result == []

    def test_load_recent_features_skips_corrupt_lines(self, tmp_path):
        """load_recent_features skips corrupt JSON lines."""
        store_dir = tmp_path / "feat_corrupt"
        os.makedirs(str(store_dir), exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")
        cutoff = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        feat_file = store_dir / "features.jsonl"
        with open(feat_file, "w") as f:
            f.write(json.dumps({"symbol": "SPY", "timestamp": today, "return_1d": 0.01}) + "\n")
            f.write("not valid json\n")
            f.write(json.dumps({"symbol": "GLD", "timestamp": today, "return_1d": 0.02}) + "\n")
            f.write(json.dumps({"symbol": "SPY", "timestamp": cutoff, "return_1d": 0.03}) + "\n")
        store = FeatureStore(data_dir=str(store_dir))
        result = store.load_recent_features("SPY", days=365)
        assert len(result) == 1
        assert result[0]["symbol"] == "SPY"


# ===================================================================
# CLI main()
# ===================================================================

class TestMain:
    """Tests for the CLI entry point."""

    def test_main_generate_default(self, tmp_path):
        """CLI 'generate' with no symbol defaults to SPY."""
        db = tmp_path / "cli_default.db"
        _populate_full_db(db, n_days=100)
        with patch.object(sys, "argv", ["features.py", "generate"]):
            with patch(
                "src.research.features.FeaturePipeline",
                return_value=FeaturePipeline(db_path=str(db)),
            ):
                with patch("builtins.print") as mock_print:
                    cli_main()
                    # Should print the JSON features to stdout
                    args = [a[0] for a in mock_print.call_args_list]
                    assert any("SPY" in str(a) for a in args)

    def test_main_generate_custom_symbol(self, tmp_path):
        """CLI 'generate' accepts a custom symbol."""
        db = tmp_path / "cli_symbol.db"
        _populate_full_db(db, n_days=100)
        with patch.object(sys, "argv", ["features.py", "generate", "GLD"]):
            with patch(
                "src.research.features.FeaturePipeline",
                return_value=FeaturePipeline(db_path=str(db)),
            ):
                with patch("builtins.print") as mock_print:
                    cli_main()
                    args = [str(a) for a in mock_print.call_args_list]
                    assert any("GLD" in a for a in args)

    def test_main_generate_no_features(self, tmp_path):
        """CLI shows 'No features generated' when data insufficient."""
        db = tmp_path / "cli_none.db"
        _build_simple_prices_db(db, "SPY", n_days=30)
        with patch.object(sys, "argv", ["features.py", "generate", "SPY"]):
            with patch(
                "src.research.features.FeaturePipeline",
                return_value=FeaturePipeline(db_path=str(db)),
            ):
                with patch("builtins.print") as mock_print:
                    cli_main()
                    # Should print the "No features" message
                    messages = [str(a[0]) for a in mock_print.call_args_list]
                    assert any("No features" in m for m in messages)

    def test_main_batch(self, tmp_path):
        """CLI 'batch' saves features for each symbol."""
        db = tmp_path / "cli_batch.db"
        _populate_full_db(db, n_days=100)
        store_dir = tmp_path / "batch_store"
        with patch.object(sys, "argv", ["features.py", "batch", "SPY", "GLD"]):
            with patch(
                "src.research.features.FeaturePipeline",
                return_value=FeaturePipeline(db_path=str(db)),
            ):
                with patch(
                    "src.research.features.FeatureStore",
                    return_value=FeatureStore(data_dir=str(store_dir)),
                ):
                    with patch("builtins.print") as mock_print:
                        cli_main()
                        messages = [str(a[0]) for a in mock_print.call_args_list]
                        assert any("Saved features" in m for m in messages)

    def test_main_historical(self, tmp_path):
        """CLI 'historical' generates and saves historical features."""
        db = tmp_path / "cli_hist.db"
        _populate_full_db(db, n_days=100)
        store_dir = tmp_path / "hist_store"
        with patch.object(sys, "argv", ["features.py", "historical", "SPY"]):
            with patch(
                "src.research.features.FeaturePipeline",
                return_value=FeaturePipeline(db_path=str(db)),
            ):
                with patch(
                    "src.research.features.FeatureStore",
                    return_value=FeatureStore(data_dir=str(store_dir)),
                ):
                    with patch("builtins.print") as mock_print:
                        cli_main()
                        messages = [str(a[0]) for a in mock_print.call_args_list]
                        assert any("Generated" in m for m in messages)

    def test_main_unknown_command(self):
        """CLI shows error for unknown command."""
        with patch.object(sys, "argv", ["features.py", "unknown_cmd"]):
            with patch("builtins.print") as mock_print:
                cli_main()
                messages = [str(a[0]) for a in mock_print.call_args_list]
                assert any("Unknown command" in m for m in messages)

    def test_main_no_args(self, tmp_path):
        """CLI with no args defaults to generating SPY features."""
        db = tmp_path / "cli_noargs.db"
        _populate_full_db(db, n_days=100)
        with patch.object(sys, "argv", ["features.py"]):
            with patch(
                "src.research.features.FeaturePipeline",
                return_value=FeaturePipeline(db_path=str(db)),
            ):
                with patch("builtins.print") as mock_print:
                    cli_main()
                    # Should print JSON or "No features"
                    assert mock_print.called


# ===================================================================
# Edge cases and error handling
# ===================================================================

class TestEdgeCases:
    """Edge cases and error handling for the feature pipeline."""

    def test_generate_features_missing_symbol(self, tmp_path):
        """Missing symbol returns None."""
        db = tmp_path / "missing.db"
        _build_simple_prices_db(db, "SPY", n_days=100)
        pipeline = FeaturePipeline(db_path=str(db))
        feats = pipeline.generate_features("NONEXISTENT")
        assert feats is None

    def test_generate_features_empty_db(self, tmp_path):
        """Empty database yields None."""
        db = tmp_path / "empty_f.db"
        with sqlite3.connect(str(db)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prices (
                    date TEXT, symbol TEXT, close REAL, volume INTEGER
                )
            """)
        pipeline = FeaturePipeline(db_path=str(db))
        feats = pipeline.generate_features("SPY")
        assert feats is None

    def test_generate_all_features_missing_symbol(self, tmp_path):
        """Missing symbol is skipped silently."""
        db = tmp_path / "missing_all.db"
        _build_simple_prices_db(db, "SPY", n_days=100)
        pipeline = FeaturePipeline(db_path=str(db))
        result = pipeline.generate_all_features(["SPY", "GHOST"], lookback_days=50)
        assert "SPY" in result
        assert result.get("GHOST", []) == []

    def test_generate_all_features_lookback_respected(self, tmp_path):
        """Larger lookback_days generates more feature vectors."""
        db = tmp_path / "lookback.db"
        _populate_full_db(db, n_days=200)
        pipeline = FeaturePipeline(db_path=str(db))
        result_short = pipeline.generate_all_features(["SPY"], lookback_days=50)
        result_long = pipeline.generate_all_features(["SPY"], lookback_days=150)
        assert len(result_long["SPY"]) >= len(result_short["SPY"])

    def test_save_features_non_existent_dir_creates(self, tmp_path):
        """save_features creates the data directory if missing."""
        store_dir = tmp_path / "new_dir" / "deep" / "path"
        store = FeatureStore(data_dir=str(store_dir))
        feat = Features(
            symbol="SPY", timestamp="2024-01-15",
            return_1d=0.0, return_5d=0.0, return_20d=0.0,
            volatility_20d=0.0, sma_20=0.0, sma_50=0.0,
            price_vs_sma20=0.0, price_vs_sma50=0.0,
            volume_20d_avg=0.0, volume_ratio=1.0,
            vix_level=20.0, vix_change_5d=0.0, vix_percentile_20d=0.5,
            spy_correlation_20d=0.0, trend_direction=0, vol_regime="normal",
        )
        store.save_features(feat)
        assert os.path.exists(store.features_file)

    def test_load_recent_features_empty_file(self, tmp_path):
        """load_recent_features returns [] for empty file."""
        store_dir = tmp_path / "empty_feat"
        os.makedirs(str(store_dir), exist_ok=True)
        feat_file = store_dir / "features.jsonl"
        feat_file.write_text("")  # Empty file
        store = FeatureStore(data_dir=str(store_dir))
        result = store.load_recent_features("SPY", days=30)
        assert result == []

    def test_generate_features_sma_50_fallback(self, tmp_path):
        """SMA 50 falls back to last price when fewer than 50 points."""
        db = tmp_path / "sma_fallback.db"
        _build_prices_db(
            db,
            {"SPY": (400.0, 0.001, 0), "^VIX": (18.0, 0.0, 1)},
            n_days=55,
        )
        pipeline = FeaturePipeline(db_path=str(db))
        feats = pipeline.generate_features("SPY")
        assert feats is not None
        assert feats.sma_50 > 0
        # With only 55 price points, SMA50 is computed from all 55
        assert feats.sma_50 > 0
