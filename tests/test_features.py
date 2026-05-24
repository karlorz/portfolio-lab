"""Tests for src/research/features.py -- feature engineering pipeline.

Tests cover:
- Features dataclass (defaults, field types, serialization)
- FeaturePipeline (init, database ops, computation, generation, batch, DataFrame)
- FeatureStore (persistent save/load, filtering, error handling)
- CLI main() entry point (all commands, unknown command, no-arg default)
- Module-level exports
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from src.research.features import (
    Features,
    FeaturePipeline,
    FeatureStore,
    main as cli_main,
)


# ---------------------------------------------------------------------------
# Helpers  (prices table builders shared across DB-backed tests)
# ---------------------------------------------------------------------------

def _make_weekdays(n: int, start: str = "2024-01-02") -> list[str]:
    """Return *n* consecutive weekday date strings (Mon-Fri)."""
    dates: list[str] = []
    d = datetime.strptime(start, "%Y-%m-%d")
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return dates


def _create_prices_table(
    db_path: str,
    symbol: str,
    prices: list[float],
    volumes: list[int] | None = None,
    dates: list[str] | None = None,
) -> None:
    """Populate a SQLite prices table for one symbol.

    If *dates* is omitted, weekday dates are auto-generated.
    """
    if dates is None:
        dates = _make_weekdays(len(prices))
    if volumes is None:
        volumes = [1_000_000] * len(prices)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS prices ("
            "  date TEXT, symbol TEXT, close REAL, volume INTEGER,"
            "  PRIMARY KEY (date, symbol)"
            ")"
        )
        for dt, px, vol in zip(dates, prices, volumes):
            conn.execute(
                "INSERT OR REPLACE INTO prices (date, symbol, close, volume) "
                "VALUES (?, ?, ?, ?)",
                (dt, symbol, round(px, 2), vol),
            )


def _populate_full_db(db_path: str, n_days: int = 100) -> None:
    """Write SPY + ^VIX (and optionally GLD/TLT) into the prices table."""
    rng = np.random.default_rng(42)
    dates = _make_weekdays(n_days)
    configs = {
        "SPY": (450.0, 0.001),
        "^VIX": (18.0, 0.000),
        "GLD": (180.0, 0.0005),
        "TLT": (95.0, -0.0002),
    }
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS prices ("
            "  date TEXT, symbol TEXT, close REAL, volume INTEGER,"
            "  PRIMARY KEY (date, symbol)"
            ")"
        )
        for sym, (start, drift) in configs.items():
            price = start
            for dt in dates:
                noise = rng.normal(0, 0.005)
                price *= 1.0 + drift + noise
                price = max(price, 0.01)
                vol = int(1_000_000 * (1.0 + rng.normal(0, 0.1)))
                conn.execute(
                    "INSERT OR REPLACE INTO prices (date, symbol, close, volume) "
                    "VALUES (?, ?, ?, ?)",
                    (dt, sym, round(price, 2), vol),
                )


def _base_feat_kw() -> dict:
    """Keyword arguments that satisfy the required Features fields."""
    return dict(
        return_1d=0.0,
        return_5d=0.0,
        return_20d=0.0,
        volatility_20d=0.0,
        sma_20=0.0,
        sma_50=0.0,
        price_vs_sma20=0.0,
        price_vs_sma50=0.0,
        volume_20d_avg=0.0,
        volume_ratio=1.0,
        vix_level=20.0,
        vix_change_5d=0.0,
        vix_percentile_20d=0.5,
        spy_correlation_20d=0.0,
        trend_direction=0,
        vol_regime="normal",
    )


def _make_feat(symbol: str = "SPY", **overrides) -> Features:
    """Build a Features instance with overridable fields."""
    timestamp = overrides.pop("timestamp", "2024-01-15")
    kw = _base_feat_kw()
    kw.update(overrides)
    return Features(symbol=symbol, timestamp=timestamp, **kw)


# ===================================================================
# Features dataclass
# ===================================================================

class TestFeaturesDataclass:
    """Field defaults, types, and serialization."""

    def test_optional_fields_default_to_none(self) -> None:
        """future_return_5d and regime_label are optional and default to None."""
        f = _make_feat()
        assert f.future_return_5d is None
        assert f.regime_label is None

    def test_optional_fields_can_be_set(self) -> None:
        """Optional fields accept explicit values."""
        f = _make_feat(future_return_5d=0.03, regime_label=2)
        assert f.future_return_5d == 0.03
        assert f.regime_label == 2

    def test_all_required_fields_accept_values(self) -> None:
        """Every required field can be read back."""
        f = Features(
            symbol="QQQ",
            timestamp="2024-06-01",
            return_1d=-0.005,
            return_5d=0.015,
            return_20d=0.04,
            volatility_20d=0.18,
            sma_20=320.0,
            sma_50=310.0,
            price_vs_sma20=0.025,
            price_vs_sma50=0.045,
            volume_20d_avg=2_000_000.0,
            volume_ratio=0.75,
            vix_level=16.0,
            vix_change_5d=0.02,
            vix_percentile_20d=0.4,
            spy_correlation_20d=0.9,
            trend_direction=1,
            vol_regime="low",
        )
        assert f.symbol == "QQQ"
        assert_almost_equal(f.return_1d, -0.005)
        assert f.sma_20 == 320.0
        assert f.volume_ratio == 0.75
        assert f.vol_regime == "low"

    def test_trend_direction_accepts_expected_values(self) -> None:
        """trend_direction holds -1, 0, or 1."""
        for val in (-1, 0, 1):
            f = _make_feat(trend_direction=val)
            assert f.trend_direction == val

    def test_vol_regime_accepts_expected_strings(self) -> None:
        """vol_regime holds 'low', 'normal', or 'high'."""
        for regime in ("low", "normal", "high"):
            f = _make_feat(vol_regime=regime)
            assert f.vol_regime == regime

    def test_field_types(self) -> None:
        """Verify critical field types."""
        f = _make_feat(future_return_5d=0.01, regime_label=1)
        assert isinstance(f.symbol, str)
        assert isinstance(f.trend_direction, int)
        assert isinstance(f.vol_regime, str)
        assert isinstance(f.return_1d, float)
        assert f.future_return_5d is None or isinstance(f.future_return_5d, float)
        assert f.regime_label is None or isinstance(f.regime_label, int)

    def test_vars_includes_all_fields(self) -> None:
        """vars() dict contains every declared dataclass field."""
        f = _make_feat(future_return_5d=0.02, regime_label=1)
        d = vars(f)
        expected_keys = {
            "symbol", "timestamp",
            "return_1d", "return_5d", "return_20d", "volatility_20d",
            "sma_20", "sma_50", "price_vs_sma20", "price_vs_sma50",
            "volume_20d_avg", "volume_ratio",
            "vix_level", "vix_change_5d", "vix_percentile_20d",
            "spy_correlation_20d", "trend_direction", "vol_regime",
            "future_return_5d", "regime_label",
        }
        assert set(d.keys()) == expected_keys
        assert d["future_return_5d"] == 0.02
        assert d["regime_label"] == 1


# ===================================================================
# FeaturePipeline -- init
# ===================================================================

class TestFeaturePipelineInit:
    """Pipeline construction and default path resolution."""

    def test_default_db_path_resolved(self) -> None:
        """When no db_path is given, src.paths.MARKET_DB is used."""
        from src.paths import MARKET_DB
        p = FeaturePipeline()
        assert p.db_path == str(MARKET_DB)

    def test_custom_db_path_honored(self) -> None:
        """An explicit db_path is stored as-is."""
        p = FeaturePipeline(db_path="/tmp/custom_test.db")
        assert p.db_path == "/tmp/custom_test.db"

    def test_features_cache_empty_dict(self) -> None:
        """features_cache starts as an empty dict."""
        p = FeaturePipeline()
        assert p.features_cache == {}


# ===================================================================
# FeaturePipeline -- database operations
# ===================================================================

class TestFeaturePipelineDB:
    """SQLite connection and price-data fetching."""

    def test_get_connection_returns_connection(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """_get_connection returns a live sqlite3.Connection."""
        db = tmp_path / "conn_test.db"
        p = FeaturePipeline(db_path=str(db))
        conn = p._get_connection()
        assert isinstance(conn, sqlite3.Connection)
        conn.close()

    def test_get_price_data_oldest_first(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """_get_price_data returns rows sorted oldest-first."""
        db = tmp_path / "oldest.db"
        _create_prices_table(str(db), "SPY", [100.0 + i for i in range(10)])
        p = FeaturePipeline(db_path=str(db))
        data = p._get_price_data("SPY", days=10)
        assert len(data) == 10
        assert data[0]["date"] < data[-1]["date"]
        assert data[0]["close"] == 100.0
        assert data[-1]["close"] == 109.0

    def test_get_price_data_no_matches(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """Empty list when symbol has no rows."""
        db = tmp_path / "no_match.db"
        _create_prices_table(str(db), "SPY", [100.0])
        p = FeaturePipeline(db_path=str(db))
        data = p._get_price_data("QQQ", days=10)
        assert data == []

    def test_get_price_data_null_volume_defaults_to_zero(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """NULL volume in DB is returned as 0."""
        db = tmp_path / "null_vol.db"
        with sqlite3.connect(str(db)) as conn:
            conn.execute(
                "CREATE TABLE prices (date TEXT, symbol TEXT, close REAL, volume INTEGER)"
            )
            conn.execute(
                "INSERT INTO prices (date, symbol, close, volume) VALUES (?, ?, ?, ?)",
                ("2024-01-03", "SPY", 101.0, None),
            )
        p = FeaturePipeline(db_path=str(db))
        data = p._get_price_data("SPY", days=5)
        assert len(data) == 1
        assert data[0]["volume"] == 0

    def test_get_price_data_respects_days_limit(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """At most *days* rows are returned."""
        db = tmp_path / "limit.db"
        _create_prices_table(str(db), "SPY", [100.0 + i for i in range(100)])
        p = FeaturePipeline(db_path=str(db))
        data = p._get_price_data("SPY", days=7)
        assert len(data) == 7

    def test_get_price_data_end_date_filter(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """end_date parameter filters rows to dates <= cutoff."""
        db = tmp_path / "enddate.db"
        _create_prices_table(str(db), "SPY", [100.0 + i for i in range(80)])
        p = FeaturePipeline(db_path=str(db))
        all_data = p._get_price_data("SPY", days=80)
        cutoff = all_data[40]["date"]
        filtered = p._get_price_data("SPY", days=80, end_date=cutoff)
        assert len(filtered) > 0
        for row in filtered:
            assert row["date"] <= cutoff

    def test_get_price_data_no_table_raises(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """Missing prices table raises OperationalError."""
        db = tmp_path / "notable.db"
        sqlite3.connect(str(db)).close()
        p = FeaturePipeline(db_path=str(db))
        with pytest.raises(sqlite3.OperationalError):
            p._get_price_data("SPY")

    def test_get_vix_data_fetches_caret_vix(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """_get_vix_data reads the ^VIX symbol."""
        db = tmp_path / "vix_fetch.db"
        _create_prices_table(str(db), "^VIX", [18.0 + 0.1 * i for i in range(25)])
        p = FeaturePipeline(db_path=str(db))
        data = p._get_vix_data(days=25)
        assert len(data) == 25
        assert all("close" in d for d in data)

    def test_get_vix_data_empty_when_no_vix(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """_get_vix_data returns [] when ^VIX has no rows."""
        db = tmp_path / "novix.db"
        _create_prices_table(str(db), "SPY", [100.0])
        p = FeaturePipeline(db_path=str(db))
        data = p._get_vix_data(days=25)
        assert data == []


# ===================================================================
# FeaturePipeline -- pure computation methods
# ===================================================================

class TestFeaturePipelineCalculations:
    """_calculate_returns, _volatility, _sma, _correlation."""

    # ---- _calculate_returns -------------------------------------------------

    def test_calc_returns_all_periods(self) -> None:
        """Returns for periods 1, 5, 20 are computed from price series."""
        p = FeaturePipeline()
        # 101 prices: 100.0, 100.5, 101.0, ..., 150.0
        prices = [100.0 + 0.5 * i for i in range(101)]
        rets = p._calculate_returns(prices, [1, 5, 20])
        assert_almost_equal(rets[1], 0.5 / 149.5, decimal=10)
        assert_almost_equal(rets[5], 2.5 / 147.5, decimal=10)
        assert_almost_equal(rets[20], 10.0 / 140.0, decimal=10)
        assert all(isinstance(v, float) for v in rets.values())

    def test_calc_returns_insufficient_data(self) -> None:
        """Returns 0.0 when not enough prices for a given period."""
        p = FeaturePipeline()
        prices = [100.0, 101.0]
        rets = p._calculate_returns(prices, [5, 20])
        assert rets[5] == 0.0
        assert rets[20] == 0.0

    def test_calc_returns_empty_list(self) -> None:
        """Returns 0.0 when price list is empty."""
        p = FeaturePipeline()
        rets = p._calculate_returns([], [1])
        assert rets[1] == 0.0

    def test_calc_returns_zero_divisor_guard(self) -> None:
        """Returns 0.0 when dividing by zero price."""
        p = FeaturePipeline()
        prices = [5.0, 10.0, 0.0, 100.0]
        rets = p._calculate_returns(prices, [1])
        # prices[-1]=100, prices[-2]=0, divisor is 0 -> guarded to 0.0
        assert rets[1] == 0.0

    # ---- _calculate_volatility ----------------------------------------------

    def test_calc_volatility_normal(self) -> None:
        """Annualized vol is computed from daily returns."""
        p = FeaturePipeline()
        # Daily returns are exactly 1% every day
        prices = [100.0 * (1.01**i) for i in range(22)]
        vol = p._calculate_volatility(prices, 20)
        assert vol > 0.0
        assert isinstance(vol, float)

    def test_calc_volatility_insufficient_prices(self) -> None:
        """Returns 0.0 when len(prices) < period + 1."""
        p = FeaturePipeline()
        assert p._calculate_volatility([100.0, 101.0], 20) == 0.0

    def test_calc_volatility_zero_prices_all_skipped(self) -> None:
        """Returns 0.0 when all prices are zero (no valid returns)."""
        p = FeaturePipeline()
        # 21 prices all zero -> all returns skipped -> < 2 returns -> 0.0
        prices = [0.0] * 21
        assert p._calculate_volatility(prices, 20) == 0.0

    def test_calc_volatility_constant_prices(self) -> None:
        """Returns 0.0 when all prices are identical."""
        p = FeaturePipeline()
        prices = [100.0] * 22
        vol = p._calculate_volatility(prices, 20)
        assert vol == 0.0

    def test_calc_volatility_single_return_after_filter(self) -> None:
        """Returns 0.0 when fewer than 2 non-zero returns survive."""
        p = FeaturePipeline()
        # 21 prices, but only 1 pair where prices[i-1] != 0
        prices = [0.0] * 20 + [100.0, 101.0]
        vol = p._calculate_volatility(prices, 20)
        assert vol == 0.0

    # ---- _calculate_sma -----------------------------------------------------

    def test_calc_sma_normal(self) -> None:
        """SMA over window returns the mean of the last N prices."""
        p = FeaturePipeline()
        prices = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert_almost_equal(p._calculate_sma(prices, 3), 40.0)  # (30+40+50)/3

    def test_calc_sma_less_data_than_period(self) -> None:
        """Returns the last price when len < period."""
        p = FeaturePipeline()
        prices = [10.0, 20.0]
        assert p._calculate_sma(prices, 20) == 20.0

    def test_calc_sma_empty_list(self) -> None:
        """Returns 0 for an empty price list."""
        p = FeaturePipeline()
        assert p._calculate_sma([], 20) == 0.0

    # ---- _calculate_correlation ---------------------------------------------

    def test_calc_corr_positive(self) -> None:
        """Two strongly co-moving series yield high positive correlation."""
        p = FeaturePipeline()
        p1 = [100.0 * (1.01**i) for i in range(22)]
        p2 = [200.0 * (1.01**i) for i in range(22)]
        corr = p._calculate_correlation(p1, p2, 20)
        assert corr > 0.9

    def test_calc_corr_negative(self) -> None:
        """Inverse-moving series yield strong negative correlation."""
        p = FeaturePipeline()
        # Build prices where returns are exact negatives
        ret1 = [0.01, -0.005, 0.02, -0.01, 0.015]
        ret2 = [-0.01, 0.005, -0.02, 0.01, -0.015]
        p1 = [100.0]
        p2 = [200.0]
        for r1, r2 in zip(ret1, ret2):
            p1.append(p1[-1] * (1.0 + r1))
            p2.append(p2[-1] * (1.0 + r2))
        corr = p._calculate_correlation(p1, p2, 5)
        assert_almost_equal(corr, -1.0, decimal=5)

    def test_calc_corr_insufficient_data(self) -> None:
        """Returns 0.0 when either series has < period points."""
        p = FeaturePipeline()
        assert p._calculate_correlation([100.0, 101.0], [200.0, 202.0], 20) == 0.0

    def test_calc_corr_fewer_than_two_returns(self) -> None:
        """Returns 0.0 when fewer than 2 valid return pairs."""
        p = FeaturePipeline()
        # period=1 gives at most 1 return pair -> < 2
        assert p._calculate_correlation([100.0, 101.0], [200.0, 202.0], 1) == 0.0

    def test_calc_corr_zero_prices_skipped(self) -> None:
        """Entries where either price is zero are not counted."""
        p = FeaturePipeline()
        p1 = [0.0, 0.0, 100.0, 101.0, 102.0]
        p2 = [0.0, 0.0, 200.0, 204.0, 208.0]
        corr = p._calculate_correlation(p1, p2, 2)
        assert corr == 0.0  # Only 1 valid return pair after skipping zeros

    def test_calc_corr_computation_failure_caught(self) -> None:
        """ValueError from np.corrcoef is caught and returns 0.0."""
        p = FeaturePipeline()
        p1 = [float(i) for i in range(25)]
        p2 = [float(i) for i in range(25)]
        with patch("src.research.features.np.corrcoef", side_effect=ValueError("constant input")):
            corr = p._calculate_correlation(p1, p2, 20)
        assert corr == 0.0


# ===================================================================
# FeaturePipeline -- generate_features
# ===================================================================

class TestFeaturePipelineGenerate:
    """The main generate_features method with various data scenarios."""

    def test_generate_basic(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """Sufficient price + VIX data produces a complete Features object."""
        _populate_full_db(str(tmp_path / "basic.db"), n_days=100)
        p = FeaturePipeline(db_path=str(tmp_path / "basic.db"))
        f = p.generate_features("SPY")
        assert f is not None
        assert f.symbol == "SPY"
        assert isinstance(f.return_1d, float)
        assert isinstance(f.volatility_20d, float)
        assert isinstance(f.spy_correlation_20d, float)
        assert f.vol_regime in ("low", "normal", "high")
        assert f.trend_direction in (-1, 0, 1)

    def test_generate_insufficient_data_returns_none(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """Fewer than 50 price points yields None."""
        db = tmp_path / "short.db"
        _create_prices_table(str(db), "SPY", [100.0 + i for i in range(30)])
        p = FeaturePipeline(db_path=str(db))
        assert p.generate_features("SPY") is None

    def test_generate_no_vix_data_defaults_level_to_20(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """When VIX data is absent, vix_level falls back to 20.0."""
        db = tmp_path / "novix.db"
        _create_prices_table(str(db), "SPY", [100.0 + 0.5 * i for i in range(80)])
        # No ^VIX rows exist
        # generate_features will also need SPY data (it queries ^VIX and SPY)
        # The VIX query returns [] -> vix_level defaults to 20.0
        # SPY query for correlation: also returns same data
        # So this should work
        p = FeaturePipeline(db_path=str(db))
        f = p.generate_features("SPY")
        assert f is not None
        assert f.vix_level == 20.0
        assert f.vix_change_5d == 0.0
        assert f.vix_percentile_20d == 0.5

    def test_generate_trend_up(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """Steadily rising price produces trend_direction == 1."""
        db = tmp_path / "tup.db"
        _create_prices_table(str(db), "SPY", [400.0 + 2.0 * i for i in range(80)])
        _create_prices_table(str(db), "^VIX", [18.0] * 50)
        p = FeaturePipeline(db_path=str(db))
        f = p.generate_features("SPY")
        assert f is not None
        assert f.trend_direction == 1

    def test_generate_trend_down(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """Steadily falling price produces trend_direction == -1."""
        db = tmp_path / "tdown.db"
        _create_prices_table(str(db), "SPY", [500.0 - 2.0 * i for i in range(80)])
        _create_prices_table(str(db), "^VIX", [18.0] * 50)
        p = FeaturePipeline(db_path=str(db))
        f = p.generate_features("SPY")
        assert f is not None
        assert f.trend_direction == -1

    def test_generate_trend_neutral(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """Flat price produces trend_direction == 0."""
        db = tmp_path / "tflat.db"
        _create_prices_table(str(db), "SPY", [100.0] * 80)
        _create_prices_table(str(db), "^VIX", [18.0] * 50)
        p = FeaturePipeline(db_path=str(db))
        f = p.generate_features("SPY")
        assert f is not None
        assert f.trend_direction == 0

    def test_generate_vol_regime_high(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """VIX > 25 sets vol_regime to 'high'."""
        db = tmp_path / "vhigh.db"
        _create_prices_table(str(db), "SPY", [100.0 + 0.5 * i for i in range(80)])
        _create_prices_table(str(db), "^VIX", [30.0] * 50)
        p = FeaturePipeline(db_path=str(db))
        f = p.generate_features("SPY")
        assert f is not None
        assert f.vol_regime == "high"

    def test_generate_vol_regime_low(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """VIX < 15 sets vol_regime to 'low'."""
        db = tmp_path / "vlow.db"
        _create_prices_table(str(db), "SPY", [100.0 + 0.5 * i for i in range(80)])
        _create_prices_table(str(db), "^VIX", [12.0] * 50)
        p = FeaturePipeline(db_path=str(db))
        f = p.generate_features("SPY")
        assert f is not None
        assert f.vol_regime == "low"

    def test_generate_vol_regime_normal(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """15 <= VIX <= 25 sets vol_regime to 'normal'."""
        db = tmp_path / "vnorm.db"
        _create_prices_table(str(db), "SPY", [100.0 + 0.5 * i for i in range(80)])
        _create_prices_table(str(db), "^VIX", [20.0] * 50)
        p = FeaturePipeline(db_path=str(db))
        f = p.generate_features("SPY")
        assert f is not None
        assert f.vol_regime == "normal"

    def test_generate_vix_change_5d(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """VIX 5-day change is computed from the last 6 VIX closes."""
        db = tmp_path / "vix5d.db"
        _create_prices_table(str(db), "SPY", [100.0 + 0.5 * i for i in range(80)])
        # VIX rises steadily in round increments to avoid rounding error
        vix_prices = [15.0 + 0.2 * i for i in range(50)]
        _create_prices_table(str(db), "^VIX", vix_prices)
        p = FeaturePipeline(db_path=str(db))
        f = p.generate_features("SPY")
        assert f is not None
        expected = (vix_prices[-1] - vix_prices[-6]) / vix_prices[-6]
        assert_almost_equal(f.vix_change_5d, expected, decimal=10)

    def test_generate_vix_percentile(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """VIX percentile is the fraction of recent closes <= current level."""
        db = tmp_path / "vixpct.db"
        _create_prices_table(str(db), "SPY", [100.0 + 0.5 * i for i in range(80)])
        _create_prices_table(str(db), "^VIX", [18.0] * 50)
        p = FeaturePipeline(db_path=str(db))
        f = p.generate_features("SPY")
        assert f is not None
        # All VIX prices are 18, and vix_level is also 18 -> 100th percentile
        assert f.vix_percentile_20d == 1.0

    def test_generate_vix_percentile_fallback(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """When fewer than 20 VIX points exist, percentile defaults to 0.5."""
        db = tmp_path / "vixpct_fb.db"
        _create_prices_table(str(db), "SPY", [100.0 + 0.5 * i for i in range(80)])
        _create_prices_table(str(db), "^VIX", [18.0] * 15)  # Only 15 VIX points
        p = FeaturePipeline(db_path=str(db))
        f = p.generate_features("SPY")
        assert f is not None
        assert f.vix_percentile_20d == 0.5

    def test_generate_spy_correlation(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """SPY correlation is computed from the price vs SPY price series."""
        db = tmp_path / "spycorr.db"
        # SPY and QQQ moving together
        _create_prices_table(str(db), "QQQ", [300.0 + 1.0 * i for i in range(80)])
        _create_prices_table(str(db), "SPY", [450.0 + 1.5 * i for i in range(80)])
        _create_prices_table(str(db), "^VIX", [18.0] * 50)
        p = FeaturePipeline(db_path=str(db))
        f = p.generate_features("QQQ")
        assert f is not None
        assert -1.0 <= f.spy_correlation_20d <= 1.0
        assert f.spy_correlation_20d > 0.0  # Both trending up

    def test_generate_missing_symbol(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """Non-existent symbol returns None."""
        _populate_full_db(str(tmp_path / "missing.db"), n_days=100)
        p = FeaturePipeline(db_path=str(tmp_path / "missing.db"))
        assert p.generate_features("NONEXISTENT") is None

    def test_generate_with_reference_date(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """A reference_date generates features for a historical point."""
        _populate_full_db(str(tmp_path / "ref.db"), n_days=100)
        p = FeaturePipeline(db_path=str(tmp_path / "ref.db"))
        all_data = p._get_price_data("SPY", days=100)
        mid_date = all_data[60]["date"]
        f = p.generate_features("SPY", reference_date=mid_date)
        assert f is not None
        assert f.timestamp <= mid_date
        assert f.symbol == "SPY"


# ===================================================================
# FeaturePipeline -- generate_all_features
# ===================================================================

class TestFeaturePipelineBatch:
    """Historical batch feature generation."""

    def test_batch_single_symbol(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """Single symbol returns a dict with one entry."""
        _populate_full_db(str(tmp_path / "single.db"), n_days=100)
        p = FeaturePipeline(db_path=str(tmp_path / "single.db"))
        result = p.generate_all_features(["SPY"], lookback_days=50)
        assert list(result.keys()) == ["SPY"]
        assert len(result["SPY"]) > 0

    def test_batch_multiple_symbols(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """Multiple symbols each get their own feature list."""
        _populate_full_db(str(tmp_path / "multi.db"), n_days=100)
        p = FeaturePipeline(db_path=str(tmp_path / "multi.db"))
        result = p.generate_all_features(["SPY", "GLD"], lookback_days=50)
        assert "SPY" in result
        assert "GLD" in result
        assert len(result["SPY"]) > 0
        assert len(result["GLD"]) > 0

    def test_batch_returns_features_instances(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """Each entry in the result list is a Features dataclass."""
        _populate_full_db(str(tmp_path / "types.db"), n_days=100)
        p = FeaturePipeline(db_path=str(tmp_path / "types.db"))
        result = p.generate_all_features(["SPY"], lookback_days=50)
        assert isinstance(result["SPY"][0], Features)

    def test_batch_skips_insufficient_data(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """Symbols with < 50 price points are skipped."""
        db = tmp_path / "skip.db"
        _create_prices_table(str(db), "SPY", [100.0 + i for i in range(80)])
        _create_prices_table(str(db), "^VIX", [18.0] * 50)
        _create_prices_table(str(db), "SHORT", [50.0 + i for i in range(30)])
        _populate_full_db(str(db), n_days=80)
        p = FeaturePipeline(db_path=str(db))
        result = p.generate_all_features(["SPY", "SHORT"], lookback_days=50)
        assert "SPY" in result
        assert "SHORT" not in result

    def test_batch_empty_symbol_list(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """Empty symbol list returns an empty dict."""
        _populate_full_db(str(tmp_path / "empty.db"), n_days=100)
        p = FeaturePipeline(db_path=str(tmp_path / "empty.db"))
        assert p.generate_all_features([], lookback_days=50) == {}

    def test_batch_empty_db(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """Empty database returns an empty dict."""
        db = tmp_path / "empty_all.db"
        with sqlite3.connect(str(db)) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS prices ("
                "  date TEXT, symbol TEXT, close REAL, volume INTEGER"
                ")"
            )
        p = FeaturePipeline(db_path=str(db))
        assert p.generate_all_features(["SPY"], lookback_days=50) == {}

    def test_batch_targets_filled(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """Future return and regime label are set for non-tail features."""
        _populate_full_db(str(tmp_path / "targets.db"), n_days=150)
        p = FeaturePipeline(db_path=str(tmp_path / "targets.db"))
        result = p.generate_all_features(["SPY"], lookback_days=100)
        with_targets = [f for f in result["SPY"] if f.future_return_5d is not None]
        assert len(with_targets) > 0
        for feat in with_targets:
            assert feat.regime_label is not None
            assert feat.regime_label in (0, 1, 2)

    def test_batch_regime_labels_consistent(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """Regime labels match the future_return_5d thresholds."""
        _populate_full_db(str(tmp_path / "reglabels.db"), n_days=150)
        p = FeaturePipeline(db_path=str(tmp_path / "reglabels.db"))
        result = p.generate_all_features(["SPY"], lookback_days=100)
        for feat in result["SPY"]:
            if feat.future_return_5d is not None:
                if feat.future_return_5d > 0.02:
                    assert feat.regime_label == 2, f"ret={feat.future_return_5d}"
                elif feat.future_return_5d < -0.02:
                    assert feat.regime_label == 0, f"ret={feat.future_return_5d}"
                else:
                    assert feat.regime_label == 1, f"ret={feat.future_return_5d}"

    def test_batch_larger_lookback_more_features(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """A larger lookback_days generates more feature vectors."""
        _populate_full_db(str(tmp_path / "lookback.db"), n_days=200)
        p = FeaturePipeline(db_path=str(tmp_path / "lookback.db"))
        short = p.generate_all_features(["SPY"], lookback_days=50)
        long = p.generate_all_features(["SPY"], lookback_days=150)
        assert len(long["SPY"]) >= len(short["SPY"])

    def test_batch_tail_features_no_targets(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """Tail-end features (no forward data) have None targets."""
        _populate_full_db(str(tmp_path / "tail.db"), n_days=55)
        p = FeaturePipeline(db_path=str(tmp_path / "tail.db"))
        result = p.generate_all_features(["SPY"], lookback_days=5)
        tail = [f for f in result["SPY"] if f.future_return_5d is None]
        assert len(tail) > 0


# ===================================================================
# FeaturePipeline -- to_dataframe
# ===================================================================

class TestFeaturePipelineDF:
    """Conversion from Features list to pandas DataFrame."""

    def test_df_empty_list(self) -> None:
        """Empty list returns an empty DataFrame."""
        p = FeaturePipeline()
        import pandas as pd
        df = p.to_dataframe([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_df_single_feature(self) -> None:
        """Single Features item becomes a one-row DataFrame."""
        p = FeaturePipeline()
        f = _make_feat(symbol="SPY", return_1d=0.01, vol_regime="normal")
        import pandas as pd
        df = p.to_dataframe([f])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "SPY"
        assert df.iloc[0]["return_1d"] == 0.01
        assert df.iloc[0]["vol_regime"] == "normal"

    def test_df_multiple_features(self) -> None:
        """N features produce N rows."""
        p = FeaturePipeline()
        feats = [
            _make_feat(symbol=s, return_5d=0.01 * i)
            for i, s in enumerate(("SPY", "GLD", "TLT", "IEF"), 1)
        ]
        df = p.to_dataframe(feats)
        assert len(df) == 4
        assert list(df["symbol"]) == ["SPY", "GLD", "TLT", "IEF"]

    def test_df_all_expected_columns(self) -> None:
        """DataFrame contains every field from the Features dataclass."""
        p = FeaturePipeline()
        f = _make_feat(future_return_5d=0.02, regime_label=1)
        expected = {
            "symbol", "timestamp",
            "return_1d", "return_5d", "return_20d", "volatility_20d",
            "sma_20", "sma_50", "price_vs_sma20", "price_vs_sma50",
            "volume_ratio", "vix_level", "vix_change_5d",
            "vix_percentile_20d", "spy_correlation_20d",
            "trend_direction", "vol_regime",
            "future_return_5d", "regime_label",
        }
        import pandas as pd
        df = p.to_dataframe([f])
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == expected

    def test_df_with_none_optionals(self) -> None:
        """Optional fields defaulting to None appear as NaN in the DataFrame."""
        p = FeaturePipeline()
        f = _make_feat()  # future_return_5d=None, regime_label=None
        import pandas as pd
        df = p.to_dataframe([f])
        assert pd.isna(df.iloc[0]["future_return_5d"])
        assert pd.isna(df.iloc[0]["regime_label"])


# ===================================================================
# FeatureStore
# ===================================================================

class TestFeatureStore:
    """Persistent JSONL storage for feature vectors."""

    def test_default_data_dir(self) -> None:
        """Default data_dir resolves from src.paths.DATA_DIR."""
        from src.paths import DATA_DIR
        s = FeatureStore()
        assert s.data_dir == str(DATA_DIR)

    def test_custom_data_dir(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """Custom data_dir is stored and features_file is constructed."""
        d = tmp_path / "my_feats"
        s = FeatureStore(data_dir=str(d))
        assert s.data_dir == str(d)
        assert s.features_file == os.path.join(str(d), "features.jsonl")

    def test_save_creates_directory(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """save_features creates the data directory if it does not exist."""
        d = tmp_path / "a" / "b" / "c"
        s = FeatureStore(data_dir=str(d))
        s.save_features(_make_feat("SPY"))
        assert os.path.isdir(str(d))
        assert os.path.isfile(s.features_file)

    def test_save_appends_multiple_records(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """Multiple save_features calls append to the same file."""
        d = tmp_path / "append"
        s = FeatureStore(data_dir=str(d))
        for sym in ("SPY", "GLD", "TLT"):
            s.save_features(_make_feat(sym))
        with open(s.features_file) as f:
            lines = f.readlines()
        assert len(lines) == 3

    def test_save_record_content(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """The saved JSON line contains all expected fields."""
        d = tmp_path / "content"
        s = FeatureStore(data_dir=str(d))
        f = _make_feat("SPY", return_1d=0.015, vix_level=16.0, vol_regime="low")
        s.save_features(f)
        with open(s.features_file) as fh:
            record = json.loads(fh.readline())
        assert record["symbol"] == "SPY"
        assert_almost_equal(record["vix_level"], 16.0)
        assert record["vol_regime"] == "low"
        assert "return_1d" in record
        assert "spy_correlation_20d" in record
        assert "future_return_5d" in record

    def test_load_no_file_returns_empty(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """load_recent_features returns [] when the JSONL file does not exist."""
        d = tmp_path / "noexist"
        s = FeatureStore(data_dir=str(d))
        assert s.load_recent_features("SPY", days=30) == []

    def test_load_empty_file_returns_empty(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """load_recent_features returns [] for an empty file."""
        d = tmp_path / "empty_f"
        os.makedirs(str(d), exist_ok=True)
        with open(os.path.join(str(d), "features.jsonl"), "w") as fh:
            fh.write("")
        s = FeatureStore(data_dir=str(d))
        assert s.load_recent_features("SPY", days=30) == []

    def test_load_filters_by_symbol(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """Only records matching the requested symbol are returned."""
        d = tmp_path / "symfilter"
        s = FeatureStore(data_dir=str(d))
        today = datetime.now().strftime("%Y-%m-%d")
        s.save_features(_make_feat("SPY", timestamp=today))
        s.save_features(_make_feat("GLD", timestamp=today))
        result = s.load_recent_features("SPY", days=30)
        assert len(result) == 1
        assert result[0]["symbol"] == "SPY"

    def test_load_filters_by_date_window(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """Records outside the day window are excluded."""
        d = tmp_path / "datefilter"
        s = FeatureStore(data_dir=str(d))
        today = datetime.now().strftime("%Y-%m-%d")
        old = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        s.save_features(_make_feat("SPY", timestamp=today))
        s.save_features(_make_feat("SPY", timestamp=old))
        result = s.load_recent_features("SPY", days=30)
        assert len(result) == 1
        assert result[0]["timestamp"] == today

    def test_load_skips_malformed_lines(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """Corrupt JSON lines are silently skipped."""
        d = tmp_path / "corrupt"
        os.makedirs(str(d), exist_ok=True)
        fp = os.path.join(str(d), "features.jsonl")
        today = datetime.now().strftime("%Y-%m-%d")
        with open(fp, "w") as fh:
            fh.write(json.dumps({"symbol": "SPY", "timestamp": today, "return_1d": 0.01}) + "\n")
            fh.write("INVALID_JSON\n")
            fh.write(json.dumps({"symbol": "SPY", "timestamp": today, "return_1d": 0.02}) + "\n")
        s = FeatureStore(data_dir=str(d))
        result = s.load_recent_features("SPY", days=30)
        assert len(result) == 2


# ===================================================================
# CLI main()
# ===================================================================

class TestMainCLI:
    """CLI entry point tests via sys.argv patching."""

    @staticmethod
    def _pipelines_and_store(
        db_path: str, store_dir: str,
    ) -> tuple:
        """Build real FeaturePipeline + FeatureStore for patch injection."""
        return (
            FeaturePipeline(db_path=db_path),
            FeatureStore(data_dir=store_dir),
        )

    def test_default_no_args(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """No arguments generates SPY features and prints them."""
        db = tmp_path / "cli_default.db"
        _populate_full_db(str(db), n_days=100)
        pipeline, store = self._pipelines_and_store(str(db), str(tmp_path / "st1"))
        with patch.object(sys, "argv", ["features.py"]):
            with patch("src.research.features.FeaturePipeline", return_value=pipeline):
                with patch("builtins.print") as mp:
                    cli_main()
                    assert mp.called
                    # Should print JSON containing SPY
                    calls_text = " ".join(str(c) for c in mp.call_args_list)
                    assert "SPY" in calls_text

    def test_generate_with_symbol(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """'generate GLD' prints features for GLD."""
        db = tmp_path / "cli_gen.db"
        _populate_full_db(str(db), n_days=100)
        pipeline, store = self._pipelines_and_store(str(db), str(tmp_path / "st2"))
        with patch.object(sys, "argv", ["features.py", "generate", "GLD"]):
            with patch("src.research.features.FeaturePipeline", return_value=pipeline):
                with patch("builtins.print") as mp:
                    cli_main()
                    calls_text = " ".join(str(c) for c in mp.call_args_list)
                    assert "GLD" in calls_text

    def test_generate_without_symbol_defaults_to_spy(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """'generate' (no symbol) defaults to SPY."""
        db = tmp_path / "cli_gen_default.db"
        _populate_full_db(str(db), n_days=100)
        pipeline, store = self._pipelines_and_store(str(db), str(tmp_path / "st3"))
        with patch.object(sys, "argv", ["features.py", "generate"]):
            with patch("src.research.features.FeaturePipeline", return_value=pipeline):
                with patch("builtins.print") as mp:
                    cli_main()
                    calls_text = " ".join(str(c) for c in mp.call_args_list)
                    assert "SPY" in calls_text

    def test_generate_no_features_message(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """Insufficient data prints a 'No features generated' message."""
        db = tmp_path / "cli_none.db"
        _create_prices_table(str(db), "SPY", [100.0 + i for i in range(30)])
        pipeline, store = self._pipelines_and_store(str(db), str(tmp_path / "st4"))
        with patch.object(sys, "argv", ["features.py", "generate", "SPY"]):
            with patch("src.research.features.FeaturePipeline", return_value=pipeline):
                with patch("builtins.print") as mp:
                    cli_main()
                    calls_text = " ".join(str(c) for c in mp.call_args_list)
                    assert "No features" in calls_text

    def test_batch_saves_features(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """'batch SPY GLD' saves features for each symbol."""
        db = tmp_path / "cli_batch.db"
        _populate_full_db(str(db), n_days=100)
        store_dir = tmp_path / "batch_store"
        pipeline, store = self._pipelines_and_store(str(db), str(store_dir))
        with patch.object(sys, "argv", ["features.py", "batch", "SPY", "GLD"]):
            with patch("src.research.features.FeaturePipeline", return_value=pipeline):
                with patch("src.research.features.FeatureStore", return_value=store):
                    with patch("builtins.print") as mp:
                        cli_main()
                        calls_text = " ".join(str(c) for c in mp.call_args_list)
                        assert "Saved features" in calls_text
                        # Both symbols should appear in the output
                        assert "SPY" in calls_text
                        assert "GLD" in calls_text

    def test_batch_default_symbols(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """'batch' (no symbols) defaults to SPY, GLD, TLT, IEF."""
        db = tmp_path / "cli_batch_def.db"
        _populate_full_db(str(db), n_days=100)
        store_dir = tmp_path / "batch_store_def"
        pipeline, store = self._pipelines_and_store(str(db), str(store_dir))
        with patch.object(sys, "argv", ["features.py", "batch"]):
            with patch("src.research.features.FeaturePipeline", return_value=pipeline):
                with patch("src.research.features.FeatureStore", return_value=store):
                    with patch("builtins.print") as mp:
                        cli_main()
                        calls_text = " ".join(str(c) for c in mp.call_args_list)
                        # Should mention all four default symbols
                        for sym in ("SPY", "GLD", "TLT", "IEF"):
                            assert sym in calls_text

    def test_historical_generates_and_saves(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """'historical SPY' generates all features and saves them."""
        db = tmp_path / "cli_hist.db"
        _populate_full_db(str(db), n_days=100)
        store_dir = tmp_path / "hist_store"
        pipeline, store = self._pipelines_and_store(str(db), str(store_dir))
        with patch.object(sys, "argv", ["features.py", "historical", "SPY"]):
            with patch("src.research.features.FeaturePipeline", return_value=pipeline):
                with patch("src.research.features.FeatureStore", return_value=store):
                    with patch("builtins.print") as mp:
                        cli_main()
                        calls_text = " ".join(str(c) for c in mp.call_args_list)
                        assert "Generated" in calls_text

    def test_historical_default_symbols(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[misc]
        """'historical' (no symbols) defaults to SPY, GLD, TLT."""
        db = tmp_path / "cli_hist_def.db"
        _populate_full_db(str(db), n_days=100)
        store_dir = tmp_path / "hist_store_def"
        pipeline, store = self._pipelines_and_store(str(db), str(store_dir))
        with patch.object(sys, "argv", ["features.py", "historical"]):
            with patch("src.research.features.FeaturePipeline", return_value=pipeline):
                with patch("src.research.features.FeatureStore", return_value=store):
                    with patch("builtins.print") as mp:
                        cli_main()
                        calls_text = " ".join(str(c) for c in mp.call_args_list)
                        assert "3 symbols" in calls_text
                        assert "Generated" in calls_text

    def test_unknown_command(self) -> None:
        """An unrecognised command prints an error message."""
        with patch.object(sys, "argv", ["features.py", "fly_to_the_moon"]):
            with patch("builtins.print") as mp:
                cli_main()
                calls_text = " ".join(str(c) for c in mp.call_args_list)
                assert "Unknown command" in calls_text


# ===================================================================
# Module exports
# ===================================================================

class TestModuleExports:
    """Verify the module's public API surface."""

    def test_public_names_accessible(self) -> None:
        """Expected public names are reachable from the module."""
        import src.research.features as mod
        expected = {"Features", "FeaturePipeline", "FeatureStore", "main"}
        for name in expected:
            assert hasattr(mod, name), f"{name} should be exported"

    def test_features_importable(self) -> None:
        """Direct imports work for all public classes."""
        # Already imported at module top; just verify they are the right types
        assert issubclass(Features, object)
        assert callable(FeaturePipeline)
        assert callable(FeatureStore)
        assert callable(cli_main)

    def test_aliased_main(self) -> None:
        """The main function is identical whether imported as main or via the module."""
        import src.research.features as mod
        assert mod.main is cli_main
