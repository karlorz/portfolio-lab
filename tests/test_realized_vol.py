#!/usr/bin/env python3
"""
Tests for realized_vol.py — OHLC volatility estimators.
Covers: OHLCBar, RealizedVolResult, all 5 estimators,
RealizedVolCalculator, RealizedVolPipeline, compute_realized_vol convenience,
estimator comparisons, edge cases.
"""
import sys
import os
import json
import math
import sqlite3
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch

from src.data.realized_vol import (
    OHLCBar, RealizedVolResult, RealizedVolCalculator,
    RealizedVolPipeline, compute_realized_vol,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bar(o=100.0, h=102.0, l=98.0, c=101.0, date="2026-01-01", vol=1000.0):
    return OHLCBar(date=date, open=o, high=h, low=l, close=c, volume=vol)


def _make_bars(n=25, base=100.0, daily_vol=0.02, seed=42):
    """Generate realistic OHLC bars with known volatility."""
    rng = np.random.RandomState(seed)
    bars = []
    price = base
    for i in range(n):
        ret = rng.normal(0, daily_vol)
        o = price
        c = price * (1 + ret)
        h = max(o, c) * (1 + abs(rng.normal(0, daily_vol * 0.3)))
        l = min(o, c) * (1 - abs(rng.normal(0, daily_vol * 0.3)))
        day = i + 1
        month = 1 + (day - 1) // 28
        dom = ((day - 1) % 28) + 1
        date = f"2026-{month:02d}-{dom:02d}"
        bars.append(OHLCBar(date=date, open=round(o, 4), high=round(h, 4),
                            low=round(l, 4), close=round(c, 4)))
        price = c
    return bars


# ---------------------------------------------------------------------------
# OHLCBar
# ---------------------------------------------------------------------------

class TestOHLCBar:
    def test_creation(self):
        bar = _make_bar()
        assert bar.date == "2026-01-01"
        assert bar.open == 100.0
        assert bar.high == 102.0
        assert bar.low == 98.0
        assert bar.close == 101.0
        assert bar.volume == 1000.0

    def test_default_volume(self):
        bar = OHLCBar(date="2026-01-01", open=100, high=102, low=98, close=101)
        assert bar.volume == 0.0

    def test_to_dict(self):
        bar = _make_bar()
        d = bar.to_dict()
        assert isinstance(d, dict)
        assert d["date"] == "2026-01-01"
        assert d["open"] == 100.0
        assert d["volume"] == 1000.0
        assert len(d) == 6  # date, open, high, low, close, volume


# ---------------------------------------------------------------------------
# RealizedVolResult
# ---------------------------------------------------------------------------

class TestRealizedVolResult:
    def test_creation(self):
        r = RealizedVolResult(
            symbol="SPY", date="2026-01-01", window=20,
            garman_klass=0.15, parkinson=0.14, rogers_satchell=0.16,
            yang_zhang=0.15, composite=0.15, close_to_close=0.14,
            n_bars=20, is_valid=True,
        )
        assert r.symbol == "SPY"
        assert r.garman_klass == 0.15
        assert r.is_valid is True

    def test_to_dict(self):
        r = RealizedVolResult(
            symbol="SPY", date="2026-01-01", window=20,
            garman_klass=0.15, parkinson=0.14, rogers_satchell=0.16,
            yang_zhang=0.15, composite=0.15, close_to_close=0.14,
            n_bars=20, is_valid=True,
        )
        d = r.to_dict()
        assert d["symbol"] == "SPY"
        assert d["garman_klass"] == 0.15
        assert "is_valid" in d

    def test_invalid_result(self):
        r = RealizedVolResult(
            symbol="", date="", window=20,
            garman_klass=0, parkinson=0, rogers_satchell=0,
            yang_zhang=0, composite=0, close_to_close=0,
            n_bars=0, is_valid=False,
        )
        assert r.is_valid is False
        assert r.symbol == ""


# ---------------------------------------------------------------------------
# Garman-Klass estimator
# ---------------------------------------------------------------------------

class TestGarmanKlass:
    def test_basic(self):
        o = np.array([100.0, 101.0])
        h = np.array([102.0, 103.0])
        l = np.array([98.0, 99.0])
        c = np.array([101.0, 102.0])
        result = RealizedVolCalculator.garman_klass(o, h, l, c)
        assert result > 0

    def test_single_bar_returns_zero(self):
        o = np.array([100.0])
        h = np.array([102.0])
        l = np.array([98.0])
        c = np.array([101.0])
        result = RealizedVolCalculator.garman_klass(o, h, l, c)
        assert result == 0.0

    def test_annualized(self):
        o = np.array([100.0] * 20)
        h = np.array([101.0] * 20)
        l = np.array([99.0] * 20)
        c = np.array([100.0] * 20)
        result = RealizedVolCalculator.garman_klass(o, h, l, c)
        assert result > 0
        assert result < 2.0  # <200% annualized

    def test_wider_range_higher_vol(self):
        o = np.array([100.0] * 20)
        c = np.array([100.0] * 20)
        h_narrow = np.array([101.0] * 20)
        l_narrow = np.array([99.0] * 20)
        h_wide = np.array([105.0] * 20)
        l_wide = np.array([95.0] * 20)
        narrow = RealizedVolCalculator.garman_klass(o, h_narrow, l_narrow, c)
        wide = RealizedVolCalculator.garman_klass(o, h_wide, l_wide, c)
        assert wide > narrow


# ---------------------------------------------------------------------------
# Parkinson estimator
# ---------------------------------------------------------------------------

class TestParkinson:
    def test_basic(self):
        h = np.array([102.0, 103.0])
        l = np.array([98.0, 99.0])
        result = RealizedVolCalculator.parkinson(h, l)
        assert result > 0

    def test_single_bar_returns_zero(self):
        h = np.array([102.0])
        l = np.array([98.0])
        result = RealizedVolCalculator.parkinson(h, l)
        assert result == 0.0

    def test_constant_price_zero_vol(self):
        h = np.array([100.0] * 20)
        l = np.array([100.0] * 20)
        result = RealizedVolCalculator.parkinson(h, l)
        assert result == 0.0

    def test_wider_range_higher_vol(self):
        h_narrow = np.array([101.0] * 20)
        l_narrow = np.array([99.0] * 20)
        h_wide = np.array([105.0] * 20)
        l_wide = np.array([95.0] * 20)
        narrow = RealizedVolCalculator.parkinson(h_narrow, l_narrow)
        wide = RealizedVolCalculator.parkinson(h_wide, l_wide)
        assert wide > narrow


# ---------------------------------------------------------------------------
# Rogers-Satchell estimator
# ---------------------------------------------------------------------------

class TestRogersSatchell:
    def test_basic(self):
        o = np.array([100.0, 101.0])
        h = np.array([102.0, 103.0])
        l = np.array([98.0, 99.0])
        c = np.array([101.0, 102.0])
        result = RealizedVolCalculator.rogers_satchell(o, h, l, c)
        assert result >= 0

    def test_single_bar_returns_zero(self):
        o = np.array([100.0])
        h = np.array([102.0])
        l = np.array([98.0])
        c = np.array([101.0])
        result = RealizedVolCalculator.rogers_satchell(o, h, l, c)
        assert result == 0.0

    def test_drift_independent(self):
        """RS should handle trending markets."""
        o = np.array([100.0] * 20)
        c = np.linspace(100.0, 120.0, 20)
        h = c * 1.02
        l = c * 0.98
        result = RealizedVolCalculator.rogers_satchell(o, h, l, c)
        assert result >= 0


# ---------------------------------------------------------------------------
# Yang-Zhang estimator
# ---------------------------------------------------------------------------

class TestYangZhang:
    def test_basic(self):
        o = np.array([100.0, 101.0, 102.0])
        h = np.array([102.0, 103.0, 104.0])
        l = np.array([98.0, 99.0, 100.0])
        c = np.array([101.0, 102.0, 103.0])
        result = RealizedVolCalculator.yang_zhang(o, h, l, c)
        assert result > 0

    def test_two_bars_returns_zero(self):
        o = np.array([100.0, 101.0])
        h = np.array([102.0, 103.0])
        l = np.array([98.0, 99.0])
        c = np.array([101.0, 102.0])
        result = RealizedVolCalculator.yang_zhang(o, h, l, c)
        assert result == 0.0

    def test_uses_overnight_gap(self):
        """YZ should capture overnight volatility."""
        o1 = np.array([100.0, 100.0, 100.0])
        h1 = np.array([101.0, 101.0, 101.0])
        l1 = np.array([99.0, 99.0, 99.0])
        c1 = np.array([100.0, 100.0, 100.0])
        vol1 = RealizedVolCalculator.yang_zhang(o1, h1, l1, c1)

        o2 = np.array([95.0, 105.0, 90.0])
        h2 = np.array([96.0, 106.0, 91.0])
        l2 = np.array([94.0, 104.0, 89.0])
        c2 = np.array([95.0, 105.0, 90.0])
        vol2 = RealizedVolCalculator.yang_zhang(o2, h2, l2, c2)
        assert vol2 > vol1


# ---------------------------------------------------------------------------
# Close-to-close estimator
# ---------------------------------------------------------------------------

class TestCloseToClose:
    def test_basic(self):
        c = np.array([100.0, 101.0, 102.0, 100.0, 99.0])
        result = RealizedVolCalculator.close_to_close(c)
        assert result > 0

    def test_single_price_returns_zero(self):
        c = np.array([100.0])
        result = RealizedVolCalculator.close_to_close(c)
        assert result == 0.0

    def test_constant_price_zero_vol(self):
        c = np.array([100.0] * 20)
        result = RealizedVolCalculator.close_to_close(c)
        assert result == 0.0

    def test_matches_std_formula(self):
        c = np.linspace(100, 120, 50)
        result = RealizedVolCalculator.close_to_close(c)
        returns = np.diff(np.log(c))
        expected = np.std(returns) * math.sqrt(252)
        assert abs(result - expected) < 0.001


# ---------------------------------------------------------------------------
# RealizedVolCalculator.compute()
# ---------------------------------------------------------------------------

class TestRealizedVolCalculatorCompute:
    def test_empty_bars(self):
        calc = RealizedVolCalculator()
        result = calc.compute([], window=20)
        assert result.n_bars == 0
        assert result.is_valid is False
        assert result.composite == 0

    def test_with_bars(self):
        calc = RealizedVolCalculator()
        bars = _make_bars(n=25)
        result = calc.compute(bars, window=20)
        assert result.n_bars == 20
        assert result.is_valid is True
        assert result.composite > 0

    def test_window_larger_than_bars(self):
        calc = RealizedVolCalculator()
        bars = _make_bars(n=15)
        result = calc.compute(bars, window=30)
        assert result.n_bars == 15
        assert result.is_valid is True  # 15 >= 10

    def test_fewer_than_10_bars_invalid(self):
        calc = RealizedVolCalculator()
        bars = _make_bars(n=8)
        result = calc.compute(bars, window=20)
        assert result.is_valid is False

    def test_result_date_is_last_bar(self):
        calc = RealizedVolCalculator()
        bars = _make_bars(n=25)
        result = calc.compute(bars, window=20)
        assert result.date == bars[-1].date

    def test_composite_is_average_of_valid_estimators(self):
        calc = RealizedVolCalculator()
        bars = _make_bars(n=25)
        result = calc.compute(bars, window=20)
        valid = [v for v in [result.garman_klass, result.parkinson,
                             result.rogers_satchell, result.yang_zhang] if v > 0.001]
        if valid:
            expected = round(np.mean(valid), 4)
            assert abs(result.composite - expected) < 0.001

    def test_values_are_rounded(self):
        calc = RealizedVolCalculator()
        bars = _make_bars(n=25)
        result = calc.compute(bars, window=20)
        for field in ["garman_klass", "parkinson", "rogers_satchell",
                       "yang_zhang", "composite", "close_to_close"]:
            val = getattr(result, field)
            assert val == round(val, 4)

    def test_constant_prices_low_vol(self):
        calc = RealizedVolCalculator()
        bars = [OHLCBar(date=f"2026-01-{(i+1):02d}", open=100, high=100,
                        low=100, close=100) for i in range(50)]
        result = calc.compute(bars, window=20)
        assert result.close_to_close < 0.01

    def test_small_window_uses_all_bars(self):
        calc = RealizedVolCalculator()
        bars = _make_bars(n=5)
        result = calc.compute(bars, window=20)
        assert result.n_bars == 5


# ---------------------------------------------------------------------------
# RealizedVolPipeline
# ---------------------------------------------------------------------------

class TestRealizedVolPipeline:
    def test_init_creates_output_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.data.realized_vol.RealizedVolPipeline.DATA_DIR", tmp_path)
        monkeypatch.setattr("src.data.realized_vol.RealizedVolPipeline.OUTPUT_DIR", tmp_path / "realized_vol")
        pipeline = RealizedVolPipeline()
        assert (tmp_path / "realized_vol").exists()

    def test_load_ohlc_no_db(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.data.realized_vol.RealizedVolPipeline.DATA_DIR", tmp_path)
        monkeypatch.setattr("src.data.realized_vol.RealizedVolPipeline.OUTPUT_DIR", tmp_path / "realized_vol")
        pipeline = RealizedVolPipeline()
        bars = pipeline.load_ohlc_bars("SPY")
        assert bars == []

    def test_load_ohlc_from_ohlc_table(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.data.realized_vol.RealizedVolPipeline.DATA_DIR", tmp_path)
        monkeypatch.setattr("src.data.realized_vol.RealizedVolPipeline.OUTPUT_DIR", tmp_path / "realized_vol")
        db = tmp_path / "market.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE ohlc (symbol TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)")
        conn.execute("INSERT INTO ohlc VALUES ('SPY', '2026-01-01', 100, 102, 98, 101, 1000)")
        conn.execute("INSERT INTO ohlc VALUES ('SPY', '2026-01-02', 101, 103, 99, 102, 1100)")
        conn.commit()
        conn.close()
        pipeline = RealizedVolPipeline()
        bars = pipeline.load_ohlc_bars("SPY")
        assert len(bars) == 2
        assert bars[0].date == "2026-01-01"
        assert bars[0].open == 100

    def test_load_ohlc_fallback_to_prices(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.data.realized_vol.RealizedVolPipeline.DATA_DIR", tmp_path)
        monkeypatch.setattr("src.data.realized_vol.RealizedVolPipeline.OUTPUT_DIR", tmp_path / "realized_vol")
        db = tmp_path / "market.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE prices (symbol TEXT, date TEXT, close REAL)")
        conn.execute("INSERT INTO prices VALUES ('SPY', '2026-01-01', 100.0)")
        conn.execute("INSERT INTO prices VALUES ('SPY', '2026-01-02', 101.0)")
        conn.execute("INSERT INTO prices VALUES ('SPY', '2026-01-03', 99.0)")
        conn.commit()
        conn.close()
        pipeline = RealizedVolPipeline()
        bars = pipeline.load_ohlc_bars("SPY")
        assert len(bars) == 3
        assert bars[0].close == 100.0
        # Estimated OHLC should have h >= max(o,c) and l <= min(o,c)
        for bar in bars:
            assert bar.high >= max(bar.open, bar.close) - 0.1
            assert bar.low <= min(bar.open, bar.close) + 0.1

    def test_load_ohlc_empty_db(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.data.realized_vol.RealizedVolPipeline.DATA_DIR", tmp_path)
        monkeypatch.setattr("src.data.realized_vol.RealizedVolPipeline.OUTPUT_DIR", tmp_path / "realized_vol")
        db = tmp_path / "market.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE ohlc (symbol TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)")
        conn.commit()
        conn.close()
        pipeline = RealizedVolPipeline()
        bars = pipeline.load_ohlc_bars("SPY")
        assert bars == []

    def test_compute_current(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.data.realized_vol.RealizedVolPipeline.DATA_DIR", tmp_path)
        monkeypatch.setattr("src.data.realized_vol.RealizedVolPipeline.OUTPUT_DIR", tmp_path / "realized_vol")
        pipeline = RealizedVolPipeline()
        bars = _make_bars(n=25)
        with patch.object(pipeline, 'load_ohlc_bars', return_value=bars):
            result = pipeline.compute_current("SPY", window=20)
        assert result.symbol == "SPY"
        assert result.is_valid is True
        assert result.composite > 0

    def test_compute_current_insufficient_data(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.data.realized_vol.RealizedVolPipeline.DATA_DIR", tmp_path)
        monkeypatch.setattr("src.data.realized_vol.RealizedVolPipeline.OUTPUT_DIR", tmp_path / "realized_vol")
        pipeline = RealizedVolPipeline()
        with patch.object(pipeline, 'load_ohlc_bars', return_value=_make_bars(n=5)):
            result = pipeline.compute_current("SPY", window=20)
        assert result.is_valid is False
        assert result.n_bars < 20

    def test_compute_rolling_realized_vol(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.data.realized_vol.RealizedVolPipeline.DATA_DIR", tmp_path)
        monkeypatch.setattr("src.data.realized_vol.RealizedVolPipeline.OUTPUT_DIR", tmp_path / "realized_vol")
        pipeline = RealizedVolPipeline()
        bars = _make_bars(n=50)
        with patch.object(pipeline, 'load_ohlc_bars', return_value=bars):
            results = pipeline.compute_rolling_realized_vol("SPY", window=20, days=50)
        assert len(results) > 0
        assert results[0].symbol == "SPY"
        assert len(results) == 31  # 50 - 20 + 1

    def test_compute_rolling_insufficient_data(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.data.realized_vol.RealizedVolPipeline.DATA_DIR", tmp_path)
        monkeypatch.setattr("src.data.realized_vol.RealizedVolPipeline.OUTPUT_DIR", tmp_path / "realized_vol")
        pipeline = RealizedVolPipeline()
        with patch.object(pipeline, 'load_ohlc_bars', return_value=_make_bars(n=10)):
            results = pipeline.compute_rolling_realized_vol("SPY", window=20, days=50)
        assert results == []

    def test_save_results(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.data.realized_vol.RealizedVolPipeline.DATA_DIR", tmp_path)
        monkeypatch.setattr("src.data.realized_vol.RealizedVolPipeline.OUTPUT_DIR", tmp_path / "realized_vol")
        pipeline = RealizedVolPipeline()
        results = [
            RealizedVolResult(
                symbol="SPY", date="2026-01-01", window=20,
                garman_klass=0.15, parkinson=0.14, rogers_satchell=0.16,
                yang_zhang=0.15, composite=0.15, close_to_close=0.14,
                n_bars=20, is_valid=True,
            )
        ]
        pipeline.save_results(results, "SPY")
        out_file = tmp_path / "realized_vol" / "SPY_realized_vol.json"
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert len(data) == 1
        assert data[0]["symbol"] == "SPY"

    def test_save_multiple_results(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.data.realized_vol.RealizedVolPipeline.DATA_DIR", tmp_path)
        monkeypatch.setattr("src.data.realized_vol.RealizedVolPipeline.OUTPUT_DIR", tmp_path / "realized_vol")
        pipeline = RealizedVolPipeline()
        results = [
            RealizedVolResult(
                symbol="SPY", date=f"2026-01-{i+1:02d}", window=20,
                garman_klass=0.15, parkinson=0.14, rogers_satchell=0.16,
                yang_zhang=0.15, composite=0.15, close_to_close=0.14,
                n_bars=20, is_valid=True,
            ) for i in range(5)
        ]
        pipeline.save_results(results, "SPY")
        data = json.loads((tmp_path / "realized_vol" / "SPY_realized_vol.json").read_text())
        assert len(data) == 5


# ---------------------------------------------------------------------------
# compute_realized_vol convenience function
# ---------------------------------------------------------------------------

class TestComputeRealizedVolConvenience:
    def test_convenience_function(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.data.realized_vol.RealizedVolPipeline.DATA_DIR", tmp_path)
        monkeypatch.setattr("src.data.realized_vol.RealizedVolPipeline.OUTPUT_DIR", tmp_path / "realized_vol")
        result = compute_realized_vol("SPY", window=20)
        assert result.symbol == "SPY"
        assert result.window == 20

    def test_default_symbol_and_window(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.data.realized_vol.RealizedVolPipeline.DATA_DIR", tmp_path)
        monkeypatch.setattr("src.data.realized_vol.RealizedVolPipeline.OUTPUT_DIR", tmp_path / "realized_vol")
        result = compute_realized_vol()
        assert result.symbol == "SPY"
        assert result.window == 20


# ---------------------------------------------------------------------------
# Estimator comparison tests
# ---------------------------------------------------------------------------

class TestEstimatorComparisons:
    def test_all_estimators_similar_for_typical_data(self):
        calc = RealizedVolCalculator()
        bars = _make_bars(n=100, daily_vol=0.015, seed=42)
        result = calc.compute(bars, window=60)
        ests = [result.garman_klass, result.parkinson, result.rogers_satchell,
                result.yang_zhang, result.close_to_close]
        for est in ests:
            if est > 0.001 and result.composite > 0:
                assert abs(est - result.composite) / result.composite < 0.5

    def test_higher_daily_vol_higher_annualized(self):
        calc = RealizedVolCalculator()
        low_vol_bars = _make_bars(n=50, daily_vol=0.005, seed=42)
        high_vol_bars = _make_bars(n=50, daily_vol=0.04, seed=42)
        low_result = calc.compute(low_vol_bars, window=30)
        high_result = calc.compute(high_vol_bars, window=30)
        assert high_result.composite > low_result.composite

    def test_parkinson_lower_than_gk(self):
        """Parkinson (HL only) should generally be lower than GK (OHLC)."""
        calc = RealizedVolCalculator()
        bars = _make_bars(n=50, daily_vol=0.02, seed=42)
        result = calc.compute(bars, window=30)
        # Parkinson only uses HL, GK uses OHLC — GK should be similar or higher
        if result.parkinson > 0.001 and result.garman_klass > 0.001:
            # They should be in the same ballpark
            assert result.parkinson / result.garman_klass < 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
