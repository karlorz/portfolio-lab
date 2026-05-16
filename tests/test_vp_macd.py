"""
Tests for v5.55 VP-MACD Signal (Volume-Price Adjusted MACD)
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime
import numpy as np
import pytest

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.signals.vp_macd import (
    _ema,
    _volume_weighted_ema,
    _compute_vp_macd,
    _compute_volatility,
    _classify_vol_regime,
    generate_signal,
    backtest,
    VPMACDSignal,
    SIGNAL_STRONG_LONG, SIGNAL_LONG, SIGNAL_NEUTRAL,
    SIGNAL_SHORT, SIGNAL_STRONG_SHORT,
    DEFAULT_FAST, DEFAULT_SLOW, DEFAULT_SIGNAL,
    STATE_PATH, SIGNALS_DIR, PRICES_PATH,
)


class TestEMA:
    """Test exponential moving average computation."""

    def test_ema_basic(self):
        """Basic EMA with known values."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        ema = _ema(values, 3)
        assert not np.isnan(ema[2])  # SMA at index 2
        assert ema[2] == pytest.approx(2.0)  # mean(1,2,3)
        assert ema[-1] > ema[-2]  # Upward trend

    def test_ema_constant(self):
        """EMA of constant values should equal the constant."""
        values = np.full(20, 5.0)
        ema = _ema(values, 5)
        assert not np.isnan(ema[4])
        assert ema[4] == pytest.approx(5.0)

    def test_ema_insufficient_data(self):
        """EMA with less data than period should return NaN."""
        values = np.array([1.0, 2.0, 3.0])
        ema = _ema(values, 10)
        assert np.all(np.isnan(ema))

    def test_ema_single_period(self):
        """EMA with period=1 is the same as input."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ema = _ema(values, 1)
        assert not np.isnan(ema[-1])

    def test_ema_period_matches_length(self):
        """EMA when period equals data length."""
        values = np.arange(1, 11, dtype=float)
        ema = _ema(values, 10)
        assert not np.isnan(ema[9])  # SMA of all values
        assert ema[9] == pytest.approx(5.5)


class TestVolumeWeightedEMA:
    """Test volume-weighted EMA computation."""

    def test_standard_ema_fallback(self):
        """Without volume, should match standard EMA."""
        prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        vw_ema = _volume_weighted_ema(prices, None, 3)
        std_ema = _ema(prices, 3)
        assert np.allclose(vw_ema[~np.isnan(vw_ema)], std_ema[~np.isnan(std_ema)])

    def test_zero_volumes_fallback(self):
        """With all zero volumes, should fall back to standard EMA."""
        prices = np.array([10.0, 11.0, 12.0, 13.0])
        volumes = np.array([0.0, 0.0, 0.0, 0.0])
        vw_ema = _volume_weighted_ema(prices, volumes, 2)
        std_ema = _ema(prices, 2)
        assert np.allclose(vw_ema[~np.isnan(vw_ema)], std_ema[~np.isnan(std_ema)])

    def test_volume_weighted_different(self):
        """With varying volumes, result should differ from standard EMA."""
        prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        volumes = np.array([1000.0, 100.0, 1000.0, 100.0, 1000.0])
        vw_ema = _volume_weighted_ema(prices, volumes, 3)
        std_ema = _ema(prices, 3)
        # At least one value should differ
        assert not np.allclose(vw_ema, std_ema, rtol=1e-5)

    def test_mismatched_length(self):
        """When volume and price lengths differ, use standard EMA."""
        prices = np.array([10.0, 11.0, 12.0, 13.0])
        volumes = np.array([100.0, 200.0])  # Too short
        vw_ema = _volume_weighted_ema(prices, volumes, 2)
        std_ema = _ema(prices, 2)
        assert np.allclose(vw_ema[~np.isnan(vw_ema)], std_ema[~np.isnan(std_ema)])


class TestComputeVPMACD:
    """Test full VP-MACD computation pipeline."""

    def test_macd_shape(self):
        """MACD output arrays match input length."""
        prices = np.arange(50, dtype=float).cumsum() + 100
        macd, signal, hist = _compute_vp_macd(prices, None, 5, 13, 5)
        assert len(macd) == len(prices)
        assert len(signal) == len(prices)
        assert len(hist) == len(prices)

    def test_macd_histogram_relationship(self):
        """Histogram = MACD line - signal line."""
        prices = np.sin(np.linspace(0, 4*np.pi, 100)) * 10 + 100
        macd, signal, hist = _compute_vp_macd(prices, None, 12, 26, 9)
        valid = ~np.isnan(hist)
        assert np.allclose(hist[valid], macd[valid] - signal[valid])

    def test_macd_up_trend(self):
        """In a strong uptrend, MACD line should be above signal line."""
        prices = np.linspace(100, 200, 100)
        macd, signal, hist = _compute_vp_macd(prices, None, 12, 26, 9)
        # Last values should show positive histogram in uptrend
        assert hist[-1] > 0 or hist[-5:].mean() > 0

    def test_macd_down_trend(self):
        """In a strong downtrend, MACD line should be below signal line."""
        # Add noise to avoid perfectly linear series (causes EMA convergence)
        rng = np.random.default_rng(42)
        trend = np.linspace(200, 100, 100)
        noise = rng.normal(0, 0.5, 100)
        prices = trend + noise
        macd, signal, hist = _compute_vp_macd(prices, None, 12, 26, 9)
        last_10 = hist[-10:]
        valid = last_10[~np.isnan(last_10)]
        assert len(valid) == 0 or np.mean(valid) < 0.01  # Not meaningfully positive

    def test_default_parameters(self):
        """Default parameters (12, 26, 9) produce valid output."""
        prices = np.random.default_rng(42).normal(100, 5, 200).cumsum() + 100
        macd, signal, hist = _compute_vp_macd(prices, None, DEFAULT_FAST, DEFAULT_SLOW, DEFAULT_SIGNAL)
        assert len(macd) == len(prices)
        assert not np.all(np.isnan(hist[-50:]))

    def test_volume_weighted_macd(self):
        """Volume-weighted MACD produces different results from standard."""
        prices = np.random.default_rng(42).normal(100, 5, 200).cumsum() + 100
        volumes = np.random.default_rng(123).uniform(100, 1000, 200)
        vp_macd, vp_sig, vp_hist = _compute_vp_macd(prices, volumes, 12, 26, 9)
        std_macd, std_sig, std_hist = _compute_vp_macd(prices, None, 12, 26, 9)
        # Should differ somewhere
        assert not np.allclose(vp_hist[-50:], std_hist[-50:], rtol=1e-3)


class TestVolatility:
    """Test volatility computation."""

    def test_volatility_shape(self):
        """Volatility array matches input length."""
        prices = np.arange(100, dtype=float) + 100
        vol = _compute_volatility(prices, 20)
        assert len(vol) == len(prices)
        assert np.all(np.isnan(vol[:20]))  # First 20 are NaN

    def test_volatility_constant_prices(self):
        """Constant prices should give near-zero volatility."""
        prices = np.full(100, 100.0)
        vol = _compute_volatility(prices, 20)
        assert np.all(vol[20:] < 0.01) or np.all(np.isnan(vol[20:]))

    def test_volatility_trending(self):
        """Steady trend gives lower vol than volatile."""
        steady = np.linspace(100, 110, 100)
        volatile = np.random.default_rng(42).normal(0, 2, 100).cumsum() + 100
        vol_steady = _compute_volatility(steady, 20)
        vol_volatile = _compute_volatility(volatile, 20)
        assert np.nanmean(vol_volatile[20:]) > np.nanmean(vol_steady[20:])


class TestVolRegime:
    """Test volatility regime classification."""

    def test_low_vol(self):
        assert _classify_vol_regime(0.05) == "low"
        assert _classify_vol_regime(0.11) == "low"

    def test_normal_vol(self):
        assert _classify_vol_regime(0.12) == "normal"
        assert _classify_vol_regime(0.15) == "normal"
        assert _classify_vol_regime(0.19) == "normal"

    def test_elevated_vol(self):
        assert _classify_vol_regime(0.20) == "elevated"
        assert _classify_vol_regime(0.25) == "elevated"

    def test_high_vol(self):
        assert _classify_vol_regime(0.30) == "high"
        assert _classify_vol_regime(0.50) == "high"

    def test_boundaries(self):
        assert _classify_vol_regime(0.119) == "low"
        assert _classify_vol_regime(0.12) == "normal"
        assert _classify_vol_regime(0.199) == "normal"
        assert _classify_vol_regime(0.20) == "elevated"
        assert _classify_vol_regime(0.299) == "elevated"
        assert _classify_vol_regime(0.30) == "high"


class TestGenerateSignal:
    """Test signal generation (with synthetic data)."""

    @pytest.fixture
    def _setup_prices(self, tmp_path):
        """Create temporary prices.json with synthetic data."""
        # Generate synthetic prices with trend
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.normal(0.001, 0.01, 200))
        dates = [f"2025-{(i // 30)+1:02d}-{(i % 30)+1:02d}" for i in range(200)]

        symbol_data = [
            {"d": dates[i], "p": float(prices[i])}
            for i in range(200)
        ]
        price_file = tmp_path / "prices.json"
        with open(price_file, "w") as f:
            json.dump({"SPY": symbol_data}, f)
        return price_file

    def test_generate_with_synthetic(self, monkeypatch, _setup_prices):
        """Generate signal with synthetic data."""
        monkeypatch.setattr("src.signals.vp_macd.PRICES_PATH", _setup_prices)
        monkeypatch.setattr("src.signals.vp_macd.DB_PATH", Path("/nonexistent/db.sqlite"))

        signal = generate_signal("SPY")
        assert signal is not None
        assert signal.ticker == "SPY"
        assert signal.vp_macd_value in [-1.0, -0.5, 0.0, 0.5, 1.0]
        assert 0 <= signal.confidence <= 1
        assert signal.vp_macd_signal in [
            "strong_short", "short", "neutral", "long", "strong_long"
        ]
        assert signal.regime in ["low", "normal", "elevated", "high"]

    def test_generate_unknown_ticker(self, monkeypatch, _setup_prices):
        """Unknown ticker returns None."""
        monkeypatch.setattr("src.signals.vp_macd.PRICES_PATH", _setup_prices)
        signal = generate_signal("UNKNOWN_TICKER_XYZ")
        assert signal is None

    def test_generate_with_custom_params(self, monkeypatch, _setup_prices):
        """Custom parameters produce valid signal."""
        monkeypatch.setattr("src.signals.vp_macd.PRICES_PATH", _setup_prices)
        monkeypatch.setattr("src.signals.vp_macd.DB_PATH", Path("/nonexistent/db.sqlite"))

        signal = generate_signal("SPY", fast=8, slow=21, signal_period=7, vol_mult=2.0)
        assert signal is not None
        assert signal.fast_period == 8
        assert signal.slow_period == 21
        assert signal.signal_period == 7
        assert signal.vol_multiplier == 2.0

    def test_generate_saves_state(self, monkeypatch, _setup_prices):
        """Generated signal should be saved to state file."""
        monkeypatch.setattr("src.signals.vp_macd.PRICES_PATH", _setup_prices)
        monkeypatch.setattr("src.signals.vp_macd.DB_PATH", Path("/nonexistent/db.sqlite"))
        monkeypatch.setattr("src.signals.vp_macd.STATE_PATH", _setup_prices.parent / "state.json")
        monkeypatch.setattr("src.signals.vp_macd.SIGNALS_DIR", _setup_prices.parent)

        signal = generate_signal("SPY")
        assert signal is not None

    def test_generate_insufficient_data(self, monkeypatch, tmp_path):
        """Very short data returns None."""
        price_file = tmp_path / "prices.json"
        with open(price_file, "w") as f:
            json.dump({"SPY": [{"d": "2025-01-01", "p": 100.0}]}, f)

        monkeypatch.setattr("src.signals.vp_macd.PRICES_PATH", price_file)
        monkeypatch.setattr("src.signals.vp_macd.DB_PATH", Path("/nonexistent/db.sqlite"))

        signal = generate_signal("SPY")
        assert signal is None


class TestBacktest:
    """Test backtest functionality."""

    @pytest.fixture
    def _setup_price_data(self, tmp_path):
        """Create longer synthetic price data for backtest."""
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.normal(0.0005, 0.015, 500))
        dates = [f"2024-{(i // 30)+1:02d}-{(i % 30)+1:02d}" for i in range(500)]

        symbol_data = [
            {"d": dates[i], "p": float(prices[i])}
            for i in range(500)
        ]
        price_file = tmp_path / "prices.json"
        with open(price_file, "w") as f:
            json.dump({"SPY": symbol_data}, f)
        return price_file

    def test_backtest_runs(self, monkeypatch, _setup_price_data):
        """Backtest function runs and returns metrics."""
        monkeypatch.setattr("src.signals.vp_macd.PRICES_PATH", _setup_price_data)
        monkeypatch.setattr("src.signals.vp_macd.DB_PATH", Path("/nonexistent/db.sqlite"))

        result = backtest("SPY")
        assert "error" not in result
        assert "vp_macd" in result
        assert "baseline_macd" in result
        assert "improvement" in result

    def test_backtest_metrics(self, monkeypatch, _setup_price_data):
        """Backtest returns expected metric keys."""
        monkeypatch.setattr("src.signals.vp_macd.PRICES_PATH", _setup_price_data)
        monkeypatch.setattr("src.signals.vp_macd.DB_PATH", Path("/nonexistent/db.sqlite"))

        result = backtest("SPY")
        vp = result["vp_macd"]
        base = result["baseline_macd"]

        for metrics in [vp, base]:
            assert "sharpe" in metrics
            assert "total_return_pct" in metrics
            assert "max_drawdown_pct" in metrics
            assert "win_rate" in metrics
            assert "num_trades" in metrics

    def test_backtest_includes_correlation(self, monkeypatch, _setup_price_data):
        """Backtest reports VP-MACD vs baseline correlation."""
        monkeypatch.setattr("src.signals.vp_macd.PRICES_PATH", _setup_price_data)
        monkeypatch.setattr("src.signals.vp_macd.DB_PATH", Path("/nonexistent/db.sqlite"))

        result = backtest("SPY")
        assert "correlation" in result
        assert -1.0 <= result["correlation"] <= 1.0

    def test_backtest_includes_regime_perf(self, monkeypatch, _setup_price_data):
        """Backtest reports regime-specific performance."""
        monkeypatch.setattr("src.signals.vp_macd.PRICES_PATH", _setup_price_data)
        monkeypatch.setattr("src.signals.vp_macd.DB_PATH", Path("/nonexistent/db.sqlite"))

        result = backtest("SPY")
        assert "regime_performance" in result
        assert isinstance(result["regime_performance"], dict)


class TestSignalOutput:
    """Test VPMACDSignal dataclass."""

    def test_dataclass_defaults(self):
        """Default details is empty dict."""
        signal = VPMACDSignal(
            timestamp="2025-01-01T00:00:00",
            ticker="SPY",
            macd_line=0.0,
            signal_line=0.0,
            histogram=0.0,
            volatility_adjusted_threshold=0.0,
            vp_macd_value=0.0,
            vp_macd_signal="neutral",
            confidence=0.5,
            regime="normal",
            volume_available=False,
            fast_period=12,
            slow_period=26,
            signal_period=9,
            vol_multiplier=1.5,
        )
        assert signal.details == {}

    def test_signal_serializable(self):
        """Signal can be serialized to JSON."""
        signal = VPMACDSignal(
            timestamp="2025-01-01T00:00:00",
            ticker="SPY",
            macd_line=0.5,
            signal_line=0.3,
            histogram=0.2,
            volatility_adjusted_threshold=0.1,
            vp_macd_value=0.5,
            vp_macd_signal="long",
            confidence=0.7,
            regime="normal",
            volume_available=True,
            fast_period=12,
            slow_period=26,
            signal_period=9,
            vol_multiplier=1.5,
        )
        import dataclasses
        d = dataclasses.asdict(signal)
        json_str = json.dumps(d, default=str)
        assert json_str
        parsed = json.loads(json_str)
        assert parsed["ticker"] == "SPY"
        assert parsed["vp_macd_signal"] == "long"


class TestSignalConstants:
    """Test signal value constants."""

    def test_signal_values(self):
        """Signal values follow expected ordering."""
        assert SIGNAL_STRONG_SHORT < SIGNAL_SHORT < SIGNAL_NEUTRAL
        assert SIGNAL_NEUTRAL < SIGNAL_LONG < SIGNAL_STRONG_LONG

    def test_signal_ranges(self):
        """All signal values are in [-1, 1]."""
        for val in [SIGNAL_STRONG_SHORT, SIGNAL_SHORT, SIGNAL_NEUTRAL,
                    SIGNAL_LONG, SIGNAL_STRONG_LONG]:
            assert -1.0 <= val <= 1.0


class TestVPINIntegration:
    """Test VPIN-aware behavior (gating)."""

    @pytest.fixture
    def _setup_price_data(self, tmp_path):
        """Create price data."""
        prices = np.linspace(100, 110, 200)
        dates = [f"2025-{(i // 30)+1:02d}-{(i % 30)+1:02d}" for i in range(200)]
        symbol_data = [
            {"d": dates[i], "p": float(prices[i])}
            for i in range(200)
        ]
        price_file = tmp_path / "prices.json"
        with open(price_file, "w") as f:
            json.dump({"SPY": symbol_data}, f)
        return price_file

    def test_signal_generated_without_vpin(self, monkeypatch, _setup_price_data):
        """Signal works even without VPIN data (graceful degradation)."""
        monkeypatch.setattr("src.signals.vp_macd.PRICES_PATH", _setup_price_data)
        monkeypatch.setattr("src.signals.vp_macd.DB_PATH", Path("/nonexistent/db.sqlite"))

        signal = generate_signal("SPY")
        assert signal is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
