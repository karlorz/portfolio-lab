"""Direct tests for src.backtest.regime_allocation_backtest."""

import numpy as np
import pytest

from src.backtest import regime_allocation_backtest as rab


def _alternating(length: int, a: float, b: float) -> np.ndarray:
    values = np.resize(np.array([a, b], dtype=float), length)
    return values[:length]


def _prices_from_returns(returns: np.ndarray, start: float = 100.0) -> np.ndarray:
    return start * np.cumprod(np.r_[1.0, 1.0 + returns])


def _synthetic_prices(n_days: int = 360) -> dict[str, np.ndarray]:
    idx = np.arange(n_days - 1)
    spy_ret = 0.0005 + 0.0010 * np.sin(idx / 7.0)
    gld_ret = 0.0003 + 0.0008 * np.cos(idx / 11.0)
    tlt_ret = 0.0002 + 0.0006 * np.sin(idx / 13.0)
    return {
        "SPY": _prices_from_returns(spy_ret, start=100.0),
        "GLD": _prices_from_returns(gld_ret, start=120.0),
        "TLT": _prices_from_returns(tlt_ret, start=90.0),
    }


class TestClassifyRegimeSimple:
    def test_short_history_defaults_to_normal(self):
        returns = np.zeros(251)
        assert rab.classify_regime_simple(returns) == "normal"

    def test_high_vol_boundary(self):
        returns = np.r_[
            _alternating(189, 0.001, -0.001),
            _alternating(63, 0.020, -0.020),
        ]
        assert rab.classify_regime_simple(returns) == "high_vol"

    def test_crisis_requires_high_vol_and_negative_recent_mean(self):
        returns = np.r_[
            _alternating(189, 0.001, -0.001),
            _alternating(42, 0.020, -0.020),
            _alternating(21, -0.030, 0.005),
        ]
        assert rab.classify_regime_simple(returns) == "crisis"

    def test_low_vol_boundary(self):
        returns = np.r_[
            _alternating(189, 0.012, -0.012),
            _alternating(63, 0.001, -0.001),
        ]
        assert rab.classify_regime_simple(returns) == "low_vol"

    def test_recovery_requires_below_median_vol_and_positive_momentum(self):
        returns = np.r_[
            _alternating(189, 0.006, -0.006),
            _alternating(63, 0.0005, 0.0115),
        ]
        assert rab.classify_regime_simple(returns) == "recovery"


class TestBacktestAllocation:
    def test_insufficient_spy_history_returns_empty_result(self):
        prices = {
            "SPY": np.ones(299) * 100.0,
            "GLD": np.ones(299) * 100.0,
            "TLT": np.ones(299) * 100.0,
        }
        result = rab.backtest_allocation(prices, allocation_map={}, default_alloc=rab.DEFAULT_ALLOCATION)
        assert result == {}

    def test_missing_asset_series_does_not_crash(self):
        prices = {
            "SPY": np.linspace(100.0, 130.0, 360),
            "GLD": np.array([]),
            "TLT": np.linspace(90.0, 100.0, 360),
        }
        result = rab.backtest_allocation(prices, allocation_map={}, default_alloc=rab.DEFAULT_ALLOCATION)

        assert result["total_return"] == pytest.approx(0.0)
        assert result["regime_counts"] == {}

    def test_allocation_weights_are_normalized(self, monkeypatch):
        prices = _synthetic_prices()
        monkeypatch.setattr(rab, "classify_regime_simple", lambda returns: "normal")

        normalized = {"normal": {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}}
        unnormalized = {"normal": {"SPY": 0.92, "GLD": 0.76, "TLT": 0.32}}

        expected = rab.backtest_allocation(prices, normalized, rab.DEFAULT_ALLOCATION)
        actual = rab.backtest_allocation(prices, unnormalized, rab.DEFAULT_ALLOCATION)

        for key in ("cagr", "vol", "sharpe", "max_dd", "total_return", "sortino", "calmar"):
            assert actual[key] == pytest.approx(expected[key])

    def test_result_metric_shape(self):
        prices = _synthetic_prices()
        result = rab.backtest_allocation(
            prices,
            allocation_map=rab.REGIME_ALLOCATIONS,
            default_alloc=rab.DEFAULT_ALLOCATION,
        )

        assert set(result) == {
            "cagr",
            "vol",
            "sharpe",
            "max_dd",
            "total_return",
            "sortino",
            "calmar",
            "regime_counts",
        }
        for key in ("cagr", "vol", "sharpe", "max_dd", "total_return", "sortino", "calmar"):
            assert isinstance(result[key], float)
        assert isinstance(result["regime_counts"], dict)
        assert sum(result["regime_counts"].values()) == len(prices["SPY"]) - 1
