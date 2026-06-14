"""Direct tests for src.backtest.rolling_vol.

The helper is used by hot paths in real_data_backtest.py,
combined_overlay_backtest.py, and combined_regime_alloc_vol_target.py.
"""

import math

import numpy as np
import pytest

from src.backtest.rolling_vol import precomputed_rolling_volatility


def _naive_legacy_volatility(
    returns,
    *,
    window: int,
    fallback_vol: float,
    warmup_std_min_index: int,
    annualization_factor: float = 252.0,
):
    """Reference implementation matching the legacy per-window np.std loop."""
    annualizer = math.sqrt(annualization_factor)
    values = [float(value) for value in returns]
    vols = []
    for i in range(len(values)):
        if i < window:
            if i >= warmup_std_min_index:
                window_values = np.asarray(values[: i + 1], dtype=float)
                vols.append(float(np.std(window_values) * annualizer))
            else:
                vols.append(fallback_vol)
        else:
            window_values = np.asarray(values[i - window : i], dtype=float)
            vols.append(float(np.std(window_values) * annualizer))
    return vols


def test_precomputed_rolling_volatility_matches_naive_np_std_loop():
    rng = np.random.default_rng(20260614)
    returns = rng.normal(loc=0.0003, scale=0.012, size=200)

    actual = precomputed_rolling_volatility(
        returns,
        window=21,
        fallback_vol=0.2,
        warmup_std_min_index=5,
    )
    expected = _naive_legacy_volatility(
        returns,
        window=21,
        fallback_vol=0.2,
        warmup_std_min_index=5,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_precomputed_rolling_volatility_uses_population_standard_deviation():
    returns = [0.01, -0.02, 0.04]

    actual = precomputed_rolling_volatility(
        returns,
        window=10,
        fallback_vol=99.0,
        warmup_std_min_index=0,
        annualization_factor=1.0,
    )

    population_std = float(np.std(returns, ddof=0))
    sample_std = float(np.std(returns, ddof=1))
    assert actual[-1] == pytest.approx(population_std)
    assert actual[-1] != pytest.approx(sample_std)


def test_precomputed_rolling_volatility_empty_input_returns_empty_list():
    assert precomputed_rolling_volatility(
        [],
        window=21,
        fallback_vol=0.2,
        warmup_std_min_index=5,
    ) == []


def test_precomputed_rolling_volatility_window_larger_than_returns_uses_warmup_contract():
    returns = [0.01, -0.02, 0.03, 0.04]

    actual = precomputed_rolling_volatility(
        returns,
        window=30,
        fallback_vol=0.2,
        warmup_std_min_index=2,
        annualization_factor=1.0,
    )

    expected = _naive_legacy_volatility(
        returns,
        window=30,
        fallback_vol=0.2,
        warmup_std_min_index=2,
        annualization_factor=1.0,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert actual[:2] == [0.2, 0.2]


def test_precomputed_rolling_volatility_window_one_is_zero_after_warmup():
    actual = precomputed_rolling_volatility(
        [0.01, -0.02, 0.03],
        window=1,
        fallback_vol=0.2,
        warmup_std_min_index=0,
        annualization_factor=1.0,
    )

    assert actual == [0.0, 0.0, 0.0]


def test_precomputed_rolling_volatility_nan_matches_np_std_window_contract():
    returns = [0.01, np.nan, 0.03, 0.04, 0.05]

    actual = precomputed_rolling_volatility(
        returns,
        window=2,
        fallback_vol=0.2,
        warmup_std_min_index=0,
        annualization_factor=1.0,
    )
    expected = _naive_legacy_volatility(
        returns,
        window=2,
        fallback_vol=0.2,
        warmup_std_min_index=0,
        annualization_factor=1.0,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
    assert np.isnan(actual[1])
    assert np.isnan(actual[2])
    assert np.isnan(actual[3])
    assert actual[4] == pytest.approx(0.005)


def test_precomputed_rolling_volatility_rejects_non_positive_window():
    with pytest.raises(ValueError, match="window must be positive"):
        precomputed_rolling_volatility(
            [0.01, -0.02],
            window=0,
            fallback_vol=0.2,
            warmup_std_min_index=0,
        )
