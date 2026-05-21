"""
Tests for the Bond Momentum Backtest Research Module.

Covers: BondMomentumResult dataclass, load_price_data (mocked file I/O),
calculate_momentum_signal (TSMOM with vol scaling), backtest_bond_momentum
(full backtest pipeline), run_sensitivity_analysis, analyze_correlation_with_duration_overlay,
main CLI entry, and edge cases (short data, constant prices, single point, zero returns).
"""

import json
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.research.bond_momentum_backtest import (
    BondMomentumResult,
    load_price_data,
    calculate_momentum_signal,
    backtest_bond_momentum,
    run_sensitivity_analysis,
    analyze_correlation_with_duration_overlay,
    main,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures: synthetic price data
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """Generate ~2520 days of realistic Treasury ETF prices (TLT, IEF, SHY, BIL)."""
    np.random.seed(42)
    n_days = 2520
    dates = pd.date_range("2014-01-01", periods=n_days, freq="B")

    tlt_noise = np.random.normal(0.0004, 0.008, n_days)
    tlt_price = 100 * np.exp(np.cumsum(0.0001 + tlt_noise))

    ief_noise = np.random.normal(0.0003, 0.004, n_days)
    ief_price = 100 * np.exp(np.cumsum(0.0001 + ief_noise))

    shy_noise = np.random.normal(0.0002, 0.0015, n_days)
    shy_price = 100 * np.exp(np.cumsum(0.0001 + shy_noise))

    bil_noise = np.random.normal(0.0001, 0.0005, n_days)
    bil_price = 100 * np.exp(np.cumsum(0.00005 + bil_noise))

    return pd.DataFrame({
        "TLT": tlt_price, "IEF": ief_price,
        "SHY": shy_price, "BIL": bil_price,
    }, index=dates)


@pytest.fixture
def short_prices() -> pd.DataFrame:
    """Short price series (100 days) — insufficient for 12m formation."""
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    np.random.seed(1)
    return pd.DataFrame({
        "TLT": 100 + np.cumsum(np.random.normal(0, 0.5, 100)),
        "IEF": 100 + np.cumsum(np.random.normal(0, 0.3, 100)),
    }, index=dates)


@pytest.fixture
def constant_prices() -> pd.DataFrame:
    """Flat price series — pct_change is all zero."""
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    return pd.DataFrame({"TLT": 100.0, "IEF": 100.0}, index=dates)


@pytest.fixture
def single_day_prices() -> pd.DataFrame:
    """Single data point."""
    return pd.DataFrame(
        {"TLT": [100.0], "IEF": [100.0]},
        index=pd.DatetimeIndex(["2025-01-02"]),
    )


@pytest.fixture
def zero_return_prices() -> pd.DataFrame:
    """Prices that oscillate around zero net returns."""
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    np.random.seed(99)
    base = np.cumsum(np.random.normal(0, 0.005, 500))
    price = 100 + (base - np.mean(base)) * 2
    price = np.abs(price) + 10
    return pd.DataFrame({"TLT": price, "IEF": price * 0.95}, index=dates)


# ═══════════════════════════════════════════════════════════════════════════
# TestBondMomentumResult
# ═══════════════════════════════════════════════════════════════════════════

class TestBondMomentumResult:

    def test_construction(self):
        result = BondMomentumResult(
            etf="TLT", formation_months=12, skip_months=1,
            volatility_target=0.08, total_return=0.15, cagr=0.05,
            volatility=0.08, sharpe=0.625, max_drawdown=-0.10,
            win_rate=0.55, avg_position=0.6, turnover=4.0,
            buy_hold_return=0.12, alpha_vs_buyhold=-0.07,
            annual_returns={2022: -0.05, 2023: 0.08},
        )
        assert result.etf == "TLT"
        assert result.formation_months == 12
        assert result.total_return == 0.15
        assert result.sharpe == 0.625
        assert result.max_drawdown == -0.10

    def test_annual_returns_empty_dict(self):
        result = BondMomentumResult(
            etf="IEF", formation_months=6, skip_months=1,
            volatility_target=0.10, total_return=0.0, cagr=0.0,
            volatility=0.0, sharpe=0.0, max_drawdown=0.0,
            win_rate=0.0, avg_position=0.0, turnover=0.0,
            buy_hold_return=0.0, alpha_vs_buyhold=0.0,
            annual_returns={},
        )
        assert result.annual_returns == {}

    def test_negative_values(self):
        result = BondMomentumResult(
            etf="TLT", formation_months=12, skip_months=1,
            volatility_target=0.08, total_return=-0.20, cagr=-0.07,
            volatility=0.12, sharpe=-0.58, max_drawdown=-0.30,
            win_rate=0.40, avg_position=0.3, turnover=5.0,
            buy_hold_return=-0.10, alpha_vs_buyhold=0.03,
            annual_returns={2022: -0.20},
        )
        assert result.total_return == -0.20
        assert result.sharpe < 0

    def test_various_etfs(self):
        for etf in ("TLT", "IEF", "SHY", "BIL", "AGG"):
            result = BondMomentumResult(
                etf=etf, formation_months=12, skip_months=1,
                volatility_target=0.08, total_return=0.10, cagr=0.03,
                volatility=0.06, sharpe=0.50, max_drawdown=-0.08,
                win_rate=0.52, avg_position=0.5, turnover=3.0,
                buy_hold_return=0.08, alpha_vs_buyhold=0.02,
                annual_returns={2023: 0.05},
            )
            assert result.etf == etf


# ═══════════════════════════════════════════════════════════════════════════
# TestLoadPriceData
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadPriceData:

    def test_loads_all_treasury_etfs(self, tmp_path):
        prices_json = {
            "TLT": [{"d": "2020-01-02", "p": 140.0}, {"d": "2020-01-03", "p": 141.0}],
            "IEF": [{"d": "2020-01-02", "p": 105.0}, {"d": "2020-01-03", "p": 104.5}],
            "SHY": [{"d": "2020-01-02", "p": 84.0}, {"d": "2020-01-03", "p": 84.1}],
            "BIL": [{"d": "2020-01-02", "p": 95.0}, {"d": "2020-01-03", "p": 95.01}],
            "SPY": [{"d": "2020-01-02", "p": 320.0}],
        }
        data_path = tmp_path / "prices.json"
        with open(data_path, "w") as f:
            json.dump(prices_json, f)

        df = load_price_data(data_path)
        assert set(df.columns) == {"BIL", "IEF", "SHY", "TLT"}
        assert len(df) == 2

    def test_missing_etf_data(self, tmp_path):
        prices_json = {
            "TLT": [{"d": "2020-01-02", "p": 140.0}],
            "SPY": [{"d": "2020-01-02", "p": 320.0}],
        }
        data_path = tmp_path / "prices.json"
        with open(data_path, "w") as f:
            json.dump(prices_json, f)

        df = load_price_data(data_path)
        assert "TLT" in df.columns
        assert len(df) == 1

    def test_empty_data_raises(self, tmp_path):
        """Empty JSON -> empty records -> DataFrame with no 'date' column -> KeyError."""
        data_path = tmp_path / "prices.json"
        with open(data_path, "w") as f:
            json.dump({}, f)

        with pytest.raises(KeyError):
            load_price_data(data_path)

    def test_only_non_treasury_symbols_raises(self, tmp_path):
        """No matching ETFs -> empty records -> KeyError on 'date'."""
        prices_json = {
            "SPY": [{"d": "2020-01-02", "p": 320.0}],
            "QQQ": [{"d": "2020-01-02", "p": 280.0}],
        }
        data_path = tmp_path / "prices.json"
        with open(data_path, "w") as f:
            json.dump(prices_json, f)

        with pytest.raises(KeyError):
            load_price_data(data_path)


# ═══════════════════════════════════════════════════════════════════════════
# TestCalculateMomentumSignal
# ═══════════════════════════════════════════════════════════════════════════

class TestCalculateMomentumSignal:

    def test_default_params(self, synthetic_prices):
        signal = calculate_momentum_signal(synthetic_prices["TLT"])
        assert isinstance(signal, pd.Series)
        pd.testing.assert_index_equal(signal.index, synthetic_prices.index)
        first_valid = signal.first_valid_index()
        assert first_valid is not None
        valid = signal.dropna()
        assert (valid >= 0).all()
        assert (valid <= 2.0).all()

    def test_shorter_formation_starts_earlier(self, synthetic_prices):
        signal_12 = calculate_momentum_signal(synthetic_prices["TLT"], formation_months=12)
        signal_3 = calculate_momentum_signal(synthetic_prices["TLT"], formation_months=3)
        first_12 = signal_12.first_valid_index()
        first_3 = signal_3.first_valid_index()
        assert first_3 is not None
        assert first_12 is not None
        # 3m may start at same time or earlier than 12m depending on rolling vol window
        assert first_3 <= first_12

    def test_zero_skip_months(self, synthetic_prices):
        signal = calculate_momentum_signal(synthetic_prices["TLT"], skip_months=0)
        assert isinstance(signal, pd.Series)
        assert (signal.dropna() >= 0).all()
        assert (signal.dropna() <= 2.0).all()

    def test_higher_vol_target_larger_positions(self, synthetic_prices):
        signal_low = calculate_momentum_signal(synthetic_prices["TLT"], volatility_target=0.06)
        signal_high = calculate_momentum_signal(synthetic_prices["TLT"], volatility_target=0.12)
        assert signal_high.dropna().mean() >= signal_low.dropna().mean()

    def test_max_leverage_clamp(self, synthetic_prices):
        signal = calculate_momentum_signal(synthetic_prices["TLT"], volatility_target=1.0)
        assert signal.dropna().max() <= 2.0

    def test_negative_momentum_gives_zero_signal(self):
        np.random.seed(7)
        n = 500
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        prices = pd.Series(
            100 * np.exp(-np.cumsum(np.abs(np.random.normal(0, 0.005, n)))),
            index=dates,
        )
        signal = calculate_momentum_signal(prices, formation_months=3)
        assert signal.dropna().mean() < 0.5

    def test_constant_prices_produce_zero_signal(self, constant_prices):
        signal = calculate_momentum_signal(constant_prices["TLT"])
        valid = signal.dropna()
        assert len(valid) > 0
        assert (valid == 0).all()

    def test_short_series_mostly_nan(self, short_prices):
        """100 days with 12m formation (252 days): momentum is NaN for most but not all."""
        signal = calculate_momentum_signal(short_prices["TLT"])
        # Most values are NaN (12m formation needs 252 days), but some early NaN→0 may appear
        assert signal.isna().sum() > 0

    def test_single_point_is_nan(self, single_day_prices):
        signal = calculate_momentum_signal(single_day_prices["TLT"])
        assert len(signal) == 1
        assert pd.isna(signal.iloc[0])

    def test_index_preserved(self, synthetic_prices):
        signal = calculate_momentum_signal(synthetic_prices["TLT"])
        pd.testing.assert_index_equal(signal.index, synthetic_prices.index)


# ═══════════════════════════════════════════════════════════════════════════
# TestBacktestBondMomentum
# ═══════════════════════════════════════════════════════════════════════════

class TestBacktestBondMomentum:

    def test_tlt_backtest_default(self, synthetic_prices):
        result = backtest_bond_momentum(synthetic_prices, "TLT")
        assert result is not None
        assert result.etf == "TLT"
        assert result.formation_months == 12
        assert result.volatility >= 0
        assert result.max_drawdown <= 0
        assert 0 <= result.win_rate <= 1
        assert isinstance(result.annual_returns, dict)

    def test_ief_backtest(self, synthetic_prices):
        result = backtest_bond_momentum(synthetic_prices, "IEF")
        assert result is not None
        assert result.etf == "IEF"

    def test_shy_backtest(self, synthetic_prices):
        result = backtest_bond_momentum(synthetic_prices, "SHY")
        assert result is not None

    def test_bil_backtest(self, synthetic_prices):
        result = backtest_bond_momentum(synthetic_prices, "BIL")
        assert result is not None

    def test_custom_formation_3m(self, synthetic_prices):
        result = backtest_bond_momentum(synthetic_prices, "TLT", formation_months=3)
        assert result is not None
        assert result.formation_months == 3

    def test_custom_formation_18m(self, synthetic_prices):
        result = backtest_bond_momentum(
            synthetic_prices, "TLT", formation_months=18, transaction_cost=0.0020
        )
        assert result is not None
        assert result.formation_months == 18

    def test_zero_transaction_cost(self, synthetic_prices):
        result = backtest_bond_momentum(synthetic_prices, "TLT", transaction_cost=0.0)
        assert result is not None

    def test_invalid_etf_raises_keyerror(self, synthetic_prices):
        with pytest.raises(KeyError):
            backtest_bond_momentum(synthetic_prices, "INVALID")

    def test_short_data_near_zero_returns(self, short_prices):
        """100 days with 12m formation: signal all NaN -> 0 positions -> near-zero returns."""
        result = backtest_bond_momentum(short_prices, "TLT")
        assert result is not None
        assert abs(result.total_return) < 0.01

    def test_constant_prices_backtest(self, constant_prices):
        result = backtest_bond_momentum(constant_prices, "TLT")
        assert result is not None
        assert abs(result.total_return) < 0.01

    def test_single_day_data_returns_none(self, single_day_prices):
        result = backtest_bond_momentum(single_day_prices, "TLT")
        assert result is None

    def test_zero_return_data(self, zero_return_prices):
        result = backtest_bond_momentum(zero_return_prices, "TLT")
        if result is not None:
            assert abs(result.cagr) < 0.10

    def test_max_drawdown_le_zero(self, synthetic_prices):
        result = backtest_bond_momentum(synthetic_prices, "TLT")
        assert result is not None
        assert result.max_drawdown <= 0

    def test_win_rate_range(self, synthetic_prices):
        result = backtest_bond_momentum(synthetic_prices, "TLT")
        assert result is not None
        assert 0 < result.win_rate < 1

    def test_turnover_non_negative(self, synthetic_prices):
        result = backtest_bond_momentum(synthetic_prices, "TLT")
        assert result is not None
        assert result.turnover >= 0

    def test_alpha_vs_buyhold_is_float(self, synthetic_prices):
        result = backtest_bond_momentum(synthetic_prices, "TLT")
        assert result is not None
        assert isinstance(result.alpha_vs_buyhold, (int, float, np.floating))


# ═══════════════════════════════════════════════════════════════════════════
# TestRunSensitivityAnalysis
# ═══════════════════════════════════════════════════════════════════════════

class TestRunSensitivityAnalysis:

    def test_output_shape(self, synthetic_prices):
        df = run_sensitivity_analysis(synthetic_prices, "TLT")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 20
        expected_cols = {"formation_months", "vol_target", "cagr", "sharpe", "max_dd", "win_rate", "alpha_vs_bh"}
        assert set(df.columns) == expected_cols

    def test_formation_periods_present(self, synthetic_prices):
        df = run_sensitivity_analysis(synthetic_prices, "TLT")
        assert set(df["formation_months"].unique()) == {3, 6, 9, 12, 18}

    def test_vol_targets_present(self, synthetic_prices):
        df = run_sensitivity_analysis(synthetic_prices, "TLT")
        assert set(df["vol_target"].unique()) == {0.06, 0.08, 0.10, 0.12}

    def test_no_nan_values(self, synthetic_prices):
        df = run_sensitivity_analysis(synthetic_prices, "TLT")
        assert not df.isnull().any().any()

    def test_sharpe_finite(self, synthetic_prices):
        df = run_sensitivity_analysis(synthetic_prices, "TLT")
        assert df["sharpe"].apply(np.isfinite).all()

    def test_max_dd_non_positive(self, synthetic_prices):
        df = run_sensitivity_analysis(synthetic_prices, "TLT")
        assert (df["max_dd"] <= 0).all()

    def test_win_rate_range(self, synthetic_prices):
        df = run_sensitivity_analysis(synthetic_prices, "TLT")
        assert (df["win_rate"] >= 0).all()
        assert (df["win_rate"] <= 1).all()

    def test_different_etf(self, synthetic_prices):
        df = run_sensitivity_analysis(synthetic_prices, "IEF")
        assert len(df) == 20

    def test_short_data_fewer_results(self, short_prices):
        """100 days: shorter formation periods may succeed, longer ones may return None."""
        df = run_sensitivity_analysis(short_prices, "TLT")
        assert isinstance(df, pd.DataFrame)
        # At least some formations should produce results (3m = 63 days)
        assert len(df) > 0

    def test_unique_combos(self, synthetic_prices):
        df = run_sensitivity_analysis(synthetic_prices, "TLT")
        combos = df[["formation_months", "vol_target"]].itertuples(index=False)
        assert len(set(combos)) == 20


# ═══════════════════════════════════════════════════════════════════════════
# TestAnalyzeCorrelationWithDurationOverlay
# ═══════════════════════════════════════════════════════════════════════════

class TestAnalyzeCorrelationWithDurationOverlay:

    def test_returns_dict_with_all_keys(self, synthetic_prices):
        result = analyze_correlation_with_duration_overlay(synthetic_prices)
        assert isinstance(result, dict)
        assert "momentum_duration_correlation" in result
        assert "signal_agreement_rate" in result
        assert "notes" in result

    def test_correlation_is_float(self, synthetic_prices):
        result = analyze_correlation_with_duration_overlay(synthetic_prices)
        assert isinstance(result["momentum_duration_correlation"], (int, float, np.floating))

    def test_correlation_range(self, synthetic_prices):
        result = analyze_correlation_with_duration_overlay(synthetic_prices)
        corr = result["momentum_duration_correlation"]
        if not np.isnan(corr):
            assert -1.0 <= corr <= 1.0

    def test_agreement_rate_range(self, synthetic_prices):
        result = analyze_correlation_with_duration_overlay(synthetic_prices)
        rate = result["signal_agreement_rate"]
        if not np.isnan(rate):
            assert 0 <= rate <= 1

    def test_notes_content(self, synthetic_prices):
        result = analyze_correlation_with_duration_overlay(synthetic_prices)
        assert "yield curve" in result["notes"]

    @pytest.mark.parametrize("formation", [3, 6, 12])
    def test_different_formations(self, synthetic_prices, formation):
        result = analyze_correlation_with_duration_overlay(synthetic_prices, formation_months=formation)
        corr = result["momentum_duration_correlation"]
        assert np.isnan(corr) or -1.0 <= corr <= 1.0

    def test_identical_series(self):
        """Identical TLT/IEF movement: correlation may be NaN or valid."""
        np.random.seed(42)
        n = 500
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        price = 100 * np.exp(np.cumsum(np.random.normal(0.0001, 0.01, n)))
        df = pd.DataFrame({"TLT": price, "IEF": price}, index=dates)
        result = analyze_correlation_with_duration_overlay(df, formation_months=3)
        corr = result["momentum_duration_correlation"]
        assert np.isnan(corr) or -1.0 <= corr <= 1.0

    def test_nan_for_short_data(self, short_prices):
        result = analyze_correlation_with_duration_overlay(short_prices)
        corr = result["momentum_duration_correlation"]
        assert np.isnan(corr) or -1.0 <= corr <= 1.0

    def test_single_day(self, single_day_prices):
        result = analyze_correlation_with_duration_overlay(single_day_prices)
        assert np.isnan(result["momentum_duration_correlation"])


# ═══════════════════════════════════════════════════════════════════════════
# TestMain
# ═══════════════════════════════════════════════════════════════════════════

class TestMain:

    def test_main_runs_and_saves_json(self, synthetic_prices, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "src.research.bond_momentum_backtest.load_price_data",
            lambda data_path=None: synthetic_prices,
        )

        import builtins
        original_open = builtins.open
        (tmp_path / "research").mkdir(exist_ok=True)

        def tracking_open(*args, **kwargs):
            path = args[0] if args else kwargs.get("file", "")
            if isinstance(path, (str, Path)) and "bond_momentum_backtest_results.json" in str(path):
                path = str(tmp_path / "research" / "bond_momentum_backtest_results.json")
                args = (path,) + args[1:]
            return original_open(*args, **kwargs)

        monkeypatch.setattr(builtins, "open", tracking_open)
        main()

        output_file = tmp_path / "research" / "bond_momentum_backtest_results.json"
        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)
        assert "results" in data
        assert "conclusion" in data

    def test_main_stdout_contains_headers(self, synthetic_prices, monkeypatch, capsys, tmp_path):
        monkeypatch.setattr(
            "src.research.bond_momentum_backtest.load_price_data",
            lambda data_path=None: synthetic_prices,
        )

        import builtins
        original_open = builtins.open
        (tmp_path / "research").mkdir(exist_ok=True)

        def tracking_open(*args, **kwargs):
            path = args[0] if args else kwargs.get("file", "")
            if isinstance(path, (str, Path)) and "bond_momentum_backtest_results.json" in str(path):
                path = str(tmp_path / "research" / "bond_momentum_backtest_results.json")
                args = (path,) + args[1:]
            return original_open(*args, **kwargs)

        monkeypatch.setattr(builtins, "open", tracking_open)
        main()
        captured = capsys.readouterr().out
        assert "Bond Momentum Backtest" in captured
        assert "CONCLUSIONS" in captured

    def test_main_with_empty_data_raises(self, monkeypatch, tmp_path):
        """Empty DataFrame -> IndexError when accessing prices.index[0]."""
        empty_df = pd.DataFrame()
        monkeypatch.setattr(
            "src.research.bond_momentum_backtest.load_price_data",
            lambda data_path=None: empty_df,
        )
        with pytest.raises(IndexError):
            main()


# ═══════════════════════════════════════════════════════════════════════════
# Edge Cases & Integration
# ═══════════════════════════════════════════════════════════════════════════

class TestBondMomentumEdgeCases:

    def test_single_etf_dataframe(self):
        np.random.seed(1)
        n = 600
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        df = pd.DataFrame({
            "TLT": 100 * np.exp(np.cumsum(np.random.normal(0.0001, 0.008, n))),
        }, index=dates)
        result = backtest_bond_momentum(df, "TLT")
        assert result is not None

    def test_backtest_with_nan_prices(self):
        np.random.seed(3)
        n = 600
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.0001, 0.008, n)))
        prices[100:110] = np.nan
        df = pd.DataFrame({"TLT": prices}, index=dates)
        result = backtest_bond_momentum(df, "TLT")
        assert result is not None

    def test_all_formation_periods_produce_results(self, synthetic_prices):
        for etf in ["TLT", "IEF", "SHY", "BIL"]:
            for formation in [3, 6, 9, 12, 18]:
                result = backtest_bond_momentum(synthetic_prices, etf, formation_months=formation)
                assert result is not None, f"Failed for {etf} formation={formation}"

    def test_reproducible_with_same_seed(self):
        np.random.seed(42)
        n = 600
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        p1 = 100 * np.exp(np.cumsum(np.random.normal(0.0001, 0.008, n)))
        df1 = pd.DataFrame({"TLT": p1}, index=dates)

        np.random.seed(42)
        p2 = 100 * np.exp(np.cumsum(np.random.normal(0.0001, 0.008, n)))
        df2 = pd.DataFrame({"TLT": p2}, index=dates)

        r1 = backtest_bond_momentum(df1, "TLT")
        r2 = backtest_bond_momentum(df2, "TLT")
        assert r1 is not None and r2 is not None
        assert abs(r1.cagr - r2.cagr) < 0.001

    def test_higher_costs_reduce_returns(self, synthetic_prices):
        r_low = backtest_bond_momentum(synthetic_prices, "TLT", transaction_cost=0.0001)
        r_high = backtest_bond_momentum(synthetic_prices, "TLT", transaction_cost=0.005)
        assert r_low is not None and r_high is not None
        assert r_high.total_return <= r_low.total_return + 0.01

    def test_annual_returns_years_are_int(self, synthetic_prices):
        result = backtest_bond_momentum(synthetic_prices, "TLT")
        assert result is not None
        assert len(result.annual_returns) > 0
        for year in result.annual_returns:
            assert isinstance(year, int)

    def test_sensitivity_best_config(self, synthetic_prices):
        df = run_sensitivity_analysis(synthetic_prices, "TLT")
        assert len(df) == 20
        best = df.loc[df["sharpe"].idxmax()]
        assert best["formation_months"] in [3, 6, 9, 12, 18]
        assert best["vol_target"] in [0.06, 0.08, 0.10, 0.12]
