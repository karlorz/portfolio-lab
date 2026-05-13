#!/usr/bin/env python3
"""
Tests for factor_data_fetcher.py — constants, FactorDataFetcher, synthetic
data generation, factor statistics, load/cache logic, and CLI.
"""
import sys
import os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.data.factor_data_fetcher import (
    FF_URLS,
    AQR_URLS,
    FactorDataFetcher,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fetcher(tmp_path=None):
    fetcher = FactorDataFetcher.__new__(FactorDataFetcher)
    fetcher.data_dir = tmp_path or Path("/tmp/test_factors")
    fetcher.cache_file = fetcher.data_dir / "factor_returns.csv"
    return fetcher


# ---------------------------------------------------------------------------
# Constants Tests
# ---------------------------------------------------------------------------

class TestConstants:

    def test_ff_urls(self):
        assert 'daily_5_factors' in FF_URLS
        assert 'monthly_5_factors' in FF_URLS
        assert 'momentum' in FF_URLS

    def test_aqr_urls(self):
        assert 'quality' in AQR_URLS
        assert 'betting_against_beta' in AQR_URLS

    def test_ff_urls_valid(self):
        for key, url in FF_URLS.items():
            assert url.startswith('https://')


# ---------------------------------------------------------------------------
# generate_synthetic_factor_data Tests
# ---------------------------------------------------------------------------

class TestSyntheticFactorData:

    def test_returns_dataframe(self):
        fetcher = _make_fetcher()
        df = fetcher.generate_synthetic_factor_data()
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self):
        fetcher = _make_fetcher()
        df = fetcher.generate_synthetic_factor_data()
        assert 'Mkt-RF' in df.columns
        assert 'SMB' in df.columns
        assert 'HML' in df.columns
        assert 'RMW' in df.columns
        assert 'CMA' in df.columns
        assert 'RF' in df.columns
        assert 'UMD' in df.columns

    def test_date_index(self):
        fetcher = _make_fetcher()
        df = fetcher.generate_synthetic_factor_data()
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == 'Date'

    def test_positive_rows(self):
        fetcher = _make_fetcher()
        df = fetcher.generate_synthetic_factor_data()
        assert len(df) > 1000

    def test_custom_start_date(self):
        fetcher = _make_fetcher()
        df = fetcher.generate_synthetic_factor_data(start_date='2020-01-01')
        assert df.index[0].year == 2020

    def test_deterministic(self):
        fetcher = _make_fetcher()
        df1 = fetcher.generate_synthetic_factor_data()
        df2 = fetcher.generate_synthetic_factor_data()
        pd.testing.assert_frame_equal(df1, df2)

    def test_rf_low_volatility(self):
        fetcher = _make_fetcher()
        df = fetcher.generate_synthetic_factor_data()
        assert df['RF'].std() < 0.001

    def test_mkt_rf_higher_vol_than_rf(self):
        fetcher = _make_fetcher()
        df = fetcher.generate_synthetic_factor_data()
        assert df['Mkt-RF'].std() > df['RF'].std() * 10


# ---------------------------------------------------------------------------
# get_factor_stats Tests
# ---------------------------------------------------------------------------

class TestGetFactorStats:

    def test_returns_dataframe(self):
        fetcher = _make_fetcher()
        df = fetcher.generate_synthetic_factor_data()
        stats = fetcher.get_factor_stats(df)
        assert isinstance(stats, pd.DataFrame)

    def test_has_expected_columns(self):
        fetcher = _make_fetcher()
        df = fetcher.generate_synthetic_factor_data()
        stats = fetcher.get_factor_stats(df)
        assert 'mean_daily' in stats.columns
        assert 'std_daily' in stats.columns
        assert 'sharpe_annual' in stats.columns
        assert 'skew' in stats.columns
        assert 'kurtosis' in stats.columns
        assert 'max_dd' in stats.columns

    def test_all_factors_in_stats(self):
        fetcher = _make_fetcher()
        df = fetcher.generate_synthetic_factor_data()
        stats = fetcher.get_factor_stats(df)
        assert len(stats) == len(df.columns)

    def test_sharpe_finite(self):
        fetcher = _make_fetcher()
        df = fetcher.generate_synthetic_factor_data()
        stats = fetcher.get_factor_stats(df)
        # Sharpe should be finite for synthetic data
        assert all(np.isfinite(s) for s in stats['sharpe_annual'])

    def test_max_dd_negative(self):
        fetcher = _make_fetcher()
        df = fetcher.generate_synthetic_factor_data()
        stats = fetcher.get_factor_stats(df)
        assert all(s <= 0 for s in stats['max_dd'])


# ---------------------------------------------------------------------------
# load_factor_data Tests
# ---------------------------------------------------------------------------

class TestLoadFactorData:

    def test_load_from_cache(self, tmp_path):
        fetcher = _make_fetcher(tmp_path)
        # Create a cache file
        df = fetcher.generate_synthetic_factor_data()
        df.to_csv(fetcher.cache_file)
        loaded = fetcher.load_factor_data(refresh=False)
        assert isinstance(loaded, pd.DataFrame)
        assert len(loaded) == len(df)

    def test_load_missing_cache_fetches(self, tmp_path):
        fetcher = _make_fetcher(tmp_path)
        with patch.object(fetcher, 'fetch_all_factors') as mock:
            mock.return_value = fetcher.generate_synthetic_factor_data()
            loaded = fetcher.load_factor_data(refresh=False)
            assert mock.called

    def test_refresh_ignores_cache(self, tmp_path):
        fetcher = _make_fetcher(tmp_path)
        df = fetcher.generate_synthetic_factor_data()
        df.to_csv(fetcher.cache_file)
        with patch.object(fetcher, 'fetch_all_factors') as mock:
            mock.return_value = df
            fetcher.load_factor_data(refresh=True)
            assert mock.called


# ---------------------------------------------------------------------------
# fetch_all_factors Tests
# ---------------------------------------------------------------------------

class TestFetchAllFactors:

    def test_falls_back_to_synthetic(self, tmp_path):
        fetcher = _make_fetcher(tmp_path)
        with patch.object(fetcher, 'download_fama_french_daily', return_value=None):
            with patch.object(fetcher, 'download_momentum_factor', return_value=None):
                result = fetcher.fetch_all_factors()
                assert isinstance(result, pd.DataFrame)
                assert 'Mkt-RF' in result.columns

    def test_combines_ff_and_momentum(self, tmp_path):
        fetcher = _make_fetcher(tmp_path)
        ff_data = pd.DataFrame({
            'Mkt-RF': [0.01, -0.005],
            'SMB': [0.003, -0.002],
            'HML': [0.002, 0.001],
            'RMW': [0.001, 0.001],
            'CMA': [0.001, 0.000],
            'RF': [0.0001, 0.0001],
        }, index=pd.to_datetime(['2026-01-02', '2026-01-03']))
        mom_data = pd.DataFrame({
            'UMD': [0.005, -0.003],
        }, index=pd.to_datetime(['2026-01-02', '2026-01-03']))
        with patch.object(fetcher, 'download_fama_french_daily', return_value=ff_data):
            with patch.object(fetcher, 'download_momentum_factor', return_value=mom_data):
                result = fetcher.fetch_all_factors()
                assert 'UMD' in result.columns

    def test_saves_cache_on_success(self, tmp_path):
        fetcher = _make_fetcher(tmp_path)
        ff_data = pd.DataFrame({
            'Mkt-RF': [0.01], 'SMB': [0.003], 'HML': [0.002],
            'RMW': [0.001], 'CMA': [0.001], 'RF': [0.0001],
        }, index=pd.to_datetime(['2026-01-02']))
        with patch.object(fetcher, 'download_fama_french_daily', return_value=ff_data):
            with patch.object(fetcher, 'download_momentum_factor', return_value=None):
                fetcher.fetch_all_factors()
                assert fetcher.cache_file.exists()


# ---------------------------------------------------------------------------
# CLI Tests
# ---------------------------------------------------------------------------

class TestCLI:

    def test_test_flag(self, capsys):
        from src.data.factor_data_fetcher import main
        with patch("sys.argv", ["factor_data_fetcher.py", "--test"]):
            main()
        captured = capsys.readouterr()
        assert "Factor" in captured.out or "Rows" in captured.out

    def test_stats_flag(self, capsys):
        from src.data.factor_data_fetcher import main
        with patch("sys.argv", ["factor_data_fetcher.py", "--test", "--stats"]):
            main()
        captured = capsys.readouterr()
        assert "Statistics" in captured.out or "sharpe" in captured.out.lower()
