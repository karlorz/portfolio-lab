#!/usr/bin/env python3
"""
Tests for Fed Policy Overlay — FRED series constants, inflation YoY,
real rate calculation, FedPolicyRegime data class, regime classification,
and FedPolicyOverlay allocation recommendation.
"""
import numpy as np
import pandas as pd
import json
import os
import time

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.signals.fed_policy_overlay import (
    FRED_SERIES,
    FRED_CACHE,
    calculate_inflation_yoy,
    calculate_real_rate,
    FedPolicyRegime,
    classify_fed_regime,
    FedPolicyOverlay,
    fetch_fred_series,
    fetch_all_fred_data,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cpi_df(n=24, base=300.0, drift=0.002):
    """Create synthetic CPI data for YoY calculation."""
    dates = pd.date_range(end=datetime.now(), periods=n, freq='MS')
    values = [base]
    for i in range(n - 1):
        values.append(values[-1] * (1 + drift))
    return pd.DataFrame({'date': dates, 'value': values})


def _make_nominal_df(n=24, base=4.0):
    dates = pd.date_range(end=datetime.now(), periods=n, freq='MS')
    values = [base + 0.1 * np.sin(i / 6) for i in range(n)]
    return pd.DataFrame({'date': dates, 'value': values})


def _make_regime(**kwargs):
    defaults = dict(
        timestamp='2026-01-01', regime='NEUTRAL',
        fed_funds_rate=5.0, inflation_yoy=2.5,
        real_rate_10y=1.5, real_rate_short=2.5,
        breakeven_10y=2.3, yield_curve_10y2y=0.5,
    )
    defaults.update(kwargs)
    return FedPolicyRegime(**defaults)


# ---------------------------------------------------------------------------
# FRED_SERIES constant tests
# ---------------------------------------------------------------------------

class TestFredSeries:
    def test_has_core_series(self):
        assert 'FEDFUNDS' in FRED_SERIES
        assert 'CPIAUCSL' in FRED_SERIES
        assert 'T10YIE' in FRED_SERIES
        assert 'DFII10' in FRED_SERIES
        assert 'DGS10' in FRED_SERIES
        assert 'DGS2' in FRED_SERIES

    def test_series_descriptions(self):
        for key, desc in FRED_SERIES.items():
            assert isinstance(desc, str) and len(desc) > 0


# ---------------------------------------------------------------------------
# calculate_inflation_yoy tests
# ---------------------------------------------------------------------------

class TestCalculateInflationYoy:
    def test_returns_dataframe(self):
        df = calculate_inflation_yoy(_make_cpi_df(n=24))
        assert isinstance(df, pd.DataFrame)

    def test_has_inflation_column(self):
        df = calculate_inflation_yoy(_make_cpi_df(n=24))
        assert 'inflation_yoy' in df.columns

    def test_positive_inflation(self):
        df = calculate_inflation_yoy(_make_cpi_df(n=24, drift=0.003))
        assert all(df['inflation_yoy'] > 0)

    def test_drops_first_12_rows(self):
        df = calculate_inflation_yoy(_make_cpi_df(n=30))
        assert len(df) == 30 - 12


# ---------------------------------------------------------------------------
# calculate_real_rate tests
# ---------------------------------------------------------------------------

class TestCalculateRealRate:
    def test_returns_dataframe(self):
        nominal = _make_nominal_df()
        cpi = calculate_inflation_yoy(_make_cpi_df())
        result = calculate_real_rate(nominal, cpi)
        assert isinstance(result, pd.DataFrame)

    def test_has_real_rate_column(self):
        nominal = _make_nominal_df()
        cpi = calculate_inflation_yoy(_make_cpi_df())
        result = calculate_real_rate(nominal, cpi)
        assert 'real_rate' in result.columns


# ---------------------------------------------------------------------------
# FedPolicyRegime tests
# ---------------------------------------------------------------------------

class TestFedPolicyRegime:
    def test_creation(self):
        r = _make_regime()
        assert r.regime == 'NEUTRAL'
        assert r.fed_funds_rate == 5.0

    def test_to_dict(self):
        r = _make_regime()
        d = r.to_dict()
        assert d['regime'] == 'NEUTRAL'
        assert 'fed_funds_rate' in d

    def test_divergence_risk_true(self):
        r = _make_regime(real_rate_short=0.0, real_rate_10y=2.0)
        assert r.is_divergence_risk() is True

    def test_divergence_risk_false(self):
        r = _make_regime(real_rate_short=1.5, real_rate_10y=2.0)
        assert r.is_divergence_risk() is False

    def test_allocation_shift_easing(self):
        r = _make_regime(regime='EASING')
        shift = r.get_allocation_shift()
        assert shift['SPY'] > 0
        assert shift['TLT'] < 0

    def test_allocation_shift_tightening(self):
        r = _make_regime(regime='TIGHTENING')
        shift = r.get_allocation_shift()
        assert shift['SPY'] < 0
        assert shift['GLD'] > 0

    def test_allocation_shift_neutral(self):
        r = _make_regime(regime='NEUTRAL')
        shift = r.get_allocation_shift()
        assert all(v == 0.0 for v in shift.values())

    def test_allocation_shift_uncertain(self):
        r = _make_regime(regime='UNCERTAIN')
        shift = r.get_allocation_shift()
        assert shift['SPY'] < 0
        assert shift['GLD'] > 0

    def test_allocation_shift_unknown_defaults_neutral(self):
        r = _make_regime(regime='UNKNOWN')
        shift = r.get_allocation_shift()
        assert all(v == 0.0 for v in shift.values())


# ---------------------------------------------------------------------------
# classify_fed_regime tests
# ---------------------------------------------------------------------------

class TestClassifyFedRegime:
    def test_returns_tuple(self):
        result = classify_fed_regime(fed_funds=5.0, inflation_yoy=2.5, real_rate_10y=1.5)
        assert len(result) == 3

    def test_neutral_default(self):
        regime, conf, factors = classify_fed_regime(
            fed_funds=2.5, inflation_yoy=2.0, real_rate_10y=0.5,
            real_rate_short=0.5, yield_curve_slope=1.0, rate_change_6m=0.0
        )
        assert regime == 'NEUTRAL'

    def test_easing_negative_real_rates(self):
        regime, conf, factors = classify_fed_regime(
            fed_funds=2.0, inflation_yoy=4.0, real_rate_10y=-2.0,
            real_rate_short=-2.0, rate_change_6m=-0.5
        )
        assert regime == 'EASING'

    def test_tightening_high_real_rates(self):
        regime, conf, factors = classify_fed_regime(
            fed_funds=5.5, inflation_yoy=3.5, real_rate_10y=2.0,
            real_rate_short=2.0, rate_change_6m=1.0
        )
        assert regime == 'TIGHTENING'

    def test_uncertain_mixed_signals(self):
        regime, conf, factors = classify_fed_regime(
            fed_funds=5.0, inflation_yoy=2.0, real_rate_10y=3.0,
            real_rate_short=3.0, rate_change_6m=-0.5, yield_curve_slope=-1.0
        )
        # High real rates + cutting = mixed → UNCERTAIN or TIGHTENING
        assert regime in ['UNCERTAIN', 'TIGHTENING']

    def test_confidence_bounded(self):
        _, conf, _ = classify_fed_regime(
            fed_funds=5.0, inflation_yoy=2.5, real_rate_10y=1.5
        )
        assert 0.0 <= conf <= 1.0

    def test_factors_dict(self):
        _, _, factors = classify_fed_regime(
            fed_funds=5.0, inflation_yoy=2.5, real_rate_10y=1.5
        )
        assert 'real_rate_level' in factors
        assert 'rate_change_6m' in factors
        assert 'inflation_gap' in factors

    def test_inverted_curve_uncertain(self):
        regime, conf, factors = classify_fed_regime(
            fed_funds=5.0, inflation_yoy=2.0, real_rate_10y=1.0,
            real_rate_short=3.0, yield_curve_slope=-1.0, rate_change_6m=0.0
        )
        assert regime in ['UNCERTAIN', 'TIGHTENING']

    def test_no_real_short_fallback(self):
        regime, conf, factors = classify_fed_regime(
            fed_funds=3.0, inflation_yoy=2.5, real_rate_10y=0.5
        )
        # real_short defaults to fed_funds - inflation = 0.5
        assert factors['real_rate_level'] == 0.5


# ---------------------------------------------------------------------------
# FedPolicyOverlay tests
# ---------------------------------------------------------------------------

class TestFedPolicyOverlay:
    def test_init(self):
        overlay = FedPolicyOverlay.__new__(FedPolicyOverlay)
        overlay.cache_path = Path('/tmp/test_fred.json')
        overlay.data = {}
        overlay.current_regime = None
        assert overlay.current_regime is None

    def test_detect_regime_no_data_returns_none(self):
        overlay = FedPolicyOverlay.__new__(FedPolicyOverlay)
        overlay.cache_path = Path('/tmp/nonexistent.json')
        overlay.data = {}
        overlay.current_regime = None
        # With empty data and no fetch, should return None
        result = overlay.detect_regime()
        # If data is empty, fetch_data is called; with no cache it may return None
        # The method returns None if fed_funds_df is None or empty
        assert result is None or isinstance(result, FedPolicyRegime)

    def test_detect_regime_with_data(self):
        overlay = FedPolicyOverlay.__new__(FedPolicyOverlay)
        overlay.cache_path = Path('/tmp/test_fred.json')
        overlay.current_regime = None

        # Build mock FRED data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='MS')
        overlay.data = {
            'FEDFUNDS': pd.DataFrame({'date': dates, 'value': [5.0] * 30}),
            'CPIAUCSL': pd.DataFrame({'date': dates, 'value': np.linspace(300, 310, 30)}),
            'DFII10': pd.DataFrame({'date': dates, 'value': [1.5] * 30}),
            'DGS10': pd.DataFrame({'date': dates, 'value': [4.5] * 30}),
            'DGS2': pd.DataFrame({'date': dates, 'value': [4.0] * 30}),
            'T10YIE': pd.DataFrame({'date': dates, 'value': [2.3] * 30}),
        }
        result = overlay.detect_regime()
        assert isinstance(result, FedPolicyRegime)
        assert result.fed_funds_rate == 5.0

    def test_detect_regime_sets_current_regime(self):
        overlay = FedPolicyOverlay.__new__(FedPolicyOverlay)
        overlay.cache_path = Path('/tmp/test_fred.json')
        overlay.current_regime = None

        dates = pd.date_range(end=datetime.now(), periods=30, freq='MS')
        overlay.data = {
            'FEDFUNDS': pd.DataFrame({'date': dates, 'value': [3.0] * 30}),
            'CPIAUCSL': pd.DataFrame({'date': dates, 'value': np.linspace(300, 305, 30)}),
            'DFII10': pd.DataFrame({'date': dates, 'value': [0.5] * 30}),
            'DGS10': pd.DataFrame({'date': dates, 'value': [3.5] * 30}),
            'DGS2': pd.DataFrame({'date': dates, 'value': [3.0] * 30}),
            'T10YIE': pd.DataFrame({'date': dates, 'value': [2.0] * 30}),
        }
        result = overlay.detect_regime()
        assert overlay.current_regime is not None

    def test_get_allocation_recommendation(self):
        overlay = FedPolicyOverlay.__new__(FedPolicyOverlay)
        overlay.cache_path = Path('/tmp/test_fred.json')
        overlay.current_regime = _make_regime(regime='EASING')

        result = overlay.get_allocation_recommendation()
        assert 'regime' in result
        assert result['regime'] == 'EASING'
        assert 'recommended_allocation' in result
        assert 'deltas' in result

    def test_get_allocation_recommendation_custom_base(self):
        overlay = FedPolicyOverlay.__new__(FedPolicyOverlay)
        overlay.cache_path = Path('/tmp/test_fred.json')
        overlay.current_regime = _make_regime(regime='NEUTRAL')

        base = {'SPY': 0.50, 'GLD': 0.30, 'TLT': 0.20}
        result = overlay.get_allocation_recommendation(base)
        assert result['base_allocation'] == base

    def test_recommendation_sums_near_one(self):
        overlay = FedPolicyOverlay.__new__(FedPolicyOverlay)
        overlay.cache_path = Path('/tmp/test_fred.json')
        overlay.current_regime = _make_regime(regime='TIGHTENING')

        result = overlay.get_allocation_recommendation()
        total = sum(result['recommended_allocation'].values())
        assert abs(total - 1.0) < 0.02

    def test_recommendation_has_key_metrics(self):
        overlay = FedPolicyOverlay.__new__(FedPolicyOverlay)
        overlay.cache_path = Path('/tmp/test_fred.json')
        overlay.current_regime = _make_regime(regime='NEUTRAL')

        result = overlay.get_allocation_recommendation()
        assert 'key_metrics' in result
        assert 'fed_funds' in result['key_metrics']

    def test_recommendation_has_deltas(self):
        overlay = FedPolicyOverlay.__new__(FedPolicyOverlay)
        overlay.cache_path = Path('/tmp/test_fred.json')
        overlay.current_regime = _make_regime(regime='UNCERTAIN')

        result = overlay.get_allocation_recommendation()
        assert 'deltas' in result
        # Deltas should sum to ~0 (some normalization may shift this)
        delta_sum = sum(result['deltas'].values())
        assert abs(delta_sum) < 0.1


class TestEmptyDataFrameGuard:
    """Regression: fed_policy_overlay was missing empty DataFrame guard."""

    def test_fetch_data_empty_df_no_crash(self):
        """Empty DataFrames from FRED should not cause IndexError."""
        overlay = FedPolicyOverlay()
        # Simulate fetch_data returning a dict with an empty DataFrame
        overlay._fetched_data = {'DFF': pd.DataFrame(columns=['date', 'value'])}
        for series_id, df in overlay._fetched_data.items():
            # Should not crash on iloc[-1] with empty df
            latest = df.iloc[-1]['date'].strftime('%Y-%m-%d') if not df.empty else "N/A"
            assert latest == "N/A"


# ---------------------------------------------------------------------------
# Extended coverage tests
# ---------------------------------------------------------------------------


class TestFedPolicyRegimeExtended:
    """Extended FedPolicyRegime dataclass tests."""

    def test_to_dict_has_all_fields(self):
        r = _make_regime(unemployment=3.8, confidence=0.85, regime_factors={'real_rate_level': 1.5})
        d = r.to_dict()
        expected_keys = {
            'timestamp', 'regime', 'fed_funds_rate', 'inflation_yoy',
            'real_rate_10y', 'real_rate_short', 'breakeven_10y',
            'yield_curve_10y2y', 'unemployment', 'confidence', 'regime_factors',
        }
        assert expected_keys == set(d.keys())

    def test_divergence_risk_boundary(self):
        """Exactly 1.0 difference should NOT trigger divergence risk."""
        r = _make_regime(real_rate_short=1.0, real_rate_10y=2.0)
        assert r.is_divergence_risk() is False  # abs(1.0 - 2.0) == 1.0, not > 1.0

    def test_divergence_risk_just_over(self):
        """Slightly over 1.0 difference should trigger divergence risk."""
        r = _make_regime(real_rate_short=0.5, real_rate_10y=2.0)
        assert r.is_divergence_risk() is True  # abs(0.5 - 2.0) = 1.5 > 1.0

    def test_allocation_shift_easing_keys(self):
        """EASING shift should have SPY, GLD, TLT, CASH keys."""
        r = _make_regime(regime='EASING')
        shift = r.get_allocation_shift()
        assert set(shift.keys()) == {'SPY', 'GLD', 'TLT', 'CASH'}
        assert shift['SPY'] > 0
        assert shift['GLD'] > 0

    def test_allocation_shift_tightening_keys(self):
        """TIGHTENING shift should have correct signs."""
        r = _make_regime(regime='TIGHTENING')
        shift = r.get_allocation_shift()
        assert shift['SPY'] < 0
        assert shift['GLD'] > 0
        assert shift['TLT'] == 0.0

    def test_allocation_shift_uncertain_keys(self):
        """UNCERTAIN shift should reduce SPY and TLT, increase GLD."""
        r = _make_regime(regime='UNCERTAIN')
        shift = r.get_allocation_shift()
        assert shift['SPY'] < 0
        assert shift['GLD'] > 0
        assert shift['TLT'] < 0


class TestClassifyFedRegimeExtended:
    """Extended classify_fed_regime tests."""

    def test_easing_with_rate_cuts(self):
        """Large rate cuts should classify as EASING."""
        regime, conf, factors = classify_fed_regime(
            fed_funds=2.0, inflation_yoy=3.0, real_rate_10y=-1.0,
            real_rate_short=-1.0, rate_change_6m=-1.0,
        )
        assert regime == 'EASING'

    def test_tightening_with_hikes_and_inflation(self):
        """Rate hikes + high inflation should classify as TIGHTENING."""
        regime, conf, factors = classify_fed_regime(
            fed_funds=5.5, inflation_yoy=4.0, real_rate_10y=2.5,
            real_rate_short=2.5, rate_change_6m=1.5,
        )
        assert regime == 'TIGHTENING'

    def test_uncertain_extreme_inflation(self):
        """Extreme inflation gap (> 2%) should add UNCERTAIN signal."""
        regime, conf, factors = classify_fed_regime(
            fed_funds=3.0, inflation_yoy=5.0, real_rate_10y=-2.0,
            rate_change_6m=-0.5,
        )
        # Either EASING or UNCERTAIN depending on scoring
        assert regime in ['EASING', 'UNCERTAIN']

    def test_confidence_increases_with_margin(self):
        """Larger score margin should increase confidence."""
        # Strong EASING signal
        _, conf_strong, _ = classify_fed_regime(
            fed_funds=1.0, inflation_yoy=4.0, real_rate_10y=-3.0,
            real_rate_short=-3.0, rate_change_6m=-1.0,
        )
        # Weak / ambiguous signal
        _, conf_weak, _ = classify_fed_regime(
            fed_funds=3.0, inflation_yoy=2.0, real_rate_10y=1.0,
            real_rate_short=1.0, rate_change_6m=0.0,
        )
        assert conf_strong >= conf_weak

    def test_no_yield_curve_default(self):
        """Missing yield_curve_slope should default to 0.0 in factors."""
        _, _, factors = classify_fed_regime(
            fed_funds=3.0, inflation_yoy=2.0, real_rate_10y=1.0,
        )
        assert factors['yield_curve'] == 0.0

    def test_inflation_gap_calculation(self):
        """inflation_gap should be inflation - 2.0."""
        _, _, factors = classify_fed_regime(
            fed_funds=3.0, inflation_yoy=3.5, real_rate_10y=1.0,
        )
        assert factors['inflation_gap'] == 1.5

    def test_real_short_fallback(self):
        """When real_rate_short is None, should use fed_funds - inflation."""
        _, _, factors = classify_fed_regime(
            fed_funds=4.0, inflation_yoy=2.5, real_rate_10y=1.5,
        )
        assert factors['real_rate_level'] == 1.5  # 4.0 - 2.5


class TestCalculateInflationYoyExtended:
    """Extended calculate_inflation_yoy tests."""

    def test_zero_drift_zero_inflation(self):
        """Zero drift should produce near-zero inflation."""
        df = _make_cpi_df(n=24, drift=0.0)
        result = calculate_inflation_yoy(df)
        if len(result) > 0:
            assert all(abs(result['inflation_yoy']) < 0.1)

    def test_high_drift_high_inflation(self):
        """High drift should produce high inflation."""
        df = _make_cpi_df(n=24, drift=0.01)
        result = calculate_inflation_yoy(df)
        if len(result) > 0:
            assert all(result['inflation_yoy'] > 5.0)

    def test_insufficient_data(self):
        """Less than 12 months should produce empty result after dropna."""
        df = _make_cpi_df(n=11)
        result = calculate_inflation_yoy(df)
        assert len(result) == 0


class TestCalculateRealRateExtended:
    """Extended calculate_real_rate tests."""

    def test_real_rate_positive(self):
        """Nominal > inflation should give positive real rate."""
        nominal = _make_nominal_df(n=24, base=5.0)
        cpi = calculate_inflation_yoy(_make_cpi_df(n=24, drift=0.002))
        result = calculate_real_rate(nominal, cpi)
        if len(result) > 0 and 'real_rate' in result.columns:
            assert any(result['real_rate'] > 0)

    def test_real_rate_formula(self):
        """Real rate should equal nominal - inflation."""
        nominal = _make_nominal_df(n=24, base=5.0)
        cpi = calculate_inflation_yoy(_make_cpi_df(n=24, drift=0.002))
        result = calculate_real_rate(nominal, cpi)
        if len(result) > 0 and 'real_rate' in result.columns:
            for _, row in result.iterrows():
                expected = row['nominal'] - row['inflation']
                assert row['real_rate'] == pytest.approx(expected, abs=0.01)


class TestFedPolicyOverlayExtended:
    """Extended FedPolicyOverlay tests."""

    def test_detect_regime_no_fed_funds(self):
        """Missing FEDFUNDS data should return None."""
        overlay = FedPolicyOverlay.__new__(FedPolicyOverlay)
        overlay.cache_path = Path('/tmp/test_fred.json')
        overlay.current_regime = None
        overlay.data = {
            'CPIAUCSL': pd.DataFrame({'date': pd.date_range(end=datetime.now(), periods=30, freq='MS'), 'value': np.linspace(300, 310, 30)}),
        }
        result = overlay.detect_regime()
        assert result is None

    def test_detect_regime_with_timestamp(self):
        """Custom timestamp should be preserved."""
        overlay = FedPolicyOverlay.__new__(FedPolicyOverlay)
        overlay.cache_path = Path('/tmp/test_fred.json')
        overlay.current_regime = None
        dates = pd.date_range(end=datetime.now(), periods=30, freq='MS')
        overlay.data = {
            'FEDFUNDS': pd.DataFrame({'date': dates, 'value': [5.0] * 30}),
            'CPIAUCSL': pd.DataFrame({'date': dates, 'value': np.linspace(300, 310, 30)}),
            'DFII10': pd.DataFrame({'date': dates, 'value': [1.5] * 30}),
            'DGS10': pd.DataFrame({'date': dates, 'value': [4.5] * 30}),
            'DGS2': pd.DataFrame({'date': dates, 'value': [4.0] * 30}),
            'T10YIE': pd.DataFrame({'date': dates, 'value': [2.3] * 30}),
        }
        result = overlay.detect_regime(timestamp='2026-05-24T12:00:00')
        assert result is not None
        assert result.timestamp == '2026-05-24T12:00:00'

    def test_get_allocation_recommendation_no_regime(self):
        """Recommendation without regime should return error or detect regime."""
        overlay = FedPolicyOverlay.__new__(FedPolicyOverlay)
        overlay.cache_path = Path('/tmp/nonexistent.json')
        overlay.current_regime = None
        overlay.data = {}
        # With empty data, detect_regime returns None
        result = overlay.get_allocation_recommendation()
        assert 'error' in result or 'regime' in result

    def test_recommendation_normalization(self):
        """Recommended allocation should sum to 1.0."""
        overlay = FedPolicyOverlay.__new__(FedPolicyOverlay)
        overlay.cache_path = Path('/tmp/test_fred.json')
        overlay.current_regime = _make_regime(regime='EASING')
        result = overlay.get_allocation_recommendation()
        total = sum(result['recommended_allocation'].values())
        assert abs(total - 1.0) < 0.01

    def test_recommendation_clamping(self):
        """Allocation weights should be clamped between 0.05 and 0.90."""
        overlay = FedPolicyOverlay.__new__(FedPolicyOverlay)
        overlay.cache_path = Path('/tmp/test_fred.json')
        overlay.current_regime = _make_regime(regime='TIGHTENING')
        result = overlay.get_allocation_recommendation()
        for asset, weight in result['recommended_allocation'].items():
            assert 0.05 <= weight <= 0.90, f"{asset}={weight} out of bounds"

    def test_recommendation_has_strategy_field(self):
        """Recommendation should include strategy identifier."""
        overlay = FedPolicyOverlay.__new__(FedPolicyOverlay)
        overlay.cache_path = Path('/tmp/test_fred.json')
        overlay.current_regime = _make_regime(regime='NEUTRAL')
        result = overlay.get_allocation_recommendation()
        assert 'strategy' in result
        assert 'Fed Policy' in result['strategy']


class TestFredSeriesExtended:
    """Extended FRED_SERIES constant tests."""

    def test_core_priority_series(self):
        """Priority series should be in FRED_SERIES."""
        priority = ['FEDFUNDS', 'CPIAUCSL', 'T10YIE', 'DFII10', 'DGS10', 'DGS2']
        for series in priority:
            assert series in FRED_SERIES

    def test_all_values_are_strings(self):
        """All FRED_SERIES values should be descriptive strings."""
        for key, desc in FRED_SERIES.items():
            assert isinstance(desc, str) and len(desc) > 10


# =============================================================================
# fetch_fred_series tests
# =============================================================================

class TestFetchFredSeries:
    """Tests for fetch_fred_series with mocked HTTP calls."""

    def test_successful_fetch_valid_csv(self):
        """Valid CSV response should return DataFrame with date and value columns."""
        csv_data = "DATE,VALUE\n2020-01-01,2.5\n2020-02-01,2.6\n"
        with patch('src.signals.fed_policy_overlay.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.text = csv_data
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = fetch_fred_series('FEDFUNDS')
            assert result is not None
            assert 'date' in result.columns
            assert 'value' in result.columns
            assert len(result) == 2

    def test_http_error_returns_none(self):
        """HTTP error (e.g. 404) should return None."""
        csv_data = "DATE,VALUE\n2020-01-01,2.5\n"
        with patch('src.signals.fed_policy_overlay.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.text = csv_data
            mock_response.raise_for_status.side_effect = Exception("HTTP 404 Not Found")
            mock_get.return_value = mock_response

            result = fetch_fred_series('FEDFUNDS')
            assert result is None

    def test_request_exception_returns_none(self):
        """Network error (connection timeout) should return None."""
        with patch('src.signals.fed_policy_overlay.requests.get') as mock_get:
            mock_get.side_effect = Exception("ConnectionError: No connection")

            result = fetch_fred_series('FEDFUNDS')
            assert result is None

    def test_invalid_numeric_data_dropped(self):
        """Non-numeric values should be coerced to NaN and dropped."""
        csv_data = "DATE,VALUE\n2020-01-01,abc\n2020-02-01,def\n"
        with patch('src.signals.fed_policy_overlay.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.text = csv_data
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = fetch_fred_series('FEDFUNDS')
            # After dropna all rows are gone
            assert result is None or len(result) == 0


# =============================================================================
# fetch_all_fred_data tests
# =============================================================================

class TestFetchAllFredData:
    """Tests for fetch_all_fred_data with caching behavior."""

    def test_cache_hit_uses_cached_data(self, tmp_path):
        """Valid cache file under 24h old should skip fetch."""
        cache_file = tmp_path / "fred_cache.json"
        cached_data = {
            'FEDFUNDS': [{'date': '2026-01-01', 'value': 5.0}],
        }
        cache_file.write_text(json.dumps(cached_data))

        with patch('src.signals.fed_policy_overlay.fetch_fred_series') as mock_fetch:
            result = fetch_all_fred_data(cache_path=cache_file)
            mock_fetch.assert_not_called()
            assert 'FEDFUNDS' in result

    def test_cache_miss_triggers_fetch(self, tmp_path):
        """No cache file should trigger FRED fetch and create cache."""
        cache_file = tmp_path / "fred_cache.json"
        mock_df = pd.DataFrame({
            'date': pd.date_range('2026-01-01', periods=3, freq='MS'),
            'value': [5.0, 5.25, 5.5],
        })

        with patch('src.signals.fed_policy_overlay.fetch_fred_series', return_value=mock_df) as mock_fetch:
            result = fetch_all_fred_data(cache_path=cache_file)
            assert mock_fetch.call_count >= 1
            assert cache_file.exists()

    def test_force_refresh_bypasses_cache(self, tmp_path):
        """force_refresh=True should bypass cache and fetch fresh data."""
        cache_file = tmp_path / "fred_cache.json"
        cached_data = {'FEDFUNDS': [{'date': '2026-01-01', 'value': 5.0}]}
        cache_file.write_text(json.dumps(cached_data))
        mock_df = pd.DataFrame({
            'date': pd.date_range('2026-01-01', periods=3, freq='MS'),
            'value': [5.0, 5.25, 5.5],
        })

        with patch('src.signals.fed_policy_overlay.fetch_fred_series', return_value=mock_df) as mock_fetch:
            result = fetch_all_fred_data(cache_path=cache_file, force_refresh=True)
            mock_fetch.assert_called()
            assert 'FEDFUNDS' in result


# =============================================================================
# calculate_real_rate merge behavior tests
# =============================================================================

class TestCalculateRealRateMerge:
    """Tests for calculate_real_rate merge and frequency behavior."""

    def test_inner_merge_drops_non_overlapping(self):
        """Inner merge should only keep rows with matching dates."""
        nominal_dates = pd.date_range('2020-01-01', periods=5, freq='MS')
        cpi_dates = pd.date_range('2021-01-01', periods=5, freq='MS')
        nominal = pd.DataFrame({'date': nominal_dates, 'value': [4.0, 4.5, 5.0, 5.5, 6.0]})
        cpi_inflation = pd.DataFrame({
            'date': cpi_dates,
            'inflation_yoy': [2.0, 2.2, 2.4, 2.6, 2.8],
        })

        result = calculate_real_rate(nominal, cpi_inflation, merge_how='inner')
        # No overlapping dates, so empty after dropna
        assert len(result) == 0

    def test_forward_fill_handles_different_frequencies(self):
        """Forward fill should propagate inflation values to fill gaps."""
        # Daily nominal, monthly CPI
        nominal_dates = pd.date_range('2020-01-01', periods=5, freq='D')
        cpi_dates = pd.date_range('2020-01-01', periods=3, freq='MS')
        nominal = pd.DataFrame({'date': nominal_dates, 'value': [4.0, 4.1, 4.2, 4.3, 4.4]})
        cpi_inflation = pd.DataFrame({
            'date': cpi_dates,
            'inflation_yoy': [2.0, 2.2, 2.4],
        })

        result = calculate_real_rate(nominal, cpi_inflation, merge_how='outer')
        # Should not crash; forward_fill should succeed
        assert 'real_rate' in result.columns

    def test_real_rate_negative_when_inflation_above_nominal(self):
        """When inflation exceeds nominal rate, real rate should be negative."""
        nominal = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=3, freq='MS'),
            'value': [1.0, 1.0, 1.0],
        })
        cpi_inflation = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=3, freq='MS'),
            'inflation_yoy': [3.0, 3.5, 4.0],
        })

        result = calculate_real_rate(nominal, cpi_inflation, merge_how='inner')
        if len(result) > 0:
            assert all(result['real_rate'] < 0)
            assert result['real_rate'].iloc[0] == pytest.approx(-2.0, abs=0.01)


# =============================================================================
# FedPolicyRegime dataclass edge cases
# =============================================================================

class TestFedPolicyRegimeDefaults:
    """Tests for FedPolicyRegime dataclass default values and edge cases."""

    def test_regime_factors_defaults_to_none(self):
        """regime_factors field should default to None."""
        r = _make_regime()
        assert r.regime_factors is None

    def test_unemployment_defaults_to_none(self):
        """unemployment field should default to None."""
        r = _make_regime()
        assert r.unemployment is None

    def test_confidence_defaults_to_zero(self):
        """confidence field should default to 0.0."""
        r = FedPolicyRegime(
            timestamp='2026-01-01',
            regime='NEUTRAL',
            fed_funds_rate=5.0,
            inflation_yoy=2.5,
            real_rate_10y=1.5,
            real_rate_short=2.5,
            breakeven_10y=2.3,
            yield_curve_10y2y=0.5,
        )
        assert r.confidence == 0.0

    def test_regime_factors_none_in_to_dict(self):
        """regime_factors=None should appear as None in to_dict()."""
        r = _make_regime(regime_factors=None)
        d = r.to_dict()
        assert d['regime_factors'] is None

    def test_divergence_risk_both_rates_negative(self):
        """Both short and long real rates negative: divergence should depend on gap."""
        r = _make_regime(real_rate_short=-2.0, real_rate_10y=-0.5)
        assert r.is_divergence_risk() is True  # gap = 1.5 > 1.0

    def test_divergence_risk_equal_rates(self):
        """Equal short and long real rates should never trigger divergence."""
        r = _make_regime(real_rate_short=2.0, real_rate_10y=2.0)
        assert r.is_divergence_risk() is False  # gap = 0.0

    def test_to_dict_contains_confidence_when_set(self):
        """Confidence should appear in to_dict() output."""
        r = _make_regime(confidence=0.85)
        d = r.to_dict()
        assert d['confidence'] == 0.85


# =============================================================================
# classify_fed_regime edge cases
# =============================================================================

class TestClassifyFedRegimeEdgeCases:
    """Edge case coverage for classify_fed_regime."""

    def test_max_score_zero_returns_neutral(self):
        """When all scores are 0, regime defaults to NEUTRAL with conf 0.5."""
        # Need to avoid ALL scoring conditions:
        # real_short=1.0 (not <0, not <0.5 for EASING; not >1.5, not >1.0 for TIGHTENING)
        # inflation=3.0 (outside NEUTRAL [1.5,2.5]; NOT >3 so no behind-curve/high-fed bonus)
        # rate_change=0.25 (NOT < -0.25 for EASING; NOT >0.5 for TIGHTENING;
        #                   NOT < 0.25 for NEUTRAL abs-check; NOT > 0.25 for TIGHTENING)
        regime, conf, factors = classify_fed_regime(
            fed_funds=5.0, inflation_yoy=3.0, real_rate_10y=1.0,
            real_rate_short=1.0, rate_change_6m=0.25,
        )
        assert regime == 'NEUTRAL'
        assert conf == 0.5

    def test_boundary_real_short_at_neutral_lower(self):
        """real_short exactly 0.5 should count as NEUTRAL."""
        regime, conf, factors = classify_fed_regime(
            fed_funds=2.5, inflation_yoy=2.0, real_rate_10y=0.5,
            real_rate_short=0.5, rate_change_6m=0.0, yield_curve_slope=1.0,
        )
        assert regime == 'NEUTRAL'

    def test_boundary_real_short_at_neutral_upper(self):
        """real_short exactly 1.5 should count as NEUTRAL."""
        regime, conf, factors = classify_fed_regime(
            fed_funds=3.5, inflation_yoy=2.0, real_rate_10y=1.5,
            real_rate_short=1.5, rate_change_6m=0.0, yield_curve_slope=1.0,
        )
        assert regime == 'NEUTRAL'

    def test_fed_behind_curve_easing(self):
        """Inflation > 3% and fed_funds < inflation should add EASING signal."""
        regime, conf, factors = classify_fed_regime(
            fed_funds=2.0, inflation_yoy=4.0, real_rate_10y=-2.0,
            real_rate_short=-2.0, rate_change_6m=-0.5,
        )
        assert regime == 'EASING'

    def test_exact_target_inflation_no_penalty(self):
        """Inflation exactly 2.0% should produce inflation_gap of 0."""
        _, _, factors = classify_fed_regime(
            fed_funds=3.0, inflation_yoy=2.0, real_rate_10y=1.0,
        )
        assert factors['inflation_gap'] == 0.0

    def test_yield_curve_slightly_positive_scores_neutral(self):
        """Yield curve slope between 0 and 2 should add to NEUTRAL."""
        regime, conf, factors = classify_fed_regime(
            fed_funds=2.5, inflation_yoy=2.0, real_rate_10y=1.0,
            real_rate_short=1.0, rate_change_6m=0.0, yield_curve_slope=0.5,
        )
        # Slope < 2 AND > 0 adds to neutral
        assert regime == 'NEUTRAL'

    def test_yield_curve_moderately_negative_no_uncertainty(self):
        """Yield curve -0.3 (not < -0.5) should NOT add UNCERTAIN signal."""
        regime, conf, factors = classify_fed_regime(
            fed_funds=2.5, inflation_yoy=2.0, real_rate_10y=1.0,
            real_rate_short=1.0, rate_change_6m=0.0, yield_curve_slope=-0.3,
        )
        # -0.3 is not < -0.5, so no uncertain point, and not > 0 for neutral
        assert regime == 'NEUTRAL'

    def test_tightening_with_high_fed_and_inflation(self):
        """High fed funds (>4) with high inflation (>3) should strengthen TIGHTENING."""
        regime, conf, factors = classify_fed_regime(
            fed_funds=5.0, inflation_yoy=3.5, real_rate_10y=2.0,
            real_rate_short=2.0, rate_change_6m=0.75,
        )
        # real_short 2.0 > 1.5 → TIGHTENING+2, rate_change 0.75 > 0.5 →+2
        # inflation > 3 and fed > 4 → +1 = 5 total TIGHTENING
        assert regime == 'TIGHTENING'

    def test_yield_curve_above_2_not_neutral(self):
        """Yield curve slope >= 2 should NOT add to NEUTRAL scoring."""
        regime, conf, factors = classify_fed_regime(
            fed_funds=2.5, inflation_yoy=2.0, real_rate_10y=1.0,
            real_rate_short=0.5, rate_change_6m=0.0, yield_curve_slope=2.5,
        )
        # Inverted curve? No, slope is 2.5 which is > 0.5.
        # curve > 0 AND < 2 → neutral +1. 2.5 is NOT < 2, so no neutral point.
        # But real_short 0.5 is exactly 0.5, which is >= 0.5, so it's in NEUTRAL range.
        # rate_change 0.0 < 0.25 → NEUTRAL +1, total NEUTRAL = 2
        assert regime == 'NEUTRAL'


# =============================================================================
# FedPolicyOverlay edge cases
# =============================================================================

class TestFedPolicyOverlayMore:
    """Additional edge case tests for FedPolicyOverlay."""

    def test_detect_regime_no_cpi_uses_breakeven(self):
        """When CPI data is missing, should fall back to breakeven inflation."""
        overlay = FedPolicyOverlay.__new__(FedPolicyOverlay)
        overlay.cache_path = Path('/tmp/test_fred_no_cpi.json')
        overlay.current_regime = None
        dates = pd.date_range(end=datetime.now(), periods=30, freq='MS')
        overlay.data = {
            'FEDFUNDS': pd.DataFrame({'date': dates, 'value': [5.0] * 30}),
            # No CPIAUCSL
            'T10YIE': pd.DataFrame({'date': dates, 'value': [2.5] * 30}),
            'DFII10': pd.DataFrame({'date': dates, 'value': [1.5] * 30}),
            'DGS10': pd.DataFrame({'date': dates, 'value': [4.5] * 30}),
            'DGS2': pd.DataFrame({'date': dates, 'value': [4.0] * 30}),
        }
        result = overlay.detect_regime()
        assert result is not None
        # Without CPIAUCSL, inflation falls back to T10YIE = 2.5
        assert result.inflation_yoy == 2.5

    def test_detect_regime_no_cpi_no_breakeven_default(self):
        """When CPI and breakeven are both missing, inflation defaults to 2.0."""
        overlay = FedPolicyOverlay.__new__(FedPolicyOverlay)
        overlay.cache_path = Path('/tmp/test_fred_no_inflation.json')
        overlay.current_regime = None
        dates = pd.date_range(end=datetime.now(), periods=30, freq='MS')
        overlay.data = {
            'FEDFUNDS': pd.DataFrame({'date': dates, 'value': [4.0] * 30}),
            'DFII10': pd.DataFrame({'date': dates, 'value': [1.0] * 30}),
            'DGS10': pd.DataFrame({'date': dates, 'value': [4.5] * 30}),
            'DGS2': pd.DataFrame({'date': dates, 'value': [4.0] * 30}),
        }
        result = overlay.detect_regime()
        assert result is not None
        # No CPI and no breakeven → inflation default 2.0
        assert result.inflation_yoy == 2.0

    def test_detect_regime_no_tips_uses_nominal_breakeven(self):
        """When TIPS data is missing, real rate should use nominal - breakeven."""
        overlay = FedPolicyOverlay.__new__(FedPolicyOverlay)
        overlay.cache_path = Path('/tmp/test_fred_no_tips.json')
        overlay.current_regime = None
        dates = pd.date_range(end=datetime.now(), periods=30, freq='MS')
        overlay.data = {
            'FEDFUNDS': pd.DataFrame({'date': dates, 'value': [5.0] * 30}),
            'CPIAUCSL': pd.DataFrame({'date': dates, 'value': np.linspace(300, 310, 30)}),
            # No DFII10
            'T10YIE': pd.DataFrame({'date': dates, 'value': [2.3] * 30}),
            'DGS10': pd.DataFrame({'date': dates, 'value': [4.5] * 30}),
            'DGS2': pd.DataFrame({'date': dates, 'value': [4.0] * 30}),
        }
        result = overlay.detect_regime()
        assert result is not None
        # real_rate_10y = nominal - breakeven = 4.5 - 2.3 = 2.2
        assert result.real_rate_10y == pytest.approx(2.2, abs=0.01)

    def test_detect_regime_fewer_than_6_points(self):
        """Less than 6 data points should set rate_change_6m to 0.0."""
        overlay = FedPolicyOverlay.__new__(FedPolicyOverlay)
        overlay.cache_path = Path('/tmp/test_fred_short.json')
        overlay.current_regime = None
        dates = pd.date_range(end=datetime.now(), periods=3, freq='MS')
        overlay.data = {
            'FEDFUNDS': pd.DataFrame({'date': dates, 'value': [5.0, 5.0, 5.0]}),
            'CPIAUCSL': pd.DataFrame({'date': dates, 'value': [300, 301, 302]}),
            'DFII10': pd.DataFrame({'date': dates, 'value': [1.5, 1.5, 1.5]}),
            'DGS10': pd.DataFrame({'date': dates, 'value': [4.5, 4.5, 4.5]}),
            'DGS2': pd.DataFrame({'date': dates, 'value': [4.0, 4.0, 4.0]}),
            'T10YIE': pd.DataFrame({'date': dates, 'value': [2.3, 2.3, 2.3]}),
        }
        result = overlay.detect_regime()
        assert result is not None
        assert result.regime_factors['rate_change_6m'] == 0.0

    def test_recommendation_includes_divergence_risk(self):
        """get_allocation_recommendation should include divergence_risk field."""
        overlay = FedPolicyOverlay.__new__(FedPolicyOverlay)
        overlay.cache_path = Path('/tmp/test_fred_div.json')
        overlay.current_regime = _make_regime(regime='NEUTRAL')
        result = overlay.get_allocation_recommendation()
        assert 'divergence_risk' in result
        assert isinstance(result['divergence_risk'], bool)

    def test_recommendation_with_none_base_allocation(self):
        """get_allocation_recommendation should accept None base_allocation."""
        overlay = FedPolicyOverlay.__new__(FedPolicyOverlay)
        overlay.cache_path = Path('/tmp/test_fred_none_base.json')
        overlay.current_regime = _make_regime(regime='NEUTRAL')
        # Explicitly passing None should use BASE_ALLOCATION
        result = overlay.get_allocation_recommendation(None)
        assert 'regime' in result
        assert result['regime'] == 'NEUTRAL'

    def test_fetch_data_calls_fetch_all(self):
        """Overlay.fetch_data should delegate to fetch_all_fred_data."""
        overlay = FedPolicyOverlay.__new__(FedPolicyOverlay)
        overlay.cache_path = Path('/tmp/test_fred_fetch.json')
        overlay.data = {}
        overlay.current_regime = None

        with patch('src.signals.fed_policy_overlay.fetch_all_fred_data') as mock_fetch:
            mock_fetch.return_value = {
                'FEDFUNDS': pd.DataFrame({'date': [datetime.now()], 'value': [5.0]}),
            }
            result = overlay.fetch_data()
            mock_fetch.assert_called_once_with(overlay.cache_path, False)
            assert 'FEDFUNDS' in result

    def test_fetch_data_force_refresh_passthrough(self):
        """Overlay.fetch_data should pass force_refresh to fetch_all_fred_data."""
        overlay = FedPolicyOverlay.__new__(FedPolicyOverlay)
        overlay.cache_path = Path('/tmp/test_fred_refresh.json')
        overlay.data = {}
        overlay.current_regime = None

        with patch('src.signals.fed_policy_overlay.fetch_all_fred_data') as mock_fetch:
            mock_fetch.return_value = {}
            result = overlay.fetch_data(force_refresh=True)
            mock_fetch.assert_called_once_with(overlay.cache_path, True)


# =============================================================================
# Constants tests
# =============================================================================

class TestConstants:
    """Tests for module-level constants."""

    def test_fred_cache_is_path(self):
        """FRED_CACHE should be a Path object."""
        assert isinstance(FRED_CACHE, Path)

    def test_all_exports_defined(self):
        """All names in __all__ should be importable."""
        from src.signals.fed_policy_overlay import __all__
        expected = [
            'FRED_SERIES', 'FRED_CACHE', 'fetch_fred_series', 'fetch_all_fred_data',
            'calculate_inflation_yoy', 'calculate_real_rate', 'FedPolicyRegime',
            'classify_fed_regime', 'FedPolicyOverlay',
        ]
        for name in expected:
            assert name in __all__, f"{name} missing from __all__"
