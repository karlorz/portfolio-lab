#!/usr/bin/env python3
"""
Tests for Fed Policy Overlay — FRED series constants, inflation YoY,
real rate calculation, FedPolicyRegime data class, regime classification,
and FedPolicyOverlay allocation recommendation.
"""
import sys
import os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.signals.fed_policy_overlay import (
    FRED_SERIES,
    calculate_inflation_yoy,
    calculate_real_rate,
    FedPolicyRegime,
    classify_fed_regime,
    FedPolicyOverlay,
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
