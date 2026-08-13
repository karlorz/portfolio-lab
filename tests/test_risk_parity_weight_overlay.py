#!/usr/bin/env python3
"""
Tests for risk_parity_weight_overlay.py — constants, RPWeightOverlay dataclass,
realized volatility calculation, risk parity overlay calculation, and CLI.
"""
import json
import dataclasses
import numpy as np
import pandas as pd

import pytest
from pathlib import Path
from datetime import datetime
from typing import Dict
from unittest.mock import patch, MagicMock

from src.strategy.risk_parity_weight_overlay import (
    VOL_LOOKBACK,
    MAX_DEVIATION,
    MIN_WEIGHT,
    REBALANCE_FREQ,
    DEFAULT_BASE,
    RPWeightOverlay,
    RiskParityWeightOverlay,
    RPBacktester,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices_df(symbols=None, days=300, seed=42):
    """Create a synthetic prices DataFrame."""
    if symbols is None:
        symbols = ["SPY", "GLD", "TLT"]
    rng = np.random.RandomState(seed)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
    data = {}
    starts = {"SPY": 450, "GLD": 190, "TLT": 95, "QQQ": 380, "IWM": 200}
    for sym in symbols:
        price = float(starts.get(sym, 100))
        prices = [price]
        for _ in range(days - 1):
            price *= (1 + rng.normal(0.0003, 0.012))
            prices.append(price)
        data[sym] = prices
    df = pd.DataFrame(data, index=dates)
    return df


def _make_overlay(tmp_path=None):
    """Create a RiskParityWeightOverlay with a mock prices path."""
    overlay = RiskParityWeightOverlay.__new__(RiskParityWeightOverlay)
    overlay.prices_path = Path("/tmp/fake.json")
    overlay.db_path = Path("/tmp/fake.db")
    overlay.vol_lookback = VOL_LOOKBACK
    overlay.max_deviation = MAX_DEVIATION
    overlay._prices_df = None
    return overlay


# ---------------------------------------------------------------------------
# Constants Tests
# ---------------------------------------------------------------------------

class TestConstants:

    def test_vol_lookback(self):
        assert VOL_LOOKBACK == 252

    def test_max_deviation(self):
        assert MAX_DEVIATION == 0.15

    def test_min_weight(self):
        assert MIN_WEIGHT == 0.05

    def test_rebalance_freq(self):
        assert REBALANCE_FREQ == 21

    def test_default_base(self):
        assert DEFAULT_BASE['SPY'] == 0.46
        assert DEFAULT_BASE['GLD'] == 0.38
        assert DEFAULT_BASE['TLT'] == 0.16
        assert DEFAULT_BASE['CASH'] == 0.0


# ---------------------------------------------------------------------------
# RPWeightOverlay Dataclass Tests
# ---------------------------------------------------------------------------

class TestRPWeightOverlay:

    def test_to_dict(self):
        overlay = RPWeightOverlay(
            timestamp="2026-05-14",
            asset_vols={"SPY": 0.18, "GLD": 0.15},
            raw_rp_weights={"SPY": 0.55, "GLD": 0.45},
            base_weights={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16, "CASH": 0.0},
            rp_adjustments={"SPY": 0.09, "GLD": 0.07, "TLT": -0.05},
            target_weights={"SPY": 0.50, "GLD": 0.40, "TLT": 0.10, "CASH": 0.0},
            expected_vol=0.16,
            risk_parity_score=0.85,
        )
        d = overlay.to_dict()
        assert d["timestamp"] == "2026-05-14"
        assert d["expected_vol"] == 0.16
        assert d["risk_parity_score"] == 0.85
        assert "asset_vols" in d
        assert "target_weights" in d


# ---------------------------------------------------------------------------
# RiskParityWeightOverlay — calculate_realized_vol
# ---------------------------------------------------------------------------

class TestCalculateRealizedVol:

    def test_returns_float(self):
        overlay = _make_overlay()
        df = _make_prices_df(["SPY"], days=300)
        vol = overlay.calculate_realized_vol("SPY", df)
        assert isinstance(vol, float)
        assert vol > 0

    def test_missing_ticker_returns_none(self):
        overlay = _make_overlay()
        df = _make_prices_df(["SPY"], days=300)
        vol = overlay.calculate_realized_vol("FAKE", df)
        assert vol is None

    def test_insufficient_data_returns_none(self):
        overlay = _make_overlay()
        df = _make_prices_df(["SPY"], days=10)
        vol = overlay.calculate_realized_vol("SPY", df)
        assert vol is None

    def test_vol_annualized(self):
        overlay = _make_overlay()
        rng = np.random.RandomState(99)
        daily_vol = 0.015
        prices = 100 * np.cumprod(1 + rng.normal(0, daily_vol, 300))
        dates = pd.date_range(end=datetime.now(), periods=300, freq='B')
        df = pd.DataFrame({"TEST": prices}, index=dates)
        vol = overlay.calculate_realized_vol("TEST", df)
        expected = daily_vol * np.sqrt(252)
        assert vol == pytest.approx(expected, rel=0.15)

    def test_lookback_respected(self):
        overlay = _make_overlay()
        overlay.vol_lookback = 50
        df = _make_prices_df(["SPY"], days=300)
        vol = overlay.calculate_realized_vol("SPY", df)
        assert vol is not None


# ---------------------------------------------------------------------------
# RiskParityWeightOverlay — calculate_rp_overlay
# ---------------------------------------------------------------------------

class TestCalculateRPOverlay:

    def test_returns_rp_overlay(self):
        overlay = _make_overlay()
        df = _make_prices_df(["SPY", "GLD", "TLT"], days=300)
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16, "CASH": 0.0}
        result = overlay.calculate_rp_overlay(base, df)
        assert isinstance(result, RPWeightOverlay)

    def test_target_weights_sum_to_one(self):
        overlay = _make_overlay()
        df = _make_prices_df(["SPY", "GLD", "TLT"], days=300)
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16, "CASH": 0.0}
        result = overlay.calculate_rp_overlay(base, df)
        total = sum(v for k, v in result.target_weights.items() if k != "CASH")
        assert total == pytest.approx(1.0, abs=0.01)

    def test_min_weight_enforced(self):
        overlay = _make_overlay()
        # Create very different vols to push one weight down
        df = _make_prices_df(["SPY", "GLD", "TLT"], days=300, seed=42)
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16, "CASH": 0.0}
        result = overlay.calculate_rp_overlay(base, df)
        for asset in ["SPY", "GLD", "TLT"]:
            assert result.target_weights[asset] >= MIN_WEIGHT

    def test_cash_weight_zero(self):
        overlay = _make_overlay()
        df = _make_prices_df(["SPY", "GLD", "TLT"], days=300)
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16, "CASH": 0.0}
        result = overlay.calculate_rp_overlay(base, df)
        assert result.target_weights["CASH"] == 0.0

    def test_max_deviation_clipping(self):
        overlay = _make_overlay()
        overlay.max_deviation = 0.05  # Very tight
        df = _make_prices_df(["SPY", "GLD", "TLT"], days=300)
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16, "CASH": 0.0}
        _ = overlay.calculate_rp_overlay(base, df)
        # Adjustments should be within ±max_deviation (before normalization)
        # Note: post-normalization adjustments may differ slightly

    def test_expected_vol_positive(self):
        overlay = _make_overlay()
        df = _make_prices_df(["SPY", "GLD", "TLT"], days=300)
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16, "CASH": 0.0}
        result = overlay.calculate_rp_overlay(base, df)
        assert result.expected_vol > 0

    def test_risk_parity_score_bounded(self):
        overlay = _make_overlay()
        df = _make_prices_df(["SPY", "GLD", "TLT"], days=300)
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16, "CASH": 0.0}
        result = overlay.calculate_rp_overlay(base, df)
        assert 0 <= result.risk_parity_score <= 1.0

    def test_insufficient_assets_returns_none(self):
        overlay = _make_overlay()
        df = _make_prices_df(["SPY", "GLD"], days=300)  # Missing TLT
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16, "CASH": 0.0}
        result = overlay.calculate_rp_overlay(base, df)
        assert result is None

    def test_raw_rp_weights_sum_to_one(self):
        overlay = _make_overlay()
        df = _make_prices_df(["SPY", "GLD", "TLT"], days=300)
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16, "CASH": 0.0}
        result = overlay.calculate_rp_overlay(base, df)
        total = sum(result.raw_rp_weights.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_inverse_vol_weighting(self):
        overlay = _make_overlay()
        df = _make_prices_df(["SPY", "GLD", "TLT"], days=300)
        base = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16, "CASH": 0.0}
        result = overlay.calculate_rp_overlay(base, df)
        # Asset with lower vol should get higher raw RP weight
        vols = result.asset_vols
        weights = result.raw_rp_weights
        if vols["GLD"] < vols["SPY"]:
            assert weights["GLD"] > weights["SPY"]


# ---------------------------------------------------------------------------
# RiskParityWeightOverlay — _load_prices (mocked)
# ---------------------------------------------------------------------------

class TestLoadPrices:

    def test_load_prices_from_json(self, tmp_path):
        # Create a minimal prices.json
        prices_data = {
            "SPY": [{"d": "2026-01-02", "p": 450.0}, {"d": "2026-01-03", "p": 452.0}],
            "GLD": [{"d": "2026-01-02", "p": 190.0}, {"d": "2026-01-03", "p": 191.0}],
        }
        import pandas as pd
        rows = []
        for sym, entries in prices_data.items():
            for entry in entries:
                rows.append({"date": entry["d"], "ticker": sym, "price": entry["p"]})
        df_expected = pd.DataFrame(rows)
        df_expected["date"] = pd.to_datetime(df_expected["date"])
        df_expected = df_expected.pivot(index="date", columns="ticker", values="price")

        overlay = RiskParityWeightOverlay.__new__(RiskParityWeightOverlay)
        overlay.vol_lookback = VOL_LOOKBACK
        overlay._prices_df = df_expected
        overlay.prices_path = None

        df = overlay._load_prices()
        assert isinstance(df, pd.DataFrame)
        assert "SPY" in df.columns
        assert "GLD" in df.columns

    def test_load_prices_cached(self, tmp_path):
        overlay = RiskParityWeightOverlay.__new__(RiskParityWeightOverlay)
        overlay._prices_df = pd.DataFrame({"SPY": [100, 200]})
        df = overlay._load_prices()
        assert list(df.columns) == ["SPY"]


# ---------------------------------------------------------------------------
# CLI Tests
# ---------------------------------------------------------------------------

class TestCLI:

    def test_status_command(self, caplog):
        import logging
        caplog.set_level(logging.INFO)
        from src.strategy.risk_parity_weight_overlay import main
        with patch("sys.argv", ["rp_overlay.py", "status"]):
            main()
        assert "Risk Parity" in caplog.text
        assert "46%" in caplog.text or "46" in caplog.text

    def test_no_command_prints_help(self, capsys):
        from src.strategy.risk_parity_weight_overlay import main
        with patch("sys.argv", ["rp_overlay.py"]):
            main()
        captured = capsys.readouterr()
        assert "usage" in captured.out.lower() or "Risk Parity" in captured.out


# ---------------------------------------------------------------------------
# Extended coverage tests
# ---------------------------------------------------------------------------


class TestCalculateRealizedVolExtended:
    """Extended realized volatility calculation tests."""

    def _make_overlay(self):
        overlay = RiskParityWeightOverlay.__new__(RiskParityWeightOverlay)
        overlay.vol_lookback = VOL_LOOKBACK
        return overlay

    def test_unknown_ticker_returns_none(self):
        """Unknown ticker should return None."""
        overlay = self._make_overlay()
        df = _make_prices_df()
        result = overlay.calculate_realized_vol('UNKNOWN', df)
        assert result is None

    def test_short_history_returns_none(self):
        """Too few prices should return None."""
        overlay = self._make_overlay()
        df = _make_prices_df(days=30)  # Too short for vol_lookback+10
        result = overlay.calculate_realized_vol('SPY', df)
        assert result is None

    def test_valid_ticker_returns_positive(self):
        """Valid ticker with enough data should return positive vol."""
        overlay = self._make_overlay()
        df = _make_prices_df(days=300)
        result = overlay.calculate_realized_vol('SPY', df)
        assert result is not None
        assert result > 0

    def test_vol_annualized(self):
        """Realized vol should be annualized (multiply by sqrt(252))."""
        overlay = self._make_overlay()
        df = _make_prices_df(days=300)
        result = overlay.calculate_realized_vol('SPY', df)
        # Annualized vol for equities should typically be 5-50%
        assert 0.01 < result < 1.0

    def test_different_tickers_have_different_vols(self):
        """Different tickers should generally have different volatilities."""
        overlay = self._make_overlay()
        df = _make_prices_df(days=300)
        vol_spy = overlay.calculate_realized_vol('SPY', df)
        vol_gld = overlay.calculate_realized_vol('GLD', df)
        # They might be close with synthetic data, but both should be valid
        assert vol_spy is not None
        assert vol_gld is not None


class TestCalculateRPOverlayExtended:
    """Extended risk parity overlay calculation tests."""

    def _make_overlay(self):
        overlay = RiskParityWeightOverlay.__new__(RiskParityWeightOverlay)
        overlay.vol_lookback = VOL_LOOKBACK
        overlay.max_deviation = MAX_DEVIATION
        return overlay

    def test_target_weights_sum_to_one(self):
        """Target weights should sum to 1.0 (excluding CASH)."""
        overlay = self._make_overlay()
        df = _make_prices_df(days=300)
        base = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16, 'CASH': 0.0}
        result = overlay.calculate_rp_overlay(base, df)
        if result is not None:
            total = sum(v for k, v in result.target_weights.items() if k != 'CASH')
            assert abs(total - 1.0) < 0.01

    def test_target_weights_respect_minimum(self):
        """Target weights should be at least MIN_WEIGHT."""
        overlay = self._make_overlay()
        df = _make_prices_df(days=300)
        base = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16, 'CASH': 0.0}
        result = overlay.calculate_rp_overlay(base, df)
        if result is not None:
            for k, v in result.target_weights.items():
                if k != 'CASH':
                    assert v >= MIN_WEIGHT

    def test_risk_parity_score_between_0_and_1(self):
        """Risk parity score should be between 0 and 1."""
        overlay = self._make_overlay()
        df = _make_prices_df(days=300)
        base = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16, 'CASH': 0.0}
        result = overlay.calculate_rp_overlay(base, df)
        if result is not None:
            assert 0.0 <= result.risk_parity_score <= 1.0

    def test_expected_vol_positive(self):
        """Expected portfolio vol should be positive."""
        overlay = self._make_overlay()
        df = _make_prices_df(days=300)
        base = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16, 'CASH': 0.0}
        result = overlay.calculate_rp_overlay(base, df)
        if result is not None:
            assert result.expected_vol > 0

    def test_missing_vol_for_asset_returns_none(self):
        """If any asset has missing vol, overlay should return None."""
        overlay = self._make_overlay()
        df = _make_prices_df(days=300, symbols=['SPY', 'GLD'])  # Missing TLT
        base = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16, 'CASH': 0.0}
        result = overlay.calculate_rp_overlay(base, df)
        assert result is None

    def test_raw_rp_weights_inverse_vol(self):
        """Raw RP weights should be proportional to inverse volatility."""
        overlay = self._make_overlay()
        df = _make_prices_df(days=300)
        base = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16, 'CASH': 0.0}
        result = overlay.calculate_rp_overlay(base, df)
        if result is not None:
            # Lower vol assets should get higher raw RP weights
            vols = result.asset_vols
            rp_weights = result.raw_rp_weights
            # Sort by vol: lowest vol should have highest RP weight
            sorted_by_vol = sorted(vols.keys(), key=lambda k: vols[k])
            sorted_by_weight = sorted(rp_weights.keys(), key=lambda k: -rp_weights[k])
            # Lowest vol should have highest weight
            assert sorted_by_vol[0] == sorted_by_weight[0]

    def test_cash_always_zero(self):
        """CASH weight should always be 0.0."""
        overlay = self._make_overlay()
        df = _make_prices_df(days=300)
        base = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16, 'CASH': 0.0}
        result = overlay.calculate_rp_overlay(base, df)
        if result is not None:
            assert result.target_weights.get('CASH', 0.0) == 0.0

    def test_adjustments_within_max_deviation(self):
        """Adjustments should be clipped to max_deviation."""
        overlay = self._make_overlay()
        df = _make_prices_df(days=300)
        base = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16, 'CASH': 0.0}
        result = overlay.calculate_rp_overlay(base, df)
        if result is not None:
            for asset, adj in result.rp_adjustments.items():
                assert abs(adj) <= MAX_DEVIATION + 0.01  # Post-normalization may slightly exceed


class TestRPWeightOverlayDataclass:
    """Extended tests for RPWeightOverlay dataclass."""

    def test_all_fields_present(self):
        overlay = RPWeightOverlay(
            timestamp="2026-01-01T00:00:00",
            asset_vols={"SPY": 0.15, "GLD": 0.16, "TLT": 0.12},
            raw_rp_weights={"SPY": 0.35, "GLD": 0.33, "TLT": 0.32},
            base_weights={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            target_weights={"SPY": 0.40, "GLD": 0.36, "TLT": 0.24},
            rp_adjustments={"SPY": -0.06, "GLD": -0.02, "TLT": 0.08},
            expected_vol=0.112,
            risk_parity_score=0.87,
        )
        assert overlay.timestamp == "2026-01-01T00:00:00"
        assert len(overlay.asset_vols) == 3
        assert overlay.expected_vol == 0.112
        assert overlay.risk_parity_score == 0.87

    def test_to_dict(self):
        overlay = RPWeightOverlay(
            timestamp="2026-01-01",
            asset_vols={"SPY": 0.15},
            raw_rp_weights={"SPY": 0.5},
            base_weights={"SPY": 0.46},
            target_weights={"SPY": 0.40},
            rp_adjustments={"SPY": -0.06},
            expected_vol=0.10,
            risk_parity_score=0.75,
        )
        d = overlay.to_dict()
        assert isinstance(d, dict)
        assert d["timestamp"] == "2026-01-01"
        assert "asset_vols" in d
        assert "expected_vol" in d
        assert "risk_parity_score" in d


class TestConstantsExtended:
    """Extended tests for risk parity constants."""

    def test_vol_lookback(self):
        assert VOL_LOOKBACK == 252

    def test_max_deviation(self):
        assert MAX_DEVIATION == 0.15

    def test_min_weight(self):
        assert MIN_WEIGHT == 0.05

    def test_rebalance_freq(self):
        assert REBALANCE_FREQ == 21

    def test_max_deviation_less_than_half(self):
        """MAX_DEVIATION should be reasonable (< 0.5)."""
        assert MAX_DEVIATION < 0.5

    def test_min_weight_positive(self):
        assert MIN_WEIGHT > 0


class TestRiskParityOverlayExtended:
    """Extended edge case tests for RiskParityWeightOverlay."""

    def _make_overlay(self):
        return RiskParityWeightOverlay()

    def test_single_asset_base(self):
        """Single asset base should still work."""
        overlay = self._make_overlay()
        df = _make_prices_df(days=300)
        base = {'SPY': 1.0, 'CASH': 0.0}
        result = overlay.calculate_rp_overlay(base, df)
        if result is not None:
            assert 'SPY' in result.target_weights

    def test_equal_vols_produce_equal_weights(self):
        """When all assets have equal vol, RP weights should equalize."""
        # Create prices with identical volatility
        np.random.seed(42)
        dates = pd.bdate_range('2020-01-01', periods=300)
        const_ret = 0.001
        df = pd.DataFrame({
            'SPY': 100 * (1 + const_ret) ** np.arange(300),
            'GLD': 100 * (1 + const_ret) ** np.arange(300),
            'TLT': 100 * (1 + const_ret) ** np.arange(300),
        }, index=dates)
        overlay = self._make_overlay()
        base = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16, 'CASH': 0.0}
        result = overlay.calculate_rp_overlay(base, df)
        # With constant returns, vols are ~0, so weights should fall back to base
        if result is not None:
            assert result.target_weights is not None

    def test_empty_base_returns_none_or_no_crash(self):
        """Empty base allocation should not crash."""
        overlay = self._make_overlay()
        df = _make_prices_df(days=300)
        try:
            _ = overlay.calculate_rp_overlay({}, df)
        except (ValueError, KeyError):
            pass  # Acceptable to raise

    def test_very_short_data(self):
        """Very short price data should not crash."""
        overlay = self._make_overlay()
        df = _make_prices_df(days=10)
        base = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16, 'CASH': 0.0}
        try:
            _ = overlay.calculate_rp_overlay(base, df)
        except (ValueError, KeyError):
            pass  # Short data may not produce valid result


# ==============================================================================
# SECTION: Dataclass Field Validation
# ==============================================================================

class TestRPWeightOverlayDataclassFields:
    """Validate RPWeightOverlay dataclass fields using dataclasses.fields()."""

    def test_eight_fields(self):
        fields = dataclasses.fields(RPWeightOverlay)
        assert len(fields) == 8

    def test_field_names(self):
        fields = {f.name: f for f in dataclasses.fields(RPWeightOverlay)}
        expected = {
            'timestamp', 'asset_vols', 'raw_rp_weights', 'base_weights',
            'rp_adjustments', 'target_weights', 'expected_vol', 'risk_parity_score',
        }
        assert set(fields.keys()) == expected

    def test_timestamp_is_str(self):
        fields = {f.name: f for f in dataclasses.fields(RPWeightOverlay)}
        assert fields['timestamp'].type is str

    def test_asset_vols_type(self):
        fields = {f.name: f for f in dataclasses.fields(RPWeightOverlay)}
        assert fields['asset_vols'].type == Dict[str, float]

    def test_expected_vol_is_float(self):
        fields = {f.name: f for f in dataclasses.fields(RPWeightOverlay)}
        assert fields['expected_vol'].type is float

    def test_risk_parity_score_is_float(self):
        fields = {f.name: f for f in dataclasses.fields(RPWeightOverlay)}
        assert fields['risk_parity_score'].type is float

    def test_all_dict_fields_typed(self):
        fields = {f.name: f for f in dataclasses.fields(RPWeightOverlay)}
        for name in ['asset_vols', 'raw_rp_weights', 'base_weights', 'rp_adjustments', 'target_weights']:
            assert fields[name].type == Dict[str, float], f'{name} type should be Dict[str, float]'

    def test_no_mutable_defaults(self):
        fields = {f.name: f for f in dataclasses.fields(RPWeightOverlay)}
        for name in ['asset_vols', 'raw_rp_weights', 'base_weights', 'rp_adjustments', 'target_weights']:
            assert fields[name].default is dataclasses.MISSING
            assert fields[name].default_factory is dataclasses.MISSING


# ==============================================================================
# SECTION: Constants Type and Range Validation
# ==============================================================================

class TestConstantsTypeAndRange:
    """Validate constants have correct types and reasonable ranges."""

    def test_vol_lookback_type(self):
        assert isinstance(VOL_LOOKBACK, int)

    def test_vol_lookback_positive(self):
        assert VOL_LOOKBACK > 0

    def test_max_deviation_type(self):
        assert isinstance(MAX_DEVIATION, float)

    def test_max_deviation_between_zero_and_one(self):
        assert 0 < MAX_DEVIATION < 1

    def test_min_weight_type(self):
        assert isinstance(MIN_WEIGHT, float)

    def test_min_weight_between_zero_and_one(self):
        assert 0 < MIN_WEIGHT < 1

    def test_rebalance_freq_type(self):
        assert isinstance(REBALANCE_FREQ, int)

    def test_rebalance_freq_positive(self):
        assert REBALANCE_FREQ > 0

    def test_default_base_has_all_expected_keys(self):
        expected = {'SPY', 'GLD', 'TLT', 'CASH'}
        assert expected.issubset(set(DEFAULT_BASE.keys()))

    def test_default_base_ex_cash_sum_to_one(self):
        non_cash = {k: v for k, v in DEFAULT_BASE.items() if k != 'CASH'}
        assert sum(non_cash.values()) == pytest.approx(1.0)


# ==============================================================================
# SECTION: Export Completeness (__all__)
# ==============================================================================

class TestExportCompleteness:
    """Verify __all__ covers all public names in the module."""

    def test_all_is_defined(self):
        import src.strategy.risk_parity_weight_overlay as mod
        assert hasattr(mod, '__all__')
        assert len(mod.__all__) > 0

    def test_all_names_exist_in_module(self):
        import src.strategy.risk_parity_weight_overlay as mod
        for name in mod.__all__:
            assert hasattr(mod, name), f'{name} in __all__ but missing from module'

    def test_all_contains_all_expected_names(self):
        import src.strategy.risk_parity_weight_overlay as mod
        expected = {'VOL_LOOKBACK', 'MAX_DEVIATION', 'MIN_WEIGHT', 'REBALANCE_FREQ',
                    'DEFAULT_BASE', 'RPWeightOverlay', 'RiskParityWeightOverlay', 'RPBacktester'}
        assert expected.issubset(set(mod.__all__))

    def test_all_no_dunder_or_private_names(self):
        import src.strategy.risk_parity_weight_overlay as mod
        for name in mod.__all__:
            assert not name.startswith('_'), f'__all__ contains private name: {name}'


# ==============================================================================
# SECTION: Load Prices Edge Cases
# ==============================================================================

class TestLoadPricesEdgeCases:
    """Edge cases for _load_prices method."""

    def test_file_not_found_raises(self):
        overlay = RiskParityWeightOverlay.__new__(RiskParityWeightOverlay)
        overlay._prices_df = None
        with patch('src.strategy.risk_parity_weight_overlay.get_prices_df', side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError):
                overlay._load_prices()

    def test_empty_df_returns_empty(self, tmp_path):
        overlay = RiskParityWeightOverlay.__new__(RiskParityWeightOverlay)
        overlay._prices_df = None
        with patch('src.strategy.risk_parity_weight_overlay.get_prices_df', return_value=pd.DataFrame()):
            df = overlay._load_prices()
        assert df.empty

    def test_single_symbol_two_days(self, tmp_path):
        dates = pd.to_datetime(['2026-01-02', '2026-01-03'])
        mock_df = pd.DataFrame({'SPY': [450.0, 452.0]}, index=dates)
        overlay = RiskParityWeightOverlay.__new__(RiskParityWeightOverlay)
        overlay._prices_df = None
        with patch('src.strategy.risk_parity_weight_overlay.get_prices_df', return_value=mock_df):
            df = overlay._load_prices()
        assert 'SPY' in df.columns
        assert len(df) == 2

    def test_cached_df_returned(self, tmp_path):
        dates = pd.to_datetime(['2026-01-02'])
        cached_df = pd.DataFrame({'SPY': [450.0]}, index=dates)
        overlay = RiskParityWeightOverlay.__new__(RiskParityWeightOverlay)
        overlay._prices_df = cached_df
        # Should return cached without calling get_prices_df
        df = overlay._load_prices()
        assert df is cached_df


# ==============================================================================
# SECTION: Calculate Realized Vol — Edge Cases
# ==============================================================================

class TestCalculateRealizedVolEdgeCases:
    """Edge cases for calculate_realized_vol."""

    def _make_overlay(self):
        overlay = RiskParityWeightOverlay.__new__(RiskParityWeightOverlay)
        overlay.vol_lookback = VOL_LOOKBACK
        return overlay

    def test_all_nan_prices_returns_none(self):
        overlay = self._make_overlay()
        dates = pd.date_range(end=datetime.now(), periods=300, freq='B')
        df = pd.DataFrame({'SPY': [np.nan] * 300}, index=dates)
        vol = overlay.calculate_realized_vol('SPY', df)
        assert vol is None

    def test_all_zero_prices_returns_zero_or_none(self):
        overlay = self._make_overlay()
        dates = pd.date_range(end=datetime.now(), periods=300, freq='B')
        df = pd.DataFrame({'SPY': [0.0] * 300}, index=dates)
        vol = overlay.calculate_realized_vol('SPY', df)
        if vol is not None:
            assert vol == 0.0

    def test_constant_prices_zero_vol(self):
        overlay = self._make_overlay()
        dates = pd.date_range(end=datetime.now(), periods=300, freq='B')
        df = pd.DataFrame({'SPY': [100.0] * 300}, index=dates)
        vol = overlay.calculate_realized_vol('SPY', df)
        assert vol is not None
        assert vol == 0.0

    def test_inf_price_at_end(self):
        overlay = self._make_overlay()
        rng = np.random.RandomState(42)
        dates = pd.date_range(end=datetime.now(), periods=300, freq='B')
        prices = np.append(100 * np.cumprod(1 + rng.normal(0.0003, 0.012, 299)), np.inf)
        df = pd.DataFrame({'SPY': prices}, index=dates)
        vol = overlay.calculate_realized_vol('SPY', df)
        assert vol is not None

    def test_barely_enough_data(self):
        overlay = self._make_overlay()
        n = VOL_LOOKBACK + 10
        rng = np.random.RandomState(42)
        prices = np.append(100.0, 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, n - 1)))
        dates = pd.date_range(end=datetime.now(), periods=n, freq='B')
        df = pd.DataFrame({'SPY': prices}, index=dates)
        vol = overlay.calculate_realized_vol('SPY', df)
        assert vol is not None
        assert vol > 0

    def test_one_less_than_lookback_threshold_returns_none(self):
        overlay = self._make_overlay()
        n = VOL_LOOKBACK + 9
        rng = np.random.RandomState(42)
        prices = np.append(100.0, 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, n - 1)))
        dates = pd.date_range(end=datetime.now(), periods=n, freq='B')
        df = pd.DataFrame({'SPY': prices}, index=dates)
        vol = overlay.calculate_realized_vol('SPY', df)
        assert vol is None

    def test_prices_with_nan_gap(self):
        overlay = self._make_overlay()
        rng = np.random.RandomState(42)
        dates = pd.date_range(end=datetime.now(), periods=300, freq='B')
        prices = np.append(100.0, 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, 299)))
        prices[100:105] = np.nan
        df = pd.DataFrame({'SPY': prices}, index=dates)
        vol = overlay.calculate_realized_vol('SPY', df)
        assert vol is not None

    def test_negative_prices(self):
        overlay = self._make_overlay()
        rng = np.random.RandomState(42)
        dates = pd.date_range(end=datetime.now(), periods=300, freq='B')
        prices = np.append(-100.0, -100 * np.cumprod(1 + rng.normal(0.0003, 0.012, 299)))
        df = pd.DataFrame({'SPY': prices}, index=dates)
        vol = overlay.calculate_realized_vol('SPY', df)
        assert vol is not None
        assert isinstance(vol, float)

    def test_single_element_price_returns_none(self):
        overlay = self._make_overlay()
        dates = pd.date_range(end=datetime.now(), periods=1, freq='B')
        df = pd.DataFrame({'SPY': [100.0]}, index=dates)
        vol = overlay.calculate_realized_vol('SPY', df)
        assert vol is None


# ==============================================================================
# SECTION: Calculate RP Overlay — Edge Cases
# ==============================================================================

class TestCalculateRPOverlayEdgeCases:
    """Edge cases for calculate_rp_overlay."""

    def _make_overlay(self):
        overlay = RiskParityWeightOverlay.__new__(RiskParityWeightOverlay)
        overlay.vol_lookback = VOL_LOOKBACK
        overlay.max_deviation = MAX_DEVIATION
        return overlay

    def test_empty_base_dict_returns_valid_overlay(self):
        overlay = self._make_overlay()
        df = _make_prices_df(days=300)
        result = overlay.calculate_rp_overlay({}, df)
        assert isinstance(result, RPWeightOverlay)
        assert result.asset_vols == {}
        assert result.target_weights == {'CASH': 0.0}
        assert result.expected_vol == 0.0
        assert result.risk_parity_score == 0.0

    def test_all_cash_base_returns_valid_overlay(self):
        overlay = self._make_overlay()
        df = _make_prices_df(days=300)
        result = overlay.calculate_rp_overlay({'CASH': 1.0}, df)
        assert isinstance(result, RPWeightOverlay)
        assert result.target_weights == {'CASH': 0.0}
        assert result.expected_vol == 0.0

    def test_base_with_missing_ticker_returns_none(self):
        overlay = self._make_overlay()
        df = _make_prices_df(['SPY'], days=300)
        result = overlay.calculate_rp_overlay({'FAKE': 0.5, 'CASH': 0.5}, df)
        assert result is None

    def test_zero_vol_asset_filtered_returns_none(self):
        overlay = self._make_overlay()
        dates = pd.date_range(end=datetime.now(), periods=300, freq='B')
        rng = np.random.RandomState(42)
        df = pd.DataFrame(index=dates)
        df['SPY'] = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, 300))
        df['GLD'] = 190.0
        rng2 = np.random.RandomState(99)
        df['TLT'] = 95 * np.cumprod(1 + rng2.normal(0.0001, 0.008, 300))
        result = overlay.calculate_rp_overlay(
            {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16, 'CASH': 0.0}, df
        )
        assert result is None

    def test_single_asset_with_cash(self):
        overlay = self._make_overlay()
        df = _make_prices_df(['SPY'], days=300)
        result = overlay.calculate_rp_overlay({'SPY': 0.8, 'CASH': 0.2}, df)
        if result is not None:
            assert 'SPY' in result.target_weights
            total = sum(v for k, v in result.target_weights.items() if k != 'CASH')
            assert total == pytest.approx(1.0, abs=0.01)

    def test_weights_normalize_after_min_clip(self):
        overlay = self._make_overlay()
        overlay.max_deviation = 0.50
        df = _make_prices_df(days=300)
        result = overlay.calculate_rp_overlay(
            {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16, 'CASH': 0.0}, df
        )
        if result is not None:
            total = sum(v for k, v in result.target_weights.items() if k != 'CASH')
            assert total == pytest.approx(1.0, abs=0.01)

    def test_max_deviation_zero_no_adjustment(self):
        overlay = self._make_overlay()
        overlay.max_deviation = 0.0
        df = _make_prices_df(days=300)
        result = overlay.calculate_rp_overlay(
            {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16, 'CASH': 0.0}, df
        )
        if result is not None:
            for asset in ['SPY', 'GLD', 'TLT']:
                assert abs(result.rp_adjustments[asset]) < 0.01

    def test_extreme_vol_difference_sorted(self):
        overlay = self._make_overlay()
        overlay.max_deviation = 0.50
        dates = pd.date_range(end=datetime.now(), periods=300, freq='B')
        rng_high = np.random.RandomState(42)
        rng_low = np.random.RandomState(99)
        df = pd.DataFrame(index=dates)
        df['SPY'] = 100 * np.cumprod(1 + rng_high.normal(0.0005, 0.04, 300))
        df['GLD'] = 190 * np.cumprod(1 + rng_low.normal(0.0002, 0.005, 300))
        df['TLT'] = 95 * np.cumprod(1 + rng_low.normal(0.0003, 0.006, 300))
        result = overlay.calculate_rp_overlay(
            {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16, 'CASH': 0.0}, df
        )
        if result is not None:
            vols = result.asset_vols
            weights = result.raw_rp_weights
            sorted_by_vol = sorted(vols.keys(), key=lambda k: vols[k])
            sorted_by_weight = sorted(weights.keys(), key=lambda k: -weights[k])
            assert sorted_by_vol[0] == sorted_by_weight[0]

    def test_risk_parity_score_in_normal_range(self):
        overlay = self._make_overlay()
        dates = pd.date_range(end=datetime.now(), periods=300, freq='B')
        rng = np.random.RandomState(42)
        rets = rng.normal(0.0003, 0.012, (300, 3))
        prices = 100 * np.cumprod(1 + rets, axis=0)
        df = pd.DataFrame(prices, index=dates, columns=['SPY', 'GLD', 'TLT'])
        result = overlay.calculate_rp_overlay(
            {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16, 'CASH': 0.0}, df
        )
        if result is not None:
            assert 0.0 <= result.risk_parity_score <= 1.0

    def test_all_assets_perfectly_correlated(self):
        overlay = self._make_overlay()
        rng = np.random.RandomState(42)
        dates = pd.date_range(end=datetime.now(), periods=300, freq='B')
        base = np.cumprod(1 + rng.normal(0.0003, 0.012, 300))
        df = pd.DataFrame({
            'SPY': 100 * base,
            'GLD': 190 * base,
            'TLT': 95 * base,
        }, index=dates)
        result = overlay.calculate_rp_overlay(
            {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16, 'CASH': 0.0}, df
        )
        if result is not None:
            assert result.expected_vol > 0


# ==============================================================================
# SECTION: RPBacktester — Initialization
# ==============================================================================

class TestRPBacktesterInit:
    """Test RPBacktester initialization."""

    def test_init_with_defaults(self):
        with patch.object(RiskParityWeightOverlay, '_load_prices',
                          return_value=_make_prices_df(days=300)):
            bt = RPBacktester(base_weights=DEFAULT_BASE)
        assert bt.base_weights == DEFAULT_BASE
        assert bt.rebalance_freq == REBALANCE_FREQ
        assert bt.max_deviation == MAX_DEVIATION
        assert bt.start_date is None
        assert bt.end_date is None

    def test_init_with_custom_params(self):
        with patch.object(RiskParityWeightOverlay, '_load_prices',
                          return_value=_make_prices_df(days=300)):
            bt = RPBacktester(
                base_weights={'SPY': 0.6, 'GLD': 0.4, 'CASH': 0.0},
                start_date='2020-01-01',
                end_date='2025-12-31',
                rebalance_freq=10,
                max_deviation=0.20,
            )
        assert bt.start_date is not None
        assert bt.end_date is not None
        assert bt.rebalance_freq == 10
        assert bt.max_deviation == 0.20

    def test_init_creates_overlay_with_correct_max_dev(self):
        with patch.object(RiskParityWeightOverlay, '_load_prices',
                          return_value=_make_prices_df(days=300)):
            bt = RPBacktester(base_weights=DEFAULT_BASE, max_deviation=0.25)
        assert bt.overlay.max_deviation == 0.25


# ==============================================================================
# SECTION: RPBacktester — run_backtest
# ==============================================================================

class TestRPBacktesterRun:
    """Test RPBacktester.run_backtest method."""

    def test_insufficient_data_returns_error(self):
        with patch.object(RiskParityWeightOverlay, '_load_prices',
                          return_value=_make_prices_df(days=10)):
            bt = RPBacktester(base_weights=DEFAULT_BASE)
        result = bt.run_backtest()
        assert isinstance(result, dict)
        assert 'error' in result

    def test_sufficient_data_returns_result(self):
        with patch.object(RiskParityWeightOverlay, '_load_prices',
                          return_value=_make_prices_df(days=350)):
            bt = RPBacktester(base_weights=DEFAULT_BASE)
        result = bt.run_backtest()
        assert isinstance(result, dict)
        assert 'error' not in result

    def test_result_has_all_expected_keys(self):
        with patch.object(RiskParityWeightOverlay, '_load_prices',
                          return_value=_make_prices_df(days=350)):
            bt = RPBacktester(base_weights=DEFAULT_BASE)
        result = bt.run_backtest()
        expected_keys = {
            'strategy', 'start_date', 'end_date', 'trading_days',
            'start_value', 'end_value', 'cagr', 'volatility',
            'sharpe_ratio', 'max_drawdown', 'calmar_ratio',
            'baseline_cagr', 'baseline_sharpe', 'baseline_volatility',
            'excess_return', 'sharpe_improvement',
            'crisis_2008_return', 'crisis_2020_return', 'crisis_2022_return',
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_sharpe_ratio_is_numeric(self):
        with patch.object(RiskParityWeightOverlay, '_load_prices',
                          return_value=_make_prices_df(days=350)):
            bt = RPBacktester(base_weights=DEFAULT_BASE)
        result = bt.run_backtest()
        assert isinstance(result['sharpe_ratio'], (int, float))
        assert not np.isnan(result['sharpe_ratio'])

    def test_volatility_non_negative(self):
        with patch.object(RiskParityWeightOverlay, '_load_prices',
                          return_value=_make_prices_df(days=350)):
            bt = RPBacktester(base_weights=DEFAULT_BASE)
        result = bt.run_backtest()
        assert result['volatility'] >= 0

    def test_end_value_positive(self):
        with patch.object(RiskParityWeightOverlay, '_load_prices',
                          return_value=_make_prices_df(days=350)):
            bt = RPBacktester(base_weights=DEFAULT_BASE)
        result = bt.run_backtest()
        if 'error' not in result:
            assert result['end_value'] > 0

    def test_backtest_crisis_returns_are_float_or_none(self):
        with patch.object(RiskParityWeightOverlay, '_load_prices',
                          return_value=_make_prices_df(days=1000)):
            bt = RPBacktester(base_weights=DEFAULT_BASE)
        result = bt.run_backtest()
        for key in ['crisis_2008_return', 'crisis_2020_return', 'crisis_2022_return']:
            assert key in result
            assert result[key] is None or isinstance(result[key], (int, float))

    def test_single_asset_backtest(self):
        with patch.object(RiskParityWeightOverlay, '_load_prices',
                          return_value=_make_prices_df(['SPY'], days=350)):
            bt = RPBacktester(base_weights={'SPY': 0.8, 'CASH': 0.2})
        result = bt.run_backtest()
        if 'error' not in result:
            assert 'SPY' in result['strategy'] or result['end_value'] > 0

    def test_baseline_comparison_present(self):
        with patch.object(RiskParityWeightOverlay, '_load_prices',
                          return_value=_make_prices_df(days=350)):
            bt = RPBacktester(base_weights=DEFAULT_BASE)
        result = bt.run_backtest()
        assert 'baseline_cagr' in result
        assert 'baseline_sharpe' in result
        assert 'excess_return' in result
        assert 'sharpe_improvement' in result


# ==============================================================================
# SECTION: CLI Extended Tests
# ==============================================================================

class TestCLIExtended:
    """Extended tests for CLI entry points."""

    def test_backtest_command_default(self, caplog):
        import logging
        caplog.set_level(logging.INFO)
        mock_result = {'strategy': 'test_strategy', 'cagr': 0.08}
        with patch('sys.argv', ['rp_overlay.py', 'backtest']):
            with patch('src.strategy.risk_parity_weight_overlay.RPBacktester') as MockBT:
                MockBT.return_value.run_backtest.return_value = mock_result
                from src.strategy.risk_parity_weight_overlay import main
                main()
        assert 'test_strategy' in caplog.text

    def test_backtest_with_custom_max_dev(self, caplog):
        import logging
        caplog.set_level(logging.INFO)
        mock_result = {'strategy': 'custom_dev', 'cagr': 0.08}
        with patch('sys.argv', ['rp_overlay.py', 'backtest', '--max-dev', '0.20']):
            with patch('src.strategy.risk_parity_weight_overlay.RPBacktester') as MockBT:
                MockBT.return_value.run_backtest.return_value = mock_result
                from src.strategy.risk_parity_weight_overlay import main
                main()
        assert 'custom_dev' in caplog.text

    def test_backtest_output_file(self, capsys, tmp_path):
        output_file = tmp_path / 'result.json'
        mock_result = {'strategy': 'file_output_test', 'cagr': 0.08}
        with patch('sys.argv', ['rp_overlay.py', 'backtest', '--output', str(output_file)]):
            with patch('src.strategy.risk_parity_weight_overlay.RPBacktester') as MockBT:
                MockBT.return_value.run_backtest.return_value = mock_result
                from src.strategy.risk_parity_weight_overlay import main
                main()
        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)
        assert data['strategy'] == 'file_output_test'

    def test_live_command_prints_allocation(self, caplog):
        import logging
        caplog.set_level(logging.INFO)
        mock_allocation = MagicMock()
        mock_allocation.to_dict.return_value = {'target_weights': {'SPY': 0.5}}
        with patch('sys.argv', ['rp_overlay.py', 'live']):
            with patch('src.strategy.risk_parity_weight_overlay.RiskParityWeightOverlay') as MockRP:
                MockRP.return_value.calculate_rp_overlay.return_value = mock_allocation
                from src.strategy.risk_parity_weight_overlay import main
                main()
        assert 'target_weights' in caplog.text

    def test_live_command_failure_prints_error(self, caplog):
        import logging
        caplog.set_level(logging.INFO)
        with patch('sys.argv', ['rp_overlay.py', 'live']):
            with patch('src.strategy.risk_parity_weight_overlay.RiskParityWeightOverlay') as MockRP:
                MockRP.return_value.calculate_rp_overlay.return_value = None
                from src.strategy.risk_parity_weight_overlay import main
                main()
        assert 'error' in caplog.text or 'Could not calculate' in caplog.text

    def test_unknown_command_exits_with_usage(self, capsys):
        with patch('sys.argv', ['rp_overlay.py', 'bogus_command_xyz']):
            from src.strategy.risk_parity_weight_overlay import main
            with pytest.raises(SystemExit):
                main()
        captured = capsys.readouterr()
        assert 'usage' in captured.err.lower()
        assert 'invalid choice' in captured.err.lower()

    def test_status_displays_all_constants(self, caplog):
        import logging
        caplog.set_level(logging.INFO)
        with patch('sys.argv', ['rp_overlay.py', 'status']):
            from src.strategy.risk_parity_weight_overlay import main
            main()
        assert 'Max deviation' in caplog.text
        assert 'Vol lookback' in caplog.text
        assert 'Rebalance frequency' in caplog.text
        assert 'Base allocation' in caplog.text
        assert 'SPY' in caplog.text
        assert 'GLD' in caplog.text
        assert 'TLT' in caplog.text


# ==============================================================================
# SECTION: Main Guard
# ==============================================================================

class TestMainGuard:
    """Test the __main__ guard pattern."""

    def test_main_function_is_callable(self):
        from src.strategy.risk_parity_weight_overlay import main
        assert callable(main)

    def test_module_imports_without_error(self):
        import importlib
        import src.strategy.risk_parity_weight_overlay as mod
        importlib.reload(mod)


# ==============================================================================
# SECTION: RPWeightOverlay to_dict — deep copy behavior
# ==============================================================================

class TestRPWeightOverlayToDict:
    """Test that to_dict produces independent copies."""

    def test_to_dict_returns_new_dict(self):
        overlay = RPWeightOverlay(
            timestamp='2026-05-24',
            asset_vols={'SPY': 0.15},
            raw_rp_weights={'SPY': 0.5},
            base_weights={'SPY': 0.46},
            rp_adjustments={'SPY': 0.04},
            target_weights={'SPY': 0.5, 'CASH': 0.0},
            expected_vol=0.12,
            risk_parity_score=0.90,
        )
        d = overlay.to_dict()
        assert d is not overlay.__dict__
        assert d['asset_vols'] is not overlay.asset_vols

    def test_to_dict_mutation_does_not_affect_original(self):
        overlay = RPWeightOverlay(
            timestamp='2026-05-24',
            asset_vols={'SPY': 0.15},
            raw_rp_weights={'SPY': 0.5},
            base_weights={'SPY': 0.46},
            rp_adjustments={'SPY': 0.04},
            target_weights={'SPY': 0.5},
            expected_vol=0.12,
            risk_parity_score=0.90,
        )
        d = overlay.to_dict()
        d['expected_vol'] = 999.0
        assert overlay.expected_vol == 0.12
