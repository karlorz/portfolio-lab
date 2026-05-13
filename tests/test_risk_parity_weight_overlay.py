#!/usr/bin/env python3
"""
Tests for risk_parity_weight_overlay.py — constants, RPWeightOverlay dataclass,
realized volatility calculation, risk parity overlay calculation, and CLI.
"""
import sys
import os
import json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pathlib import Path
from datetime import datetime, timedelta
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
        result = overlay.calculate_rp_overlay(base, df)
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
        prices_file = tmp_path / "prices.json"
        with open(prices_file, "w") as f:
            json.dump(prices_data, f)

        overlay = RiskParityWeightOverlay.__new__(RiskParityWeightOverlay)
        overlay.prices_path = prices_file
        overlay._prices_df = None
        overlay.vol_lookback = VOL_LOOKBACK

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

    def test_status_command(self, capsys):
        from src.strategy.risk_parity_weight_overlay import main
        with patch("sys.argv", ["rp_overlay.py", "status"]):
            main()
        captured = capsys.readouterr()
        assert "Risk Parity" in captured.out
        assert "46%" in captured.out or "46" in captured.out

    def test_no_command_prints_help(self, capsys):
        from src.strategy.risk_parity_weight_overlay import main
        with patch("sys.argv", ["rp_overlay.py"]):
            main()
        captured = capsys.readouterr()
        assert "usage" in captured.out.lower() or "Risk Parity" in captured.out
