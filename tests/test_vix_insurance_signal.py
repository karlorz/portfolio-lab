#!/usr/bin/env python3
"""
Tests for vix_insurance_signal.py — InsuranceSignal enum, VIXInsuranceSignal dataclass,
VIXInsuranceSignalGenerator constants, allocation logic, regime classification,
signal generation (mocked DB), and export.
"""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.signals.vix_insurance_signal import (
    InsuranceSignal,
    VIXInsuranceSignal,
    VIXInsuranceSignalGenerator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_generator(portfolio_value=100000):
    gen = VIXInsuranceSignalGenerator.__new__(VIXInsuranceSignalGenerator)
    gen.portfolio_value = portfolio_value
    gen.annual_budget = portfolio_value * 0.01
    gen.budget_used_ytd = 0.0
    return gen


def _make_signal(**overrides):
    defaults = dict(
        timestamp=datetime.now().isoformat(),
        signal="no_position",
        vix_spot=18.0,
        vix_regime="fair",
        portfolio_value=100000,
        allocation_percent=0.0,
        allocation_dollars=0.0,
        selected_strike=None,
        selected_expiration=None,
        days_to_expiration=None,
        premium_cost=None,
        delta=None,
        breakeven_vix=None,
        max_portfolio_allocation=0.01,
        annual_budget_used=0.0,
        budget_remaining=1000.0,
        portfolio_near_ath=True,
        existing_position=False,
        days_to_roll=None,
    )
    defaults.update(overrides)
    return VIXInsuranceSignal(**defaults)


# ---------------------------------------------------------------------------
# Enum Tests
# ---------------------------------------------------------------------------

class TestInsuranceSignalEnum:

    def test_values(self):
        assert InsuranceSignal.NO_POSITION.value == "no_position"
        assert InsuranceSignal.ENTER_FULL.value == "enter_full"
        assert InsuranceSignal.ENTER_HALF.value == "enter_half"
        assert InsuranceSignal.HOLD.value == "hold"
        assert InsuranceSignal.ROLL.value == "roll"
        assert InsuranceSignal.EXIT_PROFIT.value == "exit_profit"
        assert InsuranceSignal.EXIT_EXPIRE.value == "exit_expire"


# ---------------------------------------------------------------------------
# Dataclass Tests
# ---------------------------------------------------------------------------

class TestVIXInsuranceSignal:

    def test_to_dict(self):
        s = _make_signal()
        d = s.to_dict()
        assert "signal" in d
        assert "vix_spot" in d
        assert "allocation_percent" in d

    def test_to_dict_preserves_none(self):
        s = _make_signal(selected_strike=None, delta=None)
        d = s.to_dict()
        assert d["selected_strike"] is None


# ---------------------------------------------------------------------------
# Constants Tests
# ---------------------------------------------------------------------------

class TestConstants:

    def test_max_allocation(self):
        assert VIXInsuranceSignalGenerator.MAX_ALLOCATION == 0.01

    def test_max_single_trade(self):
        assert VIXInsuranceSignalGenerator.MAX_SINGLE_TRADE == 0.005

    def test_vix_thresholds(self):
        assert VIXInsuranceSignalGenerator.VIX_CHEAP == 16
        assert VIXInsuranceSignalGenerator.VIX_FAIR == 20
        assert VIXInsuranceSignalGenerator.VIX_EXPENSIVE == 22

    def test_profit_take(self):
        assert VIXInsuranceSignalGenerator.VIX_PROFIT_TAKE == 35

    def test_roll_days(self):
        assert VIXInsuranceSignalGenerator.ROLL_DAYS_BEFORE_EXPIRY == 5


# ---------------------------------------------------------------------------
# _calculate_allocation Tests
# ---------------------------------------------------------------------------

class TestCalculateAllocation:

    def test_cheap_vol_full_allocation(self):
        gen = _make_generator()
        assert gen._calculate_allocation(14) == 0.01
        assert gen._calculate_allocation(15.99) == 0.01

    def test_fair_vol_half_allocation(self):
        gen = _make_generator()
        assert gen._calculate_allocation(16) == 0.005
        assert gen._calculate_allocation(18) == 0.005
        assert gen._calculate_allocation(19.99) == 0.005

    def test_expensive_vol_no_allocation(self):
        gen = _make_generator()
        assert gen._calculate_allocation(20) == 0.0
        assert gen._calculate_allocation(25) == 0.0
        assert gen._calculate_allocation(35) == 0.0

    def test_boundary_vix_16(self):
        gen = _make_generator()
        assert gen._calculate_allocation(16) == 0.005  # Exactly at CHEAP → fair

    def test_boundary_vix_20(self):
        gen = _make_generator()
        assert gen._calculate_allocation(20) == 0.0  # Exactly at FAIR → expensive


# ---------------------------------------------------------------------------
# _determine_vix_regime Tests
# ---------------------------------------------------------------------------

class TestDetermineVIXRegime:

    def test_cheap(self):
        gen = _make_generator()
        assert gen._determine_vix_regime(12) == "cheap"
        assert gen._determine_vix_regime(15.99) == "cheap"

    def test_fair(self):
        gen = _make_generator()
        assert gen._determine_vix_regime(16) == "fair"
        assert gen._determine_vix_regime(19.99) == "fair"

    def test_elevated(self):
        gen = _make_generator()
        assert gen._determine_vix_regime(20) == "elevated"
        assert gen._determine_vix_regime(21.99) == "elevated"

    def test_expensive(self):
        gen = _make_generator()
        assert gen._determine_vix_regime(22) == "expensive"
        assert gen._determine_vix_regime(35) == "expensive"

    def test_boundary_16(self):
        gen = _make_generator()
        assert gen._determine_vix_regime(16) == "fair"

    def test_boundary_20(self):
        gen = _make_generator()
        assert gen._determine_vix_regime(20) == "elevated"

    def test_boundary_22(self):
        gen = _make_generator()
        assert gen._determine_vix_regime(22) == "expensive"


# ---------------------------------------------------------------------------
# generate_signal Tests (mocked DB)
# ---------------------------------------------------------------------------

class TestGenerateSignal:

    def test_no_context_returns_no_position(self):
        gen = _make_generator()
        with patch.object(gen, '_load_vix_context', return_value={}):
            with patch.object(gen, '_load_candidates', return_value=[]):
                signal = gen.generate_signal()
                assert signal.signal == "no_position"
                assert signal.vix_spot == 0.0

    def test_cheap_vix_enter_full(self):
        gen = _make_generator()
        context = {
            'vix_spot': 14.0,
            'vix_9day': 13.0,
            'vix_3m': 15.0,
            'contango': 0.05,
            'history_30d': [14.0] * 30,
            'vix_30d_avg': 14.0,
            'timestamp': '2026-05-14',
        }
        candidates = [{'strike': 20.0, 'expiration_date': '2026-06-20',
                        'days_to_expiration': 37, 'premium': 1.5, 'delta': 0.30,
                        'breakeven_vix': 21.5}]
        with patch.object(gen, '_load_vix_context', return_value=context):
            with patch.object(gen, '_load_candidates', return_value=candidates):
                signal = gen.generate_signal()
                assert signal.signal == "enter_full"
                assert signal.allocation_percent == 0.01

    def test_fair_vix_enter_half(self):
        gen = _make_generator()
        context = {
            'vix_spot': 18.0,
            'vix_9day': 17.0,
            'vix_3m': 19.0,
            'contango': 0.03,
            'history_30d': [18.0] * 30,
            'vix_30d_avg': 18.0,
            'timestamp': '2026-05-14',
        }
        candidates = [{'strike': 22.0, 'expiration_date': '2026-06-20',
                        'days_to_expiration': 37, 'premium': 1.0, 'delta': 0.25,
                        'breakeven_vix': 23.0}]
        with patch.object(gen, '_load_vix_context', return_value=context):
            with patch.object(gen, '_load_candidates', return_value=candidates):
                signal = gen.generate_signal()
                assert signal.signal == "enter_half"
                assert signal.allocation_percent == 0.005

    def test_expensive_vix_no_position(self):
        gen = _make_generator()
        context = {
            'vix_spot': 25.0,
            'vix_9day': 24.0,
            'vix_3m': 26.0,
            'contango': 0.02,
            'history_30d': [25.0] * 30,
            'vix_30d_avg': 25.0,
            'timestamp': '2026-05-14',
        }
        with patch.object(gen, '_load_vix_context', return_value=context):
            with patch.object(gen, '_load_candidates', return_value=[]):
                signal = gen.generate_signal()
                assert signal.signal == "no_position"
                assert signal.allocation_percent == 0.0

    def test_allocation_dollars(self):
        gen = _make_generator(portfolio_value=200000)
        context = {
            'vix_spot': 14.0,
            'vix_9day': 13.0,
            'vix_3m': 15.0,
            'contango': 0.05,
            'history_30d': [14.0] * 30,
            'vix_30d_avg': 14.0,
            'timestamp': '2026-05-14',
        }
        candidates = [{'strike': 20.0, 'expiration_date': '2026-06-20',
                        'days_to_expiration': 37, 'premium': 1.5, 'delta': 0.30,
                        'breakeven_vix': 21.5}]
        with patch.object(gen, '_load_vix_context', return_value=context):
            with patch.object(gen, '_load_candidates', return_value=candidates):
                signal = gen.generate_signal()
                assert signal.allocation_dollars == pytest.approx(2000.0)

    def test_budget_tracking(self):
        gen = _make_generator()
        gen.budget_used_ytd = 500.0
        context = {
            'vix_spot': 14.0,
            'vix_9day': 13.0,
            'vix_3m': 15.0,
            'contango': 0.05,
            'history_30d': [14.0] * 30,
            'vix_30d_avg': 14.0,
            'timestamp': '2026-05-14',
        }
        candidates = [{'strike': 20.0, 'expiration_date': '2026-06-20',
                        'days_to_expiration': 37, 'premium': 1.5, 'delta': 0.30,
                        'breakeven_vix': 21.5}]
        with patch.object(gen, '_load_vix_context', return_value=context):
            with patch.object(gen, '_load_candidates', return_value=candidates):
                signal = gen.generate_signal()
                assert signal.annual_budget_used == 500.0


# ---------------------------------------------------------------------------
# export_signal Tests
# ---------------------------------------------------------------------------

class TestExportSignal:

    def test_export_creates_file(self, tmp_path):
        gen = _make_generator()
        import src.signals.vix_insurance_signal as mod
        old_path = mod.VIXInsuranceSignalGenerator.OUTPUT_PATH
        mod.VIXInsuranceSignalGenerator.OUTPUT_PATH = tmp_path / "signal.json"
        try:
            signal = _make_signal()
            gen.export_signal(signal)
            assert (tmp_path / "signal.json").exists()
        finally:
            mod.VIXInsuranceSignalGenerator.OUTPUT_PATH = old_path

    def test_export_valid_json(self, tmp_path):
        gen = _make_generator()
        import src.signals.vix_insurance_signal as mod
        old_path = mod.VIXInsuranceSignalGenerator.OUTPUT_PATH
        mod.VIXInsuranceSignalGenerator.OUTPUT_PATH = tmp_path / "signal.json"
        try:
            signal = _make_signal()
            gen.export_signal(signal)
            with open(tmp_path / "signal.json") as f:
                data = json.load(f)
            assert data["signal"] == "no_position"
        finally:
            mod.VIXInsuranceSignalGenerator.OUTPUT_PATH = old_path


# ---------------------------------------------------------------------------
# CLI Tests
# ---------------------------------------------------------------------------

class TestCLI:

    def test_no_args_prints_help(self, capsys):
        from src.signals.vix_insurance_signal import main
        with patch("sys.argv", ["vix_insurance.py"]):
            main()
        captured = capsys.readouterr()
        assert "usage" in captured.out.lower() or "VIX" in captured.out
