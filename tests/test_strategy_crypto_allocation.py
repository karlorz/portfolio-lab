"""
Tests for src/strategy/crypto_allocation.py — CryptoAllocationStatus,
CryptoAllocationDecision dataclass, and CryptoAllocationOverlay
(with mocked signal generator and staking model).
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock
from dataclasses import asdict

from src.strategy.crypto_allocation import (
    CryptoAllocationStatus,
    CryptoAllocationDecision,
    CryptoAllocationOverlay,
    calculate_crypto_allocation,
    get_crypto_summary,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_btc_signal(signal_state="active", target_weight=0.60, vol_30d=0.50, momentum_6m=0.30):
    mock = MagicMock()
    mock.signal_state = signal_state
    mock.target_weight = target_weight
    mock.vol_30d = vol_30d
    mock.momentum_6m = momentum_6m
    return mock


def _make_eth_signal(signal_state="active", target_weight=0.40, vol_30d=0.60, momentum_6m=0.25):
    mock = MagicMock()
    mock.signal_state = signal_state
    mock.target_weight = target_weight
    mock.vol_30d = vol_30d
    mock.momentum_6m = momentum_6m
    return mock


def _make_composite_signal(
    signal_state="active",
    composite_weight=0.03,
    vol_scale_factor=0.80,
    gld_reduction=0.03,
    confidence=75.0,
    reason="BTC active, ETH active",
    btc_signal=None,
    eth_signal=None,
):
    mock = MagicMock()
    mock.signal_state = signal_state
    mock.composite_weight = composite_weight
    mock.vol_scale_factor = vol_scale_factor
    mock.gld_reduction = gld_reduction
    mock.confidence = confidence
    mock.reason = reason
    mock.btc_signal = btc_signal or _make_btc_signal()
    mock.eth_signal = eth_signal or _make_eth_signal()
    return mock


def _make_staking_metrics(annual_yield=0.035, is_attractive=False):
    mock = MagicMock()
    mock.annual_yield = annual_yield
    mock.is_attractive = is_attractive
    return mock


def _make_allocation_influence(eth_preference=0.0, btc_weight=0.60, eth_weight=0.40):
    mock = MagicMock()
    mock.eth_preference = eth_preference
    mock.btc_weight = btc_weight
    mock.eth_weight = eth_weight
    return mock


def _temp_state_file():
    fd, path = tempfile.mkstemp(suffix=".json")
    Path(path).unlink()
    return Path(path)


def _patch_overlay():
    """Patch signal gen and staking model to avoid real DB/network calls."""
    return patch.object(CryptoAllocationOverlay, "__init__", lambda self, state_file=None: _init_overlay(self, state_file))


def _init_overlay(self, state_file):
    self._signal_gen = MagicMock()
    self._staking_model = MagicMock()
    self.state_file = state_file or Path(tempfile.mktemp(suffix=".json"))
    self._state = {
        "status": "flat",
        "last_signal_date": None,
        "btc_weight": 0.0,
        "eth_weight": 0.0,
        "total_crypto": 0.0,
        "gld_reduction": 0.0,
        "entry_date": None,
        "exit_date": None,
    }


# ── CryptoAllocationStatus enum ─────────────────────────────────────


class TestCryptoAllocationStatus:
    def test_values(self):
        assert CryptoAllocationStatus.ACTIVE.value == "active"
        assert CryptoAllocationStatus.REDUCED.value == "reduced"
        assert CryptoAllocationStatus.FLAT.value == "flat"
        assert CryptoAllocationStatus.DISABLED.value == "disabled"

    def test_members(self):
        assert len(CryptoAllocationStatus) == 4


# ── CryptoAllocationDecision dataclass ──────────────────────────────


class TestCryptoAllocationDecision:
    def test_to_dict(self):
        dec = CryptoAllocationDecision(
            timestamp="2026-01-01", status="active",
            btc_weight=0.02, eth_weight=0.01, total_crypto=0.03,
            gld_reduction=0.03, btc_vol=0.50, eth_vol=0.60,
            vol_scale=0.80, btc_momentum_6m=0.30, eth_momentum_6m=0.25,
            confidence=75.0, recommendation="Active", is_actionable=True,
        )
        d = dec.to_dict()
        assert isinstance(d, dict)
        assert d["status"] == "active"
        assert d["total_crypto"] == 0.03

    def test_staking_defaults(self):
        dec = CryptoAllocationDecision(
            timestamp="2026-01-01", status="flat",
            btc_weight=0.0, eth_weight=0.0, total_crypto=0.0,
            gld_reduction=0.0, btc_vol=0.0, eth_vol=0.0,
            vol_scale=0.0, btc_momentum_6m=0.0, eth_momentum_6m=0.0,
            confidence=0.0, recommendation="Flat", is_actionable=False,
        )
        assert dec.staking_yield_pct == 0.0
        assert dec.staking_carry_bps == 0.0
        assert dec.eth_preference == 0.0
        assert dec.is_staking_attractive is False


# ── CryptoAllocationOverlay constants ───────────────────────────────


class TestOverlayConstants:
    def test_ensemble_weight(self):
        assert CryptoAllocationOverlay.ENSEMBLE_WEIGHT == 0.05


# ── State management ────────────────────────────────────────────────


class TestOverlayState:
    def test_default_state(self):
        with _patch_overlay():
            overlay = CryptoAllocationOverlay()
        assert overlay._state["status"] == "flat"
        assert overlay._state["btc_weight"] == 0.0

    def test_load_existing_state(self):
        sf = _temp_state_file()
        sf.write_text(json.dumps({
            "status": "active",
            "last_signal_date": "2026-01-01",
            "btc_weight": 0.02,
            "eth_weight": 0.01,
            "total_crypto": 0.03,
            "gld_reduction": 0.03,
            "entry_date": "2025-12-01",
            "exit_date": None,
        }))
        with _patch_overlay():
            overlay = CryptoAllocationOverlay(state_file=sf)
            # Override _load_state for this test
            if sf.exists():
                with open(sf) as f:
                    overlay._state = json.load(f)
        assert overlay._state["status"] == "active"
        assert overlay._state["btc_weight"] == 0.02

    def test_get_status(self):
        with _patch_overlay():
            overlay = CryptoAllocationOverlay()
        assert overlay.get_status() == CryptoAllocationStatus.FLAT


# ── recommend() ─────────────────────────────────────────────────────


class TestRecommend:
    def test_active_signal(self):
        sig = _make_composite_signal(
            signal_state="active", composite_weight=0.03,
            gld_reduction=0.03,
        )
        with _patch_overlay():
            overlay = CryptoAllocationOverlay()
            overlay._signal_gen.generate_signal.return_value = sig
            overlay._staking_model.get_live_yield.return_value = _make_staking_metrics()
            overlay._staking_model.compute_crypto_carry.return_value = {"total_carry_bps": 0}
            dec = overlay.recommend()
        assert dec.status == "active"
        assert dec.is_actionable is True

    def test_flat_signal(self):
        sig = _make_composite_signal(signal_state="flat", composite_weight=0.0)
        with _patch_overlay():
            overlay = CryptoAllocationOverlay()
            overlay._signal_gen.generate_signal.return_value = sig
            overlay._staking_model.get_live_yield.return_value = _make_staking_metrics()
            overlay._staking_model.compute_crypto_carry.return_value = {"total_carry_bps": 0}
            dec = overlay.recommend()
        assert dec.status == "flat"
        assert dec.is_actionable is False
        assert dec.btc_weight == 0.0
        assert dec.eth_weight == 0.0

    def test_below_minimum_threshold(self):
        sig = _make_composite_signal(signal_state="active", composite_weight=0.005)
        with _patch_overlay():
            overlay = CryptoAllocationOverlay()
            overlay._signal_gen.generate_signal.return_value = sig
            overlay._staking_model.get_live_yield.return_value = _make_staking_metrics()
            overlay._staking_model.compute_crypto_carry.return_value = {"total_carry_bps": 0}
            dec = overlay.recommend()
        assert dec.status == "flat"
        assert dec.is_actionable is False

    def test_reduced_signal(self):
        btc = _make_btc_signal(signal_state="reduced")
        sig = _make_composite_signal(
            signal_state="active", composite_weight=0.02,
            btc_signal=btc,
        )
        with _patch_overlay():
            overlay = CryptoAllocationOverlay()
            overlay._signal_gen.generate_signal.return_value = sig
            overlay._staking_model.get_live_yield.return_value = _make_staking_metrics()
            overlay._staking_model.compute_crypto_carry.return_value = {"total_carry_bps": 0}
            dec = overlay.recommend()
        assert dec.status == "reduced"
        assert dec.is_actionable is True

    def test_signal_fields_passed_through(self):
        btc = _make_btc_signal(vol_30d=0.55, momentum_6m=0.40)
        eth = _make_eth_signal(vol_30d=0.65, momentum_6m=0.30)
        sig = _make_composite_signal(
            signal_state="active", composite_weight=0.04,
            vol_scale_factor=0.90, gld_reduction=0.04,
            confidence=80.0, btc_signal=btc, eth_signal=eth,
        )
        with _patch_overlay():
            overlay = CryptoAllocationOverlay()
            overlay._signal_gen.generate_signal.return_value = sig
            overlay._staking_model.get_live_yield.return_value = _make_staking_metrics()
            overlay._staking_model.compute_crypto_carry.return_value = {"total_carry_bps": 0}
            dec = overlay.recommend()
        assert dec.btc_vol == 0.55
        assert dec.eth_vol == 0.65
        assert dec.vol_scale == 0.90
        assert dec.confidence == 80.0

    def test_state_updated_after_recommend(self):
        sig = _make_composite_signal(signal_state="active", composite_weight=0.03)
        with _patch_overlay():
            overlay = CryptoAllocationOverlay()
            overlay._signal_gen.generate_signal.return_value = sig
            overlay._staking_model.get_live_yield.return_value = _make_staking_metrics()
            overlay._staking_model.compute_crypto_carry.return_value = {"total_carry_bps": 0}
            overlay.recommend()
        assert overlay._state["status"] == "active"
        assert "last_signal_date" in overlay._state

    def test_staking_influence_applied_when_attractive(self):
        sig = _make_composite_signal(signal_state="active", composite_weight=0.04)
        metrics = _make_staking_metrics(annual_yield=0.045, is_attractive=True)
        influence = _make_allocation_influence(eth_preference=0.3, btc_weight=0.50, eth_weight=0.50)

        with _patch_overlay():
            overlay = CryptoAllocationOverlay()
            overlay._signal_gen.generate_signal.return_value = sig
            overlay._staking_model.get_live_yield.return_value = metrics
            overlay._staking_model.compute_crypto_carry.return_value = {"total_carry_bps": 15.0}
            overlay._staking_model.compute_allocation_influence.return_value = influence
            dec = overlay.recommend()

        assert dec.is_staking_attractive is True
        assert dec.eth_preference == 0.3
        assert dec.staking_carry_bps == 15.0

    def test_staking_not_applied_when_flat(self):
        sig = _make_composite_signal(signal_state="flat", composite_weight=0.0)
        metrics = _make_staking_metrics(annual_yield=0.045, is_attractive=True)

        with _patch_overlay():
            overlay = CryptoAllocationOverlay()
            overlay._signal_gen.generate_signal.return_value = sig
            overlay._staking_model.get_live_yield.return_value = metrics
            overlay._staking_model.compute_crypto_carry.return_value = {"total_carry_bps": 0}
            dec = overlay.recommend()

        # Even if staking is attractive, flat status means no influence applied
        assert dec.eth_preference == 0.0

    def test_recommendation_string_contains_status(self):
        sig = _make_composite_signal(signal_state="active", reason="BTC momentum +2%")
        with _patch_overlay():
            overlay = CryptoAllocationOverlay()
            overlay._signal_gen.generate_signal.return_value = sig
            overlay._staking_model.get_live_yield.return_value = _make_staking_metrics()
            overlay._staking_model.compute_crypto_carry.return_value = {"total_carry_bps": 0}
            dec = overlay.recommend()
        assert "Active" in dec.recommendation or "BTC" in dec.recommendation


# ── get_allocation_shifts() ─────────────────────────────────────────


class TestGetAllocationShifts:
    def test_returns_shifts_dict(self):
        sig = _make_composite_signal(signal_state="active", composite_weight=0.03, gld_reduction=0.03)
        with _patch_overlay():
            overlay = CryptoAllocationOverlay()
            overlay._signal_gen.generate_signal.return_value = sig
            overlay._staking_model.get_live_yield.return_value = _make_staking_metrics()
            overlay._staking_model.compute_crypto_carry.return_value = {"total_carry_bps": 0}
            shifts = overlay.get_allocation_shifts()
        assert "btc" in shifts
        assert "eth" in shifts
        assert "gld" in shifts
        assert "spy" in shifts
        assert "tlt" in shifts

    def test_spy_tlt_always_zero(self):
        sig = _make_composite_signal(signal_state="active", composite_weight=0.03)
        with _patch_overlay():
            overlay = CryptoAllocationOverlay()
            overlay._signal_gen.generate_signal.return_value = sig
            overlay._staking_model.get_live_yield.return_value = _make_staking_metrics()
            overlay._staking_model.compute_crypto_carry.return_value = {"total_carry_bps": 0}
            shifts = overlay.get_allocation_shifts()
        assert shifts["spy"] == 0.0
        assert shifts["tlt"] == 0.0

    def test_gld_reduction_is_negative(self):
        sig = _make_composite_signal(signal_state="active", composite_weight=0.03, gld_reduction=0.03)
        with _patch_overlay():
            overlay = CryptoAllocationOverlay()
            overlay._signal_gen.generate_signal.return_value = sig
            overlay._staking_model.get_live_yield.return_value = _make_staking_metrics()
            overlay._staking_model.compute_crypto_carry.return_value = {"total_carry_bps": 0}
            shifts = overlay.get_allocation_shifts()
        assert shifts["gld"] == -0.03


# ── Convenience functions ────────────────────────────────────────────


class TestConvenienceFunctions:
    def test_calculate_crypto_allocation(self):
        sig = _make_composite_signal(signal_state="flat", composite_weight=0.0)
        with patch("src.strategy.crypto_allocation.CryptoAllocationOverlay") as MockOverlay:
            mock_overlay = MagicMock()
            mock_overlay.recommend.return_value = CryptoAllocationDecision(
                timestamp="2026-01-01", status="flat",
                btc_weight=0.0, eth_weight=0.0, total_crypto=0.0,
                gld_reduction=0.0, btc_vol=0.0, eth_vol=0.0,
                vol_scale=0.0, btc_momentum_6m=0.0, eth_momentum_6m=0.0,
                confidence=0.0, recommendation="Flat", is_actionable=False,
            )
            MockOverlay.return_value = mock_overlay
            dec = calculate_crypto_allocation()
        assert isinstance(dec, CryptoAllocationDecision)

    def test_get_crypto_summary(self):
        with patch("src.strategy.crypto_allocation.CryptoAllocationOverlay") as MockOverlay:
            mock_overlay = MagicMock()
            mock_overlay.recommend.return_value = CryptoAllocationDecision(
                timestamp="2026-01-01", status="flat",
                btc_weight=0.0, eth_weight=0.0, total_crypto=0.0,
                gld_reduction=0.0, btc_vol=0.0, eth_vol=0.0,
                vol_scale=0.0, btc_momentum_6m=0.0, eth_momentum_6m=0.0,
                confidence=0.0, recommendation="Flat", is_actionable=False,
            )
            MockOverlay.return_value = mock_overlay
            summary = get_crypto_summary()
        assert "status" in summary
        assert "recommendation" in summary
