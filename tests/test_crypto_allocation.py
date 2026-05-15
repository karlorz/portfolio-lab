"""
Tests for Crypto Tactical Allocation Overlay (v4.70)
"""

import json
import pytest
import numpy as np
from datetime import datetime, date
from pathlib import Path

from src.strategy.crypto_allocation import (
    CryptoAllocationOverlay,
    CryptoAllocationDecision,
    CryptoAllocationStatus,
    calculate_crypto_allocation,
    get_crypto_summary,
)
from src.signals.crypto_momentum import CryptoCompositeSignal


class TestCryptoAllocationStatus:
    """Test allocation status enum."""

    def test_status_values(self):
        assert CryptoAllocationStatus.ACTIVE.value == "active"
        assert CryptoAllocationStatus.REDUCED.value == "reduced"
        assert CryptoAllocationStatus.FLAT.value == "flat"
        assert CryptoAllocationStatus.DISABLED.value == "disabled"


class TestCryptoAllocationRecommend:
    """Test allocation recommendations."""

    @pytest.fixture
    def overlay(self, tmp_path):
        state_file = tmp_path / "crypto_state.json"
        return CryptoAllocationOverlay(state_file=state_file)

    def test_generates_decision(self, overlay):
        decision = overlay.recommend()
        assert isinstance(decision, CryptoAllocationDecision)

    def test_total_crypto_matches_btc_plus_eth(self, overlay):
        decision = overlay.recommend()
        assert abs(decision.total_crypto - (decision.btc_weight + decision.eth_weight)) < 0.01

    def test_gld_reduction_non_negative(self, overlay):
        decision = overlay.recommend()
        assert decision.gld_reduction >= 0

    def test_crypto_weight_capped_at_5_pct(self, overlay):
        decision = overlay.recommend()
        assert decision.total_crypto <= 0.05

    def test_decision_serializable(self, overlay):
        decision = overlay.recommend()
        d = decision.to_dict()
        assert isinstance(d, dict)
        assert "btc_weight" in d
        assert "eth_weight" in d

    def test_status_in_known_values(self, overlay):
        decision = overlay.recommend()
        assert decision.status in ("active", "reduced", "flat", "disabled")

    def test_recommendation_is_string(self, overlay):
        decision = overlay.recommend()
        assert isinstance(decision.recommendation, str)
        assert len(decision.recommendation) > 0

    def test_state_persistence(self, overlay):
        overlay._state["total_crypto"] = 0.03
        overlay._save_state()

        overlay2 = CryptoAllocationOverlay(state_file=overlay.state_file)
        assert overlay2._state["total_crypto"] == 0.03

    def test_default_state_structure(self, tmp_path):
        overlay = CryptoAllocationOverlay(state_file=tmp_path / "crypto_state.json")
        assert "status" in overlay._state
        assert "btc_weight" in overlay._state
        assert "eth_weight" in overlay._state
        assert "total_crypto" in overlay._state


class TestAllocationShifts:
    """Test allocation shift generation."""

    @pytest.fixture
    def overlay(self, tmp_path):
        return CryptoAllocationOverlay(state_file=tmp_path / "crypto_state.json")

    def test_shifts_sum_to_zero(self, overlay):
        shifts = overlay.get_allocation_shifts()
        total = sum(shifts.values())
        assert abs(total) < 0.01  # Should be roughly neutral

    def test_gld_reduction_matches_crypto(self, overlay):
        shifts = overlay.get_allocation_shifts()
        crypto_total = shifts.get("btc", 0) + shifts.get("eth", 0)
        assert abs(crypto_total - abs(shifts.get("gld", 0))) < 0.01

    def test_spy_tlt_unchanged(self, overlay):
        shifts = overlay.get_allocation_shifts()
        assert shifts.get("spy", 0) == 0.0
        assert shifts.get("tlt", 0) == 0.0


class TestGetStatus:
    """Test status retrieval."""

    def test_initial_status(self, tmp_path):
        overlay = CryptoAllocationOverlay(state_file=tmp_path / "crypto_state.json")
        assert overlay.get_status() == CryptoAllocationStatus.FLAT


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_calculate_crypto_allocation(self):
        decision = calculate_crypto_allocation()
        assert isinstance(decision, CryptoAllocationDecision)

    def test_get_crypto_summary(self):
        summary = get_crypto_summary()
        assert isinstance(summary, dict)
        assert "status" in summary
        assert "btc_weight" in summary
        assert "eth_weight" in summary


class TestCryptoAllocationBacktest:
    """Test backtest with simulated data."""

    @pytest.fixture
    def overlay(self, tmp_path):
        return CryptoAllocationOverlay(state_file=tmp_path / "crypto_state.json")

    def test_backtest_with_simulated_data(self, overlay):
        rng = np.random.RandomState(42)
        n = 300

        # Generate correlated-ish returns
        spy_rets = rng.normal(0.0003, 0.01, n)
        gld_rets = rng.normal(0.0002, 0.012, n)
        tlt_rets = rng.normal(0.00015, 0.011, n)
        btc_rets = rng.normal(0.0008, 0.04, n)  # Higher return and vol
        eth_rets = rng.normal(0.0006, 0.045, n)

        spy = (550 * np.cumprod(1 + spy_rets)).tolist()
        gld = (200 * np.cumprod(1 + gld_rets)).tolist()
        tlt = (95 * np.cumprod(1 + tlt_rets)).tolist()
        btc = (85000 * np.cumprod(1 + btc_rets)).tolist()
        eth = (3200 * np.cumprod(1 + eth_rets)).tolist()

        dates = [f"2025-{(i // 21)+1:02d}-{(i % 21)+1:02d}" for i in range(n)]

        results = overlay.backtest(btc, eth, gld, spy, tlt, dates)

        assert "summary" in results or "error" in results
        if "summary" in results:
            s = results["summary"]
            assert "cagr_baseline" in s
            assert "cagr_crypto" in s
            assert "avg_crypto_weight" in s
            assert s["avg_crypto_weight"] <= 5.0  # Max 5% average weight

    def test_backtest_insufficient_data(self, overlay):
        results = overlay.backtest(
            [50000, 51000], [3000, 3100],
            [200, 201], [550, 551], [95, 96],
            ["2025-01-01", "2025-01-02"],
        )
        assert "error" in results

    def test_backtest_crypto_weights_in_range(self, overlay):
        rng = np.random.RandomState(42)
        n = 250
        spy = (550 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))).tolist()
        gld = (200 * np.cumprod(1 + rng.normal(0.0002, 0.012, n))).tolist()
        tlt = (95 * np.cumprod(1 + rng.normal(0.00015, 0.011, n))).tolist()
        btc = (85000 * np.cumprod(1 + rng.normal(0.0008, 0.04, n))).tolist()
        eth = (3200 * np.cumprod(1 + rng.normal(0.0006, 0.045, n))).tolist()
        dates = [f"2025-{(i//21)+1:02d}-{(i%21)+1:02d}" for i in range(n)]

        results = overlay.backtest(btc, eth, gld, spy, tlt, dates)
        if "summary" in results:
            s = results["summary"]
            assert s["max_crypto_weight"] <= 5.0


class TestEdgeCases:
    """Edge cases for crypto allocation."""

    def test_multiple_recommends_persist(self, tmp_path):
        """Multiple calls should not corrupt state."""
        overlay = CryptoAllocationOverlay(state_file=tmp_path / "crypto_state.json")
        d1 = overlay.recommend()
        d2 = overlay.recommend()
        assert d1 is not None
        assert d2 is not None
