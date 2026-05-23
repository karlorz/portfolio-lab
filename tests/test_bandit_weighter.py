"""Tests for BanditWeighter — epsilon-greedy contextual bandit for ensemble signals."""
import numpy as np
import pytest
from src.strategy.ensemble_voter import BanditWeighter


class TestBanditWeighter:
    def test_init_with_signal_list(self):
        bw = BanditWeighter(["TSFM_MOMENTUM", "CROSS_ASSET_RV", "INTL_MOMENTUM"])
        assert bw.epsilon == 0.1
        assert bw.window == 252
        assert len(bw.signals) == 3

    def test_cold_start_returns_none(self):
        bw = BanditWeighter(["sig_a", "sig_b"])
        result = bw.get_weights("NORMAL")
        assert result is None  # No data yet

    def test_select_exploit_mode_returns_best(self):
        bw = BanditWeighter(["sig_a", "sig_b"])
        # Feed noisy data where sig_a has higher mean
        rng = np.random.RandomState(42)
        for _ in range(100):
            bw.update("sig_a", "NORMAL", rng.normal(0.002, 0.01))
            bw.update("sig_b", "NORMAL", rng.normal(-0.001, 0.01))
        # With enough data and epsilon = 0 (force exploit), picks sig_a
        bw.epsilon = 0.0
        selected = bw.select("NORMAL")
        assert selected == "sig_a"

    def test_select_explore_mode_can_pick_any(self):
        bw = BanditWeighter(["sig_a", "sig_b"])
        # Feed enough data for Sharpe to compute
        rng = np.random.RandomState(42)
        for _ in range(25):
            bw.update("sig_a", "NORMAL", rng.normal(0.001, 0.01))
            bw.update("sig_b", "NORMAL", rng.normal(-0.001, 0.01))
        bw.epsilon = 1.0  # Always explore
        picks = [bw.select("NORMAL") for _ in range(50)]
        assert "sig_a" in picks
        assert "sig_b" in picks

    def test_get_weights_sums_to_one(self):
        bw = BanditWeighter(["sig_a", "sig_b", "sig_c"])
        rng = np.random.RandomState(42)
        for sig in ["sig_a", "sig_b", "sig_c"]:
            for _ in range(63):
                bw.update(sig, "NORMAL", rng.normal(0.0001, 0.01))
        weights = bw.get_weights("NORMAL")
        assert weights is not None
        assert abs(sum(weights.values()) - 1.0) < 0.001
        for sig in ["sig_a", "sig_b", "sig_c"]:
            assert sig in weights
            assert weights[sig] > 0
            assert weights[sig] < 1.0

    def test_get_weights_unknown_regime_returns_none(self):
        bw = BanditWeighter(["sig_a"])
        rng = np.random.RandomState(42)
        for _ in range(25):
            bw.update("sig_a", "NORMAL", rng.normal(0.001, 0.01))
        result = bw.get_weights("MARS_REGIME")  # Non-existent
        assert result is None

    def test_update_creates_regime_entry(self):
        bw = BanditWeighter(["sig_a"])
        bw.update("sig_a", "ELEVATED", 0.001)
        # Should not raise, should create the regime entry
        assert "ELEVATED" in bw._history

    def test_softmax_temperature_effect(self):
        bw = BanditWeighter(["sig_a", "sig_b", "sig_c"], temperature=0.1)
        rng = np.random.RandomState(42)
        for _ in range(63):
            bw.update("sig_a", "NORMAL", rng.normal(0.003, 0.005))  # best Sharpe
            bw.update("sig_b", "NORMAL", rng.normal(0.001, 0.005))
            bw.update("sig_c", "NORMAL", rng.normal(-0.002, 0.005))  # worst Sharpe
        weights = bw.get_weights("NORMAL")
        # Low temperature concentrates weight on best signal
        assert weights["sig_a"] > weights["sig_c"]

    def test_low_temperature_concentrates_weight(self):
        bw = BanditWeighter(["sig_a", "sig_b"], temperature=0.001)
        rng = np.random.RandomState(42)
        for _ in range(63):
            bw.update("sig_a", "NORMAL", rng.normal(0.003, 0.005))  # clearly better
            bw.update("sig_b", "NORMAL", rng.normal(-0.002, 0.005))  # clearly worse
        weights = bw.get_weights("NORMAL")
        assert weights["sig_a"] > 0.9

    def test_high_temperature_spreads_weight(self):
        bw = BanditWeighter(["sig_a", "sig_b"], temperature=100.0)
        rng = np.random.RandomState(42)
        for _ in range(63):
            bw.update("sig_a", "NORMAL", rng.normal(0.003, 0.005))
            bw.update("sig_b", "NORMAL", rng.normal(-0.002, 0.005))
        weights = bw.get_weights("NORMAL")
        # High temperature makes weights more uniform
        assert 0.3 < weights["sig_a"] < 0.7
        assert 0.3 < weights["sig_b"] < 0.7
