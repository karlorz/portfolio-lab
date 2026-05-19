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
        assert result is None

    def test_select_exploit_mode_picks_best(self):
        bw = BanditWeighter(["sig_a", "sig_b"])
        np.random.seed(42)
        for _ in range(100):
            bw.update("sig_a", "NORMAL", 0.001)
            bw.update("sig_b", "NORMAL", -0.001)
        bw.epsilon = 0.0  # pure exploit
        selected = bw.select("NORMAL")
        assert selected == "sig_a"

    def test_select_explore_mode_picks_any(self):
        bw = BanditWeighter(["sig_a", "sig_b"])
        bw.update("sig_a", "NORMAL", 0.001)
        bw.update("sig_b", "NORMAL", -0.001)
        bw.epsilon = 1.0  # pure explore
        picks = [bw.select("NORMAL") for _ in range(50)]
        assert "sig_a" in picks
        assert "sig_b" in picks

    def test_get_weights_sums_to_one(self):
        bw = BanditWeighter(["sig_a", "sig_b", "sig_c"])
        np.random.seed(42)
        for sig in ["sig_a", "sig_b", "sig_c"]:
            for _ in range(63):
                bw.update(sig, "NORMAL", np.random.normal(0.0001, 0.01))
        weights = bw.get_weights("NORMAL")
        assert weights is not None
        assert abs(sum(weights.values()) - 1.0) < 0.001
        for sig in ["sig_a", "sig_b", "sig_c"]:
            assert sig in weights
            assert weights[sig] > 0
            assert weights[sig] < 1.0

    def test_get_weights_unknown_regime_returns_none(self):
        bw = BanditWeighter(["sig_a"])
        bw.update("sig_a", "NORMAL", 0.001)
        result = bw.get_weights("MARS_REGIME")
        assert result is None

    def test_update_creates_regime_entry(self):
        bw = BanditWeighter(["sig_a"])
        bw.update("sig_a", "ELEVATED", 0.001)
        assert "ELEVATED" in bw._history

    def test_low_temperature_concentrates_weight(self):
        bw = BanditWeighter(["sig_a", "sig_b"], temperature=0.1)
        np.random.seed(42)
        for _ in range(63):
            bw.update("sig_a", "NORMAL", 0.002 + np.random.normal(0, 0.005))
            bw.update("sig_b", "NORMAL", -0.001 + np.random.normal(0, 0.005))
        weights = bw.get_weights("NORMAL")
        assert weights is not None
        # Low temp: best signal gets more weight
        assert weights["sig_a"] > weights["sig_b"]

    def test_high_temperature_spreads_weight(self):
        bw = BanditWeighter(["sig_a", "sig_b"], temperature=100.0)
        np.random.seed(42)
        for _ in range(63):
            bw.update("sig_a", "NORMAL", 0.002)
            bw.update("sig_b", "NORMAL", -0.001)
        weights = bw.get_weights("NORMAL")
        assert 0.35 < weights["sig_a"] < 0.65
        assert 0.35 < weights["sig_b"] < 0.65

    def test_insufficient_history_returns_none(self):
        bw = BanditWeighter(["sig_a"])
        # Only 10 data points (need 21 minimum)
        for _ in range(10):
            bw.update("sig_a", "NORMAL", 0.001)
        weights = bw.get_weights("NORMAL")
        assert weights is None
