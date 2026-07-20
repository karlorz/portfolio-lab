"""Tests for BanditWeighter — epsilon-greedy contextual bandit for ensemble signals."""
import numpy as np
import pytest
from src.strategy.ensemble_voter import BanditWeighter, EnsembleVoter, Regime, SignalSource


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

    def test_select_exploit_mode_returns_best(self, monkeypatch):
        bw = BanditWeighter(["sig_a", "sig_b"])
        # Need n>=2 so select takes the Thompson path (not cold-start fallback)
        for _ in range(30):
            bw.update("sig_a", "NORMAL", 0.002)
            bw.update("sig_b", "NORMAL", -0.001)
        bw.epsilon = 0.0  # disable epsilon random explore

        # select() ranks by sampled posterior Sharpe; pin samples so the
        # better arm wins deterministically (Thompson sampling is stochastic).
        def _fake_sample(sig: str, regime: str) -> float:
            return 2.0 if sig == "sig_a" else -1.0

        monkeypatch.setattr(bw, "_sample_sharpe", _fake_sample)
        assert bw.select("NORMAL") == "sig_a"

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


class TestBanditWeighterRollingSharpe:
    """Tests for _rolling_sharpe edge cases."""

    def test_insufficient_data_returns_zero(self):
        bw = BanditWeighter(["sig_a"])
        bw.update("sig_a", "NORMAL", 0.001)
        bw.update("sig_a", "NORMAL", 0.002)
        # Only 2 observations (< 21 minimum)
        sharpe = bw._rolling_sharpe("sig_a", "NORMAL")
        assert sharpe == 0.0

    def test_zero_std_returns_zero(self):
        bw = BanditWeighter(["sig_a"])
        for _ in range(30):
            bw.update("sig_a", "NORMAL", 0.001)  # Constant returns
        sharpe = bw._rolling_sharpe("sig_a", "NORMAL")
        assert sharpe == 0.0  # Zero std → Sharpe = 0

    def test_negative_returns_negative_sharpe(self):
        bw = BanditWeighter(["sig_a"])
        rng = np.random.RandomState(42)
        for _ in range(30):
            bw.update("sig_a", "NORMAL", rng.normal(-0.005, 0.002))
        sharpe = bw._rolling_sharpe("sig_a", "NORMAL")
        assert sharpe < 0

    def test_positive_returns_positive_sharpe(self):
        bw = BanditWeighter(["sig_a"])
        rng = np.random.RandomState(42)
        for _ in range(30):
            bw.update("sig_a", "NORMAL", rng.normal(0.003, 0.001))
        sharpe = bw._rolling_sharpe("sig_a", "NORMAL")
        assert sharpe > 0

    def test_regime_isolation(self):
        """Returns for one regime don't affect another."""
        bw = BanditWeighter(["sig_a"])
        rng = np.random.RandomState(42)
        for _ in range(30):
            bw.update("sig_a", "NORMAL", rng.normal(0.005, 0.001))
        for _ in range(30):
            bw.update("sig_a", "CRISIS", rng.normal(-0.010, 0.002))
        assert bw._rolling_sharpe("sig_a", "NORMAL") > 0
        assert bw._rolling_sharpe("sig_a", "CRISIS") < 0


class TestBanditWeighterWindowTrimming:
    """Tests for window-based history trimming."""

    def test_history_trimmed_to_window(self):
        bw = BanditWeighter(["sig_a"], window=50)
        for i in range(100):
            bw.update("sig_a", "NORMAL", 0.001)
        assert len(bw._history["NORMAL"]["sig_a"]) == 50

    def test_recent_values_preserved_after_trim(self):
        bw = BanditWeighter(["sig_a"], window=10)
        for i in range(20):
            bw.update("sig_a", "NORMAL", float(i))
        history = bw._history["NORMAL"]["sig_a"]
        assert history[0] == 10.0  # Last 10 values: 10..19
        assert history[-1] == 19.0

    def test_large_window_no_trim(self):
        bw = BanditWeighter(["sig_a"], window=1000)
        for _ in range(100):
            bw.update("sig_a", "NORMAL", 0.001)
        assert len(bw._history["NORMAL"]["sig_a"]) == 100  # Under window, no trim


class TestBanditWeighterSoftmax:
    """Tests for _softmax edge cases."""

    def test_all_zero_sharpes_equal_weights(self):
        bw = BanditWeighter(["sig_a", "sig_b", "sig_c"])
        sharpes = {"sig_a": 0.0, "sig_b": 0.0, "sig_c": 0.0}
        weights = bw._softmax(sharpes)
        for sig in ["sig_a", "sig_b", "sig_c"]:
            assert abs(weights[sig] - 1.0 / 3) < 0.01

    def test_single_dominant_signal(self):
        bw = BanditWeighter(["sig_a", "sig_b"], temperature=0.01)
        sharpes = {"sig_a": 5.0, "sig_b": -1.0}
        weights = bw._softmax(sharpes)
        assert weights["sig_a"] > 0.99

    def test_negative_sharpes_produce_valid_weights(self):
        bw = BanditWeighter(["sig_a", "sig_b"])
        sharpes = {"sig_a": -2.0, "sig_b": -3.0}
        weights = bw._softmax(sharpes)
        assert abs(sum(weights.values()) - 1.0) < 0.001
        assert weights["sig_a"] > weights["sig_b"]


class TestEnsembleVoterGetBlendedWeights:
    """Tests for EnsembleVoter.get_blended_weights()."""

    def test_cold_start_returns_static_weights(self, tmp_path):
        voter = EnsembleVoter(data_path=tmp_path)
        weights = voter.get_blended_weights("NORMAL")
        assert weights is not None
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01

    def test_bandit_blend_increases_with_observations(self, tmp_path):
        voter = EnsembleVoter(data_path=tmp_path)
        static = voter.get_blended_weights("NORMAL")

        # Feed 252 bandit observations
        rng = np.random.RandomState(42)
        for i in range(252):
            for src in SignalSource:
                voter.update_bandit(src.value, "NORMAL", rng.normal(0.001, 0.005))

        blended = voter.get_blended_weights("NORMAL")
        # After 252 observations, blend should be ~0.7
        # Weights should differ from pure static (unless bandit learned exactly static)
        assert blended is not None
        total = sum(blended.values())
        assert abs(total - 1.0) < 0.01

    def test_blend_zero_observations_is_pure_static(self, tmp_path):
        voter = EnsembleVoter(data_path=tmp_path)
        weights = voter.get_blended_weights("NORMAL")
        # All weights should be the static regime weights
        from src.strategy.ensemble_voter import REGIME_WEIGHTS
        static = REGIME_WEIGHTS.get(Regime.NORMAL, {})
        for k, v in weights.items():
            if k in static:
                assert abs(v - static[k]) < 0.001

    def test_unknown_regime_defaults_to_normal(self, tmp_path):
        voter = EnsembleVoter(data_path=tmp_path)
        weights = voter.get_blended_weights("NONEXISTENT")
        # Should fallback to NORMAL weights
        assert weights is not None


class TestEnsembleVoterUpdateBandit:
    """Tests for EnsembleVoter.update_bandit()."""

    def test_increment_observations(self, tmp_path):
        voter = EnsembleVoter(data_path=tmp_path)
        initial = voter.bandit_observations
        voter.update_bandit("multi_speed_momentum", "NORMAL", 0.001)
        assert voter.bandit_observations == initial + 1

    def test_multiple_updates_increment_count(self, tmp_path):
        voter = EnsembleVoter(data_path=tmp_path)
        voter.update_bandit("multi_speed_momentum", "NORMAL", 0.001)
        voter.update_bandit("cross_asset_rv", "NORMAL", 0.002)
        voter.update_bandit("international_momentum", "NORMAL", -0.001)
        assert voter.bandit_observations == 3


class TestEnsembleBanditDailyRewardPersistence:
    """Production daily reward path + durable bandit state."""

    def test_apply_daily_bandit_rewards_increments_and_persists(self, tmp_path):
        voter = EnsembleVoter(data_path=tmp_path)
        assert voter.bandit_observations == 0
        summary = voter.apply_daily_bandit_rewards(0.001, regime_name="NORMAL", persist=True)
        assert summary["skipped"] is False
        assert summary["updates"] >= 1
        assert summary["observations"] == voter.bandit_observations
        assert voter.bandit_observations == summary["updates"]
        assert (tmp_path / "ensemble_bandit_state.json").exists()

        # Cold start reloads observations
        voter2 = EnsembleVoter(data_path=tmp_path)
        assert voter2.bandit_observations == voter.bandit_observations
        status = voter2.get_adaptive_learning_status("NORMAL")
        # With history, should not stay pure cold_start zero-obs forever
        assert status["bandit"]["observations"] == voter.bandit_observations

    def test_load_latest_daily_return_from_performance(self, tmp_path):
        perf = tmp_path / "performance.jsonl"
        perf.write_text(
            '{"timestamp":"2026-07-01","daily_return":0.0}\n'
            '{"timestamp":"2026-07-02","daily_return":0.012}\n',
            encoding="utf-8",
        )
        ret = EnsembleVoter.load_latest_daily_return_from_performance(perf)
        assert abs(ret - 0.012) < 1e-9

    def test_bandit_get_load_state_roundtrip(self):
        bw = BanditWeighter(["sig_a", "sig_b"])
        bw.update("sig_a", "NORMAL", 0.01)
        bw.update("sig_b", "NORMAL", -0.005)
        state = bw.get_state()
        bw2 = BanditWeighter(["sig_a", "sig_b"])
        bw2.load_state(state)
        assert len(bw2._history["NORMAL"]["sig_a"]) == 1
        assert abs(bw2._history["NORMAL"]["sig_a"][0] - 0.01) < 1e-9


class TestEnsembleVoterGoalRiskBudget:
    """Tests for EnsembleVoter.apply_goal_risk_budget()."""

    def test_risk_mult_one_no_change(self):
        voter = EnsembleVoter()
        alloc = {"SPY": 0.46, "GLD": 0.38, "TLT": 0.16}
        # With risk_mult >= 1.0, returns unchanged
        result = voter.apply_goal_risk_budget(alloc)
        assert result == alloc

    def test_empty_allocation_returns_unchanged(self):
        voter = EnsembleVoter()
        result = voter.apply_goal_risk_budget({})
        assert result == {}

    def test_zero_total_returns_unchanged(self):
        voter = EnsembleVoter()
        result = voter.apply_goal_risk_budget({"SPY": 0.0})
        # total=0 early return
        assert result == {"SPY": 0.0}


class TestBanditWarmupDaySemantics:
    """Warmup blend must use calendar reward days, not arm×day updates."""

    def test_apply_daily_counts_one_warmup_day(self, tmp_path):
        voter = EnsembleVoter(data_path=tmp_path)
        assert getattr(voter, "bandit_days", 0) == 0
        summary = voter.apply_daily_bandit_rewards(0.001, regime_name="NORMAL", persist=True)
        # One calendar step regardless of source count
        assert summary.get("days", summary.get("bandit_days")) == 1 or voter.bandit_days == 1
        assert voter.bandit_days == 1
        # Arm history may still receive N source updates
        assert summary["updates"] >= 2
        assert voter.bandit_observations == summary["updates"]

    def test_blend_uses_days_not_arm_observations(self, tmp_path):
        from src.strategy.ensemble_voter import BANDIT_WARMUP_DAYS, BANDIT_MAX_BLEND
        voter = EnsembleVoter(data_path=tmp_path)
        # Simulate 9 arm updates via one daily apply (production path)
        voter.apply_daily_bandit_rewards(0.002, regime_name="NORMAL", persist=False)
        status = voter.get_adaptive_learning_status("NORMAL")
        bandit = status["bandit"]
        # Day progress: 1/252 of max blend, NOT 9/252
        expected = min(BANDIT_MAX_BLEND, 1 / BANDIT_WARMUP_DAYS * BANDIT_MAX_BLEND)
        # status publishes current_blend rounded to 4 decimals
        assert abs(bandit["current_blend"] - round(expected, 4)) < 1e-9
        assert bandit["current_blend"] < min(
            BANDIT_MAX_BLEND, 9 / BANDIT_WARMUP_DAYS * BANDIT_MAX_BLEND
        ) - 1e-6  # must not use arm×day (9 sources)
        # Disclose day counter
        assert bandit.get("reward_days", bandit.get("days", voter.bandit_days)) == 1

    def test_bandit_days_persist_across_reload(self, tmp_path):
        voter = EnsembleVoter(data_path=tmp_path)
        voter.apply_daily_bandit_rewards(0.001, regime_name="NORMAL", persist=True)
        voter.apply_daily_bandit_rewards(0.001, regime_name="NORMAL", persist=True)
        assert voter.bandit_days == 2
        voter2 = EnsembleVoter(data_path=tmp_path)
        assert voter2.bandit_days == 2
