"""Unit tests for src.strategy.ensemble_voter_bandit BanditMixin."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.signals.regime_spec import Regime
from src.strategy.ensemble_voter_bandit import BanditMixin


class DummyBandit:
    """Mock bandit for BanditMixin testing."""

    def __init__(self):
        self.updates = []
        self._history = {}
        self.state = {"dummy": "data"}

    def update(self, signal_value: str, regime_name: str, daily_return: float):
        self.updates.append((signal_value, regime_name, daily_return))

    def load_state(self, state: dict):
        self.state = state

    def get_state(self) -> dict:
        return self.state


class DummyVoter(BanditMixin):
    """Test harness incorporating BanditMixin."""

    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.bandit_state_path = data_path / "ensemble_bandit_state.json"
        self.current_regime = Regime.NORMAL
        self.current_regime_confidence = 0.85
        self.bandit = DummyBandit()
        self.bandit_observations = 0
        self.bandit_days = 0


class TestGetRebalanceConfig:
    """Test get_rebalance_config regime mapping."""

    @pytest.mark.parametrize(
        ("regime_enum", "expected_str"),
        [
            (Regime.LOW_VOL, "low_vol"),
            (Regime.NORMAL, "normal"),
            (Regime.HIGH_VOL, "high_vol"),
            (Regime.CRISIS, "crisis"),
            (Regime.RECOVERY, "recovery"),
        ],
    )
    def test_regime_mapping(self, tmp_path: Path, regime_enum: Regime, expected_str: str):
        voter = DummyVoter(tmp_path)
        voter.current_regime = regime_enum
        voter.current_regime_confidence = 0.92
        cfg = voter.get_rebalance_config()
        assert cfg["regime"] == expected_str
        assert cfg["regime_confidence"] == 0.92

    def test_unknown_regime_defaults_to_normal(self, tmp_path: Path):
        voter = DummyVoter(tmp_path)
        voter.current_regime = "unknown_regime"
        cfg = voter.get_rebalance_config()
        assert cfg["regime"] == "normal"


class TestUpdateBandit:
    """Test update_bandit delegation and observation increment."""

    def test_update_increments_and_delegates(self, tmp_path: Path):
        voter = DummyVoter(tmp_path)
        assert voter.bandit_observations == 0
        voter.update_bandit("multi_speed_momentum", "normal", 0.015)
        assert voter.bandit_observations == 1
        assert voter.bandit.updates == [("multi_speed_momentum", "normal", 0.015)]


class TestLoadSaveBanditState:
    """Test _load_bandit_state and save_bandit_state."""

    def test_load_missing_file_returns_false(self, tmp_path: Path):
        voter = DummyVoter(tmp_path)
        assert voter._load_bandit_state() is False

    def test_load_corrupt_file_returns_false(self, tmp_path: Path):
        voter = DummyVoter(tmp_path)
        voter.bandit_state_path.write_text("not json", encoding="utf-8")
        assert voter._load_bandit_state() is False

    def test_load_non_dict_returns_false(self, tmp_path: Path):
        voter = DummyVoter(tmp_path)
        voter.bandit_state_path.write_text("[1, 2, 3]", encoding="utf-8")
        assert voter._load_bandit_state() is False

    def test_load_valid_v1_schema(self, tmp_path: Path):
        voter = DummyVoter(tmp_path)
        payload = {
            "schema_version": "ensemble-bandit-state/v1",
            "observations": 42,
            "reward_days": 10,
            "bandit": {"saved_arm": 1.23},
        }
        voter.bandit_state_path.write_text(json.dumps(payload), encoding="utf-8")
        assert voter._load_bandit_state() is True
        assert voter.bandit_observations == 42
        assert voter.bandit_days == 10
        assert voter.bandit.state == {"saved_arm": 1.23}

    def test_load_derives_observations_and_days_from_history_if_missing(self, tmp_path: Path):
        voter = DummyVoter(tmp_path)
        voter.bandit._history = {
            "regime_a": {"sig1": [0.01, 0.02], "sig2": [0.03]},
            "regime_b": {"sig1": [0.01]},
        }
        payload = {"bandit": {"foo": "bar"}}
        voter.bandit_state_path.write_text(json.dumps(payload), encoding="utf-8")
        assert voter._load_bandit_state() is True
        assert voter.bandit_observations == 4
        assert voter.bandit_days >= 0

    def test_save_and_reload_roundtrip(self, tmp_path: Path):
        voter = DummyVoter(tmp_path)
        voter.bandit_observations = 100
        voter.bandit_days = 20
        voter.bandit.state = {"alpha": 1, "beta": 2}
        assert voter.save_bandit_state() is True
        assert voter.bandit_state_path.is_file()

        voter2 = DummyVoter(tmp_path)
        assert voter2._load_bandit_state() is True
        assert voter2.bandit_observations == 100
        assert voter2.bandit_days == 20
        assert voter2.bandit.state == {"alpha": 1, "beta": 2}


class TestContributionRewards:
    """Test contribution_reward_decimal and compute_daily_contribution_rewards."""

    def test_contribution_reward_directional(self):
        r = BanditMixin.contribution_reward_decimal(0.01, value=0.5, weight=0.2)
        assert r == pytest.approx(0.01 * 0.5)

    def test_contribution_reward_neutral(self):
        r = BanditMixin.contribution_reward_decimal(0.01, value=0.02, weight=0.2)
        assert r == pytest.approx(0.01 * 0.2 * 2.0)

    def test_compute_daily_contribution_rewards_insufficient_signals(self):
        signals = [{"source": "s1", "value": 0.5, "weight": 0.2}]
        res = BanditMixin.compute_daily_contribution_rewards(signals, 0.01)
        assert res is None

    def test_compute_daily_contribution_rewards_zero_spread_returns_none(self):
        signals = [
            {"source": "s1", "value": 0.5, "weight": 0.2},
            {"source": "s2", "value": 0.5, "weight": 0.2},
        ]
        res = BanditMixin.compute_daily_contribution_rewards(signals, 0.01)
        assert res is None

    def test_compute_daily_contribution_rewards_valid_spread(self):
        signals = [
            {"source": "s1", "value": 0.8, "weight": 0.2},
            {"source": "s2", "value": 0.2, "weight": 0.2},
        ]
        res = BanditMixin.compute_daily_contribution_rewards(signals, 0.01)
        assert res is not None
        assert "s1" in res and "s2" in res
        assert res["s1"] > res["s2"]


class TestAttributionSourceRewards:
    """Test load_attribution_source_rewards."""

    def test_load_attribution_missing_returns_none(self, tmp_path: Path):
        res = BanditMixin.load_attribution_source_rewards(tmp_path)
        assert res is None

    def test_load_attribution_valid_identifying_file(self, tmp_path: Path):
        attr_dir = tmp_path / "attribution"
        attr_dir.mkdir(parents=True)
        payload = {
            "sources": {
                "multi_speed_momentum": {"avg_return_bps": 120.0},
                "trend_filter": {"avg_return_bps": 40.0},
            }
        }
        (attr_dir / "latest.json").write_text(json.dumps(payload), encoding="utf-8")
        res = BanditMixin.load_attribution_source_rewards(tmp_path)
        assert res is not None
        assert res["multi_speed_momentum"] == pytest.approx(0.012)
        assert res["trend_filter"] == pytest.approx(0.004)

    def test_load_attribution_non_identifying_returns_none(self, tmp_path: Path):
        attr_dir = tmp_path / "attribution"
        attr_dir.mkdir(parents=True)
        payload = {
            "sources": {
                "sig1": {"avg_return_bps": 50.0},
                "sig2": {"avg_return_bps": 50.0},
            }
        }
        (attr_dir / "latest.json").write_text(json.dumps(payload), encoding="utf-8")
        res = BanditMixin.load_attribution_source_rewards(tmp_path)
        assert res is None
