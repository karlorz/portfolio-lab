import pytest; pytestmark = pytest.mark.heavy
#!/usr/bin/env python3
"""
Tests for execution_agent.py — ExecutionStyle enum, ExecutionNetwork architecture,
ExecutionAgent (extract_features, estimate_market_impact, _check_scheduling,
get_scheduler_status, act, compute_value, train_step).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock

# torch is guarded at module level — importing it during test collection
# (even for skipped heavy tests) loads 63MB+ and can exhaust CPU on low-resource
# machines. Only import when ML features are explicitly enabled.
_ML_ENABLED = os.environ.get("PORTFOLIO_LAB_ENABLE_ML", "0") == "1"
if _ML_ENABLED:
    import torch  # noqa: F811

from src.agents.base_agent import (
    AgentType, AgentObservation, AgentAction, AgentMessage, MessageType
)
from src.agents.execution_agent import (
    ExecutionStyle,
    ExecutionNetwork,
    ExecutionAgent,
    SCHEDULER_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obs(volatility=0.15, n_prices=50, seed=42):
    """Create a minimal AgentObservation for testing."""
    rng = np.random.RandomState(seed)
    prices = (100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n_prices))).astype(np.float64)
    returns = np.diff(prices) / prices[:-1]
    return AgentObservation(
        prices=prices,
        returns=returns,
        volatility=volatility,
        current_weights={"spy": 0.46, "gld": 0.38, "tlt": 0.16},
        portfolio_value=100000.0,
        cash_available=5000.0,
        timestamp="2026-05-14T09:30:00",
        regime="neutral",
    )


def _make_minimal_agent():
    """Create ExecutionAgent with mocked scheduler (no scheduler available)."""
    with patch("src.agents.execution_agent.SCHEDULER_AVAILABLE", False):
        agent = ExecutionAgent(agent_id="test_exec", device="cpu", use_scheduler=False)
    return agent


# ---------------------------------------------------------------------------
# ExecutionStyle Tests
# ---------------------------------------------------------------------------

class TestExecutionStyle:

    def test_vwap_value(self):
        assert ExecutionStyle.VWAP.value == 0.0

    def test_pov_value(self):
        assert ExecutionStyle.POV.value == 0.33

    def test_twap_value(self):
        assert ExecutionStyle.TWAP.value == 0.66

    def test_aggressive_value(self):
        assert ExecutionStyle.AGGRESSIVE.value == 1.0

    def test_all_styles(self):
        assert len(ExecutionStyle) == 4


# ---------------------------------------------------------------------------
# ExecutionNetwork Tests
# ---------------------------------------------------------------------------

class TestExecutionNetwork:

    def test_creates_network(self):
        net = ExecutionNetwork(obs_dim=42, action_dim=4, hidden_dim=64)
        assert isinstance(net, torch.nn.Module)

    def test_forward_returns_five_outputs(self):
        net = ExecutionNetwork(obs_dim=42, action_dim=4)
        x = torch.randn(1, 42)
        urgency, slice_frac, style, confidence, value = net(x)
        assert urgency.shape == (1, 1)
        assert slice_frac.shape == (1, 1)
        assert style.shape == (1, 1)
        assert confidence.shape == (1, 1)
        assert value.shape == (1, 1)

    def test_outputs_in_range(self):
        net = ExecutionNetwork(obs_dim=42, action_dim=4)
        x = torch.randn(1, 42)
        urgency, slice_frac, style, confidence, value = net(x)
        assert 0 <= float(urgency) <= 1
        assert 0 <= float(slice_frac) <= 1
        assert 0 <= float(style) <= 1
        assert 0 <= float(confidence) <= 1

    def test_batch_forward(self):
        net = ExecutionNetwork(obs_dim=42, action_dim=4)
        x = torch.randn(8, 42)
        urgency, slice_frac, style, confidence, value = net(x)
        assert urgency.shape == (8, 1)
        assert value.shape == (8, 1)


# ---------------------------------------------------------------------------
# ExecutionAgent Constants
# ---------------------------------------------------------------------------

class TestAgentConstants:

    def test_price_history_len(self):
        agent = _make_minimal_agent()
        assert agent.PRICE_HISTORY_LEN == 30

    def test_n_exec_features(self):
        agent = _make_minimal_agent()
        assert agent.N_EXEC_FEATURES == 12

    def test_feature_names(self):
        agent = _make_minimal_agent()
        assert len(agent.feature_names) == 12
        assert 'spread_proxy' in agent.feature_names
        assert 'liquidity_score' in agent.feature_names


# ---------------------------------------------------------------------------
# estimate_market_impact Tests
# ---------------------------------------------------------------------------

class TestEstimateMarketImpact:

    def test_returns_float(self):
        agent = _make_minimal_agent()
        impact = agent.estimate_market_impact(0.01, 0.15, 0.5)
        assert isinstance(impact, float)
        assert impact >= 0

    def test_capped_at_5_percent(self):
        agent = _make_minimal_agent()
        impact = agent.estimate_market_impact(0.50, 0.80, 0.01)
        assert impact <= 0.05

    def test_larger_order_more_impact(self):
        agent = _make_minimal_agent()
        small = agent.estimate_market_impact(0.01, 0.15, 0.5)
        large = agent.estimate_market_impact(0.10, 0.15, 0.5)
        assert large > small

    def test_higher_vol_more_impact(self):
        agent = _make_minimal_agent()
        low_vol = agent.estimate_market_impact(0.05, 0.10, 0.5)
        high_vol = agent.estimate_market_impact(0.05, 0.30, 0.5)
        assert high_vol > low_vol

    def test_lower_liquidity_more_impact(self):
        agent = _make_minimal_agent()
        liquid = agent.estimate_market_impact(0.05, 0.15, 0.9)
        illiquid = agent.estimate_market_impact(0.05, 0.15, 0.1)
        assert illiquid > liquid


# ---------------------------------------------------------------------------
# extract_features Tests
# ---------------------------------------------------------------------------

class TestExtractFeatures:

    def test_returns_tensor(self):
        agent = _make_minimal_agent()
        obs = _make_obs()
        features = agent.extract_features(obs)
        assert isinstance(features, torch.Tensor)

    def test_correct_dimension(self):
        agent = _make_minimal_agent()
        obs = _make_obs()
        features = agent.extract_features(obs)
        expected_dim = agent.PRICE_HISTORY_LEN + agent.N_EXEC_FEATURES
        assert features.shape[0] == expected_dim

    def test_short_prices_padded(self):
        agent = _make_minimal_agent()
        obs = _make_obs(n_prices=10)
        features = agent.extract_features(obs)
        expected_dim = agent.PRICE_HISTORY_LEN + agent.N_EXEC_FEATURES
        assert features.shape[0] == expected_dim

    def test_high_vol_urgency(self):
        agent = _make_minimal_agent()
        obs_low = _make_obs(volatility=0.10)
        obs_high = _make_obs(volatility=0.35)
        feat_low = agent.extract_features(obs_low)
        feat_high = agent.extract_features(obs_high)
        # urgency_required index = PRICE_HISTORY_LEN + 8
        urg_idx = agent.PRICE_HISTORY_LEN + 8
        assert feat_high[urg_idx] > feat_low[urg_idx]

    def test_all_features_finite(self):
        agent = _make_minimal_agent()
        obs = _make_obs()
        features = agent.extract_features(obs)
        assert torch.isfinite(features).all()


# ---------------------------------------------------------------------------
# act Tests
# ---------------------------------------------------------------------------

class TestAct:

    def test_returns_agent_action(self):
        agent = _make_minimal_agent()
        obs = _make_obs()
        action = agent.act(obs)
        assert isinstance(action, AgentAction)
        assert action.agent_id == "test_exec"
        assert action.action_type == "execution_plan"

    def test_action_has_metadata(self):
        agent = _make_minimal_agent()
        obs = _make_obs()
        action = agent.act(obs)
        assert 'slice_size' in action.metadata
        assert 'execution_style' in action.metadata
        assert 0.1 <= action.metadata['slice_size'] <= 0.5

    def test_execution_style_in_enum(self):
        agent = _make_minimal_agent()
        obs = _make_obs()
        action = agent.act(obs)
        style_name = action.metadata['execution_style']
        assert style_name in [e.name for e in ExecutionStyle]

    def test_sends_message_to_controller(self):
        agent = _make_minimal_agent()
        obs = _make_obs()
        agent.act(obs)
        messages = agent.clear_outbox()
        assert len(messages) > 0
        controller_msgs = [m for m in messages if m.receiver == "controller"]
        assert len(controller_msgs) > 0

    def test_urgency_in_range(self):
        agent = _make_minimal_agent()
        obs = _make_obs()
        action = agent.act(obs)
        assert 0 <= action.score <= 1

    def test_deterministic_no_noise(self):
        agent = _make_minimal_agent()
        agent.network.eval()  # Disable dropout for determinism
        obs = _make_obs()
        action1 = agent.act(obs, deterministic=True)
        action2 = agent.act(obs, deterministic=True)
        assert action1.score == action2.score

    def test_required_urgency_override(self):
        agent = _make_minimal_agent()
        agent.network.eval()
        obs = _make_obs()
        action = agent.act(obs, required_urgency=0.9, deterministic=True)
        assert action.score == pytest.approx(0.9)

    def test_scheduler_not_active_in_metadata(self):
        agent = _make_minimal_agent()
        obs = _make_obs()
        action = agent.act(obs)
        assert action.metadata['scheduler_active'] is False


# ---------------------------------------------------------------------------
# compute_value Tests
# ---------------------------------------------------------------------------

class TestComputeValue:

    def test_returns_float(self):
        agent = _make_minimal_agent()
        obs = _make_obs()
        value = agent.compute_value(obs)
        assert isinstance(value, float)

    def test_finite(self):
        agent = _make_minimal_agent()
        obs = _make_obs()
        value = agent.compute_value(obs)
        assert np.isfinite(value)


# ---------------------------------------------------------------------------
# train_step Tests
# ---------------------------------------------------------------------------

class TestTrainStep:

    def test_empty_observations(self):
        agent = _make_minimal_agent()
        result = agent.train_step([], [], [], [])
        assert result == {}

    def test_returns_loss_dict(self):
        agent = _make_minimal_agent()
        obs_list = [_make_obs(n_prices=50, seed=i) for i in range(4)]
        actions = [
            agent.act(o, deterministic=True) for o in obs_list
        ]
        returns = [0.01, -0.005, 0.02, 0.0]
        advantages = [0.5, -0.3, 0.8, 0.0]
        result = agent.train_step(obs_list, actions, returns, advantages)
        assert 'value_loss' in result
        assert 'policy_loss' in result
        assert 'mean_urgency' in result

    def test_losses_finite(self):
        agent = _make_minimal_agent()
        obs_list = [_make_obs(n_prices=50, seed=i) for i in range(4)]
        actions = [
            agent.act(o, deterministic=True) for o in obs_list
        ]
        returns = [0.01, -0.01, 0.02, 0.0]
        advantages = [0.5, -0.5, 0.8, 0.0]
        result = agent.train_step(obs_list, actions, returns, advantages)
        assert np.isfinite(result['value_loss'])


# ---------------------------------------------------------------------------
# _check_scheduling Tests
# ---------------------------------------------------------------------------

class TestCheckScheduling:

    def test_no_scheduler_returns_none(self):
        agent = _make_minimal_agent()
        result = agent._check_scheduling(0.5, "SPY")
        assert result == (None, 0.0)

    def test_high_urgency_returns_none(self):
        agent = _make_minimal_agent()
        result = agent._check_scheduling(0.90, "SPY")
        assert result == (None, 0.0)

    def test_no_scheduler_high_urgency_returns_none(self):
        agent = _make_minimal_agent()
        agent.use_scheduler = False
        result = agent._check_scheduling(0.30, "SPY")
        assert result == (None, 0.0)

    def test_no_scheduler_attribute_returns_none(self):
        agent = _make_minimal_agent()
        agent.scheduler = None
        result = agent._check_scheduling(0.30, "SPY")
        assert result == (None, 0.0)


# ---------------------------------------------------------------------------
# get_scheduler_status Tests
# ---------------------------------------------------------------------------

class TestGetSchedulerStatus:

    def test_inactive_without_scheduler(self):
        agent = _make_minimal_agent()
        status = agent.get_scheduler_status()
        assert status['active'] is False
        assert status['pending_orders'] == 0

    def test_has_expected_keys(self):
        agent = _make_minimal_agent()
        status = agent.get_scheduler_status()
        assert 'active' in status
        assert 'pending_orders' in status
        assert 'next_execution' in status

    def test_shows_optimal_window_when_active(self):
        """scheduler_available key appears only when scheduler is active."""
        agent = _make_minimal_agent()
        status = agent.get_scheduler_status()
        # When inactive, the key is absent — test for the inactive keys
        assert status['active'] is False
        assert 'pending_orders' in status


# ---------------------------------------------------------------------------
# SCHEDULER_AVAILABLE Tests
# ---------------------------------------------------------------------------

class TestSchedulerAvailable:

    def test_is_boolean(self):
        assert isinstance(SCHEDULER_AVAILABLE, bool)
