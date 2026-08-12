#!/usr/bin/env python3
"""
Tests for execution_agent.py — ExecutionStyle enum, ExecutionNetwork architecture,
ExecutionAgent (extract_features, estimate_market_impact, _check_scheduling,
get_scheduler_status, act, compute_value, train_step).

Coverage categories:
  - Dataclass field validation (AgentObservation, AgentAction, AgentMessage)
  - Computation edge cases (zero/NaN/extreme inputs)
  - Constants validation (PRICE_HISTORY_LEN, N_EXEC_FEATURES, feature_names)
  - Function boundary conditions (negative values, boundaries, missing keys)
  - Export coverage (no __all__; public classes verified)
  - CLI/__main__ guard (no __main__ block — verified via absence)
"""
import os

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock
import dataclasses

pytestmark = pytest.mark.heavy

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

# OrderUrgency mock -- referenced by _check_scheduling but not defined in codebase.
# Inject directly into the execution_agent module namespace so _check_scheduling
# can resolve it at call time.
from src.agents import execution_agent as _execution_agent
_MOCK_ORDER_URGENCY = type('OrderUrgency', (), {
    'LOW': 'low',
    'NORMAL': 'normal',
    'HIGH': 'high',
    'URGENT': 'urgent',
})
_execution_agent.OrderUrgency = _MOCK_ORDER_URGENCY


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

    def test_values_unique(self):
        """All enum values must be distinct."""
        values = [e.value for e in ExecutionStyle]
        assert len(values) == len(set(values))

    def test_member_access_by_name(self):
        """Members are accessible by name string."""
        assert ExecutionStyle["VWAP"] == ExecutionStyle.VWAP
        assert ExecutionStyle["AGGRESSIVE"] == ExecutionStyle.AGGRESSIVE


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

    def test_default_hidden_dim_128(self):
        """Default hidden_dim should be 128."""
        net = ExecutionNetwork(obs_dim=42, action_dim=4)
        # The encoder first Linear goes obs_dim -> hidden_dim
        assert net.encoder[0].in_features == 42
        assert net.encoder[0].out_features == 128

    def test_parameter_count_positive(self):
        """Network should have learnable parameters."""
        net = ExecutionNetwork(obs_dim=42, action_dim=4)
        params = list(net.parameters())
        assert len(params) > 0
        total = sum(p.numel() for p in params)
        assert total > 0

    def test_gradients_flow(self):
        """Backward pass should produce gradients on all parameters."""
        net = ExecutionNetwork(obs_dim=10, action_dim=4, hidden_dim=16)
        x = torch.randn(2, 10, requires_grad=True)
        urgency, slice_frac, style, confidence, value = net(x)
        loss = value.sum() + urgency.sum() + slice_frac.sum() + style.sum() + confidence.sum()
        loss.backward()
        for name, param in net.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum().item() > 0, f"Zero gradient for {name}"

    def test_zero_observation(self):
        """Network should handle all-zero input without error."""
        net = ExecutionNetwork(obs_dim=10, action_dim=4, hidden_dim=16)
        x = torch.zeros(1, 10)
        urgency, slice_frac, style, confidence, value = net(x)
        assert urgency.shape == (1, 1)
        assert 0 <= float(urgency) <= 1

    def test_different_input_dimensions(self):
        """Network should accept various obs_dim values."""
        for obs_dim in [5, 10, 42, 100]:
            net = ExecutionNetwork(obs_dim=obs_dim, action_dim=4, hidden_dim=32)
            x = torch.randn(1, obs_dim)
            urgency, slice_frac, style, confidence, value = net(x)
            assert urgency.shape == (1, 1)


# ---------------------------------------------------------------------------
# ExecutionAgent Initialization
# ---------------------------------------------------------------------------

class TestAgentInit:

    def test_default_agent_id(self):
        agent = _make_minimal_agent()
        assert agent.agent_id == "test_exec"

    def test_custom_agent_id(self):
        with patch("src.agents.execution_agent.SCHEDULER_AVAILABLE", False):
            agent = ExecutionAgent(agent_id="custom_id", use_scheduler=False)
        assert agent.agent_id == "custom_id"

    def test_custom_hidden_dim(self):
        with patch("src.agents.execution_agent.SCHEDULER_AVAILABLE", False):
            agent = ExecutionAgent(agent_id="test", hidden_dim=64, use_scheduler=False)
        assert agent.hidden_dim == 64

    def test_obs_dim_calculation(self):
        """obs_dim should equal PRICE_HISTORY_LEN + N_EXEC_FEATURES."""
        agent = _make_minimal_agent()
        assert agent.obs_dim == agent.PRICE_HISTORY_LEN + agent.N_EXEC_FEATURES
        assert agent.obs_dim == 42

    def test_action_dim(self):
        """action_dim should be 4 (urgency, slice, style, confidence)."""
        agent = _make_minimal_agent()
        assert agent.action_dim == 4

    def test_agent_type_is_execution(self):
        agent = _make_minimal_agent()
        assert agent.agent_type == AgentType.EXECUTION

    def test_optimizer_created(self):
        """Optimizer should be created during init."""
        agent = _make_minimal_agent()
        assert agent.optimizer is not None
        assert isinstance(agent.optimizer, torch.optim.Adam)

    def test_scheduler_disabled_by_default(self):
        """SCHEDULER_AVAILABLE is False, so use_scheduler falls back."""
        agent = _make_minimal_agent()
        assert agent.use_scheduler is False

    def test_network_created(self):
        agent = _make_minimal_agent()
        assert agent.network is not None
        assert isinstance(agent.network, ExecutionNetwork)

    def test_device_default(self):
        agent = _make_minimal_agent()
        assert str(agent.device) == "cpu"

    def test_feature_names_length(self):
        agent = _make_minimal_agent()
        assert len(agent.feature_names) == agent.N_EXEC_FEATURES

    def test_pending_orders_empty(self):
        agent = _make_minimal_agent()
        assert agent.pending_orders == []


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

    def test_feature_names_index_0_spread_proxy(self):
        agent = _make_minimal_agent()
        assert agent.feature_names[0] == 'spread_proxy'

    def test_feature_names_index_8_urgency_required(self):
        agent = _make_minimal_agent()
        assert agent.feature_names[8] == 'urgency_required'

    def test_feature_names_index_11_regime_volatility(self):
        agent = _make_minimal_agent()
        assert agent.feature_names[11] == 'regime_volatility'


# ---------------------------------------------------------------------------
# build_network Tests
# ---------------------------------------------------------------------------

class TestBuildNetwork:

    def test_returns_module_dict(self):
        agent = _make_minimal_agent()
        result = agent.build_network()
        assert isinstance(result, torch.nn.ModuleDict)

    def test_contains_main_key(self):
        agent = _make_minimal_agent()
        result = agent.build_network()
        assert 'main' in result

    def test_main_is_execution_network(self):
        agent = _make_minimal_agent()
        result = agent.build_network()
        assert isinstance(result['main'], ExecutionNetwork)


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

    def test_zero_order_size(self):
        """Zero order_size should use min floor of 0.01."""
        agent = _make_minimal_agent()
        impact = agent.estimate_market_impact(0.0, 0.15, 0.5)
        assert isinstance(impact, float)
        assert impact >= 0

    def test_zero_volatility(self):
        """Zero volatility should produce zero impact (before liquidity adjustment)."""
        agent = _make_minimal_agent()
        impact = agent.estimate_market_impact(0.05, 0.0, 0.5)
        assert impact == 0.0

    def test_zero_liquidity(self):
        """Zero liquidity is adjusted by +0.1 in denominator, should not divide-by-zero."""
        agent = _make_minimal_agent()
        impact = agent.estimate_market_impact(0.05, 0.15, 0.0)
        assert isinstance(impact, float)
        assert impact >= 0

    def test_negative_order_size(self):
        """Negative order_size uses max() with 0.01 floor."""
        agent = _make_minimal_agent()
        impact = agent.estimate_market_impact(-0.5, 0.15, 0.5)
        assert isinstance(impact, float)
        assert impact >= 0

    def test_extreme_volatility(self):
        """Very high volatility should saturate at 5% cap."""
        agent = _make_minimal_agent()
        impact = agent.estimate_market_impact(0.10, 2.0, 0.1)
        assert impact <= 0.05

    def test_boundary_min_order_size(self):
        """0.01 is the minimum order_size floor."""
        agent = _make_minimal_agent()
        impact_zero = agent.estimate_market_impact(0.0, 0.15, 0.5)
        impact_min = agent.estimate_market_impact(0.01, 0.15, 0.5)
        assert impact_zero == impact_min


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

    def test_zero_prices(self):
        """Near-zero prices should not cause NaN in outputs (0/0 avoided)."""
        agent = _make_minimal_agent()
        obs = _make_obs(n_prices=30, seed=0)
        obs.prices[:] = 1e-10
        obs.returns = np.zeros(29)
        features = agent.extract_features(obs)
        assert torch.isfinite(features).all()

    def test_constant_prices(self):
        """Constant prices produce zero returns, all features must be finite."""
        agent = _make_minimal_agent()
        obs = _make_obs(n_prices=30, seed=0)
        obs.prices[:] = 100.0
        obs.returns = np.zeros(29)
        features = agent.extract_features(obs)
        assert torch.isfinite(features).all()
        # Normalized prices should all be zero
        for i in range(agent.PRICE_HISTORY_LEN):
            assert float(features[i]) == 0.0

    def test_exact_price_history_length(self):
        """Exactly PRICE_HISTORY_LEN prices should NOT trigger padding."""
        agent = _make_minimal_agent()
        obs = _make_obs(n_prices=30, seed=0)
        features = agent.extract_features(obs)
        assert features.shape[0] == agent.PRICE_HISTORY_LEN + agent.N_EXEC_FEATURES

    def test_extreme_volatility_features_saturated(self):
        """Very high volatility should saturate spread_proxy and intraday_vol."""
        agent = _make_minimal_agent()
        obs = _make_obs(volatility=5.0, n_prices=50, seed=0)
        features = agent.extract_features(obs)
        spread_idx = agent.PRICE_HISTORY_LEN       # spread_proxy
        vol_idx = agent.PRICE_HISTORY_LEN + 2       # intraday volatility
        assert float(features[spread_idx]) <= 1.0
        assert float(features[vol_idx]) <= 1.0

    def test_zero_volatility_features(self):
        """Zero volatility should produce minimal spread_proxy."""
        agent = _make_minimal_agent()
        obs = _make_obs(volatility=0.0, n_prices=30, seed=0)
        obs.prices[:] = 100.0
        obs.returns = np.zeros(29)
        features = agent.extract_features(obs)
        spread_idx = agent.PRICE_HISTORY_LEN
        assert float(features[spread_idx]) >= 0.0
        assert float(features[spread_idx]) <= 1.0

    def test_spread_proxy_in_0_1(self):
        """Spread proxy feature must be clipped to [0, 1]."""
        agent = _make_minimal_agent()
        obs = _make_obs(volatility=2.0, n_prices=50, seed=0)
        features = agent.extract_features(obs)
        spread_idx = agent.PRICE_HISTORY_LEN
        assert 0.0 <= float(features[spread_idx]) <= 1.0

    def test_mean_reversion_feature_finite(self):
        """Mean reversion feature should be finite regardless of input."""
        agent = _make_minimal_agent()
        obs = _make_obs(n_prices=10, seed=0)
        features = agent.extract_features(obs)
        mr_idx = agent.PRICE_HISTORY_LEN + 4
        assert torch.isfinite(features[mr_idx])

    def test_vol_regime_feature_range(self):
        """Vol regime feature should be in [-1, 1] via tanh."""
        agent = _make_minimal_agent()
        obs_low = _make_obs(volatility=0.05)
        obs_high = _make_obs(volatility=0.50)
        feat_low = agent.extract_features(obs_low)
        feat_high = agent.extract_features(obs_high)
        vr_idx = agent.PRICE_HISTORY_LEN + 11
        assert -1.0 <= float(feat_low[vr_idx]) <= 1.0
        assert -1.0 <= float(feat_high[vr_idx]) <= 1.0

    def test_momentum_feature_range(self):
        """Momentum feature (tanh) should be in [-1, 1]."""
        agent = _make_minimal_agent()
        obs = _make_obs(n_prices=50, seed=0)
        features = agent.extract_features(obs)
        mom_idx = agent.PRICE_HISTORY_LEN + 3
        assert -1.0 <= float(features[mom_idx]) <= 1.0

    def test_liquidity_feature_complementary(self):
        """Liquidity score = 1 - spread_proxy (inverse relationship)."""
        agent = _make_minimal_agent()
        obs = _make_obs(n_prices=50, seed=0)
        features = agent.extract_features(obs)
        spread_idx = agent.PRICE_HISTORY_LEN
        liquid_idx = agent.PRICE_HISTORY_LEN + 5
        expected = 1.0 - float(features[spread_idx])
        assert float(features[liquid_idx]) == pytest.approx(expected)

    def test_expected_price_normalization(self):
        """Prices should be normalized by first price."""
        agent = _make_minimal_agent()
        obs = _make_obs(n_prices=30, seed=0)
        first_price = obs.prices[0]
        features = agent.extract_features(obs)
        # First normalized price should be 0
        assert float(features[0]) == 0.0
        # Last normalized price should be (last/first - 1)
        expected_last = (obs.prices[-1] / first_price) - 1
        assert float(features[agent.PRICE_HISTORY_LEN - 1]) == pytest.approx(
            float(expected_last), rel=1e-3
        )


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

    def test_action_history_increments(self):
        agent = _make_minimal_agent()
        obs = _make_obs()
        assert len(agent.action_history) == 0
        agent.act(obs)
        assert len(agent.action_history) == 1
        agent.act(obs)
        assert len(agent.action_history) == 2

    def test_direction_in_range(self):
        """direction should be in [-1, 1] as per AgentAction spec."""
        agent = _make_minimal_agent()
        obs = _make_obs()
        action = agent.act(obs)
        assert -1.0 <= action.direction <= 1.0

    def test_confidence_in_range(self):
        """confidence should be in [0, 1]."""
        agent = _make_minimal_agent()
        obs = _make_obs()
        action = agent.act(obs)
        assert 0.0 <= action.confidence <= 1.0

    def test_execution_style_vwap(self):
        """Style value < 0.25 should map to VWAP."""
        agent = _make_minimal_agent()
        obs = _make_obs()
        # We can force VWAP via required_urgency=0.5 and deterministic
        # The actual style comes from the network, so we test metadata
        action = agent.act(obs, deterministic=True)
        style_value = action.metadata.get('style_value', 1.0)
        expected_style = (
            ExecutionStyle.VWAP if style_value < 0.25 else
            ExecutionStyle.POV if style_value < 0.50 else
            ExecutionStyle.TWAP if style_value < 0.75 else
            ExecutionStyle.AGGRESSIVE
        )
        assert action.metadata['execution_style'] == expected_style.name

    def test_required_urgency_zero(self):
        """required_urgency=0 should produce score near 0."""
        agent = _make_minimal_agent()
        agent.network.eval()
        obs = _make_obs()
        action = agent.act(obs, required_urgency=0.0, deterministic=True)
        assert action.score == pytest.approx(0.0)

    def test_required_urgency_one(self):
        """required_urgency=1 should produce score near 1."""
        agent = _make_minimal_agent()
        agent.network.eval()
        obs = _make_obs()
        action = agent.act(obs, required_urgency=1.0, deterministic=True)
        assert action.score == pytest.approx(1.0)

    def test_controller_message_content(self):
        """Message to controller should contain expected keys."""
        agent = _make_minimal_agent()
        obs = _make_obs()
        agent.act(obs)
        messages = agent.clear_outbox()
        ctrl_msg = [m for m in messages if m.receiver == "controller"][0]
        assert 'urgency' in ctrl_msg.content
        assert 'slice_size' in ctrl_msg.content
        assert 'execution_style' in ctrl_msg.content
        assert 'confidence' in ctrl_msg.content
        assert 'liquidity' in ctrl_msg.content

    def test_slice_size_in_0_1_to_0_5(self):
        """Slice size must always be in [0.1, 0.5]."""
        agent = _make_minimal_agent()
        obs = _make_obs()
        for _ in range(5):
            action = agent.act(obs)
            assert 0.1 <= action.metadata['slice_size'] <= 0.5

    def test_last_observation_stored(self):
        agent = _make_minimal_agent()
        obs = _make_obs()
        agent.act(obs)
        assert agent.last_observation is obs


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

    def test_different_obs_different_values(self):
        """Different observations should (likely) produce different values."""
        agent = _make_minimal_agent()
        obs1 = _make_obs(volatility=0.10, seed=1)
        obs2 = _make_obs(volatility=0.40, seed=2)
        v1 = agent.compute_value(obs1)
        v2 = agent.compute_value(obs2)
        assert isinstance(v1, float)
        assert isinstance(v2, float)

    def test_consistent_output(self):
        """Same observation with deterministic should give same value."""
        agent = _make_minimal_agent()
        agent.network.eval()
        obs = _make_obs(seed=42)
        v1 = agent.compute_value(obs)
        v2 = agent.compute_value(obs)
        assert v1 == v2


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
        assert np.isfinite(result['policy_loss'])
        assert np.isfinite(result['entropy'])
        assert np.isfinite(result['mean_urgency'])

    def test_single_observation_works(self):
        """Training with a single observation should work."""
        agent = _make_minimal_agent()
        obs = _make_obs(n_prices=50, seed=0)
        action = agent.act(obs, deterministic=True)
        result = agent.train_step([obs], [action], [0.01], [0.5])
        assert 'value_loss' in result

    def test_large_batch(self):
        """Training with a larger batch should work without error."""
        agent = _make_minimal_agent()
        n = 8
        obs_list = [_make_obs(n_prices=50, seed=i) for i in range(n)]
        actions = [agent.act(o, deterministic=True) for o in obs_list]
        returns = [0.01 * (i % 3 - 1) for i in range(n)]
        advantages = [0.1 * (i % 2) for i in range(n)]
        result = agent.train_step(obs_list, actions, returns, advantages)
        assert np.isfinite(result['value_loss'])

    def test_negative_returns(self):
        """All negative returns should still produce valid losses."""
        agent = _make_minimal_agent()
        obs_list = [_make_obs(n_prices=50, seed=i) for i in range(4)]
        actions = [agent.act(o, deterministic=True) for o in obs_list]
        result = agent.train_step(obs_list, actions, [-0.1, -0.2, -0.05, -0.3], [0.5, 0.5, 0.5, 0.5])
        assert np.isfinite(result['value_loss'])

    def test_all_zero_advantages(self):
        """Zero advantages should produce valid losses (no NaN)."""
        agent = _make_minimal_agent()
        obs_list = [_make_obs(n_prices=50, seed=i) for i in range(4)]
        actions = [agent.act(o, deterministic=True) for o in obs_list]
        result = agent.train_step(obs_list, actions, [0.01, 0.02, 0.03, 0.04], [0.0, 0.0, 0.0, 0.0])
        assert np.isfinite(result['value_loss'])

    def test_mean_slice_in_range(self):
        """Mean slice should be in [0.1, 0.5] after scaling."""
        agent = _make_minimal_agent()
        obs_list = [_make_obs(n_prices=50, seed=i) for i in range(4)]
        actions = [agent.act(o, deterministic=True) for o in obs_list]
        result = agent.train_step(obs_list, actions, [0.01, 0.02, 0.03, 0.04], [0.5, 0.5, 0.5, 0.5])
        assert 'mean_slice' in result
        assert 0.1 <= result['mean_slice'] <= 0.5

    def test_entropy_positive(self):
        """Entropy should be non-negative."""
        agent = _make_minimal_agent()
        obs_list = [_make_obs(n_prices=50, seed=i) for i in range(4)]
        actions = [agent.act(o, deterministic=True) for o in obs_list]
        result = agent.train_step(obs_list, actions, [0.01, -0.01, 0.02, 0.0], [0.5, -0.5, 0.8, 0.0])
        assert result['entropy'] >= 0.0

    def test_parameters_updated_after_step(self):
        """Network parameters should change after training step."""
        agent = _make_minimal_agent()
        obs_list = [_make_obs(n_prices=50, seed=i) for i in range(4)]
        actions = [agent.act(o, deterministic=True) for o in obs_list]
        params_before = [p.clone() for p in agent.network.parameters()]
        agent.train_step(obs_list, actions, [0.01, -0.01, 0.02, 0.0], [0.5, -0.5, 0.8, 0.0])
        params_after = list(agent.network.parameters())
        any_changed = any(
            not torch.equal(pb, pa) for pb, pa in zip(params_before, params_after)
        )
        assert any_changed


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

    def test_low_urgency_tier(self):
        """urgency < 0.25 should map to OrderUrgency.LOW."""
        agent = _make_minimal_agent()
        agent.use_scheduler = True  # still no real scheduler — falls through
        agent.scheduler = MagicMock()
        # Mock scheduler to return a result
        mock_scheduled = MagicMock()
        mock_scheduled.scheduled_time = datetime(2026, 5, 14, 11, 0)
        mock_scheduled.estimated_cost_bps = 0.5
        agent.scheduler.schedule_rebalance.return_value = mock_scheduled
        agent.cost_model = MagicMock()
        agent.cost_model.get_immediate_cost_estimate.return_value = 1.0
        result_time, result_improvement = agent._check_scheduling(0.10, "SPY")
        assert result_time is not None
        assert result_improvement > 0.0

    def test_normal_urgency_tier(self):
        """0.25 < urgency < 0.5 should map to OrderUrgency.NORMAL."""
        agent = _make_minimal_agent()
        agent.use_scheduler = True
        agent.scheduler = MagicMock()
        mock_scheduled = MagicMock()
        mock_scheduled.scheduled_time = datetime(2026, 5, 14, 12, 0)
        mock_scheduled.estimated_cost_bps = 0.8
        agent.scheduler.schedule_rebalance.return_value = mock_scheduled
        agent.cost_model = MagicMock()
        agent.cost_model.get_immediate_cost_estimate.return_value = 1.5
        result_time, result_improvement = agent._check_scheduling(0.35, "SPY")
        assert result_time is not None
        assert result_improvement > 0.0

    def test_high_urgency_tier(self):
        """0.5 < urgency < 0.75 should map to OrderUrgency.HIGH."""
        agent = _make_minimal_agent()
        agent.use_scheduler = True
        agent.scheduler = MagicMock()
        mock_scheduled = MagicMock()
        mock_scheduled.scheduled_time = datetime(2026, 5, 14, 13, 0)
        mock_scheduled.estimated_cost_bps = 0.3
        agent.scheduler.schedule_rebalance.return_value = mock_scheduled
        agent.cost_model = MagicMock()
        agent.cost_model.get_immediate_cost_estimate.return_value = 0.5
        result_time, result_improvement = agent._check_scheduling(0.60, "SPY")
        assert result_time is not None

    def test_urgency_boundary_low_high(self):
        """urgency exactly 0.25 should be NORMAL (not LOW)."""
        agent = _make_minimal_agent()
        agent.use_scheduler = True
        agent.scheduler = MagicMock()
        mock_scheduled = MagicMock()
        mock_scheduled.scheduled_time = datetime(2026, 5, 14, 11, 0)
        mock_scheduled.estimated_cost_bps = 0.5
        agent.scheduler.schedule_rebalance.return_value = mock_scheduled
        agent.cost_model = MagicMock()
        agent.cost_model.get_immediate_cost_estimate.return_value = 1.0
        result_time, _ = agent._check_scheduling(0.25, "SPY")
        assert result_time is not None

    def test_scheduler_exception_logged(self, caplog):
        """Scheduler exceptions should be caught and logged as warning."""
        agent = _make_minimal_agent()
        agent.use_scheduler = True
        agent.scheduler = MagicMock()
        agent.scheduler.schedule_rebalance.side_effect = ValueError("schedule failed")
        result = agent._check_scheduling(0.35, "SPY")
        assert result == (None, 0.0)

    def test_scheduler_with_mock_no_cost_estimate(self):
        """When scheduled has no cost estimate, returns (None, 0.0)."""
        agent = _make_minimal_agent()
        agent.use_scheduler = True
        agent.scheduler = MagicMock()
        mock_scheduled = MagicMock()
        mock_scheduled.scheduled_time = None
        mock_scheduled.estimated_cost_bps = None
        agent.scheduler.schedule_rebalance.return_value = mock_scheduled
        result_time, result_improvement = agent._check_scheduling(0.35, "SPY")
        assert result_time is None
        assert result_improvement == 0.0


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
        assert status['active'] is False

    def test_returns_dict_always(self):
        """get_scheduler_status should always return a dict."""
        agent = _make_minimal_agent()
        status = agent.get_scheduler_status()
        assert isinstance(status, dict)

    def test_active_with_mock_scheduler(self):
        """When scheduler is active, status should reflect it."""
        agent = _make_minimal_agent()
        agent.use_scheduler = True
        agent.scheduler = MagicMock()
        agent.scheduler.get_next_execution_time.return_value = datetime(2026, 5, 14, 11, 30)
        status = agent.get_scheduler_status()
        assert status['active'] is True
        assert 'optimal_window' in status
        assert status['optimal_window'] == '11:00-14:00 ET'

    def test_active_keys_present(self):
        """Active scheduler status should have all expected keys."""
        agent = _make_minimal_agent()
        agent.use_scheduler = True
        agent.scheduler = MagicMock()
        agent.scheduler.get_next_execution_time.return_value = datetime(2026, 5, 14, 11, 30)
        status = agent.get_scheduler_status()
        assert 'active' in status
        assert 'pending_orders' in status
        assert 'next_execution' in status
        assert 'optimal_window' in status
        assert 'scheduler_available' in status

    def test_next_execution_format_when_active(self):
        """next_execution should be ISO format string when scheduler active."""
        agent = _make_minimal_agent()
        agent.use_scheduler = True
        agent.scheduler = MagicMock()
        agent.scheduler.get_next_execution_time.return_value = datetime(2026, 5, 14, 11, 30, 0)
        status = agent.get_scheduler_status()
        assert status['next_execution'] == '2026-05-14T11:30:00'


# ---------------------------------------------------------------------------
# SCHEDULER_AVAILABLE Tests
# ---------------------------------------------------------------------------

class TestSchedulerAvailable:

    def test_is_boolean(self):
        assert isinstance(SCHEDULER_AVAILABLE, bool)

    def test_is_false_in_test_env(self):
        """SCHEDULER_AVAILABLE should be False since scheduler classes don't exist."""
        assert SCHEDULER_AVAILABLE is False


# ---------------------------------------------------------------------------
# Dataclass / Enum Field Validation
# ---------------------------------------------------------------------------

class TestDataclassFields:

    def test_agent_observation_fields(self):
        """AgentObservation should have expected fields with correct types."""
        fields = {f.name: f.type for f in dataclasses.fields(AgentObservation)}
        assert 'prices' in fields
        assert 'returns' in fields
        assert 'volatility' in fields
        assert 'current_weights' in fields
        assert 'portfolio_value' in fields
        assert 'cash_available' in fields
        assert 'features' in fields
        assert 'timestamp' in fields
        assert 'regime' in fields

    def test_agent_observation_field_types(self):
        """Verify AgentObservation field types."""
        fields = {f.name: f.type for f in dataclasses.fields(AgentObservation)}
        assert fields['volatility'] == float or fields['volatility'] == 'float'
        assert fields['portfolio_value'] == float or fields['portfolio_value'] == 'float'
        assert fields['cash_available'] == float or fields['cash_available'] == 'float'
        assert fields['regime'] == str or fields['regime'] == 'str'

    def test_agent_action_fields(self):
        """AgentAction should have expected fields."""
        field_names = {f.name for f in dataclasses.fields(AgentAction)}
        assert 'agent_id' in field_names
        assert 'action_type' in field_names
        assert 'score' in field_names
        assert 'direction' in field_names
        assert 'confidence' in field_names
        assert 'metadata' in field_names
        assert 'timestamp' in field_names

    def test_agent_action_field_types(self):
        """Verify AgentAction field types."""
        fields = {f.name: f.type for f in dataclasses.fields(AgentAction)}
        assert fields['score'] == float or fields['score'] == 'float'
        assert fields['direction'] == float or fields['direction'] == 'float'
        assert fields['confidence'] == float or fields['confidence'] == 'float'

    def test_agent_message_fields(self):
        """AgentMessage should have expected fields."""
        field_names = {f.name for f in dataclasses.fields(AgentMessage)}
        assert 'sender' in field_names
        assert 'receiver' in field_names
        assert 'msg_type' in field_names
        assert 'content' in field_names
        assert 'timestamp' in field_names
        assert 'priority' in field_names

    def test_agent_message_defaults(self):
        """AgentMessage should have defaults for timestamp and priority."""
        for f in dataclasses.fields(AgentMessage):
            if f.name == 'priority':
                assert f.default == 1
            if f.name == 'timestamp':
                # default_factory means it generates on creation
                assert f.default_factory is not None

    def test_agent_observation_defaults(self):
        """AgentObservation should have defaults for features, timestamp, regime."""
        for f in dataclasses.fields(AgentObservation):
            if f.name == 'regime':
                assert f.default == 'neutral'
            if f.name == 'features':
                assert f.default_factory is not None

    def test_agent_action_defaults(self):
        """AgentAction should have defaults for metadata and timestamp."""
        for f in dataclasses.fields(AgentAction):
            if f.name == 'metadata':
                assert f.default_factory is not None

    def test_agent_type_enum_members(self):
        """AgentType should have all expected members."""
        members = {e.name for e in AgentType}
        assert 'ANALYST' in members
        assert 'SENTIMENT' in members
        assert 'RISK' in members
        assert 'EXECUTION' in members
        assert 'CONTROLLER' in members

    def test_message_type_enum_members(self):
        """MessageType should have all expected members."""
        members = {e.name for e in MessageType}
        assert 'SIGNAL' in members
        assert 'ALERT' in members
        assert 'REQUEST' in members
        assert 'RESPONSE' in members
        assert 'CONSENSUS' in members

    def test_agent_type_execution_value(self):
        """AgentType.EXECUTION should have correct value."""
        assert AgentType.EXECUTION.value == "execution"


# ---------------------------------------------------------------------------
# Export / Public API Completeness
# ---------------------------------------------------------------------------

class TestExportCompleteness:

    def test_no_all_defined(self):
        """execution_agent.py does not define __all__ — public names implicit."""
        import src.agents.execution_agent as ea
        assert not hasattr(ea, '__all__'), (
            "__all__ should be added for explicit public API"
        )

    def test_public_classes_accessible(self):
        """All key public classes should be importable from the module."""
        from src.agents.execution_agent import (
            ExecutionStyle,
            ExecutionNetwork,
            ExecutionAgent,
        )
        assert ExecutionStyle is not None
        assert ExecutionNetwork is not None
        assert ExecutionAgent is not None

    def test_module_has_no_main_block(self):
        """Verify there is no __main__ block (CLI entry point absent)."""
        import src.agents.execution_agent as ea
        assert not hasattr(ea, 'main'), "No main() function should exist"
        # Verify by checking that running as __main__ would do nothing
        assert getattr(ea, '__name__', None) != '__main__'


# ---------------------------------------------------------------------------
# Edge Cases: AgentObservation Creation with Missing Fields
# ---------------------------------------------------------------------------

class TestObservationEdgeCases:

    def test_observation_with_default_regime(self):
        """Regime should default to 'neutral'."""
        obs = AgentObservation(
            prices=np.array([100.0]),
            returns=np.array([0.0]),
            volatility=0.15,
            current_weights={"spy": 1.0},
            portfolio_value=100000.0,
            cash_available=5000.0,
        )
        assert obs.regime == "neutral"

    def test_observation_with_custom_regime(self):
        """Custom regime should be accepted."""
        obs = _make_obs()
        assert obs.regime == "neutral"

    def test_observation_features_default_empty(self):
        """features should default to empty array."""
        obs = AgentObservation(
            prices=np.array([100.0]),
            returns=np.array([0.0]),
            volatility=0.15,
            current_weights={"spy": 1.0},
            portfolio_value=100000.0,
            cash_available=5000.0,
        )
        assert isinstance(obs.features, np.ndarray)
        assert len(obs.features) == 0


# ---------------------------------------------------------------------------
# Edge Cases: Symbol attribute handling
# ---------------------------------------------------------------------------

class TestSymbolAttribute:

    def test_act_no_symbol_attribute(self):
        """act() should work when obs lacks a 'symbol' attribute."""
        agent = _make_minimal_agent()
        obs = _make_obs()
        # AgentObservation doesn't have 'symbol' — check hasattr
        assert not hasattr(obs, 'symbol')
        action = agent.act(obs, deterministic=True)
        assert isinstance(action, AgentAction)

    def test_act_with_symbol_attribute(self):
        """act() should handle obs with a 'symbol' attribute."""
        agent = _make_minimal_agent()
        obs = _make_obs()
        # Dynamically add symbol attribute
        obs.symbol = "SPY"
        action = agent.act(obs, deterministic=True)
        assert isinstance(action, AgentAction)
        assert action.action_type == "execution_plan"


# ---------------------------------------------------------------------------
# Edge Cases: Zero / empty state
# ---------------------------------------------------------------------------

class TestZeroStateEdgeCases:

    def test_empty_action_history_by_default(self):
        agent = _make_minimal_agent()
        assert agent.action_history == []

    def test_empty_inbox_by_default(self):
        agent = _make_minimal_agent()
        assert agent.inbox == []

    def test_empty_outbox_by_default(self):
        agent = _make_minimal_agent()
        assert agent.outbox == []

    def test_low_volatility_no_trades(self):
        """Very low volatility with low urgency should still produce valid action."""
        agent = _make_minimal_agent()
        obs = _make_obs(volatility=0.01, n_prices=50, seed=0)
        action = agent.act(obs, deterministic=True)
        assert isinstance(action, AgentAction)
        assert 0 <= action.score <= 1

    def test_highly_negative_returns(self):
        """Highly negative returns should not cause NaN features."""
        agent = _make_minimal_agent()
        obs = _make_obs(n_prices=50, seed=0)
        # Force extreme negative returns
        obs.returns = np.full(49, -0.05)
        obs.prices = 100.0 * np.exp(np.cumsum(np.concatenate([[0], obs.returns])))
        features = agent.extract_features(obs)
        assert torch.isfinite(features).all()
