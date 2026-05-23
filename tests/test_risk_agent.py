"""
Tests for src/agents/risk_agent.py — Risk monitoring agent.

Pure-numpy methods (calculate_var, calculate_cvar, calculate_drawdown)
are fully testable without ML. Network/feature-dependent methods are
marked @pytest.mark.heavy and require PORTFOLIO_LAB_ENABLE_ML=1.

Coverage:
- Class attributes and thresholds (safe)
- calculate_var (safe — pure numpy)
- calculate_cvar (safe — pure numpy)
- calculate_drawdown (safe — pure numpy)
- extract_features, act, compute_value, train_step, RiskNetwork (heavy)
"""
import os
import pytest
import numpy as np

os.environ["PORTFOLIO_LAB_ENABLE_ML"] = "0"

from src.agents.base_agent import AgentType, AgentObservation, AgentAction
from src.agents.risk_agent import RiskAgent, RiskNetwork


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices(n=100, drift=0.0004, vol=0.015, seed=42):
    np.random.seed(seed)
    prices = [100.0]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + np.random.normal(drift, vol)))
    return np.array(prices)


def _make_obs(prices=None, n=100):
    if prices is None:
        prices = _make_prices(n)
    returns = np.diff(prices) / prices[:-1] if len(prices) > 1 else np.array([0.0])
    return AgentObservation(
        prices=prices,
        returns=returns,
        volatility=float(np.std(returns) * np.sqrt(252)) if len(returns) > 1 else 0.15,
        current_weights={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
        portfolio_value=100000.0,
        cash_available=10000.0,
    )


def _make_stub():
    """Create a RiskAgent stub for pure-numpy method testing."""
    agent = RiskAgent.__new__(RiskAgent)
    agent.agent_id = "risk"
    agent.agent_type = AgentType.RISK
    agent.obs_dim = RiskAgent.PRICE_HISTORY_LEN + RiskAgent.N_RISK_FEATURES
    agent.action_dim = 3
    agent.hidden_dim = 128
    agent.device = "cpu"
    agent.message_queue = []
    agent.action_history = []
    agent.observation_history = []
    agent.reward_history = []
    agent.last_observation = None
    agent.last_action = None
    agent.feature_names = [
        'var_95', 'var_99', 'cvar_95', 'current_dd', 'max_dd_1y',
        'dd_duration', 'volatility_20d', 'vol_regime', 'skewness',
        'kurtosis', 'tail_risk', 'correlation_stress', 'sharpe_recent',
        'risk_regime',
    ]
    agent.portfolio_high = None
    from src.agents.base_agent import torch, nn
    agent.network = RiskNetwork(agent.obs_dim, agent.action_dim, agent.hidden_dim)
    agent.optimizer = None
    return agent


# ===========================================================================
# SAFE TESTS — no ML required
# ===========================================================================

class TestClassAttributes:

    def test_feature_count(self):
        assert RiskAgent.N_RISK_FEATURES == 14

    def test_price_history_len(self):
        assert RiskAgent.PRICE_HISTORY_LEN == 60

    def test_var_threshold(self):
        assert RiskAgent.VAR_THRESHOLD == 0.02

    def test_drawdown_warning(self):
        assert RiskAgent.DRAWDOWN_WARNING == 0.10

    def test_drawdown_critical(self):
        assert RiskAgent.DRAWDOWN_CRITICAL == 0.20

    def test_feature_names_count(self):
        agent = _make_stub()
        assert len(agent.feature_names) == RiskAgent.N_RISK_FEATURES


class TestCalculateVaR:

    def test_normal_returns(self):
        agent = _make_stub()
        np.random.seed(42)
        returns = np.random.normal(0.0004, 0.01, 100)
        var_95 = agent.calculate_var(returns, alpha=0.05)
        assert var_95 < 0

    def test_empty_returns_default(self):
        agent = _make_stub()
        assert agent.calculate_var(np.array([])) == 0.02

    def test_positive_only_returns(self):
        agent = _make_stub()
        returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
        assert agent.calculate_var(returns, alpha=0.05) > 0

    def test_extreme_negative_returns(self):
        agent = _make_stub()
        returns = np.array([-0.10, -0.08, -0.05, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
        assert agent.calculate_var(returns, alpha=0.05) <= -0.08

    def test_var_99_more_extreme_than_95(self):
        agent = _make_stub()
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 500)
        assert agent.calculate_var(returns, alpha=0.01) <= agent.calculate_var(returns, alpha=0.05)

    def test_single_return(self):
        agent = _make_stub()
        var = agent.calculate_var(np.array([0.01]), alpha=0.05)
        assert np.isfinite(var)

    def test_uniform_returns(self):
        agent = _make_stub()
        returns = np.array([-0.02] * 50 + [0.02] * 50)
        assert agent.calculate_var(returns, alpha=0.05) == -0.02


class TestCalculateCVaR:

    def test_cvar_more_extreme_than_var(self):
        agent = _make_stub()
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 500)
        var_95 = agent.calculate_var(returns, alpha=0.05)
        cvar_95 = agent.calculate_cvar(returns, alpha=0.05)
        assert cvar_95 <= var_95 + 1e-10

    def test_cvar_empty_returns(self):
        agent = _make_stub()
        assert agent.calculate_cvar(np.array([])) == 0.02

    def test_cvar_with_left_tail(self):
        agent = _make_stub()
        returns = np.array([0.01, 0.02, -0.05, 0.01, 0.02, 0.03, 0.01, 0.02, 0.03, 0.04])
        var = agent.calculate_var(returns, alpha=0.1)
        cvar = agent.calculate_cvar(returns, alpha=0.1)
        assert cvar <= var + 1e-10

    def test_cvar_symmetric_returns(self):
        agent = _make_stub()
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 1000)
        cvar_95 = agent.calculate_cvar(returns, alpha=0.05)
        var_95 = agent.calculate_var(returns, alpha=0.05)
        assert cvar_95 < var_95 + 1e-8


class TestCalculateDrawdown:

    def test_no_drawdown_monotonic(self):
        agent = _make_stub()
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        dd, dur = agent.calculate_drawdown(prices)
        assert dd == 0.0
        assert dur == 0

    def test_drawdown_from_peak(self):
        agent = _make_stub()
        prices = np.array([100.0, 110.0, 105.0, 95.0, 90.0])
        dd, dur = agent.calculate_drawdown(prices)
        assert dd < -0.18
        assert dur > 0

    def test_empty_prices(self):
        agent = _make_stub()
        dd, dur = agent.calculate_drawdown(np.array([]))
        assert dd == 0.0
        assert dur == 0

    def test_drawdown_duration(self):
        agent = _make_stub()
        prices = np.array([100.0, 105.0, 110.0, 108.0, 105.0, 100.0])
        dd, dur = agent.calculate_drawdown(prices)
        assert dd < 0
        assert dur >= 3

    def test_recovery_resets_duration(self):
        agent = _make_stub()
        prices = np.array([100.0, 90.0, 95.0, 105.0, 110.0])
        dd, dur = agent.calculate_drawdown(prices)
        assert dd == 0.0
        assert dur == 0

    def test_single_price(self):
        agent = _make_stub()
        dd, dur = agent.calculate_drawdown(np.array([100.0]))
        assert dd == 0.0

    def test_flat_prices(self):
        agent = _make_stub()
        dd, dur = agent.calculate_drawdown(np.array([100.0] * 10))
        assert dd == 0.0

    def test_drawdown_after_new_peak(self):
        agent = _make_stub()
        prices = np.array([100.0, 110.0, 100.0, 115.0, 112.0])
        dd, dur = agent.calculate_drawdown(prices)
        assert abs(dd - (-3.0 / 115.0)) < 0.001

    def test_deep_crash_drawdown(self):
        agent = _make_stub()
        # 50% crash from peak
        prices = np.array([100.0, 150.0, 120.0, 75.0])
        dd, dur = agent.calculate_drawdown(prices)
        assert dd <= -0.50  # (75-150)/150 = -50%

    def test_two_drawdowns_current_recovered(self):
        agent = _make_stub()
        # Two dips but current price is at peak
        prices = np.array([100.0, 80.0, 110.0, 90.0, 120.0])
        dd, dur = agent.calculate_drawdown(prices)
        assert dd == 0.0  # At peak


# ===========================================================================
# HEAVY TESTS — require real torch (PORTFOLIO_LAB_ENABLE_ML=1)
# ===========================================================================

@pytest.mark.heavy
class TestRiskNetworkHeavy:

    def test_network_creation(self):
        net = RiskNetwork(obs_dim=74, action_dim=3, hidden_dim=128)
        assert hasattr(net, 'encoder')
        assert hasattr(net, 'risk_budget_head')
        assert hasattr(net, 'hedge_head')
        assert hasattr(net, 'confidence_head')
        assert hasattr(net, 'dd_warning_head')
        assert hasattr(net, 'value_head')

    def test_forward_output_shapes(self):
        import torch
        net = RiskNetwork(obs_dim=74, action_dim=3, hidden_dim=128)
        x = torch.randn(1, 74)
        rb, h, c, dw, v = net(x)
        assert rb.shape == (1, 1)
        assert h.shape == (1, 1)
        assert c.shape == (1, 1)
        assert dw.shape == (1, 1)
        assert v.shape == (1, 1)

    def test_forward_output_ranges(self):
        import torch
        net = RiskNetwork(obs_dim=74, action_dim=3, hidden_dim=64)
        x = torch.randn(1, 74)
        rb, h, c, dw, _ = net(x)
        assert 0.5 <= float(rb.squeeze()) <= 1.5
        assert 0.0 <= float(h.squeeze()) <= 1.0
        assert 0.0 <= float(c.squeeze()) <= 1.0
        assert 0.0 <= float(dw.squeeze()) <= 1.0

    def test_batch_forward(self):
        import torch
        net = RiskNetwork(obs_dim=74, action_dim=3, hidden_dim=64)
        x = torch.randn(8, 74)
        outputs = net(x)
        assert len(outputs) == 5
        assert outputs[0].shape == (8, 1)


@pytest.mark.heavy
class TestExtractFeaturesHeavy:

    def test_output_length(self):
        agent = RiskAgent()
        obs = _make_obs(n=100)
        features = agent.extract_features(obs)
        expected = RiskAgent.PRICE_HISTORY_LEN + RiskAgent.N_RISK_FEATURES
        assert features.shape == (expected,)

    def test_no_nan_valid_data(self):
        agent = RiskAgent()
        obs = _make_obs(n=100)
        features = agent.extract_features(obs)
        assert not np.isnan(features.numpy()).any()

    def test_short_price_series(self):
        agent = RiskAgent()
        obs = _make_obs(prices=_make_prices(n=30))
        features = agent.extract_features(obs)
        expected = RiskAgent.PRICE_HISTORY_LEN + RiskAgent.N_RISK_FEATURES
        assert features.shape == (expected,)

    def test_feature_positions_var(self):
        agent = RiskAgent()
        obs = _make_obs(n=100)
        arr = agent.extract_features(obs).numpy()
        h = RiskAgent.PRICE_HISTORY_LEN
        assert arr[h] < 0  # var_95 negative for typical returns

    def test_feature_positions_volatility(self):
        agent = RiskAgent()
        obs = _make_obs(n=100)
        arr = agent.extract_features(obs).numpy()
        h = RiskAgent.PRICE_HISTORY_LEN
        assert arr[h + 6] > 0  # volatility_20d positive


@pytest.mark.heavy
class TestRiskAgentActHeavy:

    def test_act_returns_action(self):
        agent = RiskAgent()
        obs = _make_obs(n=100)
        action = agent.act(obs)
        assert isinstance(action, AgentAction)

    def test_action_metadata_keys(self):
        agent = RiskAgent()
        obs = _make_obs(n=100)
        action = agent.act(obs)
        assert 'hedge_level' in action.metadata
        assert 'drawdown_warning' in action.metadata
        assert 'var_95' in action.metadata

    def test_risk_budget_score(self):
        agent = RiskAgent()
        obs = _make_obs(n=100)
        action = agent.act(obs, deterministic=True)
        assert 0.0 <= action.score <= 2.0

    def test_hedge_direction_negative(self):
        agent = RiskAgent()
        obs = _make_obs(n=100)
        action = agent.act(obs, deterministic=True)
        assert action.direction <= 0.01

    def test_deterministic_reproducible(self):
        agent = RiskAgent()
        obs = _make_obs(n=100)
        a1 = agent.act(obs, deterministic=True)
        a2 = agent.act(obs, deterministic=True)
        assert abs(a1.score - a2.score) < 0.01

    def test_compute_value_returns_float(self):
        agent = RiskAgent()
        obs = _make_obs(n=100)
        value = agent.compute_value(obs)
        assert isinstance(value, float)
        assert np.isfinite(value)

    def test_train_step_metrics(self):
        import torch
        agent = RiskAgent()
        obs = _make_obs(n=80)
        action = AgentAction(
            agent_id="risk", action_type="risk_signal",
            score=1.0, direction=-0.3, confidence=0.8,
        )
        result = agent.train_step([obs], [action], [1.0], [0.5])
        assert 'value_loss' in result
        assert 'policy_loss' in result
        assert 'entropy' in result

    def test_train_step_empty_batch(self):
        import torch
        agent = RiskAgent()
        result = agent.train_step([], [], [], [])
        assert result == {}
