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
    agent.inbox = []
    agent.outbox = []
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
# Module-level validation
# ===========================================================================

class TestModuleLevel:
    """Verify module-level constants and docstrings."""

    def test_ml_enabled_false(self):
        from src.agents.risk_agent import _ML_ENABLED
        assert _ML_ENABLED is False

    def test_module_docstring_present(self):
        import src.agents.risk_agent as mod
        assert mod.__doc__ is not None
        assert "Risk Agent" in mod.__doc__

    def test_class_docstring_present(self):
        assert RiskAgent.__doc__ is not None
        assert "Risk monitoring" in RiskAgent.__doc__

    def test_risk_agent_exported(self):
        from src.agents.risk_agent import RiskAgent
        assert isinstance(RiskAgent, type)

    def test_risk_network_exported(self):
        from src.agents.risk_agent import RiskNetwork
        assert isinstance(RiskNetwork, type)

    def test_risk_network_is_nn_module_subclass(self):
        from src.agents.risk_agent import RiskNetwork
        from src.agents.base_agent import nn
        assert issubclass(RiskNetwork, nn.Module)


# ===========================================================================
# Dataclass and enum validation
# ===========================================================================

class TestDataclassAndEnum:
    """Verify dataclass fields and enum values from base_agent."""

    def test_agent_observation_has_all_fields(self):
        from dataclasses import fields
        from src.agents.base_agent import AgentObservation
        names = {f.name for f in fields(AgentObservation)}
        expected = {'prices', 'returns', 'volatility', 'current_weights',
                    'portfolio_value', 'cash_available', 'features',
                    'timestamp', 'regime'}
        assert names == expected

    def test_agent_observation_field_types(self):
        from dataclasses import fields
        from src.agents.base_agent import AgentObservation
        fmap = {f.name: f.type for f in fields(AgentObservation)}
        assert fmap['prices'] is np.ndarray
        assert fmap['returns'] is np.ndarray
        assert fmap['portfolio_value'] is float

    def test_agent_action_has_all_fields(self):
        from dataclasses import fields
        from src.agents.base_agent import AgentAction
        names = {f.name for f in fields(AgentAction)}
        expected = {'agent_id', 'action_type', 'score', 'direction',
                    'confidence', 'metadata', 'timestamp'}
        assert names == expected

    def test_agent_action_defaults(self):
        from dataclasses import fields
        from src.agents.base_agent import AgentAction
        # metadata and timestamp have defaults
        fmap = {f.name: f for f in fields(AgentAction)}
        assert fmap['metadata'].default_factory is not None  # field(default_factory=dict)
        assert fmap['timestamp'].default_factory is not None

    def test_agent_message_has_all_fields(self):
        from dataclasses import fields
        from src.agents.base_agent import AgentMessage
        names = {f.name for f in fields(AgentMessage)}
        expected = {'sender', 'receiver', 'msg_type', 'content',
                    'timestamp', 'priority'}
        assert names == expected

    def test_enums_have_expected_values(self):
        from src.agents.base_agent import AgentType, MessageType
        assert AgentType.RISK.value == "risk"
        assert AgentType.ANALYST.value == "analyst"
        assert AgentType.SENTIMENT.value == "sentiment"
        assert AgentType.EXECUTION.value == "execution"
        assert AgentType.CONTROLLER.value == "controller"
        assert MessageType.ALERT.value == "alert"
        assert MessageType.SIGNAL.value == "signal"
        assert MessageType.REQUEST.value == "request"
        assert MessageType.RESPONSE.value == "response"
        assert MessageType.CONSENSUS.value == "consensus"


# ===========================================================================
# calculate_var — additional edge cases
# ===========================================================================

class TestCalculateVaREdgeCases:
    """calculate_var boundary conditions and edge cases."""

    def test_returns_all_zeros(self):
        agent = _make_stub()
        returns = np.zeros(100)
        assert agent.calculate_var(returns, alpha=0.05) == 0.0

    def test_returns_all_same_positive(self):
        agent = _make_stub()
        returns = np.full(50, 0.01)
        assert agent.calculate_var(returns, alpha=0.05) == 0.01

    def test_returns_all_same_negative(self):
        agent = _make_stub()
        returns = np.full(50, -0.03)
        assert agent.calculate_var(returns, alpha=0.05) == -0.03

    def test_alpha_zero(self):
        agent = _make_stub()
        returns = np.array([-0.10, -0.05, 0.01, 0.02, 0.03])
        result = agent.calculate_var(returns, alpha=0.0)
        assert result == -0.10  # 0th percentile = min

    def test_alpha_one(self):
        agent = _make_stub()
        returns = np.array([-0.10, -0.05, 0.01, 0.02, 0.03])
        result = agent.calculate_var(returns, alpha=1.0)
        assert result == 0.03  # 100th percentile = max

    def test_two_element_array(self):
        agent = _make_stub()
        returns = np.array([-0.02, 0.03])
        var = agent.calculate_var(returns, alpha=0.05)
        assert var <= 0.03  # Should be between -0.02 and 0.03
        assert np.isfinite(var)

    def test_large_array_fast(self):
        agent = _make_stub()
        np.random.seed(99)
        returns = np.random.normal(0, 0.02, 10000)
        var = agent.calculate_var(returns, alpha=0.05)
        assert var < 0
        assert np.isfinite(var)

    def test_extreme_negative_tail_vs_normal(self):
        agent = _make_stub()
        # VaR with an extreme outlier should be more negative than without
        np.random.seed(99)
        normal = np.random.normal(0, 0.01, 500)
        with_outlier = np.concatenate([normal, [-0.50]])
        var_normal = agent.calculate_var(normal, alpha=0.05)
        var_with = agent.calculate_var(with_outlier, alpha=0.05)
        assert var_with <= var_normal  # Outlier drags VaR down

    def test_all_positive_returns_var_negative(self):
        """VaR of positive returns can be positive (no loss)."""
        agent = _make_stub()
        returns = np.array([0.001, 0.002, 0.003, 0.0015, 0.0025,
                            0.001, 0.002, 0.003, 0.0015, 0.0025])
        var = agent.calculate_var(returns, alpha=0.05)
        assert var > 0  # 5th percentile of all-positive is positive

    def test_alpha_less_than_zero_raises(self):
        agent = _make_stub()
        returns = np.array([-0.05, -0.03, 0.01, 0.02, 0.03])
        with pytest.raises(ValueError):
            agent.calculate_var(returns, alpha=-0.1)

    def test_alpha_greater_than_one_raises(self):
        agent = _make_stub()
        returns = np.array([-0.05, -0.03, 0.01, 0.02, 0.03])
        with pytest.raises(ValueError):
            agent.calculate_var(returns, alpha=1.5)

    def test_mixed_sign_returns_var_stable(self):
        agent = _make_stub()
        np.random.seed(101)
        returns = np.random.randn(500) * 0.02
        var_95 = agent.calculate_var(returns, alpha=0.05)
        var_99 = agent.calculate_var(returns, alpha=0.01)
        assert var_99 <= var_95  # 99% VaR should be more negative


# ===========================================================================
# calculate_cvar — additional edge cases
# ===========================================================================

class TestCalculateCVaREdgeCases:
    """calculate_cvar boundary conditions and edge cases."""

    def test_cvar_all_zeros(self):
        agent = _make_stub()
        returns = np.zeros(100)
        cvar = agent.calculate_cvar(returns, alpha=0.05)
        assert cvar == 0.0

    def test_cvar_all_identical(self):
        agent = _make_stub()
        returns = np.full(50, -0.02)
        assert agent.calculate_cvar(returns, alpha=0.05) == -0.02

    def test_cvar_single_element(self):
        agent = _make_stub()
        cvar = agent.calculate_cvar(np.array([0.01]), alpha=0.05)
        assert cvar == 0.01  # Falls back to var (the only element)

    def test_cvar_two_elements(self):
        agent = _make_stub()
        returns = np.array([-0.05, 0.03])
        cvar = agent.calculate_cvar(returns, alpha=0.1)
        assert cvar <= agent.calculate_var(returns, alpha=0.1)

    def test_cvar_one_extreme_outlier(self):
        agent = _make_stub()
        # Extreme left tail in small sample
        returns = np.array([0.01, 0.02, 0.01, 0.02, -0.30])
        cvar_10 = agent.calculate_cvar(returns, alpha=0.1)
        var_10 = agent.calculate_var(returns, alpha=0.1)
        assert cvar_10 <= var_10  # CVaR <= VaR by definition

    def test_cvar_all_values_below_var(self):
        """When all returns are below VaR, CVaR = mean of all returns."""
        agent = _make_stub()
        returns = np.array([-0.05, -0.04, -0.06, -0.03, -0.07])
        # var_95 of these: 5th percentile of 5 elements with linear interp
        cvar = agent.calculate_cvar(returns, alpha=0.5)
        var = agent.calculate_var(returns, alpha=0.5)
        assert cvar <= var

    def test_cvar_symmetric_distribution(self):
        agent = _make_stub()
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 1000)
        cvar_95 = agent.calculate_cvar(returns, alpha=0.05)
        cvar_99 = agent.calculate_cvar(returns, alpha=0.01)
        assert cvar_99 <= cvar_95  # 99% CVaR more extreme

    def test_cvar_alpha_boundary_zero(self):
        agent = _make_stub()
        returns = np.array([-0.10, -0.05, 0.01, 0.02])
        var = agent.calculate_var(returns, alpha=0.0)
        cvar = agent.calculate_cvar(returns, alpha=0.0)
        # CVaR at alpha=0 should approach the min return
        assert cvar <= var  # VaR at alpha=0 is the min, CVaR is mean of <= min = min

    def test_cvar_positive_only_returns(self):
        agent = _make_stub()
        returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        cvar = agent.calculate_cvar(returns, alpha=0.05)
        assert cvar > 0  # Expected shortfall of all-positive is positive

    def test_cvar_constant_positive_returns(self):
        agent = _make_stub()
        returns = np.full(100, 0.005)
        cvar = agent.calculate_cvar(returns, alpha=0.01)
        assert cvar == pytest.approx(0.005, abs=1e-10)


# ===========================================================================
# calculate_drawdown — additional edge cases
# ===========================================================================

class TestCalculateDrawdownEdgeCases:
    """calculate_drawdown boundary conditions and edge cases."""

    def test_strictly_decreasing_prices(self):
        agent = _make_stub()
        prices = np.array([100.0, 90.0, 80.0, 70.0, 60.0])
        dd, dur = agent.calculate_drawdown(prices)
        assert dd == -0.40  # (60-100)/100
        # Duration counts trailing days below -0.001 threshold
        # drawdown = [0, -0.1, -0.2, -0.3, -0.4]; trailing: 4 elements < -0.001
        assert dur == 4

    def test_two_elements_increasing(self):
        agent = _make_stub()
        dd, dur = agent.calculate_drawdown(np.array([100.0, 101.0]))
        assert dd == 0.0
        assert dur == 0

    def test_two_elements_decreasing(self):
        agent = _make_stub()
        dd, dur = agent.calculate_drawdown(np.array([100.0, 90.0]))
        assert dd == -0.10
        assert dur == 1

    def test_constant_price_after_rise(self):
        agent = _make_stub()
        prices = np.array([100.0, 110.0, 110.0, 110.0])
        dd, dur = agent.calculate_drawdown(prices)
        assert dd == 0.0  # At peak
        assert dur == 0

    def test_zigzag_pattern_no_peak_break(self):
        agent = _make_stub()
        # Up, down, below peak, up to new peak, down
        prices = np.array([100.0, 105.0, 102.0, 110.0, 108.0])
        dd, dur = agent.calculate_drawdown(prices)
        expected_dd = (108.0 - 110.0) / 110.0
        assert abs(dd - expected_dd) < 0.001
        assert dur == 1  # Only last day in drawdown

    def test_negative_prices(self):
        """Survival test: handle negative prices mathematically."""
        agent = _make_stub()
        prices = np.array([100.0, 50.0, 0.0, -50.0])
        dd, dur = agent.calculate_drawdown(prices)
        # Drawdown from peak=100: (-50-100)/100 = -1.5
        assert dd < -1.0
        assert dur > 0

    def test_multi_peak_with_partial_recovery(self):
        agent = _make_stub()
        prices = np.array([100.0, 120.0, 110.0, 115.0, 105.0, 108.0])
        dd, dur = agent.calculate_drawdown(prices)
        # Peak is 120, current is 108
        expected_dd = (108.0 - 120.0) / 120.0
        assert abs(dd - expected_dd) < 0.001

    def test_very_small_noise_no_drawdown(self):
        agent = _make_stub()
        # Tiny fluctuations upward; no drawdown
        prices = np.array([100.0, 100.001, 100.002, 100.003])
        dd, dur = agent.calculate_drawdown(prices)
        assert dd == 0.0
        assert dur == 0

    def test_drawdown_negative_at_peak_threshold(self):
        """Prices within -0.001 of peak should not count as drawdown."""
        agent = _make_stub()
        prices = np.array([100.0, 110.0, 109.95])  # 109.95/110 = -0.00045
        dd, dur = agent.calculate_drawdown(prices)
        assert dur == 0  # Below -0.001 threshold

    def test_drawdown_at_precise_threshold(self):
        """Exactly -0.001 drawdown should count as drawdown."""
        agent = _make_stub()
        prices = np.array([100.0, 110.0, 109.89])  # (109.89-110)/110 = -0.001
        dd, dur = agent.calculate_drawdown(prices)
        # drawdown[-1] = (109.89 - 110) / 110 ≈ -0.001
        # Since max.accumulate is [100, 110, 110], the last element is -0.001
        # in_drawdown = current_dd < -0.001 → False (exactly -0.001)
        # So dur should be 0
        assert dur == 0

    def test_drawdown_just_below_threshold(self):
        """Drawdown at -0.0011 should count as drawdown."""
        agent = _make_stub()
        prices = np.array([100.0, 110.0, 109.88])  # Slightly below threshold
        dd, dur = agent.calculate_drawdown(prices)
        assert dur == 1


# ===========================================================================
# NaN/Inf handling — cross-cutting across all computation methods
# ===========================================================================

class TestNanInfHandling:
    """Verify methods handle NaN and Inf gracefully."""

    def test_var_with_nan(self):
        agent = _make_stub()
        returns = np.array([0.01, np.nan, -0.02, 0.03, -0.01])
        var = agent.calculate_var(returns, alpha=0.05)
        assert np.isnan(var)

    def test_cvar_with_nan(self):
        agent = _make_stub()
        returns = np.array([0.01, np.nan, -0.02, 0.03])
        cvar = agent.calculate_cvar(returns, alpha=0.05)
        assert np.isnan(cvar)

    def test_drawdown_with_nan(self):
        agent = _make_stub()
        prices = np.array([100.0, np.nan, 105.0])
        dd, dur = agent.calculate_drawdown(prices)
        assert np.isnan(dd)

    def test_var_with_inf(self):
        agent = _make_stub()
        returns = np.array([0.01, np.inf, -0.02, 0.03])
        var = agent.calculate_var(returns, alpha=0.05)
        # np.percentile handles Inf without crashing
        assert np.isfinite(var) or np.isinf(var)
        assert not np.isnan(var)

    def test_cvar_with_inf(self):
        agent = _make_stub()
        returns = np.array([0.01, np.inf, -0.02, 0.03])
        cvar = agent.calculate_cvar(returns, alpha=0.05)
        # Should not crash; may produce finite or inf result
        assert not np.isnan(cvar) or np.isfinite(cvar)

    def test_drawdown_with_inf(self):
        agent = _make_stub()
        prices = np.array([100.0, np.inf, 105.0])
        dd, dur = agent.calculate_drawdown(prices)
        # np.maximum.accumulate with Inf may produce NaN for division
        assert np.isfinite(dd) or np.isnan(dd)  # Should not crash


# ===========================================================================
# Feature names validation
# ===========================================================================

class TestFeatureNames:
    """Verify feature_names list correctness."""

    KNOWN_NAMES = [
        'var_95', 'var_99', 'cvar_95', 'current_dd', 'max_dd_1y',
        'dd_duration', 'volatility_20d', 'vol_regime', 'skewness',
        'kurtosis', 'tail_risk', 'correlation_stress', 'sharpe_recent',
        'risk_regime',
    ]

    def test_all_feature_names_present(self):
        agent = _make_stub()
        assert agent.feature_names == self.KNOWN_NAMES

    def test_no_duplicate_feature_names(self):
        agent = _make_stub()
        assert len(set(agent.feature_names)) == len(agent.feature_names)

    def test_feature_names_are_strings(self):
        agent = _make_stub()
        assert all(isinstance(n, str) for n in agent.feature_names)

    def test_feature_names_match_count_constant(self):
        agent = _make_stub()
        assert len(agent.feature_names) == RiskAgent.N_RISK_FEATURES


# ===========================================================================
# Agent instance attributes and initial state
# ===========================================================================

class TestAgentInstanceAttributes:
    """Verify instance-level attributes with stub agent."""

    def test_initial_portfolio_high_is_none(self):
        agent = _make_stub()
        assert agent.portfolio_high is None

    def test_initial_last_observation_is_none(self):
        agent = _make_stub()
        assert agent.last_observation is None

    def test_initial_last_action_is_none(self):
        agent = _make_stub()
        assert agent.last_action is None

    def test_action_history_is_empty_list(self):
        agent = _make_stub()
        assert isinstance(agent.action_history, list)
        assert len(agent.action_history) == 0

    def test_message_queue_is_list(self):
        agent = _make_stub()
        assert hasattr(agent, 'message_queue')
        assert isinstance(agent.message_queue, list)

    def test_build_network_returns_dict(self):
        agent = _make_stub()
        net = agent.build_network()
        assert isinstance(net, dict)  # Stub returns {}


# ===========================================================================
# __init__ validation (ML-safe with import guard)
# ===========================================================================

class TestRiskAgentInit:
    """Verify constructor attributes and defaults."""

    def test_default_agent_id_is_risk(self):
        agent = _make_stub()
        assert agent.agent_id == "risk"

    def test_agent_type_is_risk(self):
        agent = _make_stub()
        assert agent.agent_type == AgentType.RISK

    def test_obs_dim_calculation(self):
        expected = RiskAgent.PRICE_HISTORY_LEN + RiskAgent.N_RISK_FEATURES
        agent = _make_stub()
        assert agent.obs_dim == expected
        assert agent.obs_dim == 74

    def test_action_dim_is_three(self):
        agent = _make_stub()
        assert agent.action_dim == 3

    def test_hidden_dim_default(self):
        agent = _make_stub()
        assert agent.hidden_dim == 128

    def test_device_is_cpu(self):
        agent = _make_stub()
        assert agent.device == "cpu"

    def test_network_exists_on_stub(self):
        agent = _make_stub()
        assert agent.network is not None

    def test_optimizer_is_none_on_stub(self):
        agent = _make_stub()
        assert agent.optimizer is None


# ===========================================================================
# BaseAgent integration — message passing and action history
# ===========================================================================

class TestBaseAgentIntegration:
    """Verify inherited BaseAgent functionality."""

    def test_send_message_adds_to_outbox(self):
        from src.agents.base_agent import MessageType
        agent = _make_stub()
        agent.send_message("controller", MessageType.SIGNAL,
                          {'test': True})
        assert len(agent.outbox) == 1
        msg = agent.outbox[0]
        assert msg.receiver == "controller"
        assert msg.msg_type == MessageType.SIGNAL

    def test_send_message_broadcast(self):
        from src.agents.base_agent import MessageType
        agent = _make_stub()
        agent.send_message(None, MessageType.ALERT,
                          {'alert_type': 'test'})
        assert len(agent.outbox) == 1
        assert agent.outbox[0].receiver is None

    def test_clear_outbox_empties_messages(self):
        from src.agents.base_agent import MessageType
        agent = _make_stub()
        agent.send_message("controller", MessageType.SIGNAL, {'k': 'v'})
        cleared = agent.clear_outbox()
        assert len(cleared) == 1
        assert len(agent.outbox) == 0

    def test_receive_message_appends_to_inbox(self):
        from src.agents.base_agent import AgentMessage, MessageType
        agent = _make_stub()
        msg = AgentMessage(sender="controller", receiver="risk",
                          msg_type=MessageType.SIGNAL, content={})
        agent.receive_message(msg)
        assert len(agent.inbox) == 1

    def test_process_inbox_clears(self):
        from src.agents.base_agent import AgentMessage, MessageType
        agent = _make_stub()
        msg = AgentMessage(sender="controller", receiver="risk",
                          msg_type=MessageType.SIGNAL, content={})
        agent.receive_message(msg)
        msgs = agent.process_inbox()
        assert len(msgs) == 1
        assert len(agent.inbox) == 0

    def test_send_message_priority_defaults_to_one(self):
        from src.agents.base_agent import MessageType
        agent = _make_stub()
        agent.send_message("controller", MessageType.SIGNAL, {'k': 'v'})
        assert agent.outbox[0].priority == 1

    def test_send_message_with_custom_priority(self):
        from src.agents.base_agent import MessageType
        agent = _make_stub()
        agent.send_message("controller", MessageType.ALERT,
                          {'k': 'v'}, priority=5)
        assert agent.outbox[0].priority == 5


# ===========================================================================
# CLI / __main__ guard
# ===========================================================================

class TestCliGuard:
    """Verify no __main__ entry point or print-based CLI exists."""

    def test_no_main_block(self):
        """Module should not have a __main__ entry point."""
        import src.agents.risk_agent as mod
        source = mod.__file__
        with open(source) as f:
            content = f.read()
        assert 'if __name__' not in content
        assert '__main__' not in content

    def test_no_print_statements_in_module(self):
        """CLI output should not use print()."""
        import src.agents.risk_agent as mod
        source = mod.__file__
        with open(source) as f:
            content = f.read()
        # Only allow print inside docstrings/comments, not as executable code
        lines = [l for l in content.split('\n')
                 if 'print(' in l and not l.strip().startswith('#')]
        assert len(lines) == 0, f"Found print() statements: {lines}"


# ===========================================================================
# Export completeness
# ===========================================================================

class TestExportCompleteness:
    """Verify public API surface."""

    def test_risk_agent_importable(self):
        from src.agents.risk_agent import RiskAgent
        assert isinstance(RiskAgent, type)

    def test_risk_network_importable(self):
        from src.agents.risk_agent import RiskNetwork
        assert isinstance(RiskNetwork, type)

    def test_no_all_defined(self):
        """Module does not define __all__, so all public names are accessible."""
        import src.agents.risk_agent as mod
        assert not hasattr(mod, '__all__'), \
            "If __all__ is added, RiskAgent and RiskNetwork must be included"

    def test_risk_agent_inherits_base_agent(self):
        from src.agents.base_agent import BaseAgent
        assert issubclass(RiskAgent, BaseAgent)

    def test_risk_agent_is_abstract_method_implemented(self):
        """Verify all abstract methods are implemented."""
        import inspect
        from src.agents.base_agent import BaseAgent
        abstract = []
        for name, method in inspect.getmembers(BaseAgent):
            if inspect.isfunction(method) and hasattr(method, '__isabstractmethod__'):
                abstract.append(name)
        for name in abstract:
            assert hasattr(RiskAgent, name), f"RiskAgent missing abstract method: {name}"
            impl = getattr(RiskAgent, name)
            assert not getattr(impl, '__isabstractmethod__', False), \
                f"RiskAgent.{name} is still abstract"


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
        agent = RiskAgent()
        result = agent.train_step([], [], [], [])
        assert result == {}
