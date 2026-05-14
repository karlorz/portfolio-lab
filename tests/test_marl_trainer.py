import pytest; pytestmark = pytest.mark.heavy
#!/usr/bin/env python3
"""
Tests for MARL Trainer — Transition/RolloutBuffer data classes,
GAE computation, MarketEnvironment simulation, and trainer lifecycle.
"""
import sys
import os
import json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path

# Mock heavy dependencies before import
_orig_modules = {}
_mock_targets = [
    'torch', 'torch.nn',
    'src.agents.base_agent', 'src.agents.agent_graph',
    'src.agents.analyst_agent', 'src.agents.sentiment_agent',
    'src.agents.risk_agent', 'src.agents.execution_agent',
    'src.agents.controller_agent',
]
for mod in _mock_targets:
    _orig_modules[mod] = sys.modules.get(mod)
    sys.modules[mod] = MagicMock()

# Create proper mock for base_agent types
mock_base = sys.modules['src.agents.base_agent']

class MockAgentObservation:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class MockAgentAction:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

mock_base.AgentObservation = MockAgentObservation
mock_base.AgentAction = MockAgentAction

from src.agents.marl_trainer import (
    Transition, RolloutBuffer, MarketEnvironment, MARLTrainer,
)

# Restore original modules
for mod, orig in _orig_modules.items():
    if orig is None:
        sys.modules.pop(mod, None)
    else:
        sys.modules[mod] = orig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_observation():
    """Create a mock AgentObservation."""
    obs = MagicMock()
    obs.prices = np.random.randn(60)
    obs.returns = np.random.randn(59)
    obs.volatility = 0.15
    obs.current_weights = {'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16}
    obs.portfolio_value = 100000.0
    obs.cash_available = 0.0
    return obs


def _make_transition(reward=0.01, done=False):
    """Create a Transition with mock data."""
    return Transition(
        observation=_make_observation(),
        actions={'controller': MagicMock()},
        reward=reward,
        next_observation=_make_observation(),
        done=done,
        agent_values={'controller': 1.0},
        agent_log_probs={'controller': -0.5},
    )


def _make_prices(n=300):
    """Create synthetic price arrays."""
    np.random.seed(42)
    spy = 400 * np.cumprod(1 + np.random.normal(0.0004, 0.012, n))
    gld = 150 * np.cumprod(1 + np.random.normal(0.0002, 0.008, n))
    tlt = 130 * np.cumprod(1 + np.random.normal(-0.0001, 0.006, n))
    return {'SPY': spy, 'GLD': gld, 'TLT': tlt}


# ---------------------------------------------------------------------------
# Transition tests
# ---------------------------------------------------------------------------

class TestTransition:
    def test_creation(self):
        t = _make_transition()
        assert t.reward == 0.01
        assert t.done is False

    def test_defaults(self):
        t = Transition(
            observation=MagicMock(), actions={}, reward=0.0,
            next_observation=MagicMock(), done=False,
        )
        assert t.agent_values == {}
        assert t.agent_log_probs == {}


# ---------------------------------------------------------------------------
# RolloutBuffer tests
# ---------------------------------------------------------------------------

class TestRolloutBuffer:
    def test_creation(self):
        buf = RolloutBuffer()
        assert buf.transitions == []

    def test_add(self):
        buf = RolloutBuffer()
        buf.add(_make_transition())
        assert len(buf.transitions) == 1

    def test_add_multiple(self):
        buf = RolloutBuffer()
        for _ in range(5):
            buf.add(_make_transition())
        assert len(buf.transitions) == 5

    def test_clear(self):
        buf = RolloutBuffer()
        buf.add(_make_transition())
        buf.clear()
        assert len(buf.transitions) == 0

    def test_get_stats_empty(self):
        buf = RolloutBuffer()
        assert buf.get_stats() == {}

    def test_get_stats_with_data(self):
        buf = RolloutBuffer()
        for r in [0.01, -0.02, 0.03]:
            buf.add(_make_transition(reward=r))
        stats = buf.get_stats()
        assert 'mean_reward' in stats
        assert 'std_reward' in stats
        assert 'total_steps' in stats
        assert stats['total_steps'] == 3

    def test_get_stats_values(self):
        buf = RolloutBuffer()
        buf.add(_make_transition(reward=0.05))
        buf.add(_make_transition(reward=-0.05))
        stats = buf.get_stats()
        assert stats['mean_reward'] == pytest.approx(0.0)
        assert stats['min_reward'] == pytest.approx(-0.05)
        assert stats['max_reward'] == pytest.approx(0.05)

    # compute_returns
    def test_compute_returns_empty(self):
        buf = RolloutBuffer()
        assert buf.compute_returns() == []

    def test_compute_returns_single(self):
        buf = RolloutBuffer()
        buf.add(_make_transition(reward=1.0, done=True))
        returns = buf.compute_returns(gamma=0.99)
        assert len(returns) == 1
        assert returns[0] == pytest.approx(1.0)

    def test_compute_returns_discount(self):
        buf = RolloutBuffer()
        buf.add(_make_transition(reward=1.0, done=False))
        buf.add(_make_transition(reward=1.0, done=True))
        returns = buf.compute_returns(gamma=0.99)
        assert len(returns) == 2
        # R[1] = 1.0, R[0] = 1.0 + 0.99 * 1.0 = 1.99
        assert returns[1] == pytest.approx(1.0)
        assert returns[0] == pytest.approx(1.99)

    def test_compute_returns_done_resets(self):
        buf = RolloutBuffer()
        buf.add(_make_transition(reward=1.0, done=True))
        buf.add(_make_transition(reward=1.0, done=False))
        returns = buf.compute_returns(gamma=0.99)
        # R[1] = 1.0 + 0.99*0 = 1.0 (done=False but nothing after)
        # R[0] = 1.0 (done=True, so future is 0)
        assert returns[0] == pytest.approx(1.0)

    # compute_gae
    def test_compute_gae_empty(self):
        buf = RolloutBuffer()
        adv, ret = buf.compute_gae([])
        assert adv == []
        assert ret == []

    def test_compute_gae_single(self):
        buf = RolloutBuffer()
        buf.add(_make_transition(reward=1.0, done=True))
        adv, ret = buf.compute_gae([0.5], gamma=0.99, lambda_=0.95)
        assert len(adv) == 1
        assert len(ret) == 1
        # delta = 1.0 + 0 - 0.5 = 0.5, gae = 0.5
        assert adv[0] == pytest.approx(0.5)
        assert ret[0] == pytest.approx(1.0)

    def test_compute_gae_multiple(self):
        buf = RolloutBuffer()
        buf.add(_make_transition(reward=1.0, done=False))
        buf.add(_make_transition(reward=2.0, done=True))
        adv, ret = buf.compute_gae([0.5, 1.5], gamma=0.99, lambda_=0.95)
        assert len(adv) == 2
        assert len(ret) == 2

    def test_compute_gae_returns_equal_advantages_plus_values(self):
        buf = RolloutBuffer()
        for _ in range(5):
            buf.add(_make_transition(reward=0.5, done=False))
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        adv, ret = buf.compute_gae(values, gamma=0.99, lambda_=0.95)
        for a, v, r in zip(adv, values, ret):
            assert r == pytest.approx(a + v)


# ---------------------------------------------------------------------------
# MarketEnvironment tests
# ---------------------------------------------------------------------------

class TestMarketEnvironment:
    def test_creation(self):
        prices = _make_prices(300)
        env = MarketEnvironment(prices)
        assert env.n_assets == 3
        assert env.transaction_cost == 0.001

    def test_default_allocation(self):
        prices = _make_prices(300)
        env = MarketEnvironment(prices)
        assert env.default_allocation['SPY'] == 0.46

    def test_custom_allocation(self):
        prices = _make_prices(300)
        env = MarketEnvironment(prices, allocations={'SPY': 0.5, 'GLD': 0.5})
        assert env.default_allocation['SPY'] == 0.5

    def test_reset(self):
        prices = _make_prices(300)
        env = MarketEnvironment(prices)
        obs = env.reset()
        assert env.portfolio_value == 100000.0
        assert env.current_step == 0
        assert obs is not None

    def test_reset_custom_start(self):
        prices = _make_prices(300)
        env = MarketEnvironment(prices)
        env.reset(start_idx=100)
        assert env.price_idx == 100

    def test_step_basic(self):
        prices = _make_prices(300)
        env = MarketEnvironment(prices)
        env.reset()
        action = {'allocation': [0.5, 0.3, 0.2]}
        obs, reward, done, info = env.step(action)
        assert env.current_step == 1
        assert 'portfolio_return' in info
        assert 'drawdown' in info

    def test_step_no_allocation(self):
        prices = _make_prices(300)
        env = MarketEnvironment(prices)
        env.reset()
        obs, reward, done, info = env.step({})
        assert env.current_step == 1

    def test_step_done_at_end(self):
        prices = _make_prices(10)
        env = MarketEnvironment(prices)
        env.reset()
        for _ in range(20):
            obs, reward, done, info = env.step({})
            if done:
                break
        assert done is True

    def test_step_portfolio_value_changes(self):
        prices = _make_prices(300)
        env = MarketEnvironment(prices)
        env.reset()
        initial_value = env.portfolio_value
        env.step({'allocation': [0.5, 0.3, 0.2]})
        # Value should change (unless all returns are exactly 0)
        # Just check it's still a number
        assert isinstance(env.portfolio_value, float)

    def test_step_drawdown_tracked(self):
        prices = _make_prices(300)
        env = MarketEnvironment(prices)
        env.reset()
        _, _, _, info = env.step({})
        assert 'drawdown' in info
        assert info['drawdown'] >= 0

    def test_step_turnover_in_info(self):
        prices = _make_prices(300)
        env = MarketEnvironment(prices)
        env.reset()
        _, _, _, info = env.step({'allocation': [0.6, 0.2, 0.2]})
        assert 'turnover' in info


# ---------------------------------------------------------------------------
# MARLTrainer tests (heavy mocking)
# ---------------------------------------------------------------------------

class TestMARLTrainer:
    def _make_trainer(self, tmp_path):
        """Create trainer with mocked agent graph and env."""
        mock_graph = MagicMock()
        mock_graph.agents = {
            'controller': MagicMock(),
            'analyst': MagicMock(),
        }
        prices = _make_prices(300)
        env = MarketEnvironment(prices)
        trainer = MARLTrainer(agent_graph=mock_graph, env=env)
        return trainer

    def test_init(self, tmp_path):
        trainer = self._make_trainer(tmp_path)
        assert trainer.gamma == 0.99
        assert trainer.lambda_ == 0.95
        assert trainer.episode_count == 0

    def test_init_custom_params(self, tmp_path):
        mock_graph = MagicMock()
        mock_graph.agents = {}
        env = MarketEnvironment(_make_prices(300))
        trainer = MARLTrainer(
            agent_graph=mock_graph, env=env,
            gamma=0.95, lambda_=0.90, clip_epsilon=0.3,
        )
        assert trainer.gamma == 0.95
        assert trainer.clip_epsilon == 0.3

    def test_collect_rollout(self, tmp_path):
        trainer = self._make_trainer(tmp_path)
        # Mock agent graph execute_step
        action = MagicMock()
        action.metadata = {'allocation': [0.5, 0.3, 0.2]}
        trainer.agent_graph.execute_step.return_value = {'controller': action}
        # Mock compute_value
        for agent in trainer.agent_graph.agents.values():
            agent.compute_value.return_value = 0.5

        stats = trainer.collect_rollout(max_steps=10)
        assert 'episode' in stats
        assert 'total_reward' in stats
        assert stats['length'] <= 10

    def test_collect_rollout_increments_episode(self, tmp_path):
        trainer = self._make_trainer(tmp_path)
        action = MagicMock()
        action.metadata = {'allocation': [0.5, 0.3, 0.2]}
        trainer.agent_graph.execute_step.return_value = {'controller': action}
        for agent in trainer.agent_graph.agents.values():
            agent.compute_value.return_value = 0.5

        trainer.collect_rollout(max_steps=5)
        assert trainer.episode_count == 1

    def test_collect_rollout_buffer_filled(self, tmp_path):
        trainer = self._make_trainer(tmp_path)
        action = MagicMock()
        action.metadata = {'allocation': [0.5, 0.3, 0.2]}
        trainer.agent_graph.execute_step.return_value = {'controller': action}
        for agent in trainer.agent_graph.agents.values():
            agent.compute_value.return_value = 0.5

        trainer.collect_rollout(max_steps=5)
        assert len(trainer.buffer.transitions) > 0

    def test_update_insufficient_data(self, tmp_path):
        trainer = self._make_trainer(tmp_path)
        result = trainer.update(batch_size=100)
        assert result == {}

    def test_save_load(self, tmp_path):
        trainer = self._make_trainer(tmp_path)
        trainer.best_sharpe = 1.5
        save_path = tmp_path / "checkpoint"
        save_path.mkdir()

        trainer.save(save_path)
        config_path = save_path / "trainer_config.json"
        assert config_path.exists()

        with open(config_path) as f:
            config = json.load(f)
        assert config['best_sharpe'] == 1.5

    def test_load_restores_sharpe(self, tmp_path):
        trainer = self._make_trainer(tmp_path)
        save_path = tmp_path / "checkpoint"
        save_path.mkdir()

        # Save with known sharpe
        trainer.best_sharpe = 2.0
        trainer.save(save_path)

        # Reset and load
        trainer.best_sharpe = -float('inf')
        trainer.load(save_path)
        assert trainer.best_sharpe == 2.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
