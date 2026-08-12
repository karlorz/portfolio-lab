#!/usr/bin/env python3
"""Tests for src/agents/sentiment_agent.py and analyst_agent.py.

Both agents use torch stubs from base_agent, so they're importable
under PORTFOLIO_LAB_ENABLE_ML=0. Tests verify initialization, structure,
and methods that work with stubs (without full torch functionality).
"""
import os
from unittest.mock import patch, MagicMock

os.environ["PORTFOLIO_LAB_ENABLE_ML"] = "0"

from src.agents.sentiment_agent import SentimentAgent, SentimentNetwork
from src.agents.analyst_agent import AnalystAgent, AnalystNetwork
from src.agents.base_agent import AgentType


def _stub_torch_optim():
    """Patch torch.optim.Adam for agent __init__ (unavailable in stubs)."""
    from src.agents.base_agent import torch
    if not hasattr(torch, 'optim'):
        torch.optim = MagicMock()
        torch.optim.Adam = MagicMock(return_value=MagicMock())
    if not hasattr(torch, 'FloatTensor'):
        torch.FloatTensor = MagicMock(return_value=MagicMock())
    if not hasattr(torch, 'no_grad'):
        torch.no_grad = MagicMock()


def _make_sentiment_agent(**kwargs):
    _stub_torch_optim()
    return SentimentAgent(**kwargs)


def _make_analyst_agent(**kwargs):
    _stub_torch_optim()
    return AnalystAgent(**kwargs)


class TestSentimentAgentInit:
    def test_creates_agent(self):
        agent = _make_sentiment_agent(agent_id="test_sentiment")
        assert agent.agent_id == "test_sentiment"
        assert agent.agent_type == AgentType.SENTIMENT

    def test_has_network(self):
        agent = _make_sentiment_agent()
        assert agent.network is not None

    def test_build_network_returns_module(self):
        agent = _make_sentiment_agent()
        network = agent.build_network()
        assert network is not None

    def test_default_hidden_dim(self):
        agent = _make_sentiment_agent()
        assert agent.hidden_dim == 128

    def test_custom_hidden_dim(self):
        agent = _make_sentiment_agent(hidden_dim=64)
        assert agent.hidden_dim == 64

    def test_has_optimizer(self):
        agent = _make_sentiment_agent()
        assert hasattr(agent, 'optimizer')


class TestSentimentAgentTrainStep:
    def test_train_step_with_empty_data_returns_dict(self):
        agent = _make_sentiment_agent()
        result = agent.train_step([], [], [], [])
        assert isinstance(result, dict)

    def test_train_step_skips_empty_observations(self):
        agent = _make_sentiment_agent()
        result = agent.train_step([], [], [], [])
        # Empty observations -> early return with empty or minimal dict
        assert isinstance(result, dict)


class TestAnalystAgentInit:
    def test_creates_agent(self):
        agent = _make_analyst_agent(agent_id="test_analyst")
        assert agent.agent_id == "test_analyst"
        assert agent.agent_type == AgentType.ANALYST

    def test_has_network(self):
        agent = _make_analyst_agent()
        assert agent.network is not None

    def test_build_network_returns_module(self):
        agent = _make_analyst_agent()
        network = agent.build_network()
        assert network is not None

    def test_default_hidden_dim(self):
        agent = _make_analyst_agent()
        assert agent.hidden_dim == 128

    def test_custom_hidden_dim(self):
        agent = _make_analyst_agent(hidden_dim=64)
        assert agent.hidden_dim == 64

    def test_has_optimizer(self):
        agent = _make_analyst_agent()
        assert hasattr(agent, 'optimizer')


class TestAnalystAgentTrainStep:
    def test_train_step_with_empty_data_returns_dict(self):
        agent = _make_analyst_agent()
        result = agent.train_step([], [], [], [])
        assert isinstance(result, dict)

    def test_train_step_skips_empty_observations(self):
        agent = _make_analyst_agent()
        result = agent.train_step([], [], [], [])
        assert isinstance(result, dict)


class TestAgentIntegration:
    """Test agents work together in AIController with mocked graph."""

    def test_controller_creates_with_mocked_graph(self):
        from src.agents.ai_controller import AIController
        with patch("src.agents.ai_controller.AgentGraph") as mock_graph_cls:
            mock_graph = MagicMock()
            mock_graph.agents = {"analyst": MagicMock(), "sentiment": MagicMock()}
            mock_graph_cls.return_value = mock_graph
            ctrl = AIController(use_signal_integrator=False)
            status = ctrl.get_status()
            assert "analyst" in status["agents_loaded"]
            assert "sentiment" in status["agents_loaded"]

    def test_sentiment_network_init(self):
        """SentimentNetwork initializes without error."""
        net = SentimentNetwork(obs_dim=60, action_dim=3, hidden_dim=32)
        assert net is not None

    def test_analyst_network_init(self):
        """AnalystNetwork initializes without error."""
        net = AnalystNetwork(obs_dim=60, action_dim=3, hidden_dim=32)
        assert net is not None
