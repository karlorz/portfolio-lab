"""
Tests for src/agents/base_agent.py — Base MARL agent infrastructure.
Runs without ML enabled (uses torch stubs).
"""
import pytest
import numpy as np
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

# Ensure torch stubs are used
os.environ["PORTFOLIO_LAB_ENABLE_ML"] = "0"

from src.agents.base_agent import (
    AgentType,
    MessageType,
    AgentMessage,
    AgentObservation,
    AgentAction,
    BaseAgent,
)


# Concrete subclass for testing abstract BaseAgent
class _TestAgent(BaseAgent):
    """Minimal concrete agent for testing BaseAgent functionality."""

    def __init__(self, agent_id="test_agent", agent_type=AgentType.ANALYST):
        super().__init__(
            agent_id=agent_id,
            agent_type=agent_type,
            obs_dim=10,
            action_dim=4,
            hidden_dim=64,
        )

    def build_network(self):
        from torch.nn import ModuleDict
        return ModuleDict({})

    def extract_features(self, obs):
        import torch
        return torch.zeros(10)

    def act(self, obs, deterministic=False):
        return AgentAction(
            agent_id=self.agent_id,
            action_type="hold",
            score=0.5,
            direction=0.0,
            confidence=0.5,
        )

    def compute_value(self, obs):
        return 0.0


class TestAgentType:
    """AgentType enum values."""

    def test_five_agent_types(self):
        assert len(AgentType) == 5

    def test_analyst_value(self):
        assert AgentType.ANALYST.value == "analyst"

    def test_sentiment_value(self):
        assert AgentType.SENTIMENT.value == "sentiment"

    def test_risk_value(self):
        assert AgentType.RISK.value == "risk"

    def test_execution_value(self):
        assert AgentType.EXECUTION.value == "execution"

    def test_controller_value(self):
        assert AgentType.CONTROLLER.value == "controller"


class TestMessageType:
    """MessageType enum values."""

    def test_five_message_types(self):
        assert len(MessageType) == 5

    def test_signal_value(self):
        assert MessageType.SIGNAL.value == "signal"

    def test_alert_value(self):
        assert MessageType.ALERT.value == "alert"

    def test_request_value(self):
        assert MessageType.REQUEST.value == "request"

    def test_response_value(self):
        assert MessageType.RESPONSE.value == "response"

    def test_consensus_value(self):
        assert MessageType.CONSENSUS.value == "consensus"


class TestAgentMessage:
    """AgentMessage dataclass."""

    def test_create_directed_message(self):
        msg = AgentMessage(
            sender="agent1",
            receiver="agent2",
            msg_type=MessageType.SIGNAL,
            content={"score": 0.75},
        )
        assert msg.sender == "agent1"
        assert msg.receiver == "agent2"
        assert msg.msg_type == MessageType.SIGNAL
        assert msg.content["score"] == 0.75
        assert msg.priority == 1  # default

    def test_create_broadcast_message(self):
        msg = AgentMessage(
            sender="controller",
            receiver=None,
            msg_type=MessageType.CONSENSUS,
            content={"decision": "hold"},
        )
        assert msg.receiver is None

    def test_priority_default_and_custom(self):
        msg = AgentMessage(
            sender="a", receiver="b", msg_type=MessageType.ALERT,
            content={}, priority=5,
        )
        assert msg.priority == 5

    def test_timestamp_auto_generated(self):
        msg = AgentMessage(
            sender="a", receiver="b",
            msg_type=MessageType.REQUEST, content={},
        )
        assert msg.timestamp is not None
        assert isinstance(msg.timestamp, str)


class TestAgentObservation:
    """AgentObservation dataclass."""

    def test_create_observation(self):
        obs = AgentObservation(
            prices=np.array([100.0, 101.0, 102.0]),
            returns=np.array([0.01, 0.0099]),
            volatility=0.15,
            current_weights={"SPY": 0.46, "GLD": 0.38, "TLT": 0.16},
            portfolio_value=100000.0,
            cash_available=5000.0,
        )
        assert obs.volatility == 0.15
        assert obs.portfolio_value == 100000.0
        assert obs.cash_available == 5000.0
        assert obs.regime == "neutral"  # default

    def test_default_features_empty_array(self):
        obs = AgentObservation(
            prices=np.array([100.0]),
            returns=np.array([0.01]),
            volatility=0.1,
            current_weights={},
            portfolio_value=0,
            cash_available=0,
        )
        assert len(obs.features) == 0

    def test_default_timestamp(self):
        obs = AgentObservation(
            prices=np.array([100.0]),
            returns=np.array([0.0]),
            volatility=0.1,
            current_weights={},
            portfolio_value=0,
            cash_available=0,
        )
        assert obs.timestamp is not None

    def test_custom_regime(self):
        obs = AgentObservation(
            prices=np.array([100.0]),
            returns=np.array([0.0]),
            volatility=0.25,
            current_weights={},
            portfolio_value=0,
            cash_available=0,
            regime="crisis",
        )
        assert obs.regime == "crisis"


class TestAgentAction:
    """AgentAction dataclass."""

    def test_create_action(self):
        action = AgentAction(
            agent_id="analyst_1",
            action_type="buy",
            score=0.85,
            direction=0.7,
            confidence=0.9,
        )
        assert action.agent_id == "analyst_1"
        assert action.action_type == "buy"
        assert action.score == 0.85
        assert action.direction == 0.7
        assert action.confidence == 0.9
        assert action.metadata == {}

    def test_metadata_field(self):
        action = AgentAction(
            agent_id="risk_1",
            action_type="alert",
            score=0.1,
            direction=-0.8,
            confidence=0.95,
            metadata={"reason": "var_breach"},
        )
        assert action.metadata["reason"] == "var_breach"

    def test_timestamp_auto_generated(self):
        action = AgentAction(
            agent_id="a", action_type="hold",
            score=0.5, direction=0.0, confidence=0.5,
        )
        assert isinstance(action.timestamp, str)


class TestBaseAgentInit:
    """BaseAgent initialization."""

    def test_create_concrete_agent(self):
        agent = _TestAgent()
        assert agent.agent_id == "test_agent"
        assert agent.agent_type == AgentType.ANALYST
        assert agent.obs_dim == 10
        assert agent.action_dim == 4
        assert agent.hidden_dim == 64

    def test_inbox_starts_empty(self):
        agent = _TestAgent()
        assert len(agent.inbox) == 0

    def test_outbox_starts_empty(self):
        agent = _TestAgent()
        assert len(agent.outbox) == 0

    def test_action_history_starts_empty(self):
        agent = _TestAgent()
        assert len(agent.action_history) == 0

    def test_no_last_observation_initially(self):
        agent = _TestAgent()
        assert agent.last_observation is None

    def test_no_last_action_initially(self):
        agent = _TestAgent()
        assert agent.last_action is None

    def test_accuracy_history_starts_empty(self):
        agent = _TestAgent()
        assert len(agent.accuracy_history) == 0

    def test_reward_history_starts_empty(self):
        agent = _TestAgent()
        assert len(agent.reward_history) == 0

    def test_different_agent_types(self):
        a1 = _TestAgent(agent_type=AgentType.RISK)
        assert a1.agent_type == AgentType.RISK
        a2 = _TestAgent(agent_type=AgentType.CONTROLLER)
        assert a2.agent_type == AgentType.CONTROLLER


class TestBaseAgentMessaging:
    """Inter-agent message passing."""

    def test_receive_directed_message(self):
        agent = _TestAgent()
        msg = AgentMessage(
            sender="other", receiver="test_agent",
            msg_type=MessageType.REQUEST, content={"query": "status"},
        )
        agent.receive_message(msg)
        assert len(agent.inbox) == 1
        assert agent.inbox[0].sender == "other"

    def test_receive_broadcast_message(self):
        agent = _TestAgent()
        msg = AgentMessage(
            sender="controller", receiver=None,
            msg_type=MessageType.CONSENSUS, content={"decision": "buy"},
        )
        agent.receive_message(msg)
        assert len(agent.inbox) == 1

    def test_receive_message_for_other_agent_ignored(self):
        agent = _TestAgent()
        msg = AgentMessage(
            sender="a", receiver="other_agent",
            msg_type=MessageType.SIGNAL, content={"score": 0.5},
        )
        agent.receive_message(msg)
        assert len(agent.inbox) == 0

    def test_send_message_queues_to_outbox(self):
        agent = _TestAgent()
        agent.send_message(
            receiver="risk_agent",
            msg_type=MessageType.ALERT,
            content={"level": "high"},
            priority=4,
        )
        assert len(agent.outbox) == 1
        msg = agent.outbox[0]
        assert msg.sender == "test_agent"
        assert msg.receiver == "risk_agent"
        assert msg.msg_type == MessageType.ALERT
        assert msg.content["level"] == "high"
        assert msg.priority == 4

    def test_send_message_default_priority(self):
        agent = _TestAgent()
        agent.send_message("x", MessageType.SIGNAL, {})
        assert agent.outbox[0].priority == 1

    def test_process_inbox_returns_and_clears(self):
        agent = _TestAgent()
        msg = AgentMessage(sender="a", receiver="test_agent",
                           msg_type=MessageType.SIGNAL, content={})
        agent.receive_message(msg)
        messages = agent.process_inbox()
        assert len(messages) == 1
        assert len(agent.inbox) == 0

    def test_clear_outbox_returns_and_clears(self):
        agent = _TestAgent()
        agent.send_message("a", MessageType.SIGNAL, {})
        messages = agent.clear_outbox()
        assert len(messages) == 1
        assert len(agent.outbox) == 0

    def test_multiple_messages_queued(self):
        agent = _TestAgent()
        for i in range(5):
            agent.send_message(f"agent{i}", MessageType.SIGNAL, {"i": i})
        assert len(agent.outbox) == 5


class TestBaseAgentAccuracy:
    """Accuracy tracking."""

    def test_initial_accuracy_is_default(self):
        agent = _TestAgent()
        assert agent.get_accuracy() == 0.5

    def test_correct_prediction_increases_accuracy(self):
        agent = _TestAgent()
        agent.update_accuracy(0.8, 0.3)  # Both positive → correct
        assert agent.get_accuracy() == 1.0

    def test_incorrect_prediction_reduces_accuracy(self):
        agent = _TestAgent()
        agent.update_accuracy(0.8, -0.3)  # Signs differ → wrong
        assert agent.get_accuracy() == 0.0

    def test_accuracy_window_is_last_20(self):
        agent = _TestAgent()
        # First 80 wrong
        for _ in range(80):
            agent.update_accuracy(0.5, -0.5)  # wrong
        # Last 20 right
        for _ in range(20):
            agent.update_accuracy(0.5, 0.5)  # correct
        assert agent.get_accuracy() == 1.0

    def test_accuracy_history_capped_at_100(self):
        agent = _TestAgent()
        for _ in range(150):
            agent.update_accuracy(0.5, 0.5)
        assert len(agent.accuracy_history) == 100

    def test_zero_inputs(self):
        agent = _TestAgent()
        agent.update_accuracy(0.0, 0.0)
        # 0 * 0 = 0, not > 0, so accuracy = 0
        assert agent.get_accuracy() == 0.0

    def test_mixed_accuracy_averaged(self):
        agent = _TestAgent()
        agent.update_accuracy(0.5, 0.5)   # correct
        agent.update_accuracy(0.5, -0.5)  # wrong
        agent.update_accuracy(-0.5, -0.5) # correct (both negative)
        agent.update_accuracy(-0.5, 0.5)  # wrong
        assert agent.get_accuracy() == 0.5


class TestBaseAgentSaveLoad:
    """Save/load functionality with torch stubs."""

    def test_save_does_not_raise(self, tmp_path):
        agent = _TestAgent()
        agent.update_accuracy(0.5, 0.5)
        path = tmp_path / "agent.pt"
        # torch.save is stubbed as no-op — should not crash
        agent.save(path)

    def test_load_with_mocked_checkpoint(self):
        agent = _TestAgent()
        mock_checkpoint = {
            "agent_id": "test_agent",
            "agent_type": "analyst",
            "state_dict": {},
            "accuracy_history": [1.0],
            "action_history_len": 0,
        }
        with patch("src.agents.base_agent.torch.load", return_value=mock_checkpoint):
            agent.load(Path("/fake/path.pt"))
        assert agent.accuracy_history == [1.0]


class TestBaseAgentAbstractMethods:
    """Abstract methods are callable on concrete subclass."""

    def test_build_network_returns_dict(self):
        agent = _TestAgent()
        net = agent.build_network()
        assert isinstance(net, dict)

    def test_extract_features_returns_array(self):
        agent = _TestAgent()
        obs = AgentObservation(
            prices=np.array([100.0]),
            returns=np.array([0.0]),
            volatility=0.1,
            current_weights={},
            portfolio_value=0,
            cash_available=0,
        )
        features = agent.extract_features(obs)
        import torch
        assert isinstance(features, (np.ndarray, torch.Tensor))

    def test_act_returns_agent_action(self):
        agent = _TestAgent()
        obs = AgentObservation(
            prices=np.array([100.0]),
            returns=np.array([0.0]),
            volatility=0.1,
            current_weights={},
            portfolio_value=0,
            cash_available=0,
        )
        action = agent.act(obs)
        assert isinstance(action, AgentAction)
        assert action.agent_id == "test_agent"

    def test_compute_value_returns_float(self):
        agent = _TestAgent()
        obs = AgentObservation(
            prices=np.array([100.0]),
            returns=np.array([0.0]),
            volatility=0.1,
            current_weights={},
            portfolio_value=0,
            cash_available=0,
        )
        value = agent.compute_value(obs)
        assert isinstance(value, float)


class TestDirectInstantiationForbidden:
    """Cannot instantiate BaseAgent directly (abstract)."""

    def test_base_agent_is_abstract(self):
        with pytest.raises(TypeError):
            BaseAgent(
                agent_id="x",
                agent_type=AgentType.ANALYST,
                obs_dim=10,
                action_dim=4,
            )
