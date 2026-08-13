#!/usr/bin/env python3
"""
Tests for src/agents/agent_graph.py — LangGraph-style agent communication layer.

Tests cover: NodeType enum, GraphEdge dataclass, AgentGraph init/registration/
topology/message routing/execution/save-load/viz without real torch.
ML-gated agent classes (AnalystAgent, SentimentAgent, etc.) are mocked.
"""

import os
import json
import threading
import tempfile
import numpy as np
from pathlib import Path
from collections import deque
from unittest.mock import MagicMock, patch

# Ensure ML stubs are active
os.environ["PORTFOLIO_LAB_ENABLE_ML"] = "0"

# ── Import the module under test ──────────────────────────────────────────
# The import works without real torch because all agent modules use the
# base_agent torch stubs (nn.Module -> _StubModule) when ML is disabled.
from src.agents.base_agent import AgentMessage, MessageType
from src.agents.agent_graph import (
    AgentGraph,
    NodeType,
    GraphEdge,
)
# Import ControllerAgent for isinstance checks in consensus tests.
# This module works without ML via base_agent torch stubs.
from src.agents.controller_agent import ControllerAgent


# ═══════════════════════════════════════════════════════════════════════════
# Helper: create a mock agent with the bare attributes agent_graph touches
# ═══════════════════════════════════════════════════════════════════════════

def _make_mock_agent(agent_id: str = "test_agent"):
    """Create a MagicMock that looks like a BaseAgent for graph operations.

    Uses a plain MagicMock (no spec) because BaseAgent inherits from the
    stub nn.Module which itself has a stub __init__ that takes *args/**kwargs.
    A spec=BaseAgent mock would still work but we avoid it for simplicity.
    """
    agent = MagicMock()
    agent.agent_id = agent_id
    agent.inbox = []
    agent.outbox = []
    agent.act.return_value = MagicMock(
        score=0.5, direction=0.0, confidence=0.5
    )
    agent.clear_outbox.return_value = []
    agent.process_inbox.return_value = []
    agent.receive_message = MagicMock()
    agent.save = MagicMock()
    agent.load = MagicMock()
    return agent


def _agent_side_effect(**kwargs):
    """Side-effect factory for patched agent classes.

    Returns a mock whose ``agent_id`` matches the ``agent_id`` kwarg so that
    ``AgentGraph.register_agent`` inserts it under the correct key.
    """
    aid = kwargs.get("agent_id", "unknown")
    return _make_mock_agent(aid)


# ═══════════════════════════════════════════════════════════════════════════
# NodeType Enum Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestNodeType:
    """NodeType enum -- 5 values for agent node types."""

    def test_five_node_types(self):
        assert len(NodeType) == 5

    def test_analyst_value(self):
        assert NodeType.ANALYST.value == "analyst"

    def test_sentiment_value(self):
        assert NodeType.SENTIMENT.value == "sentiment"

    def test_risk_value(self):
        assert NodeType.RISK.value == "risk"

    def test_execution_value(self):
        assert NodeType.EXECUTION.value == "execution"

    def test_controller_value(self):
        assert NodeType.CONTROLLER.value == "controller"

    def test_membership_contains(self):
        assert "analyst" in {e.value for e in NodeType}

    def test_from_string_roundtrip(self):
        for nt in NodeType:
            assert NodeType(nt.value) == nt


# ═══════════════════════════════════════════════════════════════════════════
# GraphEdge Dataclass Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestGraphEdge:
    """GraphEdge dataclass -- directed edge between agent nodes."""

    def test_default_creation(self):
        edge = GraphEdge(source="a", target="b")
        assert edge.source == "a"
        assert edge.target == "b"
        assert edge.msg_types == set()
        assert edge.filter_fn is None
        assert edge.priority_boost == 0

    def test_with_msg_types(self):
        edge = GraphEdge(
            source="risk",
            target="controller",
            msg_types={MessageType.ALERT, MessageType.SIGNAL}
        )
        assert MessageType.ALERT in edge.msg_types
        assert MessageType.SIGNAL in edge.msg_types

    def test_with_filter_fn(self):
        def fn(m):
            return m.priority > 2
        edge = GraphEdge(source="a", target="b", filter_fn=fn)
        assert edge.filter_fn is not None
        assert edge.filter_fn(MagicMock(priority=3)) is True
        assert edge.filter_fn(MagicMock(priority=1)) is False

    def test_with_priority_boost(self):
        edge = GraphEdge(source="risk", target="all", priority_boost=2)
        assert edge.priority_boost == 2

    def test_default_factory_isolation(self):
        """Each edge gets its own msg_types set, not shared."""
        e1 = GraphEdge(source="a", target="b")
        e2 = GraphEdge(source="c", target="d")
        e1.msg_types.add(MessageType.SIGNAL)
        assert MessageType.SIGNAL not in e2.msg_types


# ═══════════════════════════════════════════════════════════════════════════
# AgentGraph.__init__ Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAgentGraphInit:
    """AgentGraph initial state."""

    def test_empty_agents(self):
        g = AgentGraph()
        assert g.agents == {}

    def test_empty_edges(self):
        g = AgentGraph()
        assert g.edges == []

    def test_empty_node_types(self):
        g = AgentGraph()
        assert g.node_types == {}

    def test_message_bus_maxlen(self):
        g = AgentGraph()
        assert g.message_bus.maxlen == 1000

    def test_message_bus_is_deque(self):
        g = AgentGraph()
        assert isinstance(g.message_bus, deque)

    def test_default_metrics(self):
        g = AgentGraph()
        assert g.metrics["messages_routed"] == 0
        assert g.metrics["alerts_triggered"] == 0
        assert g.metrics["consensus_reached"] == 0
        assert g.metrics["conflicts_detected"] == 0

    def test_device_default(self):
        g = AgentGraph()
        assert g.device == "cpu"

    def test_device_custom(self):
        g = AgentGraph(device="cuda:0")
        assert g.device == "cuda:0"

    def test_message_lock(self):
        g = AgentGraph()
        assert isinstance(g.message_lock, type(threading.Lock()))

    def test_execution_history_empty(self):
        g = AgentGraph()
        assert g.execution_history == []

    def test_max_history(self):
        g = AgentGraph()
        assert g.max_history == 1000


# ═══════════════════════════════════════════════════════════════════════════
# AgentGraph Registration Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAgentGraphRegistration:
    """register_agent and add_edge."""

    def test_register_single_agent(self):
        g = AgentGraph()
        agent = _make_mock_agent("analyst")
        g.register_agent(agent, NodeType.ANALYST)
        assert "analyst" in g.agents
        assert g.agents["analyst"] is agent

    def test_register_sets_node_type(self):
        g = AgentGraph()
        agent = _make_mock_agent("risk_monitor")
        g.register_agent(agent, NodeType.RISK)
        assert g.node_types["risk_monitor"] == NodeType.RISK

    def test_register_multiple_agents(self):
        g = AgentGraph()
        a1 = _make_mock_agent("a1")
        a2 = _make_mock_agent("a2")
        g.register_agent(a1, NodeType.ANALYST)
        g.register_agent(a2, NodeType.SENTIMENT)
        assert len(g.agents) == 2
        assert g.node_types["a1"] == NodeType.ANALYST
        assert g.node_types["a2"] == NodeType.SENTIMENT

    def test_register_overwrite_existing(self):
        g = AgentGraph()
        agent = _make_mock_agent("dup")
        g.register_agent(agent, NodeType.ANALYST)
        agent2 = _make_mock_agent("dup")
        g.register_agent(agent2, NodeType.RISK)
        assert g.agents["dup"] is agent2
        assert g.node_types["dup"] == NodeType.RISK

    def test_add_edge(self):
        g = AgentGraph()
        edge = GraphEdge(source="a", target="b")
        g.add_edge(edge)
        assert len(g.edges) == 1
        assert g.edges[0] is edge

    def test_add_multiple_edges(self):
        g = AgentGraph()
        g.add_edge(GraphEdge(source="x", target="y"))
        g.add_edge(GraphEdge(source="y", target="z"))
        assert len(g.edges) == 2


# ═══════════════════════════════════════════════════════════════════════════
# AgentGraph Topology Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAgentGraphTopology:
    """setup_default_topology -- creates 11 edges."""

    def test_default_topology_edge_count(self):
        g = AgentGraph()
        g.setup_default_topology()
        # 3 specialist->controller + 1 risk->controller + 3 risk->specialist
        # + 1 controller->execution + 3 controller->specialist = 11 edges
        assert len(g.edges) == 11

    def test_specialist_to_controller_msg_types(self):
        g = AgentGraph()
        g.setup_default_topology()
        for src in ["analyst", "sentiment", "execution"]:
            edges = [e for e in g.edges
                     if e.source == src and e.target == "controller"]
            assert len(edges) == 1
            assert MessageType.SIGNAL in edges[0].msg_types
            assert MessageType.RESPONSE in edges[0].msg_types

    def test_risk_to_controller(self):
        g = AgentGraph()
        g.setup_default_topology()
        edges = [e for e in g.edges
                 if e.source == "risk" and e.target == "controller"]
        assert len(edges) == 1
        assert MessageType.SIGNAL in edges[0].msg_types
        assert MessageType.ALERT in edges[0].msg_types

    def test_risk_broadcasts_alert(self):
        g = AgentGraph()
        g.setup_default_topology()
        for target in ["analyst", "sentiment", "execution"]:
            edges = [e for e in g.edges
                     if e.source == "risk" and e.target == target]
            assert len(edges) == 1
            assert edges[0].msg_types == {MessageType.ALERT}

    def test_risk_alert_priority_boost(self):
        g = AgentGraph()
        g.setup_default_topology()
        for target in ["analyst", "sentiment", "execution"]:
            edges = [e for e in g.edges
                     if e.source == "risk" and e.target == target]
            assert edges[0].priority_boost == 2

    def test_controller_to_execution(self):
        g = AgentGraph()
        g.setup_default_topology()
        edges = [e for e in g.edges
                 if e.source == "controller" and e.target == "execution"]
        assert len(edges) == 1
        assert MessageType.REQUEST in edges[0].msg_types
        assert MessageType.SIGNAL in edges[0].msg_types

    def test_controller_to_specialists(self):
        g = AgentGraph()
        g.setup_default_topology()
        for target in ["analyst", "sentiment", "risk"]:
            edges = [e for e in g.edges
                     if e.source == "controller" and e.target == target]
            assert len(edges) == 1
            assert edges[0].msg_types == {MessageType.REQUEST}


# ═══════════════════════════════════════════════════════════════════════════
# AgentGraph Message Bus Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAgentGraphMessageBus:
    """broadcast_to_bus and route_messages."""

    def test_broadcast_single_message(self):
        g = AgentGraph()
        msg = AgentMessage(
            sender="analyst", receiver=None,
            msg_type=MessageType.SIGNAL, content={"score": 0.8}
        )
        g.broadcast_to_bus(msg)
        assert len(g.message_bus) == 1

    def test_broadcast_multiple_messages(self):
        g = AgentGraph()
        for i in range(5):
            msg = AgentMessage(
                sender="a", receiver=None,
                msg_type=MessageType.SIGNAL, content={"i": i}
            )
            g.broadcast_to_bus(msg)
        assert len(g.message_bus) == 5

    def test_route_messages_empty_bus(self):
        g = AgentGraph()
        result = g.route_messages()
        assert result == 0

    def test_route_by_source(self):
        """Messages are routed when edge.source == msg.sender."""
        g = AgentGraph()
        agent = _make_mock_agent("analyst")
        controller = _make_mock_agent("controller")
        g.register_agent(agent, NodeType.ANALYST)
        g.register_agent(controller, NodeType.CONTROLLER)
        g.add_edge(GraphEdge(
            source="analyst", target="controller",
            msg_types={MessageType.SIGNAL}
        ))
        g.add_edge(GraphEdge(
            source="sentiment", target="controller",
            msg_types={MessageType.SIGNAL}
        ))

        msg = AgentMessage(
            sender="analyst", receiver="controller",
            msg_type=MessageType.SIGNAL, content={"score": 0.8}
        )
        g.broadcast_to_bus(msg)
        result = g.route_messages()

        assert result >= 1
        agent.receive_message.assert_not_called()  # analyst is sender
        assert g.metrics["messages_routed"] >= 1

    def test_route_by_msg_type(self):
        """Only messages matching edge.msg_types are routed."""
        g = AgentGraph()
        controller = _make_mock_agent("controller")
        g.register_agent(controller, NodeType.CONTROLLER)
        g.add_edge(GraphEdge(
            source="sender", target="controller",
            msg_types={MessageType.SIGNAL}  # NOT ALERT
        ))

        alert_msg = AgentMessage(
            sender="sender", receiver="controller",
            msg_type=MessageType.ALERT, content={"level": "high"}
        )
        g.broadcast_to_bus(alert_msg)
        g.route_messages()

        # ALERT does not match SIGNAL-only edge, so no routing
        controller.receive_message.assert_not_called()

    def test_route_with_filter_fn(self):
        """filter_fn blocks messages that do not match."""
        g = AgentGraph()
        controller = _make_mock_agent("controller")
        g.register_agent(controller, NodeType.CONTROLLER)
        g.add_edge(GraphEdge(
            source="analyst", target="controller",
            msg_types={MessageType.SIGNAL},
            filter_fn=lambda m: m.content.get("score", 0) > 0.5
        ))

        low_msg = AgentMessage(
            sender="analyst", receiver="controller",
            msg_type=MessageType.SIGNAL, content={"score": 0.3}
        )
        g.broadcast_to_bus(low_msg)
        g.route_messages()
        assert controller.receive_message.call_count == 0

        high_msg = AgentMessage(
            sender="analyst", receiver="controller",
            msg_type=MessageType.SIGNAL, content={"score": 0.9}
        )
        g.broadcast_to_bus(high_msg)
        g.route_messages()
        assert controller.receive_message.call_count >= 1

    def test_route_with_priority_boost(self):
        """Message priority is boosted by edge.priority_boost."""
        g = AgentGraph()
        controller = _make_mock_agent("controller")
        g.register_agent(controller, NodeType.CONTROLLER)
        g.add_edge(GraphEdge(
            source="risk", target="controller",
            msg_types={MessageType.ALERT},
            priority_boost=2
        ))

        msg = AgentMessage(
            sender="risk", receiver="controller",
            msg_type=MessageType.ALERT, content={"alert": "drawdown"},
            priority=3
        )
        g.broadcast_to_bus(msg)
        g.route_messages()

        # The routed message should have priority 3+2=5
        call_msg = controller.receive_message.call_args[0][0]
        assert call_msg.priority == 5

    def test_route_broadcast_none_receiver(self):
        """receiver=None matches any edge target."""
        g = AgentGraph()
        analyst = _make_mock_agent("analyst")
        sentiment = _make_mock_agent("sentiment")
        g.register_agent(analyst, NodeType.ANALYST)
        g.register_agent(sentiment, NodeType.SENTIMENT)
        g.add_edge(GraphEdge(
            source="risk", target="analyst",
            msg_types={MessageType.ALERT}
        ))
        g.add_edge(GraphEdge(
            source="risk", target="sentiment",
            msg_types={MessageType.ALERT}
        ))

        broadcast_msg = AgentMessage(
            sender="risk", receiver=None,  # broadcast
            msg_type=MessageType.ALERT, content={"alert": "high vol"}
        )
        g.broadcast_to_bus(broadcast_msg)
        g.route_messages()

        assert analyst.receive_message.call_count >= 1
        assert sentiment.receive_message.call_count >= 1

    def test_route_max_messages(self):
        """Respects max_messages limit."""
        g = AgentGraph()
        controller = _make_mock_agent("controller")
        g.register_agent(controller, NodeType.CONTROLLER)
        g.add_edge(GraphEdge(
            source="a", target="controller",
            msg_types={MessageType.SIGNAL}
        ))

        for i in range(10):
            msg = AgentMessage(
                sender="a", receiver="controller",
                msg_type=MessageType.SIGNAL, content={"i": i}
            )
            g.broadcast_to_bus(msg)

        result = g.route_messages(max_messages=3)
        assert result <= 3
        assert len(g.message_bus) == 7  # 7 remain

    def test_route_alert_metric(self):
        """Alerts increment metrics['alerts_triggered']."""
        g = AgentGraph()
        controller = _make_mock_agent("controller")
        g.register_agent(controller, NodeType.CONTROLLER)
        g.add_edge(GraphEdge(
            source="risk", target="controller",
            msg_types={MessageType.ALERT}
        ))

        msg = AgentMessage(
            sender="risk", receiver="controller",
            msg_type=MessageType.ALERT, content={"level": "critical"}
        )
        g.broadcast_to_bus(msg)
        g.route_messages()

        assert g.metrics["alerts_triggered"] >= 1
        assert g.metrics["messages_routed"] >= 1

    def test_route_no_match_found(self):
        """Message with no matching edge is still counted in messages_routed."""
        g = AgentGraph()
        msg = AgentMessage(
            sender="orphan", receiver="nowhere",
            msg_type=MessageType.SIGNAL, content={}
        )
        g.broadcast_to_bus(msg)
        result = g.route_messages()
        assert result == 0
        # The message is popped and counted even if no edge matched
        assert g.metrics["messages_routed"] == 1

    def test_route_thread_safety(self):
        """route_messages acquires the message lock."""
        g = AgentGraph()
        msg = AgentMessage(
            sender="a", receiver="b",
            msg_type=MessageType.SIGNAL, content={}
        )
        g.broadcast_to_bus(msg)
        # Just verify it does not raise
        g.route_messages()
        assert len(g.message_bus) == 0


# ═══════════════════════════════════════════════════════════════════════════
# AgentGraph create_default_agents Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAgentGraphCreateDefaultAgents:
    """create_default_agents -- creates and registers all 5 agents."""

    @patch("src.agents.agent_graph.AnalystAgent")
    @patch("src.agents.agent_graph.SentimentAgent")
    @patch("src.agents.agent_graph.RiskAgent")
    @patch("src.agents.agent_graph.ExecutionAgent")
    @patch("src.agents.agent_graph.ControllerAgent")
    def test_creates_five_agents(
        self, MockCA, MockEA, MockRA, MockSA, MockAA
    ):
        """All 5 agent constructors are called."""
        for mc in [MockAA, MockSA, MockRA, MockEA, MockCA]:
            mc.side_effect = _agent_side_effect

        g = AgentGraph()
        result = g.create_default_agents(hidden_dim=128)

        MockAA.assert_called_once_with(
            agent_id="analyst", hidden_dim=128, device="cpu"
        )
        MockSA.assert_called_once_with(
            agent_id="sentiment", hidden_dim=128, device="cpu"
        )
        MockRA.assert_called_once_with(
            agent_id="risk", hidden_dim=128, device="cpu"
        )
        MockEA.assert_called_once_with(
            agent_id="execution", hidden_dim=128, device="cpu"
        )
        MockCA.assert_called_once_with(
            agent_id="controller", n_assets=4, hidden_dim=256, device="cpu"
        )
        assert len(result) == 5

    @patch("src.agents.agent_graph.AnalystAgent")
    @patch("src.agents.agent_graph.SentimentAgent")
    @patch("src.agents.agent_graph.RiskAgent")
    @patch("src.agents.agent_graph.ExecutionAgent")
    @patch("src.agents.agent_graph.ControllerAgent")
    def test_registers_all_agents(
        self, MockCA, MockEA, MockRA, MockSA, MockAA
    ):
        """All created agents are registered."""
        for mc in [MockAA, MockSA, MockRA, MockEA, MockCA]:
            mc.side_effect = _agent_side_effect

        g = AgentGraph()
        g.create_default_agents()

        assert "analyst" in g.agents
        assert "sentiment" in g.agents
        assert "risk" in g.agents
        assert "execution" in g.agents
        assert "controller" in g.agents

    @patch("src.agents.agent_graph.AnalystAgent")
    @patch("src.agents.agent_graph.SentimentAgent")
    @patch("src.agents.agent_graph.RiskAgent")
    @patch("src.agents.agent_graph.ExecutionAgent")
    @patch("src.agents.agent_graph.ControllerAgent")
    def test_sets_up_topology(
        self, MockCA, MockEA, MockRA, MockSA, MockAA
    ):
        """Topology edges are created."""
        for mc in [MockAA, MockSA, MockRA, MockEA, MockCA]:
            mc.side_effect = _agent_side_effect

        g = AgentGraph()
        g.create_default_agents()

        assert len(g.edges) == 11

    @patch("src.agents.agent_graph.AnalystAgent")
    @patch("src.agents.agent_graph.SentimentAgent")
    @patch("src.agents.agent_graph.RiskAgent")
    @patch("src.agents.agent_graph.ExecutionAgent")
    @patch("src.agents.agent_graph.ControllerAgent")
    def test_returns_agents_dict(
        self, MockCA, MockEA, MockRA, MockSA, MockAA
    ):
        """Returns dict with all 5 agent IDs."""
        for mc in [MockAA, MockSA, MockRA, MockEA, MockCA]:
            mc.side_effect = _agent_side_effect

        g = AgentGraph()
        result = g.create_default_agents()

        assert isinstance(result, dict)
        for key in ["analyst", "sentiment", "risk", "execution", "controller"]:
            assert key in result

    @patch("src.agents.agent_graph.AnalystAgent")
    @patch("src.agents.agent_graph.SentimentAgent")
    @patch("src.agents.agent_graph.RiskAgent")
    @patch("src.agents.agent_graph.ExecutionAgent")
    @patch("src.agents.agent_graph.ControllerAgent")
    def test_controller_node_type(
        self, MockCA, MockEA, MockRA, MockSA, MockAA
    ):
        """Controller agent gets NodeType.CONTROLLER."""
        for mc in [MockAA, MockSA, MockRA, MockEA, MockCA]:
            mc.side_effect = _agent_side_effect

        g = AgentGraph()
        g.create_default_agents()

        assert g.node_types["controller"] == NodeType.CONTROLLER
        assert g.node_types["analyst"] == NodeType.ANALYST
        assert g.node_types["sentiment"] == NodeType.SENTIMENT
        assert g.node_types["risk"] == NodeType.RISK
        assert g.node_types["execution"] == NodeType.EXECUTION

    @patch("src.agents.agent_graph.AnalystAgent")
    @patch("src.agents.agent_graph.SentimentAgent")
    @patch("src.agents.agent_graph.RiskAgent")
    @patch("src.agents.agent_graph.ExecutionAgent")
    @patch("src.agents.agent_graph.ControllerAgent")
    def test_hidden_dim_passthrough(
        self, MockCA, MockEA, MockRA, MockSA, MockAA
    ):
        """Custom hidden_dim is passed to specialist agents."""
        for mc in [MockAA, MockSA, MockRA, MockEA, MockCA]:
            mc.side_effect = _agent_side_effect

        g = AgentGraph()
        g.create_default_agents(hidden_dim=64)

        MockAA.assert_called_once_with(
            agent_id="analyst", hidden_dim=64, device="cpu"
        )
        # Controller uses hidden_dim * 2
        MockCA.assert_called_once_with(
            agent_id="controller", n_assets=4, hidden_dim=128, device="cpu"
        )


# ═══════════════════════════════════════════════════════════════════════════
# AgentGraph execute_step Tests
# ═══════════════════════════════════════════════════════════════════════════

class MockObs:
    """Minimal observation stub with .features attribute."""
    def __init__(self, features=None):
        self.features = features if features is not None else np.array([1.0])


class TestAgentGraphExecuteStep:
    """execute_step -- full execution cycle."""

    def test_execute_step_basic(self):
        """Basic execution with all specialist agents and controller."""
        g = AgentGraph()
        agents = {
            "analyst": _make_mock_agent("analyst"),
            "sentiment": _make_mock_agent("sentiment"),
            "risk": _make_mock_agent("risk"),
            "execution": _make_mock_agent("execution"),
            "controller": _make_mock_agent("controller"),
        }

        # Set up outbox: each specialist produces one signal message
        for aid in ["analyst", "sentiment", "risk", "execution"]:
            msg = AgentMessage(
                sender=aid, receiver="controller",
                msg_type=MessageType.SIGNAL,
                content={"score": 0.6, "direction": 0.3}
            )
            agents[aid].clear_outbox.return_value = [msg]

        # Controller outbox
        controller_msg = AgentMessage(
            sender="controller", receiver="execution",
            msg_type=MessageType.REQUEST,
            content={"action": "rebalance", "size": 0.05}
        )
        agents["controller"].clear_outbox.return_value = [controller_msg]

        for aid, agent in agents.items():
            g.register_agent(
                agent,
                NodeType(aid) if aid != "controller"
                else NodeType.CONTROLLER
            )

        g.setup_default_topology()

        obs = MockObs(np.array([1.0, 2.0, 3.0]))
        results = g.execute_step(obs)

        # All 5 agents produce output
        assert "analyst" in results
        assert "sentiment" in results
        assert "risk" in results
        assert "execution" in results
        assert "controller" in results

        # History recorded
        assert len(g.execution_history) >= 1

    def test_execute_step_without_controller(self):
        """Execution without a controller agent."""
        g = AgentGraph()
        analyst = _make_mock_agent("analyst")
        g.register_agent(analyst, NodeType.ANALYST)

        results = g.execute_step(MockObs())
        assert "analyst" in results
        assert "controller" not in results

    def test_execute_step_updates_history(self):
        """execution_history gets appended."""
        g = AgentGraph()
        agent = _make_mock_agent("analyst")
        g.register_agent(agent, NodeType.ANALYST)

        g.execute_step(MockObs())
        assert len(g.execution_history) == 1
        entry = g.execution_history[0]
        assert "timestamp" in entry
        assert "observation_shape" in entry
        assert "agent_outputs" in entry

    def test_execute_step_history_capped(self):
        """execution_history is capped at max_history."""
        g = AgentGraph()
        g.max_history = 3
        agent = _make_mock_agent("analyst")
        g.register_agent(agent, NodeType.ANALYST)

        for _ in range(5):
            g.execute_step(MockObs())

        assert len(g.execution_history) == 3

    def test_execute_step_observation_shape_zero(self):
        """observation_shape is 0 when obs has no features attr."""
        g = AgentGraph()
        agent = _make_mock_agent("analyst")
        g.register_agent(agent, NodeType.ANALYST)

        # observation without features
        obs = object()
        g.execute_step(obs)

        assert g.execution_history[0]["observation_shape"] == 0

    def test_execute_step_routes_messages(self):
        """Messages from specialists are routed to controller."""
        g = AgentGraph()
        analyst = _make_mock_agent("analyst")
        controller = _make_mock_agent("controller")
        g.register_agent(analyst, NodeType.ANALYST)
        g.register_agent(controller, NodeType.CONTROLLER)

        msg = AgentMessage(
            sender="analyst", receiver="controller",
            msg_type=MessageType.SIGNAL,
            content={"score": 0.8}
        )
        analyst.clear_outbox.return_value = [msg]
        g.setup_default_topology()

        g.execute_step(MockObs())
        # controller received the routed signal
        assert controller.receive_message.call_count >= 1

    def test_execute_step_agent_outputs_in_results(self):
        """Each agent's act() output is stored in the results dict."""
        g = AgentGraph()
        analyst = _make_mock_agent("analyst")
        g.register_agent(analyst, NodeType.ANALYST)

        results = g.execute_step(MockObs())
        output = results["analyst"]
        assert output.score == 0.5
        assert output.direction == 0.0
        assert output.confidence == 0.5


# ═══════════════════════════════════════════════════════════════════════════
# AgentGraph Consensus Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAgentGraphConsensus:
    """get_consensus_status."""

    def test_consensus_no_controller(self):
        """Returns empty dict when no controller agent is registered."""
        g = AgentGraph()
        result = g.get_consensus_status()
        assert result == {}

    def test_consensus_no_controller_not_controlleragent(self):
        """Returns empty dict when controller is not a ControllerAgent."""
        g = AgentGraph()
        non_controller = _make_mock_agent("controller")
        g.register_agent(non_controller, NodeType.CONTROLLER)
        result = g.get_consensus_status()
        assert result == {}

    def test_consensus_with_controller(self):
        """Returns consensus data when controller has consensus attr."""
        g = AgentGraph()
        # Use spec=ControllerAgent so isinstance check passes
        controller = MagicMock(spec=ControllerAgent)
        controller.agent_id = "controller"
        controller.receive_message = MagicMock()

        mock_consensus = MagicMock()
        mock_consensus.get_consensus_score.return_value = (0.85, 0.72)
        mock_consensus.agent_signals = {
            "analyst": {"score": 0.7, "direction": 0.5, "confidence": 0.8},
            "sentiment": {"score": 0.6, "direction": -0.2, "confidence": 0.6},
        }
        controller.consensus = mock_consensus

        g.register_agent(controller, NodeType.CONTROLLER)
        result = g.get_consensus_status()

        assert result["consensus_level"] == 0.85
        assert result["consensus_signal"] == 0.72
        assert result["agents_contributing"] == 2
        assert "analyst" in result["agent_signals"]
        assert "sentiment" in result["agent_signals"]

    def test_consensus_empty_signals(self):
        """Returns zeros when no agent signals exist."""
        g = AgentGraph()
        controller = MagicMock(spec=ControllerAgent)
        controller.agent_id = "controller"
        controller.receive_message = MagicMock()

        mock_consensus = MagicMock()
        mock_consensus.get_consensus_score.return_value = (0.0, 0.0)
        mock_consensus.agent_signals = {}
        controller.consensus = mock_consensus

        g.register_agent(controller, NodeType.CONTROLLER)
        result = g.get_consensus_status()

        assert result["consensus_level"] == 0.0
        assert result["consensus_signal"] == 0.0
        assert result["agents_contributing"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# AgentGraph Save/Load Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAgentGraphSaveLoad:
    """save and load methods."""

    def test_save_creates_directory(self):
        """save creates the target directory."""
        g = AgentGraph()
        agent = _make_mock_agent("analyst")
        g.register_agent(agent, NodeType.ANALYST)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "graph_state"
            assert not save_path.exists()
            g.save(save_path)
            assert save_path.exists()
            assert save_path.is_dir()

    def test_save_writes_metrics(self):
        """save writes graph_metrics.json."""
        g = AgentGraph()
        agent = _make_mock_agent("analyst")
        g.register_agent(agent, NodeType.ANALYST)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "graph_state"
            g.save(save_path)
            metrics_file = save_path / "graph_metrics.json"
            assert metrics_file.exists()
            with open(metrics_file) as f:
                data = json.load(f)
            assert "metrics" in data
            assert "execution_history_sample" in data

    def test_save_calls_agent_save(self):
        """Each agent's save() is called with its .pt path."""
        g = AgentGraph()
        a1 = _make_mock_agent("analyst")
        a2 = _make_mock_agent("sentiment")
        g.register_agent(a1, NodeType.ANALYST)
        g.register_agent(a2, NodeType.SENTIMENT)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "graph_state"
            g.save(save_path)
            a1.save.assert_called_once_with(save_path / "analyst.pt")
            a2.save.assert_called_once_with(save_path / "sentiment.pt")

    def test_load_calls_agent_load(self):
        """Each agent's load() is called for existing .pt files."""
        g = AgentGraph()
        a1 = _make_mock_agent("analyst")
        a2 = _make_mock_agent("sentiment")
        g.register_agent(a1, NodeType.ANALYST)
        g.register_agent(a2, NodeType.SENTIMENT)

        with tempfile.TemporaryDirectory() as tmpdir:
            load_path = Path(tmpdir) / "graph_state"
            load_path.mkdir(parents=True)
            # Create .pt files
            (load_path / "analyst.pt").write_text("stub")
            (load_path / "sentiment.pt").write_text("stub")

            g.load(load_path)
            a1.load.assert_called_once_with(load_path / "analyst.pt")
            a2.load.assert_called_once_with(load_path / "sentiment.pt")

    def test_load_skips_missing(self):
        """load silently skips missing .pt files."""
        g = AgentGraph()
        a1 = _make_mock_agent("analyst")
        g.register_agent(a1, NodeType.ANALYST)

        with tempfile.TemporaryDirectory() as tmpdir:
            load_path = Path(tmpdir) / "graph_state"
            load_path.mkdir(parents=True)
            # No .pt files created
            g.load(load_path)
            # Should not raise
            a1.load.assert_not_called()

    def test_save_with_execution_history(self):
        """save includes execution history sample in metrics."""
        g = AgentGraph()
        agent = _make_mock_agent("analyst")
        g.register_agent(agent, NodeType.ANALYST)

        # Run a step to populate history
        g.execute_step(MockObs())

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "graph_state"
            g.save(save_path)
            metrics_file = save_path / "graph_metrics.json"
            with open(metrics_file) as f:
                data = json.load(f)
            sample = data["execution_history_sample"]
            assert len(sample) >= 1
            assert sample[0]["observation_shape"] == 1
            assert "analyst" in sample[0]["agent_outputs"]


# ═══════════════════════════════════════════════════════════════════════════
# AgentGraph Topology Viz Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAgentGraphTopologyViz:
    """get_topology_viz -- ASCII visualization."""

    def test_viz_contains_header(self):
        g = AgentGraph()
        viz = g.get_topology_viz()
        assert "Agent Graph Topology" in viz

    def test_viz_contains_edges_section(self):
        g = AgentGraph()
        g.add_edge(GraphEdge(
            source="a", target="b",
            msg_types={MessageType.SIGNAL}
        ))
        viz = g.get_topology_viz()
        assert "Edges:" in viz
        assert "a -> b: signal" in viz

    def test_viz_no_edges(self):
        g = AgentGraph()
        viz = g.get_topology_viz()
        assert "Edges:" in viz
        lines = viz.split("\n")
        edges_idx = next(i for i, item in enumerate(lines) if "Edges:" in item)
        edge_lines = [item for item in lines[edges_idx:] if "->" in item]
        assert len(edge_lines) == 0

    def test_viz_with_priority_boost(self):
        g = AgentGraph()
        g.add_edge(GraphEdge(
            source="risk", target="analyst",
            msg_types={MessageType.ALERT}, priority_boost=2
        ))
        viz = g.get_topology_viz()
        assert "(+2)" in viz

    def test_viz_multiple_msg_types(self):
        g = AgentGraph()
        g.add_edge(GraphEdge(
            source="analyst", target="controller",
            msg_types={MessageType.SIGNAL, MessageType.RESPONSE}
        ))
        viz = g.get_topology_viz()
        assert "signal" in viz
        assert "response" in viz

    def test_viz_static_ascii(self):
        """Static ASCII art is present."""
        g = AgentGraph()
        viz = g.get_topology_viz()
        assert "Analyst" in viz
        assert "Controller" in viz
        assert "Execution" in viz
        assert "Risk" in viz

    def test_viz_with_default_topology(self):
        """Full topology viz includes all 11 edges."""
        g = AgentGraph()
        g.setup_default_topology()
        viz = g.get_topology_viz()
        # Count edge lines after "Edges:" header
        lines = viz.split("\n")
        edges_idx = next(i for i, item in enumerate(lines) if "Edges:" in item)
        edge_lines = [item for item in lines[edges_idx:] if "->" in item]
        assert len(edge_lines) == 11
