#!/usr/bin/env python3
"""
Portfolio-Lab v2.51: Agent Graph

LangGraph-inspired agent communication layer for multi-agent reinforcement
learning. Implements directed graph topology for message passing between
specialist agents and the controller.

Graph Topology:
- Analyst -> Controller
- Sentiment -> Controller
- Risk -> Controller (broadcasts alerts)
- Execution -> Controller
- Controller -> Execution (execution instructions)

Communication Patterns:
- Signal: Agent output to controller
- Alert: Risk broadcasts to all
- Request/Response: Inter-agent queries
- Consensus: Controller aggregates signals
"""

from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
from enum import Enum
import threading
import json
from pathlib import Path

from .base_agent import BaseAgent, AgentMessage, MessageType
from .analyst_agent import AnalystAgent
from .sentiment_agent import SentimentAgent
from .risk_agent import RiskAgent
from .execution_agent import ExecutionAgent
from .controller_agent import ControllerAgent


class NodeType(Enum):
    """Agent node types in the graph."""
    ANALYST = "analyst"
    SENTIMENT = "sentiment"
    RISK = "risk"
    EXECUTION = "execution"
    CONTROLLER = "controller"


@dataclass
class GraphEdge:
    """Directed edge between agent nodes."""
    source: str
    target: str
    msg_types: Set[MessageType] = field(default_factory=set)
    filter_fn: Optional[Callable[[AgentMessage], bool]] = None
    priority_boost: int = 0  # Add to message priority


class AgentGraph:
    """
    LangGraph-style agent communication graph.
    
    Manages:
    - Agent node registration
    - Edge routing rules
    - Message broadcasting
    - Execution order/topology
    
    Default Topology:
    ```
    Analyst ───┐
    Sentiment ──┼──> Controller ──> Execution
    Risk ───────┤      ↑
    Execution ──┘      │
                       └── (broadcast alerts from Risk)
    ```
    """
    
    def __init__(self, device: str = "cpu"):
        self.agents: Dict[str, BaseAgent] = {}
        self.edges: List[GraphEdge] = []
        self.node_types: Dict[str, NodeType] = {}
        self.device = device
        
        # Message bus for async routing
        self.message_bus: deque = deque(maxlen=1000)
        self.message_lock = threading.Lock()
        
        # Execution history
        self.execution_history: List[Dict[str, Any]] = []
        self.max_history = 1000
        
        # Performance metrics
        self.metrics = {
            'messages_routed': 0,
            'alerts_triggered': 0,
            'consensus_reached': 0,
            'conflicts_detected': 0
        }
    
    def register_agent(self, agent: BaseAgent, node_type: NodeType):
        """Register an agent node."""
        self.agents[agent.agent_id] = agent
        self.node_types[agent.agent_id] = node_type
    
    def add_edge(self, edge: GraphEdge):
        """Add a directed edge between nodes."""
        self.edges.append(edge)
    
    def setup_default_topology(self):
        """Set up default MARL topology."""
        # Specialist agents -> Controller
        for agent_type in [NodeType.ANALYST, NodeType.SENTIMENT, NodeType.EXECUTION]:
            self.add_edge(GraphEdge(
                source=agent_type.value,
                target="controller",
                msg_types={MessageType.SIGNAL, MessageType.RESPONSE}
            ))
        
        # Risk -> Controller (alerts too)
        self.add_edge(GraphEdge(
            source="risk",
            target="controller",
            msg_types={MessageType.SIGNAL, MessageType.ALERT}
        ))
        
        # Risk broadcasts alerts to all
        for target in ["analyst", "sentiment", "execution"]:
            self.add_edge(GraphEdge(
                source="risk",
                target=target,
                msg_types={MessageType.ALERT},
                priority_boost=2  # Boost alert priority
            ))
        
        # Controller -> Execution (execution instructions)
        self.add_edge(GraphEdge(
            source="controller",
            target="execution",
            msg_types={MessageType.REQUEST, MessageType.SIGNAL}
        ))
        
        # Controller can query specialists
        for target in ["analyst", "sentiment", "risk"]:
            self.add_edge(GraphEdge(
                source="controller",
                target=target,
                msg_types={MessageType.REQUEST}
            ))
    
    def create_default_agents(self, hidden_dim: int = 128) -> Dict[str, BaseAgent]:
        """Create and register default agent set."""
        agents = {}
        
        # Create agents
        agents['analyst'] = AnalystAgent(
            agent_id="analyst",
            hidden_dim=hidden_dim,
            device=self.device
        )
        
        agents['sentiment'] = SentimentAgent(
            agent_id="sentiment",
            hidden_dim=hidden_dim,
            device=self.device
        )
        
        agents['risk'] = RiskAgent(
            agent_id="risk",
            hidden_dim=hidden_dim,
            device=self.device
        )
        
        agents['execution'] = ExecutionAgent(
            agent_id="execution",
            hidden_dim=hidden_dim,
            device=self.device
        )
        
        agents['controller'] = ControllerAgent(
            agent_id="controller",
            n_assets=4,
            hidden_dim=hidden_dim * 2,  # Larger network
            device=self.device
        )
        
        # Register all
        for agent_id, agent in agents.items():
            node_type = NodeType(agent_id) if agent_id != "controller" else NodeType.CONTROLLER
            self.register_agent(agent, node_type)
        
        self.setup_default_topology()
        
        return agents
    
    def route_messages(self, max_messages: int = 100) -> int:
        """
        Route pending messages along graph edges.
        Returns number of messages routed.
        """
        routed = 0
        
        with self.message_lock:
            while self.message_bus and routed < max_messages:
                msg = self.message_bus.popleft()
                
                # Find matching edges
                for edge in self.edges:
                    if edge.source != msg.sender:
                        continue
                    
                    # Check message type
                    if msg.msg_type not in edge.msg_types:
                        continue
                    
                    # Check filter
                    if edge.filter_fn and not edge.filter_fn(msg):
                        continue
                    
                    # Check target (broadcast if None)
                    if msg.receiver is None or edge.target == msg.receiver:
                        # Route message
                        target_agent = self.agents.get(edge.target)
                        if target_agent:
                            # Boost priority if needed
                            boosted_msg = AgentMessage(
                                sender=msg.sender,
                                receiver=edge.target,
                                msg_type=msg.msg_type,
                                content=msg.content,
                                timestamp=msg.timestamp,
                                priority=msg.priority + edge.priority_boost
                            )
                            target_agent.receive_message(boosted_msg)
                            routed += 1
                
                # Update metrics
                if msg.msg_type == MessageType.ALERT:
                    self.metrics['alerts_triggered'] += 1
                
                self.metrics['messages_routed'] += 1
        
        return routed
    
    def broadcast_to_bus(self, msg: AgentMessage):
        """Add message to routing bus."""
        with self.message_lock:
            self.message_bus.append(msg)
    
    def execute_step(self, observation: Any) -> Dict[str, Any]:
        """
        Execute one step of agent graph.
        
        Order:
        1. Specialist agents observe and act
        2. Messages broadcast to bus
        3. Messages routed to targets
        4. Controller processes and decides
        5. Execution receives instructions
        
        Returns final controller action.
        """
        results = {}
        
        # Step 1: Specialist agents observe and act (can be parallel)
        for agent_id, agent in self.agents.items():
            if agent_id == "controller":
                continue  # Controller goes last
            
            # Generate action
            action = agent.act(observation, deterministic=False)
            results[agent_id] = action
            
            # Broadcast messages to bus
            for msg in agent.clear_outbox():
                self.broadcast_to_bus(msg)
        
        # Step 2 & 3: Route messages
        self.route_messages()
        
        # Step 4: Controller processes
        controller = self.agents.get("controller")
        if controller:
            # Controller processes inbox messages
            inbox_messages = controller.process_inbox()
            if isinstance(controller, ControllerAgent):
                controller.process_messages(inbox_messages)
            
            # Controller generates final decision
            controller_action = controller.act(observation, deterministic=False)
            results['controller'] = controller_action
            
            # Route controller output
            for msg in controller.clear_outbox():
                self.broadcast_to_bus(msg)
            
            self.route_messages()
        
        # Step 5: Execution receives final instructions
        execution = self.agents.get("execution")
        if execution:
            execution.process_inbox()  # Clear any controller requests
        
        # Log execution
        self.execution_history.append({
            'timestamp': datetime.now().isoformat(),
            'observation_shape': len(observation.features) if hasattr(observation, 'features') else 0,
            'agent_outputs': {
                k: {
                    'score': v.score,
                    'direction': v.direction,
                    'confidence': v.confidence
                } for k, v in results.items()
            }
        })
        
        if len(self.execution_history) > self.max_history:
            self.execution_history.pop(0)
        
        return results
    
    def get_consensus_status(self) -> Dict[str, Any]:
        """Get current consensus state from controller."""
        controller = self.agents.get("controller")
        if not isinstance(controller, ControllerAgent):
            return {}
        
        consensus_level, consensus_signal = controller.consensus.get_consensus_score()
        
        return {
            'consensus_level': consensus_level,
            'consensus_signal': consensus_signal,
            'agents_contributing': len(controller.consensus.agent_signals),
            'agent_signals': {
                k: {
                    'score': v.get('score', 0),
                    'conviction': v.get('conviction', v.get('direction', 0)),
                    'confidence': v.get('confidence', 0)
                }
                for k, v in controller.consensus.agent_signals.items()
            }
        }
    
    def save(self, path: Path):
        """Save agent graph state."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save each agent
        for agent_id, agent in self.agents.items():
            agent_path = path / f"{agent_id}.pt"
            agent.save(agent_path)
        
        # Save metrics
        metrics_path = path / "graph_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                'metrics': self.metrics,
                'execution_history_sample': self.execution_history[-100:]
            }, f, indent=2)
    
    def load(self, path: Path):
        """Load agent graph state."""
        path = Path(path)
        
        for agent_id, agent in self.agents.items():
            agent_path = path / f"{agent_id}.pt"
            if agent_path.exists():
                agent.load(agent_path)
    
    def get_topology_viz(self) -> str:
        """Generate ASCII topology visualization."""
        lines = [
            "Agent Graph Topology",
            "=" * 40,
            "",
            "    Analyst ───┐",
            "  Sentiment ───┼──> Controller",
            "       Risk ───┤      │",
            "  Execution ───┘      │",
            "                      ↓",
            "                 Execution",
            "                      ↑",
            "         (Risk alerts broadcast)",
            "",
            "Edges:"
        ]
        
        for edge in self.edges:
            msg_types = ', '.join(t.value for t in edge.msg_types)
            boost = f" (+{edge.priority_boost})" if edge.priority_boost else ""
            lines.append(f"  {edge.source} -> {edge.target}: {msg_types}{boost}")
        
        return '\n'.join(lines)
