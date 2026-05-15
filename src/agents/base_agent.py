#!/usr/bin/env python3
"""
Portfolio-Lab v2.51: Base Agent Module

Abstract base class for all MARL agents with common interfaces for
observation spaces, action spaces, and inter-agent communication.

Architecture:
- BaseAgent: Abstract foundation for all specialized agents
- AgentState: Dataclass for agent observations and actions
- MessageBus: Inter-agent communication protocol
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from datetime import datetime
from enum import Enum
import numpy as np
from pathlib import Path
import os

# Conditional ML import — disabled by default to prevent OOM in test suites.
# Set PORTFOLIO_LAB_ENABLE_ML=1 to load real torch.
_ML_ENABLED = os.environ.get("PORTFOLIO_LAB_ENABLE_ML", "0") == "1"

if _ML_ENABLED:
    import torch
    import torch.nn as nn
else:
    # Stub torch.nn.Module for type compatibility without 63MB memory cost
    import types as _types

    class _StubNoGrad:
        def __enter__(self): pass
        def __exit__(self, *a): pass

    class _StubModule:
        """Stand-in for nn.Module — supports ABC inheritance without real torch."""
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs): return None
        def forward(self, *args, **kwargs): return None
        def parameters(self): return []
        def named_parameters(self): return iter([])
        def train(self, mode=True): return self
        def eval(self): return self
        def to(self, *args, **kwargs): return self
        def state_dict(self): return {}
        def load_state_dict(self, *args, **kwargs): pass
        def zero_grad(self): pass
        def __repr__(self): return "StubModule()"

    _stub_torch = _types.ModuleType("torch")
    _stub_torch.Tensor = np.ndarray
    _stub_torch.tensor = lambda x, **kw: np.array(x)
    _stub_torch.TensorType = np.ndarray
    _stub_torch.device = lambda x: "cpu"
    _stub_torch.no_grad = _StubNoGrad
    _stub_torch.zeros = lambda *a, **kw: np.zeros(a or 1, **kw)
    _stub_torch.ones = lambda *a, **kw: np.ones(a or 1, **kw)
    _stub_torch.randn = lambda *a, **kw: np.random.randn(*(a or (1,)))
    _stub_torch.save = lambda obj, path: None
    _stub_torch.load = lambda path, **kw: {}
    _stub_torch.float32 = np.float32
    _stub_torch.float64 = np.float64
    _stub_torch.long = np.int64

    _stub_nn = _types.ModuleType("torch.nn")
    _stub_nn.Module = _StubModule
    _stub_nn.ModuleDict = lambda *a, **kw: {}
    _stub_nn.Parameter = lambda *a, **kw: None
    _stub_nn.Linear = lambda *a, **kw: _StubModule()
    _stub_nn.Sequential = lambda *a: _StubModule()
    _stub_nn.ReLU = lambda: _StubModule()
    _stub_nn.Tanh = lambda: _StubModule()
    _stub_nn.Sigmoid = lambda: _StubModule()
    _stub_nn.MSELoss = lambda: lambda x, y: 0.0
    _stub_nn.L1Loss = lambda: lambda x, y: 0.0

    torch = _stub_torch
    nn = _stub_nn

    # Register stubs in sys.modules so other agent modules find them
    # (overwrites any existing entries to prevent silent no-op from setdefault)
    import sys as _sys
    _sys.modules["torch"] = _stub_torch
    _sys.modules["torch.nn"] = _stub_nn


class AgentType(Enum):
    """Agent specialization types."""
    ANALYST = "analyst"           # Fundamental/value analysis
    SENTIMENT = "sentiment"       # News/social sentiment
    RISK = "risk"                 # Risk/drawdown monitoring
    EXECUTION = "execution"       # Order execution timing
    CONTROLLER = "controller"     # Orchestration/coordination


class MessageType(Enum):
    """Message types for inter-agent communication."""
    SIGNAL = "signal"             # Signal/score output
    ALERT = "alert"               # Risk/urgency alert
    REQUEST = "request"           # Information request
    RESPONSE = "response"         # Information response
    CONSENSUS = "consensus"       # Group decision


@dataclass
class AgentMessage:
    """Message passed between agents."""
    sender: str                   # Agent ID
    receiver: Optional[str]      # Target agent (None = broadcast)
    msg_type: MessageType
    content: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    priority: int = 1             # 1-5, higher = more urgent


@dataclass
class AgentObservation:
    """Observation space for an agent."""
    # Market data
    prices: np.ndarray            # Recent price history
    returns: np.ndarray         # Return series
    volatility: float           # Current volatility
    
    # Portfolio state
    current_weights: Dict[str, float]
    portfolio_value: float
    cash_available: float
    
    # Agent-specific features (populated by subclasses)
    features: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Context
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    regime: str = "neutral"


@dataclass
class AgentAction:
    """Action space output from an agent."""
    agent_id: str
    action_type: str
    
    # Primary outputs (agent-specific interpretation)
    score: float                # [0, 1] primary score
    direction: float            # [-1, 1] conviction/direction
    confidence: float           # [0, 1] confidence level
    
    # Secondary outputs
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class BaseAgent(ABC, nn.Module):
    """Abstract base class for all MARL agents."""
    
    def __init__(
        self,
        agent_id: str,
        agent_type: AgentType,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        device: str = "cpu"
    ):
        super().__init__()
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)
        
        # Message inbox
        self.inbox: List[AgentMessage] = []
        self.outbox: List[AgentMessage] = []
        
        # State tracking
        self.last_observation: Optional[AgentObservation] = None
        self.last_action: Optional[AgentAction] = None
        self.action_history: List[AgentAction] = []
        
        # Performance metrics
        self.accuracy_history: List[float] = []
        self.reward_history: List[float] = []
        
    @abstractmethod
    def build_network(self) -> nn.ModuleDict:
        """Build the neural network architecture."""
        pass
    
    @abstractmethod
    def extract_features(self, obs: AgentObservation) -> torch.Tensor:
        """Extract agent-specific features from observation."""
        pass
    
    @abstractmethod
    def act(self, obs: AgentObservation, deterministic: bool = False) -> AgentAction:
        """Generate action from observation."""
        pass
    
    @abstractmethod
    def compute_value(self, obs: AgentObservation) -> float:
        """Compute state value estimate (for critic)."""
        pass
    
    def receive_message(self, msg: AgentMessage):
        """Receive message from another agent."""
        if msg.receiver is None or msg.receiver == self.agent_id:
            self.inbox.append(msg)
    
    def send_message(self, receiver: Optional[str], msg_type: MessageType, content: Dict[str, Any], priority: int = 1):
        """Queue message to send to another agent."""
        msg = AgentMessage(
            sender=self.agent_id,
            receiver=receiver,
            msg_type=msg_type,
            content=content,
            priority=priority
        )
        self.outbox.append(msg)
    
    def process_inbox(self) -> List[AgentMessage]:
        """Process and clear inbox."""
        messages = self.inbox.copy()
        self.inbox.clear()
        return messages
    
    def clear_outbox(self) -> List[AgentMessage]:
        """Get and clear outbox messages."""
        messages = self.outbox.copy()
        self.outbox.clear()
        return messages
    
    def update_accuracy(self, predicted: float, actual: float):
        """Update accuracy tracking."""
        # Directional accuracy
        accuracy = 1.0 if (predicted * actual) > 0 else 0.0
        self.accuracy_history.append(accuracy)
        
        # Keep last 100
        if len(self.accuracy_history) > 100:
            self.accuracy_history.pop(0)
    
    def get_accuracy(self) -> float:
        """Get recent accuracy."""
        if not self.accuracy_history:
            return 0.5
        return np.mean(self.accuracy_history[-20:])
    
    def save(self, path: Path):
        """Save agent state."""
        torch.save({
            'agent_id': self.agent_id,
            'agent_type': self.agent_type.value,
            'state_dict': self.state_dict(),
            'accuracy_history': self.accuracy_history,
            'action_history_len': len(self.action_history)
        }, path)
    
    def load(self, path: Path):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])
        self.accuracy_history = checkpoint.get('accuracy_history', [])
