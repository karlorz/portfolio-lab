#!/usr/bin/env python3
"""
Portfolio-Lab v2.51: Controller Agent

Master orchestration agent that aggregates signals from all specialist
agents (Analyst, Sentiment, Risk, Execution) and produces final portfolio
allocation decisions. Implements centralized critic with value decomposition.

Observations:
- All agent outputs (scores, convictions, confidences)
- Portfolio state (weights, values, performance)
- Agent consensus/conflict indicators
- Historical agent accuracy

Actions:
- agent_weights [4]: relative weight per agent
- final_allocation: target portfolio weights
- confidence: overall decision confidence
- rebalancing_trigger: whether to trade
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from collections import defaultdict

from .base_agent import BaseAgent, AgentType, AgentObservation, AgentAction, AgentMessage, MessageType


class ControllerNetwork(nn.Module):
    """Neural network for controller agent."""
    
    def __init__(self, obs_dim: int, n_agents: int, n_assets: int, hidden_dim: int = 256):
        super().__init__()
        
        self.n_agents = n_agents
        self.n_assets = n_assets
        
        # Encoder for aggregated state
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Agent weighting (attention over agent outputs)
        self.agent_attention = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_agents),
            nn.Softmax(dim=-1)
        )
        
        # Allocation head (per-asset weight adjustment)
        self.allocation_head = nn.Sequential(
            nn.Linear(hidden_dim + n_agents, 128),
            nn.ReLU(),
            nn.Linear(128, n_assets)
            # Softmax applied externally to ensure sum to 1
        )
        
        # Rebalancing trigger
        self.rebalance_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Value for centralized critic
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, obs: torch.Tensor) -> tuple:
        features = self.encoder(obs)
        
        # Agent attention weights
        agent_weights = self.agent_attention(features)
        
        # Allocation (raw logits, softmax applied outside)
        allocation_input = torch.cat([features, agent_weights], dim=-1)
        allocation_logits = self.allocation_head(allocation_input)
        
        # Rebalance trigger
        rebalance_prob = self.rebalance_head(features)
        
        # Confidence
        confidence = self.confidence_head(features)
        
        # State value (centralized critic)
        state_value = self.value_head(features)
        
        return agent_weights, allocation_logits, rebalance_prob, confidence, state_value


class AgentConsensus:
    """Track agent consensus and conflict."""
    
    def __init__(self):
        self.agent_signals: Dict[str, Dict[str, Any]] = {}
        self.timestamp: Optional[str] = None
    
    def update(self, msg: AgentMessage):
        """Update with agent message."""
        if msg.msg_type == MessageType.SIGNAL:
            self.agent_signals[msg.sender] = msg.content
            self.timestamp = msg.timestamp
    
    def get_consensus_score(self) -> Tuple[float, float]:
        """
        Calculate consensus score.
        Returns: (consensus_level, weighted_signal)
        """
        if len(self.agent_signals) < 2:
            return 0.0, 0.0
        
        signals = []
        confidences = []
        
        for agent_id, content in self.agent_signals.items():
            if 'conviction' in content or 'score' in content:
                # Extract directional signal
                score = content.get('score', 0.5)
                conviction = content.get('conviction', content.get('direction', 0))
                conf = content.get('confidence', 0.5)
                
                directional = (score - 0.5) * 2 * conviction  # [-1, 1]
                signals.append(directional)
                confidences.append(conf)
        
        if not signals:
            return 0.0, 0.0
        
        # Weighted average
        weights = np.array(confidences) / sum(confidences)
        weighted_signal = np.dot(signals, weights)
        
        # Consensus = agreement level
        avg_signal = np.mean(signals)
        deviations = [abs(s - avg_signal) for s in signals]
        consensus = 1.0 - min(np.mean(deviations), 1.0)
        
        return consensus, weighted_signal
    
    def get_risk_budget(self) -> Tuple[float, float]:
        """Get aggregate risk budget from risk agent."""
        if 'risk' in self.agent_signals:
            risk_content = self.agent_signals['risk']
            return (
                risk_content.get('risk_budget', 1.0),
                risk_content.get('hedge_level', 0.0)
            )
        return 1.0, 0.0
    
    def clear(self):
        """Clear consensus state."""
        self.agent_signals.clear()
        self.timestamp = None


class ControllerAgent(BaseAgent):
    """
    Master orchestration agent.
    
    Coordinates:
    - Analyst: value/fundamental signals
    - Sentiment: news/social signals
    - Risk: drawdown/volatility monitoring
    - Execution: trade timing optimization
    
    Outputs:
    - Final portfolio allocation weights
    - Agent contribution weights
    - Rebalancing triggers
    """
    
    PRICE_HISTORY_LEN = 20
    N_ASSETS = 4  # SPY, GLD, TLT, Cash
    
    # Default allocation (46/38/16/0 baseline)
    DEFAULT_ALLOCATION = np.array([0.46, 0.38, 0.16, 0.0])
    
    def __init__(
        self,
        agent_id: str = "controller",
        n_assets: int = 4,
        hidden_dim: int = 256,
        device: str = "cpu"
    ):
        # obs_dim = price history + 4 agents * 3 outputs + portfolio state
        n_agent_outputs = 4 * 3  # 4 agents, 3 outputs each
        # Portfolio state: weights (n) + deviations (n) + value (1) + consensus (3)
        portfolio_state = n_assets * 2 + 1 + 3
        obs_dim = self.PRICE_HISTORY_LEN + n_agent_outputs + portfolio_state
        
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.CONTROLLER,
            obs_dim=obs_dim,
            action_dim=n_assets + 4 + 1,  # weights + agent_weights + rebalance
            hidden_dim=hidden_dim,
            device=device
        )
        
        self.n_assets = n_assets
        self.network = ControllerNetwork(
            obs_dim, 4, n_assets, hidden_dim
        ).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=2e-4)
        
        # Consensus tracking
        self.consensus = AgentConsensus()
        
        # Agent accuracy tracking for adaptive weighting
        self.agent_accuracy: Dict[str, List[float]] = defaultdict(list)
        
        # Current allocation
        self.current_allocation = self.DEFAULT_ALLOCATION.copy()
        self.target_allocation = self.DEFAULT_ALLOCATION.copy()
    
    def build_network(self) -> nn.ModuleDict:
        """Build network (already done in __init__)."""
        return nn.ModuleDict({'main': self.network})
    
    def process_messages(self, messages: List[AgentMessage]):
        """Process incoming agent messages."""
        for msg in messages:
            self.consensus.update(msg)
            
            # Store alerts
            if msg.msg_type == MessageType.ALERT and msg.content.get('severity') == 'critical':
                self.receive_message(msg)
    
    def extract_features(self, obs: AgentObservation) -> torch.Tensor:
        """Extract controller features from observation and consensus."""
        features = []
        
        # Price history
        prices = obs.prices[-self.PRICE_HISTORY_LEN:]
        if len(prices) < self.PRICE_HISTORY_LEN:
            prices = np.concatenate([np.ones(self.PRICE_HISTORY_LEN - len(prices)), prices])
        
        price_normalized = prices / prices[0] - 1 if prices[0] != 0 else prices
        features.extend(price_normalized)
        
        # Agent outputs (4 agents x 3 outputs: score, direction, confidence)
        agent_order = ['analyst', 'sentiment', 'risk', 'execution']
        
        for agent_id in agent_order:
            if agent_id in self.consensus.agent_signals:
                content = self.consensus.agent_signals[agent_id]
                
                # Map different agent output formats to standard 3 values
                if agent_id == 'risk':
                    score = content.get('risk_budget', 1.0) / 1.5  # Normalize [0.5,1.5] to [0.33,1]
                    direction = -content.get('hedge_level', 0.0)  # Risk direction is defensive
                    conf = content.get('confidence', 0.5)
                elif agent_id == 'execution':
                    score = content.get('urgency', 0.5)
                    direction = content.get('style_value', 0.5) * 2 - 1  # [0,1] to [-1,1]
                    conf = content.get('confidence', 0.5)
                else:
                    score = content.get('score', 0.5)
                    direction = content.get('conviction', 0.0)
                    conf = content.get('confidence', 0.5)
                
                features.extend([score, direction, conf])
            else:
                features.extend([0.5, 0.0, 0.0])  # Default if no signal
        
        # Portfolio state
        # Current weights
        current_weights = np.array([
            obs.current_weights.get('SPY', 0.46),
            obs.current_weights.get('GLD', 0.38),
            obs.current_weights.get('TLT', 0.16),
            obs.current_weights.get('CASH', 0.0)
        ])
        features.extend(current_weights)
        
        # Weight deviations from target
        deviations = current_weights - self.DEFAULT_ALLOCATION
        features.extend(deviations)
        
        # Portfolio value
        portfolio_val = obs.portfolio_value / 100000  # Normalize to ~1
        features.append(portfolio_val)
        
        # Consensus metrics
        consensus_level, consensus_signal = self.consensus.get_consensus_score()
        features.extend([consensus_level, consensus_signal, obs.volatility])
        
        return torch.FloatTensor(features).to(self.device)
    
    def act(self, obs: AgentObservation, deterministic: bool = False) -> AgentAction:
        """Generate final allocation decision."""
        self.last_observation = obs
        
        features = self.extract_features(obs)
        
        with torch.no_grad():
            agent_weights, alloc_logits, rebalance_prob, confidence, _ = self.network(
                features.unsqueeze(0)
            )
        
        # Softmax allocation ensuring sum to 1
        allocation = torch.softmax(alloc_logits, dim=-1).squeeze()
        
        # Apply risk budget from risk agent
        risk_budget, hedge_level = self.consensus.get_risk_budget()
        if risk_budget < 1.0:
            # Scale down equity exposure
            allocation = allocation * risk_budget
            # Ensure cash absorbs the difference
            allocation[-1] = 1.0 - allocation[:-1].sum()
        
        # Rebalancing decision
        should_rebalance = float(rebalance_prob.squeeze()) > 0.5
        
        # Check for critical alerts
        critical_alerts = [m for m in self.inbox 
                          if m.msg_type == MessageType.ALERT and 
                          m.content.get('severity') == 'critical']
        
        if critical_alerts:
            should_rebalance = True
            confidence = confidence * 0.8  # Reduce confidence during crisis
        
        if not deterministic:
            # Add noise to allocation
            noise = torch.randn_like(allocation) * 0.02
            allocation = torch.clamp(allocation + noise, 0, 1)
            allocation = allocation / allocation.sum()  # Renormalize
        
        agent_weights_list = [float(w) for w in agent_weights.squeeze()]
        
        action = AgentAction(
            agent_id=self.agent_id,
            action_type="portfolio_allocation",
            score=float(confidence.squeeze()),  # Overall confidence
            direction=float(rebalance_prob.squeeze()),  # Rebalance trigger
            confidence=float(confidence.squeeze()),
            metadata={
                'allocation': [float(a) for a in allocation],
                'agent_weights': agent_weights_list,
                'should_rebalance': should_rebalance,
                'risk_budget_applied': risk_budget,
                'hedge_level': hedge_level,
                'consensus_level': float(features[-3]),
                'critical_alerts': len(critical_alerts)
            }
        )
        
        self.last_action = action
        self.action_history.append(action)
        
        # Update target allocation
        self.target_allocation = np.array([float(a) for a in allocation])
        
        # Clear processed messages
        self.inbox.clear()
        self.consensus.clear()
        
        return action
    
    def compute_value(self, obs: AgentObservation) -> float:
        """Compute state value estimate (centralized critic)."""
        features = self.extract_features(obs)
        with torch.no_grad():
            _, _, _, _, value = self.network(features.unsqueeze(0))
        return float(value.squeeze())
    
    def update_agent_accuracy(self, agent_id: str, accuracy: float):
        """Update accuracy tracking for an agent."""
        self.agent_accuracy[agent_id].append(accuracy)
        if len(self.agent_accuracy[agent_id]) > 50:
            self.agent_accuracy[agent_id].pop(0)
    
    def get_agent_accuracy(self, agent_id: str) -> float:
        """Get recent accuracy for an agent."""
        if agent_id not in self.agent_accuracy or not self.agent_accuracy[agent_id]:
            return 0.5
        return np.mean(self.agent_accuracy[agent_id][-10:])
    
    def train_step(self, observations: List[AgentObservation],
                 actions: List[AgentAction],
                 returns: List[float],
                 advantages: List[float]) -> Dict[str, float]:
        """Centralized critic training step."""
        if len(observations) == 0:
            return {}
        
        # Store consensus state temporarily
        consensus_backup = self.consensus.agent_signals.copy()
        
        obs_batch = []
        for obs in observations:
            # Restore consensus for each observation
            # In practice, we'd store agent outputs with observations
            self.consensus.agent_signals = consensus_backup
            obs_batch.append(self.extract_features(obs))
        
        obs_batch = torch.stack(obs_batch)
        returns_t = torch.FloatTensor(returns).to(self.device).unsqueeze(1)
        advantages_t = torch.FloatTensor(advantages).to(self.device).unsqueeze(1)
        
        agent_weights, alloc_logits, rebalance_prob, confidence, values = self.network(obs_batch)
        
        # Centralized value loss
        value_loss = nn.MSELoss()(values, returns_t)
        
        # Policy loss: better allocations lead to better returns
        allocation_probs = torch.softmax(alloc_logits, dim=-1)
        
        # Entropy for exploration
        entropy = -torch.mean(
            torch.sum(
                allocation_probs * torch.log(allocation_probs + 1e-8),
                dim=-1
            )
        )
        
        # Confidence should correlate with |advantage|
        confidence_target = torch.sigmoid(torch.abs(advantages_t) * 2)
        confidence_loss = nn.MSELoss()(confidence, confidence_target)
        
        # Total loss
        loss = value_loss + 0.1 * confidence_loss - 0.01 * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()
        
        return {
            'value_loss': float(value_loss),
            'confidence_loss': float(confidence_loss),
            'entropy': float(entropy),
            'mean_agent_weights': [float(w) for w in torch.mean(agent_weights, dim=0)],
            'mean_confidence': float(torch.mean(confidence))
        }
