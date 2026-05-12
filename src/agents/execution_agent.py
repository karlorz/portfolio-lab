#!/usr/bin/env python3
"""
Portfolio-Lab v2.51: Execution Agent

Trade execution optimization agent. Determines optimal order timing,
sizing, and routing to minimize market impact and maximize fill quality.

Observations:
- Bid-ask spread and depth
- Volume profile (intraday patterns)
- Market impact estimates
- Liquidity conditions
- Volatility regime

Actions:
- urgency [0, 1]: execution urgency (immediate vs patient)
- slice_size [0.1, 0.5]: order slice as fraction of target
- execution_style [0, 1]: VWAP (0) vs aggressive (1)
- confidence [0, 1]: certainty in execution plan
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum

from .base_agent import BaseAgent, AgentType, AgentObservation, AgentAction, AgentMessage, MessageType


class ExecutionStyle(Enum):
    """Execution style spectrum."""
    VWAP = 0.0      # Volume-weighted average price
    POV = 0.33      # Percentage of volume
    TWAP = 0.66     # Time-weighted average price
    AGGRESSIVE = 1.0  # Market orders / aggressive


class ExecutionNetwork(nn.Module):
    """Neural network for execution agent."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Execution urgency [0, 1] - 0 = patient, 1 = immediate
        self.urgency_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Slice size [0.1, 0.5] fraction of target
        self.slice_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # [0, 1] -> [0.1, 0.5]
        )
        
        # Execution style [0, 1] VWAP -> Aggressive
        self.style_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Confidence [0, 1]
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Value for critic
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, obs: torch.Tensor) -> tuple:
        features = self.encoder(obs)
        
        urgency = self.urgency_head(features)
        slice_frac = self.slice_head(features)
        style = self.style_head(features)
        confidence = self.confidence_head(features)
        state_value = self.value_head(features)
        
        return urgency, slice_frac, style, confidence, state_value


class ExecutionAgent(BaseAgent):
    """
    Trade execution optimization agent.
    
    Optimizes for:
    - Low market impact
    - High fill rates
    - Minimal slippage
    - Adaptation to liquidity conditions
    
    In live trading, integrates with broker APIs for
    smart order routing. In backtest, simulates realistic
    execution with market impact models.
    """
    
    PRICE_HISTORY_LEN = 30  # Execution needs recent history
    N_EXEC_FEATURES = 12
    
    def __init__(
        self,
        agent_id: str = "execution",
        hidden_dim: int = 128,
        device: str = "cpu"
    ):
        obs_dim = self.PRICE_HISTORY_LEN + self.N_EXEC_FEATURES
        action_dim = 4
        
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.EXECUTION,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            device=device
        )
        
        self.network = ExecutionNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=3e-4)
        
        # Feature metadata
        self.feature_names = [
            'spread_proxy',        # Bid-ask spread proxy
            'volume_proxy',        # Volume trend
            'volatility_intra',    # Intraday volatility
            'momentum_5m',        # Short momentum
            'mean_reversion',     # Mean reversion signal
            'liquidity_score',    # Liquidity proxy
            'impact_estimate',    # Estimated market impact
            'time_of_day',        # Trading session phase
            'urgency_required',   # Required urgency from others
            'target_size',        # Order size relative to ADV
            'confidence_external', # External signal confidence
            'regime_volatility',  # Vol regime indicator
        ]
    
    def build_network(self) -> nn.ModuleDict:
        """Build network (already done in __init__)."""
        return nn.ModuleDict({'main': self.network})
    
    def extract_features(self, obs: AgentObservation) -> torch.Tensor:
        """Extract execution features from observation."""
        features = []
        
        prices = obs.prices[-self.PRICE_HISTORY_LEN:]
        if len(prices) < self.PRICE_HISTORY_LEN:
            prices = np.concatenate([np.ones(self.PRICE_HISTORY_LEN - len(prices)), prices])
        
        returns = np.diff(prices) / prices[:-1] if len(prices) > 1 else np.array([0])
        
        # Spread proxy (from volatility)
        vol_short = np.std(returns[-5:]) if len(returns) >= 5 else 0.01
        spread_proxy = vol_short * 2  # Approximate spread from volatility
        features.append(np.clip(spread_proxy * 100, 0, 1))  # [0, 1]
        
        # Volume proxy (from price volatility patterns)
        # High volume often associated with higher volatility
        vol_long = np.std(returns) if len(returns) > 0 else 0.01
        volume_proxy = 1.0 if vol_short > vol_long * 1.5 else (0.5 if vol_short > vol_long else 0.0)
        features.append(volume_proxy)
        
        # Intraday volatility
        features.append(np.clip(vol_short * 20, 0, 1))
        
        # Short momentum (5-period)
        if len(returns) >= 5:
            mom_5 = np.tanh(np.sum(returns[-5:]) * 50)
        else:
            mom_5 = 0.0
        features.append(mom_5)
        
        # Mean reversion signal
        if len(returns) >= 10:
            short_mean = np.mean(returns[-5:])
            long_mean = np.mean(returns[-10:])
            mean_rev = np.tanh((short_mean - long_mean) * 100)
        else:
            mean_rev = 0.0
        features.append(mean_rev)
        
        # Liquidity score (inverse of spread)
        liquidity = 1.0 - features[0]
        features.append(liquidity)
        
        # Market impact estimate (volatility-adjusted)
        # Kyle's lambda approximation
        impact = vol_short * (1 + volume_proxy)
        features.append(np.clip(impact * 30, 0, 1))
        
        # Time of day (simulated as random in backtest)
        # In live: 0 = open, 0.5 = mid, 1 = close
        features.append(0.5)
        
        # Urgency required (from volatility regime)
        urgency_req = 1.0 if obs.volatility > 0.30 else (0.5 if obs.volatility > 0.20 else 0.0)
        features.append(urgency_req)
        
        # Target size (normalized)
        # Default assumption: medium size relative to liquidity
        features.append(0.3)
        
        # External confidence (from other agents)
        # Default: moderate
        features.append(0.5)
        
        # Vol regime indicator
        vol_regime = np.tanh((obs.volatility - 0.15) * 10)
        features.append(vol_regime)
        
        # Normalize prices
        price_normalized = prices / prices[0] - 1 if prices[0] != 0 else prices
        
        full_features = np.concatenate([
            price_normalized,
            np.array(features)
        ])
        
        return torch.FloatTensor(full_features).to(self.device)
    
    def act(self, obs: AgentObservation, deterministic: bool = False,
            required_urgency: Optional[float] = None) -> AgentAction:
        """Generate execution action."""
        self.last_observation = obs
        
        features = self.extract_features(obs)
        
        with torch.no_grad():
            urgency, slice_frac, style, confidence, _ = self.network(features.unsqueeze(0))
        
        # Override urgency if specified
        if required_urgency is not None:
            urgency = torch.tensor([[required_urgency]], device=self.device)
        
        # Scale slice_frac to [0.1, 0.5]
        slice_size = 0.1 + slice_frac * 0.4
        
        # Determine execution style
        style_value = float(style.squeeze())
        if style_value < 0.25:
            exec_style = ExecutionStyle.VWAP
        elif style_value < 0.50:
            exec_style = ExecutionStyle.POV
        elif style_value < 0.75:
            exec_style = ExecutionStyle.TWAP
        else:
            exec_style = ExecutionStyle.AGGRESSIVE
        
        if not deterministic:
            noise = torch.randn_like(urgency) * 0.05
            urgency = torch.clamp(urgency + noise, 0, 1)
            confidence = torch.clamp(confidence + torch.randn_like(confidence) * 0.05, 0, 1)
        
        action = AgentAction(
            agent_id=self.agent_id,
            action_type="execution_plan",
            score=float(urgency.squeeze()),  # Urgency level
            direction=float(style.squeeze()),  # Execution style
            confidence=float(confidence.squeeze()),
            metadata={
                'slice_size': float(slice_size.squeeze()),
                'execution_style': exec_style.name,
                'style_value': style_value,
                'spread_proxy': float(features[self.PRICE_HISTORY_LEN]),
                'liquidity_score': float(features[self.PRICE_HISTORY_LEN + 5])
            }
        )
        
        self.last_action = action
        self.action_history.append(action)
        
        # Send execution plan to controller
        self.send_message(
            receiver="controller",
            msg_type=MessageType.SIGNAL,
            content={
                'urgency': float(urgency.squeeze()),
                'slice_size': float(slice_size.squeeze()),
                'execution_style': exec_style.name,
                'style_value': style_value,
                'confidence': float(confidence.squeeze()),
                'liquidity': float(features[self.PRICE_HISTORY_LEN + 5])
            },
            priority=1
        )
        
        return action
    
    def compute_value(self, obs: AgentObservation) -> float:
        """Compute state value estimate."""
        features = self.extract_features(obs)
        with torch.no_grad():
            _, _, _, _, value = self.network(features.unsqueeze(0))
        return float(value.squeeze())
    
    def estimate_market_impact(
        self,
        order_size_pct: float,
        volatility: float,
        liquidity: float
    ) -> float:
        """
        Estimate market impact using square-root law.
        
        Impact = alpha * volatility * sqrt(order_size / ADV)
        """
        alpha = 0.5  # Impact coefficient
        impact = alpha * volatility * np.sqrt(max(order_size_pct, 0.01))
        # Adjust for liquidity
        impact = impact / (liquidity + 0.1)
        return min(impact, 0.05)  # Cap at 5%
    
    def train_step(self, observations: List[AgentObservation],
                 actions: List[AgentAction],
                 returns: List[float],
                 advantages: List[float]) -> Dict[str, float]:
        """PPO training step."""
        if len(observations) == 0:
            return {}
        
        obs_batch = torch.stack([self.extract_features(o) for o in observations])
        returns_t = torch.FloatTensor(returns).to(self.device).unsqueeze(1)
        advantages_t = torch.FloatTensor(advantages).to(self.device).unsqueeze(1)
        
        urgency, slice_frac, style, confidence, values = self.network(obs_batch)
        
        # Value loss
        value_loss = nn.MSELoss()(values, returns_t)
        
        # Policy: minimize impact when advantage positive
        # Lower urgency/slices when good outcomes expected
        policy_loss = -torch.mean(
            confidence * (1 - urgency) * advantages_t * torch.log(confidence + 1e-8)
        )
        
        # Entropy
        entropy = -torch.mean(
            confidence * torch.log(confidence + 1e-8) +
            (1 - confidence) * torch.log(1 - confidence + 1e-8)
        )
        
        loss = value_loss + 0.3 * policy_loss - 0.01 * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()
        
        return {
            'value_loss': float(value_loss),
            'policy_loss': float(policy_loss),
            'entropy': float(entropy),
            'mean_urgency': float(torch.mean(urgency)),
            'mean_slice': float(torch.mean(0.1 + slice_frac * 0.4))
        }
