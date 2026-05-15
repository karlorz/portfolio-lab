#!/usr/bin/env python3
"""
Portfolio-Lab v2.51: Analyst Agent

Fundamental/value analysis agent that processes earnings estimates,
valuation ratios, growth signals, and quality metrics to generate
value-based investment signals.

Observations:
- Earnings surprise history
- Forward P/E, P/B, P/S ratios (z-scores vs sector)
- Revenue/EPS growth trends
- ROE, ROIC quality metrics
- FCF yield and capital allocation

Actions:
- value_score [0, 1]: attractiveness score
- conviction [-1, 1]: directional conviction (long/short/neutral)
- confidence [0, 1]: certainty in assessment
"""

import os
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path

# Conditional ML import — disabled by default to prevent OOM in test suites.
# Set PORTFOLIO_LAB_ENABLE_ML=1 to load real torch.
_ML_ENABLED = os.environ.get("PORTFOLIO_LAB_ENABLE_ML", "0") == "1"
if _ML_ENABLED:
    import torch
    import torch.nn as nn
else:
    from .base_agent import torch, nn

from .base_agent import BaseAgent, AgentType, AgentObservation, AgentAction, AgentMessage, MessageType


class AnalystNetwork(nn.Module):
    """Neural network for analyst agent."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Value scoring head (attractiveness)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # [0, 1]
        )
        
        # Conviction head (direction)
        self.conviction_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # [-1, 1]
        )
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # [0, 1]
        )
        
        # Value estimate for critic
        self.value_head_critic = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, obs: torch.Tensor) -> tuple:
        features = self.encoder(obs)
        
        value_score = self.value_head(features)
        conviction = self.conviction_head(features)
        confidence = self.confidence_head(features)
        state_value = self.value_head_critic(features)
        
        return value_score, conviction, confidence, state_value


class AnalystAgent(BaseAgent):
    """
    Fundamental/value analysis agent.
    
    Processes value-oriented signals including:
    - Valuation metrics (P/E, P/B, EV/EBITDA z-scores)
    - Earnings quality and surprise
    - Growth trends (revenue, earnings, FCF)
    - Capital efficiency (ROE, ROIC, ROA)
    """
    
    # Feature dimensions
    PRICE_HISTORY_LEN = 20
    N_FUNDAMENTAL_FEATURES = 12
    
    def __init__(
        self,
        agent_id: str = "analyst",
        hidden_dim: int = 128,
        device: str = "cpu"
    ):
        obs_dim = self.PRICE_HISTORY_LEN + self.N_FUNDAMENTAL_FEATURES
        action_dim = 3  # value_score, conviction, confidence
        
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.ANALYST,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            device=device
        )
        
        self.network = AnalystNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=3e-4)
        
        # Feature metadata
        self.feature_names = [
            'pe_zscore',           # P/E z-score vs history
            'pb_zscore',           # P/B z-score
            'ps_zscore',           # P/S z-score
            'ev_ebitda_zscore',    # EV/EBITDA z-score
            'earnings_surprise',   # Last earnings surprise %
            'eps_growth_1y',      # 1-year EPS growth
            'revenue_growth_1y',  # 1-year revenue growth
            'fcf_yield',          # FCF / Market Cap
            'roe_ttm',            # Return on equity
            'roic_ttm',           # Return on invested capital
            'debt_equity',        # Debt to equity ratio
            'quality_score',      # Composite quality metric
        ]
    
    def build_network(self) -> nn.ModuleDict:
        """Build network (already done in __init__)."""
        return nn.ModuleDict({'main': self.network})
    
    def extract_features(self, obs: AgentObservation) -> torch.Tensor:
        """
        Extract fundamental features from observation.
        
        In production, this would query fundamental data APIs.
        For simulation, we derive features from price patterns.
        """
        features = []
        
        # Price momentum as value proxy (mean reversion assumption)
        prices = obs.prices[-self.PRICE_HISTORY_LEN:]
        returns = np.diff(prices) / prices[:-1]
        
        # Simple fundamental proxies from price action
        # (In production, these come from actual fundamental APIs)
        
        # P/E proxy: inverse of recent momentum
        momentum = np.mean(returns[-5:]) if len(returns) >= 5 else 0
        pe_proxy = -momentum * 20  # Mean reversion: high momentum = high P/E
        features.append(np.clip(pe_proxy, -3, 3))
        
        # P/B proxy: longer-term trend
        long_momentum = np.mean(returns) if len(returns) > 0 else 0
        pb_proxy = -long_momentum * 15
        features.append(np.clip(pb_proxy, -3, 3))
        
        # P/S proxy: volatility adjusted
        vol = np.std(returns) if len(returns) > 0 else 0.01
        ps_proxy = -momentum / (vol + 0.01) * 5
        features.append(np.clip(ps_proxy, -3, 3))
        
        # EV/EBITDA proxy
        ev_proxy = features[-1] * 0.8
        features.append(np.clip(ev_proxy, -3, 3))
        
        # Earnings surprise: sudden moves
        if len(returns) >= 2:
            surprise = returns[-1] - np.mean(returns[:-1])
        else:
            surprise = 0
        features.append(np.clip(surprise * 10, -1, 1))
        
        # Growth proxies
        eps_growth = np.mean(returns[-10:]) * 252 if len(returns) >= 10 else 0
        features.append(np.clip(eps_growth, -0.5, 0.5))
        
        revenue_growth = long_momentum * 252
        features.append(np.clip(revenue_growth, -0.5, 0.5))
        
        # FCF yield proxy
        fcf_proxy = -features[0] * 0.05  # Inverse of P/E
        features.append(np.clip(fcf_proxy, -0.1, 0.1))
        
        # Quality proxies
        roe = 0.15 + features[1] * 0.05  # Base 15% adjusted by P/B proxy
        features.append(np.clip(roe, 0, 0.3))
        
        roic = roe * 0.9
        features.append(np.clip(roic, 0, 0.25))
        
        debt_equity = max(0, -features[1])  # Higher P/B = lower D/E
        features.append(np.clip(debt_equity, 0, 2))
        
        quality = (features[-2] + features[-3]) / 2 - features[-1] * 0.1
        features.append(np.clip(quality, 0, 1))
        
        # Add price history
        price_normalized = prices / prices[0] - 1 if prices[0] != 0 else prices
        
        full_features = np.concatenate([
            price_normalized,
            np.array(features)
        ])
        
        return torch.FloatTensor(full_features).to(self.device)
    
    def act(self, obs: AgentObservation, deterministic: bool = False) -> AgentAction:
        """Generate value-based action."""
        self.last_observation = obs
        
        features = self.extract_features(obs)
        
        with torch.no_grad():
            value_score, conviction, confidence, _ = self.network(features.unsqueeze(0))
        
        # Add exploration noise if not deterministic
        if not deterministic:
            noise = torch.randn_like(value_score) * 0.05
            value_score = torch.clamp(value_score + noise, 0, 1)
            conviction = torch.clamp(conviction + torch.randn_like(conviction) * 0.1, -1, 1)
        
        action = AgentAction(
            agent_id=self.agent_id,
            action_type="value_signal",
            score=float(value_score.squeeze()),
            direction=float(conviction.squeeze()),
            confidence=float(confidence.squeeze()),
            metadata={
                'regime': obs.regime,
                'feature_summary': {
                    'pe_proxy': float(features[self.PRICE_HISTORY_LEN]),
                    'quality_proxy': float(features[-1])
                }
            }
        )
        
        self.last_action = action
        self.action_history.append(action)
        
        # Broadcast signal to controller
        self.send_message(
            receiver="controller",
            msg_type=MessageType.SIGNAL,
            content={
                'score': action.score,
                'conviction': action.direction,
                'confidence': action.confidence,
                'implied_allocation_delta': action.score * action.direction * 0.05
            },
            priority=2 if abs(action.direction) > 0.7 else 1
        )
        
        return action
    
    def compute_value(self, obs: AgentObservation) -> float:
        """Compute state value estimate."""
        features = self.extract_features(obs)
        with torch.no_grad():
            _, _, _, value = self.network(features.unsqueeze(0))
        return float(value.squeeze())
    
    def train_step(self, observations: List[AgentObservation], 
                 actions: List[AgentAction], 
                 returns: List[float],
                 advantages: List[float]) -> Dict[str, float]:
        """PPO training step."""
        if len(observations) == 0:
            return {}
        
        # Prepare batch
        obs_batch = torch.stack([self.extract_features(o) for o in observations])
        
        # Current predictions
        value_scores, convictions, confidences, values = self.network(obs_batch)
        
        # Target returns
        returns_t = torch.FloatTensor(returns).to(self.device).unsqueeze(1)
        advantages_t = torch.FloatTensor(advantages).to(self.device).unsqueeze(1)
        
        # Value loss
        value_loss = nn.MSELoss()(values, returns_t)
        
        # Policy loss (simplified PPO)
        # We want to maximize expected return, weighted by confidence
        policy_loss = -torch.mean(
            confidences * advantages_t * torch.log(confidences + 1e-8)
        )
        
        # Entropy bonus for exploration
        entropy = -torch.mean(
            confidences * torch.log(confidences + 1e-8) +
            (1 - confidences) * torch.log(1 - confidences + 1e-8)
        )
        
        # Total loss
        loss = value_loss + 0.5 * policy_loss - 0.01 * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()
        
        return {
            'value_loss': float(value_loss),
            'policy_loss': float(policy_loss),
            'entropy': float(entropy)
        }
