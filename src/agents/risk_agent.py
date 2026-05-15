#!/usr/bin/env python3
"""
Portfolio-Lab v2.51: Risk Agent

Risk monitoring and drawdown prevention agent. Monitors VaR, CVaR,
maximum drawdown, tail risk, and correlation breakdown to generate
risk-adjusted position sizing and hedging signals.

Observations:
- VaR (Value at Risk)
- CVaR (Conditional VaR / Expected Shortfall)
- Maximum drawdown and drawdown duration
- Tail risk metrics
- Correlation regime changes
- Volatility clustering

Actions:
- risk_budget [0.5, 1.5]: position sizing multiplier
- hedging_level [0, 1]: hedge ratio recommendation
- confidence [0, 1]: certainty in risk assessment
"""

import os
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from .base_agent import BaseAgent, AgentType, AgentObservation, AgentAction, AgentMessage, MessageType

# Conditional ML import — disabled by default to prevent OOM in test suites.
_ML_ENABLED = os.environ.get("PORTFOLIO_LAB_ENABLE_ML", "0") == "1"
if _ML_ENABLED:
    import torch
    import torch.nn as nn
else:
    from .base_agent import torch, nn


class RiskNetwork(nn.Module):
    """Neural network for risk agent."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Risk budget multiplier [0.5, 1.5]
        self.risk_budget_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # [0, 1] -> scaled to [0.5, 1.5]
        )
        
        # Hedging level [0, 1]
        self.hedge_head = nn.Sequential(
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
        
        # Drawdown early warning
        self.dd_warning_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
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
        
        risk_budget_raw = self.risk_budget_head(features)  # [0, 1]
        risk_budget = risk_budget_raw * 1.0 + 0.5  # [0.5, 1.5]
        
        hedge_level = self.hedge_head(features)
        confidence = self.confidence_head(features)
        dd_warning = self.dd_warning_head(features)
        state_value = self.value_head(features)
        
        return risk_budget, hedge_level, confidence, dd_warning, state_value


class RiskAgent(BaseAgent):
    """
    Risk monitoring agent.
    
    Monitors risk metrics including:
    - VaR (95% and 99%)
    - CVaR / Expected Shortfall
    - Current and max drawdown
    - Volatility clustering
    - Tail risk indicators
    - Correlation breakdown detection
    """
    
    PRICE_HISTORY_LEN = 60  # Risk needs longer history
    N_RISK_FEATURES = 14
    
    # Risk thresholds
    VAR_THRESHOLD = 0.02  # 2% daily VaR warning
    DRAWDOWN_WARNING = 0.10  # 10% drawdown warning
    DRAWDOWN_CRITICAL = 0.20  # 20% drawdown critical
    
    def __init__(
        self,
        agent_id: str = "risk",
        hidden_dim: int = 128,
        device: str = "cpu"
    ):
        obs_dim = self.PRICE_HISTORY_LEN + self.N_RISK_FEATURES
        action_dim = 3
        
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.RISK,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            device=device
        )
        
        self.network = RiskNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=2e-4)
        
        # Feature metadata
        self.feature_names = [
            'var_95',              # 95% VaR
            'var_99',              # 99% VaR
            'cvar_95',             # Conditional VaR 95%
            'current_dd',          # Current drawdown
            'max_dd_1y',          # Max drawdown over period
            'dd_duration',        # Days in current drawdown
            'volatility_20d',     # 20-day realized vol
            'vol_regime',         # Vol regime indicator
            'skewness',           # Return skewness
            'kurtosis',           # Return kurtosis (fat tails)
            'tail_risk',          # Tail risk score
            'correlation_stress',  # Correlation breakdown
            'sharpe_recent',      # Recent Sharpe ratio
            'risk_regime',        # Overall risk regime
        ]
        
        # Track portfolio highs for drawdown
        self.portfolio_high: Optional[float] = None
    
    def build_network(self) -> nn.ModuleDict:
        """Build network (already done in __init__)."""
        return nn.ModuleDict({'main': self.network})
    
    def calculate_var(self, returns: np.ndarray, alpha: float = 0.05) -> float:
        """Calculate Value at Risk."""
        if len(returns) == 0:
            return 0.02
        return np.percentile(returns, alpha * 100)
    
    def calculate_cvar(self, returns: np.ndarray, alpha: float = 0.05) -> float:
        """Calculate Conditional VaR (Expected Shortfall)."""
        var = self.calculate_var(returns, alpha)
        return np.mean(returns[returns <= var]) if len(returns[returns <= var]) > 0 else var
    
    def calculate_drawdown(self, prices: np.ndarray) -> Tuple[float, int]:
        """Calculate current drawdown and duration."""
        if len(prices) == 0:
            return 0.0, 0
        
        peak = np.maximum.accumulate(prices)
        drawdown = (prices - peak) / peak
        current_dd = drawdown[-1]
        
        # Find drawdown duration
        in_drawdown = current_dd < -0.001
        dd_duration = 0
        if in_drawdown:
            for i in range(1, len(drawdown)):
                if drawdown[-i] < -0.001:
                    dd_duration += 1
                else:
                    break
        
        return current_dd, dd_duration
    
    def extract_features(self, obs: AgentObservation) -> torch.Tensor:
        """Extract risk features from observation."""
        features = []
        
        prices = obs.prices[-self.PRICE_HISTORY_LEN:]
        if len(prices) < 2:
            # Pad with zeros
            prices = np.concatenate([np.ones(self.PRICE_HISTORY_LEN - len(prices)), prices])
        
        returns = np.diff(prices) / prices[:-1]
        
        # VaR calculations
        var_95 = self.calculate_var(returns, 0.05)
        var_99 = self.calculate_var(returns, 0.01)
        features.extend([var_95, var_99])
        
        # CVaR
        cvar_95 = self.calculate_cvar(returns, 0.05)
        features.append(cvar_95)
        
        # Drawdown
        current_dd, dd_duration = self.calculate_drawdown(prices)
        max_dd = np.min(np.minimum.accumulate(prices) / np.maximum.accumulate(prices) - 1)
        features.extend([current_dd, max_dd, min(dd_duration, 252) / 252])  # Normalize to [0,1]
        
        # Volatility metrics
        if len(returns) >= 20:
            vol_20 = np.std(returns[-20:]) * np.sqrt(252)
        else:
            vol_20 = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.20
        features.append(vol_20)
        
        # Vol regime: low, normal, high, extreme
        vol_regime = 0.0
        if vol_20 > 0.40:
            vol_regime = 1.0  # Extreme
        elif vol_20 > 0.25:
            vol_regime = 0.5  # High
        elif vol_20 < 0.10:
            vol_regime = -0.5  # Low
        features.append(vol_regime)
        
        # Tail risk (skewness and kurtosis)
        if len(returns) >= 30:
            skew = np.mean((returns - np.mean(returns))**3) / (np.std(returns)**3 + 1e-8)
            kurt = np.mean((returns - np.mean(returns))**4) / (np.std(returns)**4 + 1e-8) - 3
        else:
            skew = 0.0
            kurt = 0.0
        features.extend([np.tanh(skew), np.tanh(kurt / 3)])
        
        # Tail risk score (higher = more left tail risk)
        tail_risk = -skew + max(0, kurt) / 3 + max(0, -var_99 * 20)
        features.append(np.clip(tail_risk, 0, 1))
        
        # Correlation stress proxy: rolling correlation breakdown
        if len(returns) >= 30:
            half = len(returns) // 2
            corr_first = np.corrcoef(np.arange(half), returns[:half])[0, 1] if half > 1 else 0
            corr_second = np.corrcoef(np.arange(len(returns) - half), returns[half:])[0, 1] if (len(returns) - half) > 1 else 0
            corr_stress = abs(corr_first - corr_second)
        else:
            corr_stress = 0.0
        features.append(corr_stress)
        
        # Recent Sharpe
        if len(returns) >= 20:
            sharpe = np.mean(returns[-20:]) / (np.std(returns[-20:]) + 1e-8) * np.sqrt(252)
        else:
            sharpe = 0.0
        features.append(np.tanh(sharpe))
        
        # Overall risk regime
        risk_score = (var_95 * 20 + abs(current_dd) + vol_20 * 0.5)
        risk_regime = np.tanh(risk_score - 1.0)
        features.append(risk_regime)
        
        # Normalize prices
        price_normalized = prices / prices[0] - 1 if prices[0] != 0 else prices
        
        full_features = np.concatenate([
            price_normalized,
            np.array(features)
        ])
        
        return torch.FloatTensor(full_features).to(self.device)
    
    def act(self, obs: AgentObservation, deterministic: bool = False) -> AgentAction:
        """Generate risk-adjusted action."""
        self.last_observation = obs
        
        features = self.extract_features(obs)
        
        with torch.no_grad():
            risk_budget, hedge_level, confidence, dd_warning, _ = self.network(features.unsqueeze(0))
        
        # Adjust based on drawdown warning
        warning_level = float(dd_warning.squeeze())
        if warning_level > 0.7:
            # Emergency de-risking
            risk_budget = risk_budget * 0.5
            hedge_level = torch.clamp(hedge_level + 0.3, 0, 1)
            confidence = confidence * 1.2  # High confidence in warning
        
        if not deterministic:
            noise = torch.randn_like(risk_budget) * 0.03
            risk_budget = torch.clamp(risk_budget + noise, 0.5, 1.5)
        
        action = AgentAction(
            agent_id=self.agent_id,
            action_type="risk_signal",
            score=float(risk_budget.squeeze()),  # Used as budget multiplier
            direction=-float(hedge_level.squeeze()),  # Negative = defensive
            confidence=float(confidence.squeeze()),
            metadata={
                'hedge_level': float(hedge_level.squeeze()),
                'drawdown_warning': warning_level,
                'var_95': float(features[self.PRICE_HISTORY_LEN]),
                'current_dd': float(features[self.PRICE_HISTORY_LEN + 3]),
                'vol_20d': float(features[self.PRICE_HISTORY_LEN + 6])
            }
        )
        
        self.last_action = action
        self.action_history.append(action)
        
        # Send alert if warning
        priority = 1
        if warning_level > 0.7:
            priority = 5  # Critical
            self.send_message(
                receiver=None,  # Broadcast
                msg_type=MessageType.ALERT,
                content={
                    'alert_type': 'drawdown_warning',
                    'severity': 'critical' if warning_level > 0.85 else 'warning',
                    'drawdown': float(features[self.PRICE_HISTORY_LEN + 3]),
                    'recommendation': 'de_risk'
                },
                priority=priority
            )
        
        # Send regular signal to controller
        self.send_message(
            receiver="controller",
            msg_type=MessageType.SIGNAL,
            content={
                'risk_budget': float(risk_budget.squeeze()),
                'hedge_level': float(hedge_level.squeeze()),
                'confidence': float(confidence.squeeze()),
                'drawdown_warning': warning_level > 0.7,
                'position_scale': float(risk_budget.squeeze())
            },
            priority=priority
        )
        
        return action
    
    def compute_value(self, obs: AgentObservation) -> float:
        """Compute state value estimate."""
        features = self.extract_features(obs)
        with torch.no_grad():
            _, _, _, _, value = self.network(features.unsqueeze(0))
        return float(value.squeeze())
    
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
        
        risk_budget, hedge_level, confidence, dd_warning, values = self.network(obs_batch)
        
        # Value loss
        value_loss = nn.MSELoss()(values, returns_t)
        
        # Policy: risk agent learns to reduce risk when bad outcomes expected
        # Lower risk budget when advantage is negative
        policy_loss = -torch.mean(
            confidence * torch.log(risk_budget + 1e-8) * advantages_t
        )
        
        # Hedge policy: increase hedge when advantage negative
        hedge_loss = -torch.mean(
            confidence * torch.log(hedge_level + 1e-8) * (-advantages_t)
        )
        
        # Entropy
        entropy = -torch.mean(
            confidence * torch.log(confidence + 1e-8) +
            (1 - confidence) * torch.log(1 - confidence + 1e-8)
        )
        
        loss = value_loss + 0.3 * (policy_loss + hedge_loss) - 0.01 * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()
        
        return {
            'value_loss': float(value_loss),
            'policy_loss': float(policy_loss),
            'hedge_loss': float(hedge_loss),
            'entropy': float(entropy),
            'mean_risk_budget': float(torch.mean(risk_budget)),
            'mean_hedge': float(torch.mean(hedge_level))
        }
