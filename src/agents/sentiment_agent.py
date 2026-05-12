#!/usr/bin/env python3
"""
Portfolio-Lab v2.51: Sentiment Agent

News and social media sentiment analysis agent. Processes alternative
data signals, news sentiment, social metrics, and anomaly detection
to generate sentiment-based investment signals.

Observations:
- News sentiment scores
- Social media sentiment
- Volume anomalies
- Options sentiment (put/call skew)
- Search trends and alternative data

Actions:
- sentiment_score [0, 1]: bullishness level
- direction [-1, 1]: directional tilt
- confidence [0, 1]: certainty in sentiment reading
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from datetime import datetime

from .base_agent import BaseAgent, AgentType, AgentObservation, AgentAction, AgentMessage, MessageType


class SentimentNetwork(nn.Module):
    """Neural network for sentiment agent."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        # Encoder with attention for recent signals
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Sentiment scoring
        self.sentiment_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # [0, 1]
        )
        
        # Direction (can be contrary for contrarian signals)
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # [-1, 1]
        )
        
        # Confidence (uncertainty in noisy sentiment)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # [0, 1]
        )
        
        # Contrarian detection
        self.contrarian_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Probability sentiment is overdone
        )
        
        # Value for critic
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, obs: torch.Tensor) -> tuple:
        features = self.encoder(obs)
        
        sentiment = self.sentiment_head(features)
        direction = self.direction_head(features)
        confidence = self.confidence_head(features)
        contrarian_prob = self.contrarian_head(features)
        state_value = self.value_head(features)
        
        return sentiment, direction, confidence, contrarian_prob, state_value


class SentimentAgent(BaseAgent):
    """
    Sentiment analysis agent.
    
    Processes sentiment-oriented signals including:
    - News sentiment aggregation
    - Social media sentiment trends
    - Volume/sentiment divergence detection
    - Options market sentiment (put/call ratio, skew)
    - Search trend indicators
    """
    
    PRICE_HISTORY_LEN = 20
    N_SENTIMENT_FEATURES = 10
    
    def __init__(
        self,
        agent_id: str = "sentiment",
        hidden_dim: int = 128,
        device: str = "cpu"
    ):
        obs_dim = self.PRICE_HISTORY_LEN + self.N_SENTIMENT_FEATURES
        action_dim = 3
        
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.SENTIMENT,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            device=device
        )
        
        self.network = SentimentNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=3e-4)
        
        # Sentiment feature names
        self.feature_names = [
            'news_sentiment',      # Aggregated news sentiment [-1, 1]
            'social_sentiment',    # Social media sentiment [-1, 1]
            'sentiment_momentum',  # Sentiment trend
            'volume_sentiment',    # Volume-weighted sentiment
            'options_skew',        # Put/call skew proxy
            'volume_anomaly',      # Volume vs average
            'price_sentiment_div', # Divergence score
            'extreme_sentiment',   # Extreme reading indicator
            'sentiment_regime',    # Bullish/bearish/neutral regime
            'contrarian_trigger',  # Contrarian opportunity
        ]
    
    def extract_features(self, obs: AgentObservation) -> torch.Tensor:
        """Extract sentiment features from observation."""
        features = []
        
        prices = obs.prices[-self.PRICE_HISTORY_LEN:]
        returns = np.diff(prices) / prices[:-1] if len(prices) > 1 else np.array([0])
        
        # Sentiment proxies derived from price action
        # In production, these come from actual news/social APIs
        
        # News sentiment: price momentum with decay
        if len(returns) >= 5:
            news_sent = np.tanh(np.mean(returns[-5:]) * 50)
        else:
            news_sent = 0.0
        features.append(news_sent)
        
        # Social sentiment: more volatile, shorter-term
        if len(returns) >= 3:
            social_sent = np.tanh(np.mean(returns[-3:]) * 80)
        else:
            social_sent = 0.0
        features.append(social_sent)
        
        # Sentiment momentum
        if len(returns) >= 10:
            sent_mom = np.tanh((np.mean(returns[-5:]) - np.mean(returns[-10:-5])) * 100)
        else:
            sent_mom = 0.0
        features.append(sent_mom)
        
        # Volume-weighted sentiment (proxy using volatility)
        vol = np.std(returns) if len(returns) > 1 else 0.01
        vol_sent = news_sent * (1 - np.tanh(vol * 20))  # Higher vol = lower confidence
        features.append(vol_sent)
        
        # Options skew proxy: high vol = fear, low vol = complacency
        options_skew = -np.tanh((vol - 0.02) * 30)  # 20% annualized is baseline
        features.append(options_skew)
        
        # Volume anomaly
        if len(returns) >= 10:
            recent_vol = np.std(returns[-5:])
            hist_vol = np.std(returns[-10:])
            vol_anomaly = np.tanh((recent_vol / (hist_vol + 1e-6) - 1) * 3)
        else:
            vol_anomaly = 0.0
        features.append(vol_anomaly)
        
        # Price-sentiment divergence
        price_trend = np.tanh(np.mean(returns) * 30) if len(returns) > 0 else 0
        sentiment_trend = news_sent
        divergence = price_trend - sentiment_trend
        features.append(divergence)
        
        # Extreme sentiment indicator
        extreme = abs(news_sent) > 0.7 or abs(social_sent) > 0.7
        features.append(float(extreme))
        
        # Sentiment regime
        avg_sent = (news_sent + social_sent) / 2
        regime = np.sign(avg_sent) if abs(avg_sent) > 0.3 else 0
        features.append(regime)
        
        # Contrarian trigger: extreme + divergence
        contrarian = float(extreme and abs(divergence) > 0.5)
        features.append(contrarian)
        
        # Normalize prices
        price_normalized = prices / prices[0] - 1 if prices[0] != 0 else prices
        
        full_features = np.concatenate([
            price_normalized,
            np.array(features)
        ])
        
        return torch.FloatTensor(full_features).to(self.device)
    
    def act(self, obs: AgentObservation, deterministic: bool = False) -> AgentAction:
        """Generate sentiment-based action."""
        self.last_observation = obs
        
        features = self.extract_features(obs)
        
        with torch.no_grad():
            sentiment, direction, confidence, contrarian_prob, _ = self.network(features.unsqueeze(0))
        
        # Adjust for contrarian signals
        contrarian = float(contrarian_prob.squeeze())
        if contrarian > 0.6:
            # Flip direction when sentiment is extreme and overdone
            direction = -direction * 0.5
            confidence = confidence * 0.8  # Reduce confidence
        
        if not deterministic:
            noise = torch.randn_like(sentiment) * 0.08
            sentiment = torch.clamp(sentiment + noise, 0, 1)
            confidence = torch.clamp(confidence + torch.randn_like(confidence) * 0.05, 0, 1)
        
        action = AgentAction(
            agent_id=self.agent_id,
            action_type="sentiment_signal",
            score=float(sentiment.squeeze()),
            direction=float(direction.squeeze()),
            confidence=float(confidence.squeeze()),
            metadata={
                'contrarian_detected': contrarian > 0.6,
                'contrarian_prob': contrarian,
                'sentiment_regime': obs.regime,
                'divergence': float(features[self.PRICE_HISTORY_LEN + 6])
            }
        )
        
        self.last_action = action
        self.action_history.append(action)
        
        # Broadcast to controller
        self.send_message(
            receiver="controller",
            msg_type=MessageType.SIGNAL,
            content={
                'score': action.score,
                'conviction': action.direction,
                'confidence': action.confidence,
                'contrarian': contrarian > 0.6,
                'implied_allocation_delta': action.score * action.direction * 0.04
            },
            priority=3 if contrarian > 0.6 else (2 if abs(action.direction) > 0.7 else 1)
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
        
        sentiment, direction, confidence, contrarian_prob, values = self.network(obs_batch)
        
        # Value loss
        value_loss = nn.MSELoss()(values, returns_t)
        
        # Policy loss with contrarian awareness
        policy_loss = -torch.mean(
            confidence * (1 - contrarian_prob * 0.3) * advantages_t * torch.log(confidence + 1e-8)
        )
        
        # Entropy
        entropy = -torch.mean(
            confidence * torch.log(confidence + 1e-8) +
            (1 - confidence) * torch.log(1 - confidence + 1e-8)
        )
        
        loss = value_loss + 0.5 * policy_loss - 0.01 * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()
        
        return {
            'value_loss': float(value_loss),
            'policy_loss': float(policy_loss),
            'entropy': float(entropy),
            'contrarian_rate': float(torch.mean((contrarian_prob > 0.6).float()))
        }
