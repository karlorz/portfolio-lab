#!/usr/bin/env python3
"""
Portfolio-Lab v2.51: MARL Trainer

Multi-Agent Reinforcement Learning training with PPO/MAPPO.
Implements centralized critic with value decomposition for
the portfolio management agent graph.

Training Architecture:
- Rollout collection: Agents interact with market environment
- Advantage estimation: GAE for stable gradients
- Centralized value update: Controller critic
- Decentralized policy updates: Per-agent PPO updates
- Consensus reward shaping: Agent agreement bonus
"""

import os
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
from collections import deque

# Conditional ML import — disabled by default to prevent OOM in test suites.
_ML_ENABLED = os.environ.get("PORTFOLIO_LAB_ENABLE_ML", "0") == "1"
if _ML_ENABLED:
    import torch
    import torch.nn as nn
else:
    from .base_agent import torch, nn

from .base_agent import BaseAgent, AgentObservation, AgentAction
from .agent_graph import AgentGraph
from .analyst_agent import AnalystAgent
from .sentiment_agent import SentimentAgent
from .risk_agent import RiskAgent
from .execution_agent import ExecutionAgent
from .controller_agent import ControllerAgent


@dataclass
class Transition:
    """Single step transition."""
    observation: AgentObservation
    actions: Dict[str, AgentAction]
    reward: float
    next_observation: AgentObservation
    done: bool
    
    # Per-agent metrics
    agent_values: Dict[str, float] = field(default_factory=dict)
    agent_log_probs: Dict[str, float] = field(default_factory=dict)


@dataclass
class RolloutBuffer:
    """Buffer for collecting training trajectories."""
    transitions: List[Transition] = field(default_factory=list)
    
    # Episode stats
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    
    def add(self, transition: Transition):
        """Add transition to buffer."""
        self.transitions.append(transition)
    
    def compute_returns(self, gamma: float = 0.99) -> List[float]:
        """Compute discounted returns (MC)."""
        returns = []
        R = 0
        for t in reversed(self.transitions):
            R = t.reward + gamma * R * (0 if t.done else 1)
            returns.insert(0, R)
        return returns
    
    def compute_gae(self, values: List[float], 
                    gamma: float = 0.99, 
                    lambda_: float = 0.95) -> Tuple[List[float], List[float]]:
        """
        Compute Generalized Advantage Estimation.
        
        Returns: (advantages, returns)
        """
        advantages = []
        gae = 0
        
        for t in reversed(range(len(self.transitions))):
            transition = self.transitions[t]
            
            if t == len(self.transitions) - 1:
                next_value = 0 if transition.done else values[t]
            else:
                next_value = values[t + 1]
            
            delta = transition.reward + gamma * next_value - values[t]
            gae = delta + gamma * lambda_ * (0 if transition.done else 1) * gae
            advantages.insert(0, gae)
        
        # Returns = advantages + values
        returns = [a + v for a, v in zip(advantages, values)]
        
        return advantages, returns
    
    def clear(self):
        """Clear buffer."""
        self.transitions.clear()
    
    def get_stats(self) -> Dict[str, float]:
        """Get buffer statistics."""
        if not self.transitions:
            return {}
        
        rewards = [t.reward for t in self.transitions]
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'total_steps': len(self.transitions)
        }


class MarketEnvironment:
    """
    Simplified market environment for training.
    
    In production, this interfaces with:
    - Live market data feeds
    - Broker APIs for paper/live trading
    - Risk management systems
    
    For training, uses historical backtest simulation.
    """
    
    def __init__(
        self,
        prices: Dict[str, np.ndarray],
        allocations: Dict[str, float] = None,
        transaction_cost: float = 0.001
    ):
        """
        Args:
            prices: Dict of price histories {ticker: prices}
            allocations: Starting allocation
            transaction_cost: Cost per trade (0.001 = 10bps)
        """
        self.prices = prices
        self.tickers = list(prices.keys())
        self.n_assets = len(self.tickers)
        self.transaction_cost = transaction_cost
        
        # Default allocation (46/38/16/0 for SPY/GLD/TLT/CASH)
        self.default_allocation = allocations or {
            'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16, 'CASH': 0.0
        }
        
        # Episode state
        self.current_step = 0
        self.portfolio_value = 100000.0
        self.allocation = self.default_allocation.copy()
        self.price_idx = 0
        
        # Track performance
        self.returns_history = []
        self.max_value = self.portfolio_value
    
    def reset(self, start_idx: int = 0) -> AgentObservation:
        """Reset environment."""
        self.current_step = 0
        self.price_idx = start_idx
        self.portfolio_value = 100000.0
        self.allocation = self.default_allocation.copy()
        self.returns_history = []
        self.max_value = self.portfolio_value
        
        return self._get_observation()
    
    def _get_observation(self) -> AgentObservation:
        """Build observation from current state."""
        # Get price history for each asset
        hist_len = 60  # Long enough for risk agent
        
        price_histories = {}
        for ticker in self.tickers:
            prices = self.prices[ticker]
            start = max(0, self.price_idx - hist_len)
            end = self.price_idx
            price_histories[ticker] = prices[start:end+1]
        
        # Use SPY as representative for observation
        spy_prices = price_histories.get('SPY', np.ones(hist_len))
        
        # Compute returns and volatility
        if len(spy_prices) > 1:
            returns = np.diff(spy_prices) / spy_prices[:-1]
            volatility = np.std(returns[-20:]) * np.sqrt(252) if len(returns) >= 20 else 0.15
        else:
            returns = np.array([0])
            volatility = 0.15
        
        obs = AgentObservation(
            prices=spy_prices,
            returns=returns,
            volatility=volatility,
            current_weights=self.allocation,
            portfolio_value=self.portfolio_value,
            cash_available=self.portfolio_value * self.allocation.get('CASH', 0)
        )
        
        return obs
    
    def step(self, action: Dict[str, Any]) -> Tuple[AgentObservation, float, bool, Dict]:
        """
        Execute one step.
        
        Args:
            action: Controller action with allocation recommendation
        
        Returns:
            (observation, reward, done, info)
        """
        # Extract allocation from controller action
        if 'allocation' in action:
            new_allocation = {
                self.tickers[i]: action['allocation'][i]
                for i in range(min(len(self.tickers), len(action['allocation'])))
            }
        else:
            new_allocation = self.allocation
        
        # Calculate rebalancing cost
        turnover = sum(abs(new_allocation.get(t, 0) - self.allocation.get(t, 0))
                      for t in self.tickers) / 2
        rebalance_cost = turnover * self.transaction_cost
        
        # Update allocation
        self.allocation = new_allocation
        
        # Advance price index
        self.price_idx += 1
        self.current_step += 1
        
        # Check if episode done
        done = self.price_idx >= len(self.prices[self.tickers[0]]) - 1
        
        # Calculate portfolio return
        portfolio_return = 0
        for ticker, weight in self.allocation.items():
            if ticker in self.prices and ticker != 'CASH':
                if self.price_idx < len(self.prices[ticker]):
                    price_return = (self.prices[ticker][self.price_idx] / 
                                   self.prices[ticker][self.price_idx - 1] - 1)
                    portfolio_return += weight * price_return
        
        # Cash return (risk-free, assume 0 for simplicity in training)
        portfolio_return += self.allocation.get('CASH', 0) * 0.0
        
        # Update portfolio value
        old_value = self.portfolio_value
        self.portfolio_value *= (1 + portfolio_return - rebalance_cost)
        
        # Calculate reward (Sharpe-like)
        self.returns_history.append(portfolio_return)
        
        # Update max value for drawdown
        if self.portfolio_value > self.max_value:
            self.max_value = self.portfolio_value
        
        drawdown = (self.max_value - self.portfolio_value) / self.max_value
        
        # Reward components
        base_reward = portfolio_return * 100  # Scale to reasonable range
        
        # Drawdown penalty
        dd_penalty = -max(0, drawdown - 0.10) * 10  # Penalty after 10% DD
        
        # Sharpe bonus if enough history
        if len(self.returns_history) >= 20:
            mean_ret = np.mean(self.returns_history[-20:])
            std_ret = np.std(self.returns_history[-20:]) + 1e-8
            sharpe = mean_ret / std_ret * np.sqrt(252)
            sharpe_bonus = sharpe * 0.1
        else:
            sharpe_bonus = 0
        
        # Rebalancing cost penalty
        cost_penalty = -rebalance_cost * 100
        
        reward = base_reward + dd_penalty + sharpe_bonus + cost_penalty
        
        # Get next observation
        next_obs = self._get_observation()
        
        info = {
            'portfolio_return': portfolio_return,
            'drawdown': drawdown,
            'rebalance_cost': rebalance_cost,
            'portfolio_value': self.portfolio_value,
            'turnover': turnover
        }
        
        return next_obs, reward, done, info


class MARLTrainer:
    """
    Multi-Agent RL trainer with MAPPO.
    
    Implements:
    - Rollout collection with agent graph
    - Centralized value estimation (controller critic)
    - PPO updates per agent
    - Consensus reward shaping
    """
    
    def __init__(
        self,
        agent_graph: AgentGraph,
        env: MarketEnvironment,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lambda_: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cpu"
    ):
        self.agent_graph = agent_graph
        self.env = env
        self.device = device
        
        # PPO hyperparameters
        self.gamma = gamma
        self.lambda_ = lambda_
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Training state
        self.episode_count = 0
        self.step_count = 0
        self.buffer = RolloutBuffer()
        
        # Metrics
        self.metrics_history = deque(maxlen=100)
        self.best_sharpe = -np.inf
    
    def collect_rollout(self, max_steps: int = 252) -> Dict[str, Any]:
        """
        Collect one episode rollout.
        
        Returns episode statistics.
        """
        obs = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            # Execute agent graph
            actions = self.agent_graph.execute_step(obs)
            
            # Get controller action (final allocation)
            controller_action = actions.get('controller')
            
            # Step environment
            next_obs, reward, done, info = self.env.step(
                controller_action.metadata if controller_action else {}
            )
            
            # Get values from agents
            agent_values = {}
            for agent_id, agent in self.agent_graph.agents.items():
                agent_values[agent_id] = agent.compute_value(obs)
            
            # Create transition
            transition = Transition(
                observation=obs,
                actions=actions,
                reward=reward,
                next_observation=next_obs,
                done=done,
                agent_values=agent_values
            )
            
            self.buffer.add(transition)
            
            episode_reward += reward
            episode_length += 1
            obs = next_obs
            
            if done:
                break
        
        # Episode stats
        returns = [t.reward for t in self.buffer.transitions[-episode_length:]]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(len(returns)) if len(returns) > 1 else 0
        
        stats = {
            'episode': self.episode_count,
            'length': episode_length,
            'total_reward': episode_reward,
            'final_value': info.get('portfolio_value', 0),
            'sharpe': sharpe,
            'max_drawdown': info.get('drawdown', 0)
        }
        
        self.episode_count += 1
        self.buffer.episode_rewards.append(episode_reward)
        self.buffer.episode_lengths.append(episode_length)
        
        return stats
    
    def update(self, batch_size: int = 32) -> Dict[str, Any]:
        """
        Update all agents using collected rollout.
        
        Returns training metrics.
        """
        if len(self.buffer.transitions) < batch_size:
            return {}
        
        # Compute values using controller critic
        values = [t.agent_values.get('controller', 0) for t in self.buffer.transitions]
        
        # GAE
        advantages, returns = self.buffer.compute_gae(
            values, self.gamma, self.lambda_
        )
        
        # Normalize advantages
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        advantages = [(a - adv_mean) / adv_std for a in advantages]
        
        # Group transitions by agent for updates
        agent_observations: Dict[str, List] = {k: [] for k in self.agent_graph.agents}
        agent_returns: Dict[str, List] = {k: [] for k in self.agent_graph.agents}
        agent_advantages: Dict[str, List] = {k: [] for k in self.agent_graph.agents}
        
        for i, transition in enumerate(self.buffer.transitions):
            for agent_id, action in transition.actions.items():
                agent_observations[agent_id].append(transition.observation)
                agent_returns[agent_id].append(returns[i])
                agent_advantages[agent_id].append(advantages[i])
        
        # Update each agent
        update_stats = {}
        
        for agent_id, agent in self.agent_graph.agents.items():
            if not hasattr(agent, 'train_step'):
                continue
            
            if len(agent_observations[agent_id]) == 0:
                continue
            
            stats = agent.train_step(
                observations=agent_observations[agent_id],
                actions=list(self.buffer.transitions[0].actions.values()),
                returns=agent_returns[agent_id],
                advantages=agent_advantages[agent_id]
            )
            
            update_stats[agent_id] = stats
        
        # Clear buffer after update
        self.buffer.clear()
        
        return update_stats
    
    def train(self, n_episodes: int = 100, log_interval: int = 10) -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns final training stats.
        """
        print(f"Starting MARL training for {n_episodes} episodes...")
        print(f"Agents: {list(self.agent_graph.agents.keys())}")
        print(f"Environment: {self.env.n_assets} assets")
        
        all_stats = []
        
        for episode in range(n_episodes):
            # Collect rollout
            stats = self.collect_rollout()
            all_stats.append(stats)
            
            # Update after each episode
            if episode > 0 and episode % 4 == 0:
                update_stats = self.update()
            else:
                update_stats = {}
            
            # Logging
            if episode % log_interval == 0:
                recent_rewards = [s['total_reward'] for s in all_stats[-log_interval:]]
                recent_sharpes = [s['sharpe'] for s in all_stats[-log_interval:] if s['sharpe'] != 0]
                
                print(f"Episode {episode}: "
                      f"Reward={stats['total_reward']:.2f}, "
                      f"Sharpe={stats['sharpe']:.3f}, "
                      f"Value=${stats['final_value']:,.0f}")
                
                if update_stats:
                    for agent_id, s in update_stats.items():
                        if s and 'value_loss' in s:
                            print(f"  {agent_id}: v_loss={s['value_loss']:.4f}")
            
            # Track best
            if stats['sharpe'] > self.best_sharpe:
                self.best_sharpe = stats['sharpe']
        
        print(f"\nTraining complete. Best Sharpe: {self.best_sharpe:.3f}")
        
        return {
            'episodes': n_episodes,
            'best_sharpe': self.best_sharpe,
            'final_stats': all_stats[-1] if all_stats else {}
        }
    
    def save(self, path: Path):
        """Save training state."""
        self.agent_graph.save(path)
        
        # Save training config
        config = {
            'gamma': self.gamma,
            'lambda': self.lambda_,
            'clip_epsilon': self.clip_epsilon,
            'episode_count': self.episode_count,
            'best_sharpe': self.best_sharpe
        }
        
        config_path = Path(path) / "trainer_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load(self, path: Path):
        """Load training state."""
        self.agent_graph.load(path)
        
        config_path = Path(path) / "trainer_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                self.best_sharpe = config.get('best_sharpe', -np.inf)
