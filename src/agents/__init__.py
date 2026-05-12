"""
Portfolio-Lab v2.51: Multi-Agent Reinforcement Learning (MARL) Module

Specialized agents coordinated via LangGraph-style communication:
- AnalystAgent: Fundamental/value analysis
- SentimentAgent: News/social sentiment  
- RiskAgent: Drawdown/volatility monitoring
- ExecutionAgent: Trade timing optimization
- ControllerAgent: Master orchestration

Usage:
    from src.agents import AIController
    
    controller = AIController()
    result = controller.infer(portfolio_value=100000)
"""

__version__ = "2.51.0"

from .base_agent import BaseAgent, AgentType, AgentObservation, AgentAction, AgentMessage, MessageType
from .analyst_agent import AnalystAgent
from .sentiment_agent import SentimentAgent
from .risk_agent import RiskAgent
from .execution_agent import ExecutionAgent
from .controller_agent import ControllerAgent, AgentConsensus
from .agent_graph import AgentGraph
from .marl_trainer import MARLTrainer, MarketEnvironment
from .ai_controller import AIController, main

__all__ = [
    'BaseAgent',
    'AgentType', 
    'AgentObservation',
    'AgentAction',
    'AgentMessage',
    'MessageType',
    'AnalystAgent',
    'SentimentAgent',
    'RiskAgent',
    'ExecutionAgent',
    'ControllerAgent',
    'AgentConsensus',
    'AgentGraph',
    'MARLTrainer',
    'MarketEnvironment',
    'AIController',
    'main',
]
