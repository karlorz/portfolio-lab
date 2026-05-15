"""
Portfolio-Lab v2.51: Multi-Agent Reinforcement Learning (MARL) Module

Specialized agents coordinated via LangGraph-style communication:
- AnalystAgent: Fundamental/value analysis
- SentimentAgent: News/social sentiment
- RiskAgent: Drawdown/volatility monitoring
- ExecutionAgent: Order timing optimization
- ControllerAgent: Master orchestration

Usage:
    from src.agents import AIController

    controller = AIController()
    result = controller.infer(portfolio_value=100000)

Lazy imports: agent submodules are loaded on-demand to prevent ML library
(torch) imports during test discovery when PORTFOLIO_LAB_ENABLE_ML=0.
"""

__version__ = "2.51.0"

from .base_agent import BaseAgent, AgentType, AgentObservation, AgentAction, AgentMessage, MessageType

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


def _lazy_import(name):
    """Lazily import agent submodules to avoid ML import cascade."""
    import importlib
    _mapping = {
        'AnalystAgent': '.analyst_agent',
        'SentimentAgent': '.sentiment_agent',
        'RiskAgent': '.risk_agent',
        'ExecutionAgent': '.execution_agent',
        'ControllerAgent': '.controller_agent',
        'AgentConsensus': '.controller_agent',
        'AgentGraph': '.agent_graph',
        'MARLTrainer': '.marl_trainer',
        'MarketEnvironment': '.marl_trainer',
        'AIController': '.ai_controller',
        'main': '.ai_controller',
    }
    if name in _mapping:
        mod = importlib.import_module(__name__ + _mapping[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Support lazy attribute access: from src.agents import AnalystAgent
import sys as _sys
_module = _sys.modules[__name__]

def __getattr__(name):
    try:
        return _lazy_import(name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")