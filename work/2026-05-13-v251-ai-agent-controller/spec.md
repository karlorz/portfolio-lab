# AI Agent Controller v2.51 - Work Item

**Status:** In Progress  
**Created:** 2026-05-13 09:00 UTC  
**Assigned:** Auto (system)  
**Priority:** Critical  
**Component:** AI Engine v2.51  

## Objective
Implement Multi-Agent Reinforcement Learning (MARL) portfolio controller with 5 specialized agents coordinated via LangGraph-style agent graph. Integrate with v2.24 signal integrator.

## Research Basis
Based on multi-agent RL for portfolio management literature:
- 5 specialized agents: Analyst, Sentiment, Risk, Execution, Controller
- State aggregation with agent-specific observation spaces
- PPO-based training with centralized critic, decentralized actors
- LangGraph-inspired directed graph for agent communication

## Requirements

### Agent Architecture
1. **Analyst Agent** (value/fundamental)
   - Obs: earnings estimates, ratios, growth signals
   - Action: value score [0,1], conviction [-1,1]
   
2. **Sentiment Agent** (news/social)
   - Obs: sentiment features, volume, anomalies
   - Action: sentiment score [0,1], direction [-1,1]
   
3. **Risk Agent** (volatility/drawdown)
   - Obs: VaR, CVaR, max drawdown, tail risk
   - Action: risk budget [0.5,1.5], hedging level [0,1]
   
4. **Execution Agent** (timing/routing)
   - Obs: spread, volume profile, market impact
   - Action: urgency [0,1], slice size [0.1,0.5]
   
5. **Controller Agent** (orchestration)
   - Obs: all agent outputs, portfolio state
   - Action: agent weights, final allocation

### Integration Points
- Consume signals from v2.24 signal_integrator.py
- Output allocations to execution layer
- Support both live trading and backtest modes

## Implementation Tasks
1. [ ] Create agent network architectures (PyTorch)
2. [ ] Build agent graph communication layer
3. [ ] Implement centralized critic with value decomposition
4. [ ] Add training loop with PPO/MAPPO
5. [ ] Integrate with signal integrator
6. [ ] Create agent state persistence
7. [ ] Add explainability (agent contribution tracking)

## Success Metrics
- Sharpe >= 0.85 (vs 0.79 champion baseline)
- Agent consensus accuracy >70%
- Inference latency <50ms
- Training convergence <500 episodes

## Files to Create
- src/agents/analyst_agent.py
- src/agents/sentiment_agent.py  
- src/agents/risk_agent.py
- src/agents/execution_agent.py
- src/agents/controller_agent.py
- src/agents/agent_graph.py
- src/agents/marl_trainer.py
- src/agents/ai_controller.py (main entry)
