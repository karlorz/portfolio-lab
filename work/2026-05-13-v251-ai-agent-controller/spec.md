---
skill_compatibility: 2
version: "2.0"
kind: feature
slug: v251-ai-agent-controller
status: in-progress
progress: 0
started: 2026-05-13
references: ["38fe7bcfaacc"]
---

# v2.51 AI Agent Controller (MARL)

## Goal
Multi-agent reinforcement learning controller for autonomous portfolio management with LLM+RL hybrid intelligence.

## Research Insights (Session 38fe7bcfaacc)
- MARL outperforms single-agent RL by 15-20% in returns
- Hybrid LLM+RL adds reasoning over unstructured data
- Framework: FinRL-inspired architecture
- Key algorithms: PPO, SAC, DDPG for continuous actions
- State: prices, indicators, sentiment, current weights
- Reward: Δ(portfolio value) - λ × risk - costs

## Agent Architecture
1. **Analyst Agent** - Fundamental/technical analysis
2. **Sentiment Agent** - LLM-processed news/social
3. **Risk Agent** - Volatility/regime monitoring
4. **Execution Agent** - Trade sizing/timing

## Implementation
- LangGraph-style agent coordination
- Hermes cron integration for continuous operation
- Paper trading → live graduation gates

## Files
- `src/agents/controller.py`
- `src/agents/analyst.py`
- `src/agents/sentiment.py`
- `src/agents/risk.py`
- `src/agents/executor.py`
