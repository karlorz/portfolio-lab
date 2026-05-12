# v2.50-2.51 Deep Research Synthesis

**Date:** 2026-05-13  
**Session IDs:** df31233fa23a (QML), 38fe7bcfaacc (AI Agents)

## 1. Quantum ML Portfolio Optimization (QAOA/VQE)

**Key Findings:**
- **QAOA Performance**: ~97.3% classical solution quality on simulators; Sharpe ratios up to 1.81 vs simulated annealing baselines
- **30-40% better risk-return** in multi-asset tests with higher-order formulations (skewness/kurtosis)
- **Optimal Scale**: 10-20 assets (NISQ era constraint); 50+ assets in subproblem decomposition
- **Hybrid Architecture**: Classical preprocessing (covariance, constraints) + quantum sampling (QAOA/VQE) + post-processing

**Implementation Strategy:**
```
Classical Frontend: Risk model, constraints, covariance
↓ QUBO/Ising formulation
↓ Quantum Sampling (QAOA p=1-3 or VQE)
↓ Classical Post-processing: Local search, refinement
↓ Portfolio Weights
```

**Competitive Edge:**
- Discrete asset selection (cardinality constraints)
- Higher-moment optimization (skewness/kurtosis)
- Better convergence in multi-constraint scenarios

---

## 2. AI Agent Trading Systems (MARL)

**Key Findings:**
- **MARL outperforms single-agent RL by 15-20%** in returns through specialized roles
- **Hybrid LLM+RL**: LLM agents process unstructured data; RL agents make allocation decisions
- **Leading Framework**: FinRL (open-source DRL for trading/portfolio)
- **Algorithms**: PPO (stable), SAC/DDPG (continuous actions)

**Agent Architecture Pattern:**
```
Controller Agent
├── Analyst Agent (fundamental/technical)
├── Sentiment Agent (LLM-processed news)
├── Risk Agent (volatility/regime)
└── Execution Agent (trade sizing)
```

**State Space:**
- Market data (prices, volumes)
- Technical indicators (regime detection)
- LLM sentiment scores
- Current portfolio weights
- Risk metrics (VaR, CVaR)

**Reward Function:**
```
Reward = Δ(Portfolio Value) - λ × Risk_Penalty - μ × Transaction_Costs
```

---

## 3. Synergy: QML + MARL for Portfolio-Lab v2.5x

**Integrated Architecture:**
```
MARL Controller (Policy Network)
├── Analyst Agent → Uses QML for asset selection (10-20 universe)
├── Sentiment Agent → LLM feature extraction
├── Risk Agent → Vol targeting (existing v2.42b)
└── Execution Agent → Order routing with cost optimization
```

**Implementation Roadmap:**
1. **v2.50** - Quantum-classical hybrid optimizer (Qiskit/Pennylane simulation)
2. **v2.51** - Multi-agent RL controller with LLM integration
3. **v2.52** - Combined system: MARL for allocation, QML for sub-universe selection
4. **v2.53** - Autonomous operation with Hermes cron + circuit breakers

**Risk Mitigation:**
- Non-stationarity: Online learning, regime detection
- Overfitting: Walk-forward validation, paper trading gates
- Interpretability: Attention visualization, SHAP values
- Costs: Explicit transaction cost modeling in reward function

---

## References
- QAOA benchmarking: arxiv.org/html/2602.14827v1
- VQE Dicke states: nature.com/articles/s41598-026-36333-4
- MARL trading: dl.acm.org/doi/10.1145/3746709.3746915
- FinRL framework: finrl.readthedocs.io
- LLM+RL hybrid: arxiv.org/html/2508.11152v1
