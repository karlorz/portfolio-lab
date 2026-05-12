---
skill_compatibility: 2
version: "2.0"
kind: feature
slug: v250-quantum-ml
status: in-progress
progress: 0
started: 2026-05-13
references: ["df31233fa23a"]
---

# v2.50 Quantum ML Portfolio Optimization

## Goal
Implement quantum-classical hybrid optimization for portfolio allocation using QAOA/VQE algorithms for 10-20 asset universes.

## Research Insights (Session df31233fa23a)
- QAOA achieves ~97.3% classical solution quality on simulators
- Backtest Sharpe ratios up to 1.81 vs classical baselines
- 30-40% better risk-return in multi-asset tests
- Best suited for 10-20 assets (NISQ era limitation)
- Higher-order formulations (skewness/kurtosis) show advantage

## Implementation
- Classical preprocessing + quantum sampling hybrid
- QUBO formulation for discrete asset selection
- Integration with existing risk parity framework
- Fallback to classical when quantum unavailable

## Files
- `src/optimization/quantum_hybrid.py`

## Dependencies
- Qiskit or Pennylane simulation backends
- Existing covariance/preference matrix pipeline
