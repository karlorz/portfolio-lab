---
title: "Regime Transition Forecaster"
status: in-progress
kind: feature
priority: high
created: 2026-05-29
---

# Regime Transition Forecaster

## Spec

Create `src/regime/regime_transition_forecaster.py` that:

1. **Empirical transition matrix** — compute 5×5 probability matrix from historical regime label sequences. Entry (i,j) = P(regime_j at t+1 | regime_i at t).

2. **Regime persistence modeling** — exponential survival model per regime with known persistence parameters:
   - NORMAL: 7.6 days
   - CRISIS: 9.9 days
   - LOW_VOL: 10.0 days
   - HIGH_VOL: 7.1 days
   - RECOVERY: 1.4 days

3. **Forward-looking regime probabilities** — given current regime and transition matrix, forecast probability distribution over next N days via matrix power.

4. **Integration point** — `get_regime_forecast(current_regime, horizon_days)` returns Dict[str, float] mapping regime names to probabilities.

## Data flow

- Input: list of historical regime labels (strings) from TwoStageKMeansRegime.predict_regime_names()
- Output: RegimeForecast dataclass with transition_matrix, persistence_params, forecast_probs

## Expected impact

+0.01-0.03 Sharpe through earlier regime-adaptive allocation shifts.

## Interview Summary (native)

- **Scope**: Feature + tests — regime transition probability modeling with full test coverage
- **Constraints**: Must match existing patterns in src/regime/ (numpy-only, no ML deps, dataclass results)
- **Acceptance**: Tests pass + integration with DashboardGenerator
