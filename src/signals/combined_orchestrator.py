# Combined Signal Orchestrator v2.54

Portfolio signal aggregation combining TSMOM (v2.52) + HMM Regime (v2.53) + base integrator (v2.51).

## Signal Weights

| Source | Weight | Purpose |
|--------|--------|---------|
| TSMOM Overlay | 0.35 | Time-series momentum (AQR style) |
| HMM Regime | 0.30 | Market state classification |
| AI Agent | 0.15 | MARL controller decisions |
| Macro | 0.10 | Fed policy, rates |
| Sentiment | 0.05 | News/social signals |
| Technical | 0.05 | Traditional indicators |

## Combined Allocation Formula

allocation_ticker = base + (TSMOM_delta * 0.35) + (Regime_shift * 0.30) + ...

## Current Status (2026-05-13)

- SPY: Neutral regime, positive TSMOM -> slight overweight
- GLD: Neutral regime, positive TSMOM -> maintain  
- TLT: Neutral regime, weak TSMOM -> slight underweight

## CLI Usage

```bash
python -m src.signals.combined_orchestrator status
python -m src.signals.combined_orchestrator recommend --portfolio 46/38/16
```
