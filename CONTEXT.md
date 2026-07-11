# CONTEXT.md — portfolio-lab

Domain glossary for dev-loop agents. Use these precise terms instead of paraphrasing.

## Strategy & Portfolio Construction

- **All-Season Portfolio** — Risk-parity-inspired allocation designed to perform across
  all economic regimes (growth/inflation/recession/disinflation). Core building blocks:
  equities (SPY), gold (GLD), long bonds (TLT), intermediate bonds (IEF).
- **Sharpe ratio** — Risk-adjusted return metric (excess return / volatility). The
  project's primary optimization target. Champion: 0.79 (SPY/GLD/TLT 46/38/16).
- **CAGR** — Compound Annual Growth Rate. Gross return before fees.
- **Max Drawdown (Max DD)** — Peak-to-trough decline. Constraint: ≤22% target.
- **Risk parity** — Allocate so each asset contributes equal risk (vol × weight).
- **Drift-based rebalancing** — Rebalance when any asset deviates ≥10% from target
  weight, rather than on a calendar schedule. Found to improve Sharpe (+0.04 vs annual).
- **Grid search** — Exhaustive parameter sweep (94 configs) to find optimal weights.
- **Rolling window** — Out-of-sample validation across sub-periods (9 windows tested).
- **Monte Carlo (FIRE)** — 1000 bootstrap simulations for retirement withdrawal rates.

## Overlays & Tactical Shifts

- **Overlay** — A tactical allocation shift applied on top of the base strategic
  portfolio. Each overlay has its own signal generator + allocation module.
- **Collar overlay** — Sell OTM call + buy OTM put, net premium near zero. VIX-gated
  (freezes above VIX 40). Targets max DD reduction.
- **Bond duration rotation** — Rotate bond sleeve (TLT/IEF/SHY) based on yield curve
  regime (steep/normal/flat/inverted) and real rate direction.
- **Crypto tactical allocation** — BTC/ETH momentum + vol-scaled position (max 5%
  portfolio). Funded from GLD.
- **Calendar seasonality** — Execution timing based on TOM, pre/post-holiday,
  quarter-end, OPEX windows. Improves execution by 5-15 bps annually.
- **TSMOM** — Time-series momentum (AQR-style). 12-month formation, 1-month skip,
  vol-scaled. Sharpe 0.96 standalone.
- **VIX term structure** — VIX/VIX3M/VIX6M slope analysis for tactical shifts.

## Ensemble & Signal Integration

- **EnsembleVoter** — Aggregates weighted signals from multiple sources into a
  directional vote. Sources: TSMOM, collar, bond duration, crypto, calendar,
  VIXY hedge, factor rotation, behavioral sentiment, stacking ensemble.
- **SignalSource** — Enum of signal origins with associated ensemble weights.
- **Combined Signal Orchestrator** — Multi-source aggregation with historical
  validation. Key finding: TSMOM alone (0.96) beats combined (0.93).
- **Signal stacking** — ML meta-learner (XGBoost) trained on 84 features from
  8 base signals. Directional accuracy improvement does not translate to
  meaningful Sharpe gains due to signal frequency constraints.

## AI / ML Agents (v2.51)

- **MARL** — Multi-Agent Reinforcement Learning. 5 specialized agents with a
  centralized critic (MAPPO training).
- **Agent types**: Analyst (fundamental/value), Sentiment (news/social, contrarian
  detection), Risk (VaR/CVaR monitoring), Execution (order timing, market impact),
  Controller (orchestration with centralized critic).
- **ML gate** — `PORTFOLIO_LAB_ENABLE_ML` env var. ML libs (torch 63MB, sklearn 78MB,
  hmmlearn 23MB) are blocked by default to prevent OOM on low-resource hosts.
  4-layer defense: collect_ignore → env var → import hook → ulimit.

## Regime & Risk

- **Regime detection** — Classifying market state (bull/bear/neutral/high_vol/crisis).
  HMM-LSTM detector trained on 26,225 samples with 5-state classification.
- **VIX regime** — Volatility buckets: NORMAL (<20), ELEVATED (20-30), STRESS (30-40),
  CRISIS (>40). Gates collar and VIXY hedge overlays.
- **VPIN** — Volume-synchronized Probability of Informed Trading. Microstructure
  toxicity metric. >0.5 triggers rebalance deferral.
- **CVaR / VaR** — Conditional Value at Risk (expected loss beyond VaR threshold).
  GARCH-filtered for volatility clustering.
- **Entropy monitor** — Shannon entropy + effective N + HHI for concentration risk.

## Data

- **prices.json** — Generated compact OHLCV data fetched from Yahoo Finance v8
  via `bun run fetch-data`. At this revision, the checked-in snapshot has 38
  symbols and 5,404 SPY rows from 2005-01-03 to 2026-06-26. Because this file is
  refreshed frequently, verify the current row/date/symbol coverage with:
  `node -e "const fs=require('fs');const p=JSON.parse(fs.readFileSync('public/data/prices.json','utf8'));console.log(p.SPY.length,p.SPY[0].d,p.SPY[p.SPY.length-1].d,Object.keys(p).length)"`.
- **market.db** — SQLite fallback for close-only data when Yahoo is unavailable.
- **Symbol universe**: AGG, DBC, EFA, FXA, FXB, FXC, FXE, FXF, FXY, GLD, IEF,
  MTUM, QQQ, QUAL, SHY, SPY, TLT, TMF, UBT, UDN, USMV, UUP, VBR, VLUE, VTI,
  VXUS, XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY, ^VIX3M.

## Project Conventions

- **Work item slugs**: `vXX-<feature>` pattern (e.g., `v292-etf-premium-monitor`)
- **Vault paths**: specs/plans go to the active SkillWiki vault returned by
  `skillwiki path`, under `projects/portfolio-lab/work/`; compound pages go
  in `projects/portfolio-lab/compound/`, ADRs in `projects/portfolio-lab/architecture/`.
  Do not use repo-local `wiki/` or `work/` directories.
- **Agent instructions**: slim `CLAUDE.md` / `AGENTS.md` (hard rules + wiki index only). Durable status/docs live in SkillWiki `projects/portfolio-lab/knowledge.md` and `compound/claude-md-agent-reference.md` — update those after features land, not a status dump in CLAUDE.md
- **No ML imports without explicit user request** — always default to `PORTFOLIO_LAB_ENABLE_ML=0`
- **Test safety**: `make test` runs the generated safe suite with ML disabled
  and a 3GB ulimit. Treat exact test counts as generated output from the latest
  run, not a static contract.
