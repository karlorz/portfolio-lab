# v2.22 Implementation Plan: LLM Sentiment Analysis

## Research Source
`/root/projects/portfolio-lab/work/2026-05-12-q3-research-synthesis/compound-synthesis.md`

## Phase 1: Client Infrastructure (2-3 hours)
1. Create `src/llm/sentiment_client.py` with OpenAI/Claude clients
2. Implement retry logic and rate limiting
3. Add prompt caching layer
4. Cost tracking and budget enforcement
5. Test with sample prompts

## Phase 2: Earnings Analyzer (3-4 hours)
1. Create `src/llm/earnings_analyzer.py`
2. Implement transcript chunking strategy
3. Build ABSA prompts (revenue, margins, guidance, risks)
4. Add tone shift detection
5. CLI interface: `earnings analyze <ticker>`
6. Test on historical transcripts

## Phase 3: Fed Analyzer (2-3 hours)
1. Create `src/llm/fed_analyzer.py`
2. FOMC statement/minutes parsing
3. Hawk-dove scoring implementation
4. Context-aware classification
5. CLI interface: `fed analyze --date <date>`

## Phase 4: News Pipeline (2-3 hours)
1. Create `src/llm/news_pipeline.py`
2. Headline ingestion (RSS/API)
3. Topic filtering and aggregation
4. Rolling sentiment windows
5. Spike detection (3-sigma)
6. CLI interface: `news stream --topics <list>`

## Phase 5: Signal Integration (2 hours)
1. Create `src/llm/signal_integrator.py`
2. Composite score aggregation
3. Time-decay weighting
4. Regime-dependent weights
5. Integration with ensemble_voter.py
6. CLI interface: `llm signal --composite`

## Phase 6: Backtesting (2 hours)
1. Create `src/llm/backtest.py`
2. Historical replay framework
3. Walk-forward validation
4. Cost and slippage modeling
5. Performance attribution
6. CLI interface: `llm backtest --start <date> --end <date>`

## Phase 7: Dashboard & Integration (1-2 hours)
1. Add sentiment panel to LiveDashboard.tsx
2. Signal strength visualization
3. Cost tracking display
4. Integration tests
5. Documentation update

## Current Status
- [x] Research synthesis: COMPLETED (13386 bytes)
- [x] Work item created with spec.md, plan.md
- [x] Phase 1: COMPLETED - sentiment_client.py (488 lines, committed)
- [x] Phase 2: COMPLETED - earnings_analyzer.py (860 lines, committed)
- [x] Phase 3: COMPLETED - fed_analyzer.py (918 lines, committed)
- [x] Phase 4: IN PROGRESS - news_pipeline.py
- [x] Phase 5: IN PROGRESS - signal_integrator.py
- [ ] Phase 6: DEFERRED - backtest.py (analytics layer)
- [ ] Phase 7: DEFERRED - Dashboard integration (analytics layer)
- [x] Git push: COMPLETED - 32 commits pushed to origin/main

## Blockers
None