---
kind: implementation
version: 2.22
status: ready
priority: critical
dependencies:
  - v221-volatility-parity-convexity
research_session: 00a385d07168
research_synthesis: ~/projects/portfolio-lab/work/2026-05-12-q3-research-synthesis/compound-synthesis.md
---

# v2.22: LLM Sentiment Analysis Module

## Objective
Implement institutional-grade LLM sentiment analysis for portfolio timing based on Q3 2026 research findings. Target: GPT-4o-mini for cost-effective high-volume processing with Sharpe 3.05+ potential.

## Research Basis
From `/root/projects/portfolio-lab/work/2026-05-12-q3-research-synthesis/compound-synthesis.md`:
- GPT-4o-mini: 76% accuracy, Sharpe 3.05, $0.75/$4.50 per 1M tokens
- Cumulative returns: +355% (Aug 2021 - Jul 2023, net of 10bps costs)
- Chain-of-Thought (CoT) prompting dominates standard approaches
- Finance-specialized models (FinBERT) underperform general LLMs with proper prompting

## Implementation Tasks

### 1. LLM Client Infrastructure (src/llm/sentiment_client.py)
- [ ] OpenAI GPT-4o-mini client with retry logic
- [ ] Claude 3.5 Sonnet client for complex documents
- [ ] Prompt caching layer (~90% savings on repeated prompts)
- [ ] Tiered routing: nano/mini for volume, larger for edge cases
- [ ] Cost tracking and budget enforcement
- [ ] Rate limiting and backoff handling

### 2. Earnings Call Analysis (src/llm/earnings_analyzer.py)
- [ ] Transcript ingestion (Seeking Alpha, S&P Global APIs)
- [ ] Chunking strategy: CEO remarks, CFO review, Q&A segments
- [ ] Aspect-based sentiment analysis (ABSA):
  - Revenue guidance sentiment (-1 to +1)
  - Margin outlook sentiment
  - Risk factor sentiment
  - Management tone/hedging detection
- [ ] Quarter-over-quarter tone shift detection
- [ ] Structured JSON output with confidence scores
- [ ] CLI: `earnings analyze <ticker> --quarter Q4-2025`

### 3. Fed Communications Analysis (src/llm/fed_analyzer.py)
- [ ] FOMC statement parsing
- [ ] Meeting minutes section extraction
- [ ] Hawk-dove scoring (-1 to +1 scale)
- [ ] Context-aware classification (e.g., "strong labor" = dovish in weak economy)
- [ ] Chair speech vs other speaker weighting
- [ ] Uncertainty quantification
- [ ] CLI: `fed analyze --date 2026-05-07 --type minutes`

### 4. News Sentiment Pipeline (src/llm/news_pipeline.py)
- [ ] Reuters/Dow Jones headline ingestion
- [ ] Topic filtering (macro, sector-specific)
- [ ] Rolling sentiment aggregation (1h, 4h, 1d windows)
- [ ] Source quality weighting (Reuters > social media)
- [ ] Spike detection (3-sigma sentiment moves)
- [ ] CLI: `news stream --topics SPY,GLD,TLT --window 4h`

### 5. Signal Integration (src/llm/signal_integrator.py)
- [ ] Aggregate earnings, Fed, news signals into composite score
- [ ] Time-decay weighting (recent > historical)
- [ ] Confidence thresholding (only act on >0.7 confidence)
- [ ] Regime-dependent signal weights:
  - Normal: Earnings 50%, News 30%, Fed 20%
  - FOMC week: Fed 60%, Earnings 25%, News 15%
- [ ] Integration with ensemble_voter.py
- [ ] CLI: `llm signal --composite --format json`

### 6. Cost Management & Optimization
- [ ] Daily cost budget: $50 (configurable)
- [ ] Prompt templates with caching keys
- [ ] Batch API usage for non-real-time processing (50% discount)
- [ ] JSON mode for structured output (token reduction)
- [ ] Cost reporting: daily/weekly/monthly dashboards
- [ ] Alert on 80% budget consumption

### 7. Backtesting Framework (src/llm/backtest.py)
- [ ] Historical transcript replay (2020-2026)
- [ ] Walk-forward validation with 3-month embargo
- [ ] Transaction cost modeling (10bps)
- [ ] Sharpe, max DD, win rate calculation
- [ ] Comparison vs FinBERT baseline
- [ ] CLI: `llm backtest --start 2024-01-01 --end 2026-05-01`

### 8. Testing & Validation
- [ ] Accuracy validation on labeled earnings dataset
- [ ] Fed communication backtest vs actual market moves
- [ ] Live paper trading integration
- [ ] Cost tracking validation
- [ ] Performance attribution analysis

## Files to Create/Modify
- Create: `src/llm/__init__.py`
- Create: `src/llm/sentiment_client.py`
- Create: `src/llm/earnings_analyzer.py`
- Create: `src/llm/fed_analyzer.py`
- Create: `src/llm/news_pipeline.py`
- Create: `src/llm/signal_integrator.py`
- Create: `src/llm/backtest.py`
- Create: `config/llm_config.yaml`
- Modify: `src/strategy/ensemble_voter.py` (integrate LLM signals)
- Modify: `src/components/LiveDashboard.tsx` (add sentiment panel)

## Acceptance Criteria
- [ ] GPT-4o-mini client operational with <2s latency
- [ ] Earnings ABSA achieves >75% accuracy on validation set
- [ ] Fed hawk-dove correlation to market moves >0.6
- [ ] Composite signal Sharpe >2.0 in backtest (vs FinBERT >1.5)
- [ ] Daily processing cost <$50 at 10,000 documents/day
- [ ] All CLI commands have --help and error handling
- [ ] Integration tests pass with mock LLM responses

## Performance Targets
| Metric | Target | Research Benchmark |
|--------|--------|-------------------|
| Accuracy | >75% | GPT-4o-mini: 76% |
| Sharpe | >2.0 | GPT-3/OPT: 3.05 |
| Latency | <2s | 0.5-2s range |
| Cost/day | <$50 | $1.20-1.50 baseline |
| Win Rate | >60% | 66-90% (condor ref) |

## ETA
10-14 hours

## References
- `/root/projects/portfolio-lab/work/2026-05-12-q3-research-synthesis/compound-synthesis.md` Section 1
- arXiv 2024-2025 LLM financial sentiment papers
- JPMorgan "Hawk-Dove Score" methodology
- S&P Global earnings sentiment research

## Budget Planning
| Component | Monthly Cost |
|-----------|-------------|
| GPT-4o-mini (primary) | $1,000 |
| Claude 3.5 (complex docs) | $500 |
| API caching savings | -$1,350 |
| **Net Monthly** | **~$150** |

---

## Implementation Status
- [ ] Phase 1: Client infrastructure (2-3h)
- [ ] Phase 2: Earnings analyzer (3-4h)
- [ ] Phase 3: Fed analyzer (2-3h)
- [ ] Phase 4: News pipeline (2-3h)
- [ ] Phase 5: Signal integration (2h)
- [ ] Phase 6: Backtesting (2h)
- [ ] Phase 7: Dashboard integration (1-2h)
