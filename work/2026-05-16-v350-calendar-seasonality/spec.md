---
type: work-item
project: portfolio-lab
version: v3.50
title: Calendar Seasonality Overlay for Rebalancing Timing
created: 2026-05-16
status: ready-for-implementation
priority: medium
tags: [seasonality, execution, rebalancing, calendar-effects, implementation]
---

# v3.50: Calendar Seasonality Overlay

## Overview

Implement calendar-based seasonality filters for rebalancing execution timing. Uses well-documented market anomalies (Turn-of-Month, Holiday effects) to delay rebalances during unfavorable windows, improving execution quality.

**Expected Impact**: +0.01-0.02 Sharpe through 5-15 bps better execution annually  
**Risk**: Low (execution-only, no allocation changes)  
**Complexity**: Medium (requires historical backtest validation)

## Background

Deep research completed: `compound/calendar-seasonality-effects-quant-2026.md`

Key findings:
- **Turn-of-Month (TOM)**: Last trading day + first 3 days of month show +0.5% excess returns
- **Holiday Effects**: Pre-holiday returns 5-10x normal (Thanksgiving strongest at 9.3x)
- **Monday Effect**: Historically weakest day (decaying but persistent)
- **Quarter-End**: Window dressing creates artificial price pressure

## Implementation Plan

### Phase 1: Signal Module (3 hours)

Create `src/signals/calendar_seasonality.py`:

```python
class CalendarSeasonalitySignal:
    """
    Calendar-based seasonality for rebalancing timing.
    
    Modifies rebalancing urgency based on documented seasonal effects:
    - Turn-of-Month (TOM): Last trading day + first 3 days
    - Pre-Holiday: Day before major US market holidays
    - Quarter-End: Last week of quarter (window dressing)
    - Monday: Historically weaker returns
    - Pre-FOMC: Day before Fed announcements
    """
    
    URGENCY_MODIFIERS = {
        'tom_window': 0.70,      # Turn-of-month
        'pre_holiday': 0.50,    # Day before holiday
        'quarter_end': 0.60,    # Last week of quarter
        'monday': 0.80,         # Monday effect
        'pre_fomc': 0.65,       # Day before Fed announcement
        'december': 0.75,       # Tax-loss season
    }
    
    HOLIDAYS = [
        'New Years', 'MLK Day', 'Presidents Day', 'Good Friday',
        'Memorial Day', 'Juneteenth', 'Independence Day', 'Labor Day',
        'Thanksgiving', 'Christmas'
    ]
    
    def get_urgency_modifier(self, date: datetime) -> float:
        """Return urgency modifier (0.0-1.0) for given date."""
        # Check all windows, return minimum modifier
        pass
    
    def is_tom_window(self, date: datetime) -> bool:
        """Check if date is in Turn-of-Month window."""
        pass
    
    def is_pre_holiday(self, date: datetime) -> bool:
        """Check if date is trading day before market holiday."""
        pass
    
    def is_quarter_end(self, date: datetime) -> bool:
        """Check if date is in last week of quarter."""
        pass
```

**Requirements**:
- [ ] Implement all window detection methods
- [ ] Handle trading calendar (skip weekends, holidays)
- [ ] Unit tests: 20+ scenarios covering all windows
- [ ] Performance: <1ms per call

### Phase 2: Integration (2 hours)

Modify `src/execution/rebalance_scheduler.py`:

```python
def calculate_optimal_rebalance_window(
    urgency: float,
    drift_scores: Dict[str, float],
    calendar_signal: Optional[CalendarSeasonalitySignal] = None
) -> RebalanceWindow:
    """
    Calculate optimal rebalancing window considering:
    1. Urgency level (0.0-1.0)
    2. Calendar seasonality effects
    3. VPIN microstructure toxicity
    4. Intraday seasonality (existing)
    """
    # Apply calendar modifier
    if calendar_signal:
        modifier = calendar_signal.get_urgency_modifier(datetime.now())
        adjusted_urgency = urgency * modifier
    else:
        adjusted_urgency = urgency
    
    # Continue with existing logic...
```

**Integration Points**:
- [ ] Import and instantiate CalendarSeasonalitySignal
- [ ] Apply modifier to urgency calculation
- [ ] Log calendar-based adjustments
- [ ] Ensure backward compatibility (disabled if module fails)

### Phase 3: Backtest Validation (4 hours)

Create `src/backtest/calendar_seasonality_backtest.py`:

```python
"""
Backtest: Rebalancing with Calendar Seasonality Overlay (2015-2025)

Compare three strategies:
1. Baseline: Drift-based rebalancing only
2. VPIN-only: Drift + VPIN timing
3. Full: Drift + VPIN + Calendar seasonality

Metrics:
- Execution slippage vs. target price
- Number of rebalances deferred
- Sharpe ratio impact
"""
```

**Validation Requirements**:
- [ ] 2015-2025 historical simulation
- [ ] Simulate 50+ rebalancing events
- [ ] Measure slippage reduction: target 5-15 bps
- [ ] Statistical significance test

### Phase 4: Dashboard Integration (2 hours)

Update dashboard to show calendar seasonality status:

```typescript
// src/components/CalendarSeasonalityStatus.tsx
interface CalendarStatus {
  currentDate: string;
  activeModifiers: string[];
  currentModifier: number;
  nextWindow: string;
  nextModifier: number;
  recommendation: 'proceed' | 'delay' | 'wait';
}
```

**Requirements**:
- [ ] Show current calendar modifiers
- [ ] Display next upcoming window
- [ ] Visual indicator (green/yellow/red)

### Phase 5: Documentation (1 hour)

- [ ] Update CLAUDE.md with v3.50 entry
- [ ] Create compound page: `compound/v350-calendar-seasonality-implementation.md`
- [ ] Add to knowledge.md index

## Acceptance Criteria

1. **Signal Accuracy**: Correctly identifies >95% of seasonal windows
2. **Test Coverage**: >90% line coverage, 20+ unit tests
3. **Backtest Results**: Demonstrate 5+ bps execution improvement
4. **Integration**: No regressions in existing rebalancing logic
5. **Performance**: <1ms latency added to scheduler

## Test Scenarios

### Unit Tests (test_calendar_seasonality.py)

```python
class TestCalendarSeasonalitySignal:
    def test_tom_window_last_trading_day(self):
        """Dec 31 (or last trading day) should be TOM window."""
        
    def test_tom_window_first_three_days(self):
        """Jan 2-4 (if trading days) should be TOM window."""
        
    def test_pre_holiday_thanksgiving(self):
        """Day before Thanksgiving should have 0.5 modifier."""
        
    def test_pre_holiday_christmas(self):
        """Christmas Eve (if trading day) should have 0.5 modifier."""
        
    def test_quarter_end_march(self):
        """March 25-31 should be quarter-end window."""
        
    def test_monday_effect(self):
        """Mondays should have 0.8 modifier."""
        
    def test_normal_day(self):
        """Random Tuesday mid-month should have 1.0 modifier."""
        
    def test_multiple_modifiers(self):
        """Day that is both TOM and Monday should use min modifier."""
        
    def test_trading_calendar_holiday(self):
        """July 4th should not be considered (market closed)."""
        
    def test_weekend_skipping(self):
        """Saturday/Sunday should be skipped in window calculations."""
```

## References

- Deep Research: `compound/calendar-seasonality-effects-quant-2026.md`
- Related Modules:
  - `src/execution/rebalance_scheduler.py` (integration target)
  - `src/signals/vpin_bvc.py` (existing timing overlay)
  - `src/execution/intraday_cost_model.py` (existing seasonality)

## Estimated Timeline

| Phase | Hours | Cumulative |
|-------|-------|------------|
| 1. Signal Module | 3 | 3 |
| 2. Integration | 2 | 5 |
| 3. Backtest | 4 | 9 |
| 4. Dashboard | 2 | 11 |
| 5. Docs | 1 | 12 |

**Total**: 12 hours (1.5 developer-days)

## Decision Points

1. **After Phase 1**: Review signal module with team
2. **After Phase 3**: Backtest must show >5 bps improvement to proceed to Phase 4
3. **Go/No-Go**: If backtest negative, abort and document lessons

## Notes

- Uses only NYSE trading calendar (US-focused portfolio)
- Modifiers are multiplicative (not additive) to maintain urgency scale
- All existing urgency thresholds remain unchanged
- If calendar module fails, scheduler falls back to baseline behavior
