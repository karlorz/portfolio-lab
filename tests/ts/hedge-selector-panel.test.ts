import { describe, expect, it } from 'bun:test';
import {
  buildHedgeSelectorDisplay,
  formatHedgeBps,
  formatHedgeRatioPct,
  formatHedgeSizePct,
} from '../../src/components/HedgeSelectorPanel';
import type { HedgeSelectorData } from '../../src/types/live';

const validHedgeSelector = (): HedgeSelectorData => ({
  available: true,
  generated_at: '2026-06-08T12:00:00Z',
  regime: 'stress',
  regime_confidence: 0.8,
  primary_hedge: 'put_spread',
  primary_size_pct: 6.0,
  secondary_hedge: 'vixy',
  secondary_size_pct: 4.0,
  expected_benefit_bps: 300,
  expected_cost_bps: 12,
  net_benefit_bps: 288,
  cost_benefit_gate: true,
  kelly_fraction: 0.24,
  confidence_scaled_size: 6.0,
  min_hold_days: 5,
  transition_cost_bps: 25,
});

describe('hedge selector panel display helpers', () => {
  it('returns a stable empty-state display when data is missing', () => {
    const display = buildHedgeSelectorDisplay(null);

    expect(display.available).toBe(false);
    expect(display.title).toBe('Hedge Selector');
    expect(display.emptyMessage).toBe('Hedge selector data not available');
  });

  it('formats valid hedge selector payload fields without changing size units', () => {
    const display = buildHedgeSelectorDisplay(validHedgeSelector());

    expect(display.available).toBe(true);
    expect(display.gateLabel).toBe('GATE OPEN');
    expect(display.regimeLabel).toBe('stress');
    expect(display.primaryHedge).toEqual({
      label: 'Primary Hedge',
      value: 'put_spread',
      detail: '6.00%',
    });
    expect(display.secondaryHedge).toEqual({
      label: 'Secondary Hedge',
      value: 'vixy',
      detail: '4.00%',
    });
    expect(display.netBenefit).toEqual({
      label: 'Net Benefit',
      value: '288.0 bps',
    });
    expect(display.regimeConfidence).toEqual({
      label: 'Regime Confidence',
      value: '80.00%',
    });
    expect(display.kellyFraction).toEqual({
      label: 'Kelly Fraction',
      value: '24.00%',
    });
    expect(display.minimumHold).toEqual({
      label: 'Minimum Hold',
      value: '5 trading days',
    });
  });

  it('uses explicit formatters for size percentages, ratios, and basis points', () => {
    expect(formatHedgeSizePct(6)).toBe('6.00%');
    expect(formatHedgeRatioPct(0.8)).toBe('80.00%');
    expect(formatHedgeBps(12)).toBe('12.0 bps');
  });
});
