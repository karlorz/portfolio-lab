/**
 * Regression: public signals.json.volatility_parity stores percentage points
 * (spy_pct: 40 means 40%), not decimal fractions. Panel must not *100 again.
 */
import { describe, expect, it } from 'bun:test';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import {
  VolatilityParityPanel,
  normalizeVolatilityParityData,
} from '../../src/components/VolatilityParityPanel';
import { VolatilityParitySchema } from '../../src/schemas/signals';

/** Shape currently emitted into public/data/signals.json (producer allocation). */
function producerPayload() {
  return {
    date: '2026-07-12',
    target_volatility: 10.0,
    spy_pct: 40.0,
    gld_pct: 28.0,
    tlt_pct: 12.0,
    core_vol_contribution: 11.36,
    vix_short_pct: 0.0,
    vix_tail_pct: 2.0,
    vix_vol_contribution: -30.0,
    cash_pct: 18.0,
    expected_portfolio_vol: 11.66,
    expected_max_dd: 17.49,
    rebalance_triggered: false,
    rebalance_reason: null,
  };
}

/** Nested legacy fixture used by older panel types. */
function nestedLegacyPayload() {
  return {
    allocation: producerPayload(),
    summary: {
      total_capital_allocation: 100,
      total_vol_contribution: -18.64,
      target_vol: 10.0,
      vol_gap: -1.66,
      vix_regime: 'backwardation',
    },
  };
}

describe('normalizeVolatilityParityData', () => {
  it('maps flat producer percentage-point fields without re-scaling', () => {
    const view = normalizeVolatilityParityData(producerPayload());
    expect(view).not.toBeNull();
    expect(view!.allocation.spy_pct).toBeCloseTo(40, 5);
    expect(view!.allocation.cash_pct).toBeCloseTo(18, 5);
    expect(view!.allocation.target_volatility).toBeCloseTo(10, 5);
    expect(view!.allocation.expected_portfolio_vol).toBeCloseTo(11.66, 5);
    expect(view!.allocation.expected_max_dd).toBeCloseTo(17.49, 5);
    expect(view!.summary.target_vol).toBeCloseTo(10, 5);
    expect(view!.summary.vol_gap).toBeCloseTo(10 - 11.66, 5);
  });

  it('preserves nested legacy shape without *100', () => {
    const view = normalizeVolatilityParityData(nestedLegacyPayload());
    expect(view).not.toBeNull();
    expect(view!.allocation.spy_pct).toBe(40);
    expect(view!.summary.vol_gap).toBeCloseTo(-1.66, 5);
    expect(view!.summary.vix_regime).toBe('backwardation');
  });

  it('returns null for empty/invalid payloads', () => {
    expect(normalizeVolatilityParityData(null)).toBeNull();
    expect(normalizeVolatilityParityData({})).toBeNull();
    expect(normalizeVolatilityParityData({ foo: 1 })).toBeNull();
  });
});

describe('VolatilityParitySchema producer contract', () => {
  it('accepts public percentage-point payload', () => {
    const parsed = VolatilityParitySchema.safeParse(producerPayload());
    expect(parsed.success).toBe(true);
  });
});

describe('VolatilityParityPanel display units', () => {
  it('renders ~40% SPY and ~10% target vol for producer payload (not 4000%/1000%)', () => {
    const html = renderToStaticMarkup(
      <VolatilityParityPanel data={producerPayload() as never} />,
    );
    expect(html).toContain('40.0%');
    expect(html).toContain('10%');
    expect(html).toContain('11.7%'); // expected vol 11.66 → 11.7
    expect(html).toContain('18.0%'); // cash
    expect(html).not.toContain('4000');
    expect(html).not.toContain('1000%');
    expect(html).not.toContain('1166');
  });

  it('renders nested legacy payload with percentage-point labels', () => {
    const html = renderToStaticMarkup(
      <VolatilityParityPanel data={nestedLegacyPayload() as never} />,
    );
    expect(html).toContain('40.0%');
    expect(html).toContain('10%');
    expect(html).not.toContain('4000');
  });

  it('shows empty state when data missing', () => {
    const html = renderToStaticMarkup(<VolatilityParityPanel data={null} />);
    expect(html).toContain('No volatility parity data available');
  });
});
