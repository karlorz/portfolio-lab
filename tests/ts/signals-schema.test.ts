import { describe, expect, it } from 'bun:test';
import {
  GarchCvarSchema,
  IcDecaySchema,
  RegimeSchema,
  SmartRebalanceSchema,
  YieldCurveSchema,
} from '../../src/schemas/signals';

describe('signals.ts Zod schemas (non-ML contract)', () => {
  it('RegimeSchema accepts minimal regime payload', () => {
    const r = RegimeSchema.safeParse({ regime: 'NORMAL', vix: 18.2, detected: '2026-07-01' });
    expect(r.success).toBe(true);
  });

  it('GarchCvarSchema rejects missing required tail fields', () => {
    const bad = GarchCvarSchema.safeParse({ cvar_95: -0.02 });
    expect(bad.success).toBe(false);
  });

  it('GarchCvarSchema accepts full garch_cvar panel', () => {
    const good = GarchCvarSchema.safeParse({
      cvar_95: -0.04,
      cvar_95_garch: -0.0215,
      var_95: -0.69,
      var_95_garch: -0.0142,
      cvar_ratio: 1.51,
      garch_active: true,
      current_volatility: 0.0054,
      forecast_volatility: 0.015,
      volatility_clustering: 'elevated',
    });
    expect(good.success).toBe(true);
  });

  it('SmartRebalanceSchema accepts defer hold decision', () => {
    const good = SmartRebalanceSchema.safeParse({
      should_execute: false,
      decision: 'defer',
      urgency: 'moderate',
      max_drift: 0.03,
      estimated_cost_bps: 12,
      reason: 'vpin_high',
      drift_details: { SPY: 0.02 },
      vpin: 0.8,
      in_optimal_window: true,
      ytd_cost_bps: 5,
      remaining_budget_pct: 0.9,
      status: {
        ytd_cost_bps: 5,
        ytd_cost_pct: 0.0005,
        remaining_budget_pct: 0.9,
        is_over_budget: false,
        is_warning: false,
        last_rebalance: null,
        deferred_until: null,
        config: {
          drift_threshold: 0.1,
          vpin_threshold: 0.7,
          optimal_window: '10:00-15:30',
          annual_cost_limit: '50bps',
        },
      },
    });
    expect(good.success).toBe(true);
  });

  it('YieldCurveSchema allows nullable spreads', () => {
    expect(YieldCurveSchema.safeParse({ spread2s10s: null, dgs2: null, dgs10: null, duration_regime: null }).success).toBe(
      true,
    );
  });

  it('IcDecaySchema accepts empty signals map', () => {
    const good = IcDecaySchema.safeParse({ signals: {} });
    expect(good.success).toBe(true);
  });
});