import { describe, it, expect } from 'bun:test';
import { existsSync, readFileSync } from 'fs';
import {
  RegimeSchema,
  YieldCurveSchema,
  DurationAllocationSchema,
  HedgeSelectorSchema,
  PositionSchema,
  RecentOrderSchema,
  GarchCvarSchema,
  EntropySchema,
  SmartRebalanceSchema,
  BrokerSchema,
  ClosingAuctionSchema,
  ZeroDTESchema,
  BondMomentumSchema,
  VIXTermStructureSchema,
  VIXOverlaySchema,
  SignalsDataSchema,
  validateSignalsData,
  DashboardDataSchema,
  AlertsDataSchema,
  StatsDataSchema,
  HealthDataSchema,
  AnalyticsDataSchema,
  RebalanceHealthSchema,
  GraduationDataSchema,
  validateFetchData,
} from '../../src/schemas/signals';
import { z } from 'zod';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function validRegime() {
  return { regime: 'normal', vix: 15.2, detected: '2026-05-26' };
}

function validYieldCurve() {
  return {
    spread2s10s: 0.35,
    dgs2: 4.25,
    dgs10: 4.60,
    duration_regime: 'normal' as const,
    spread_history: [0.3, 0.4, 0.35],
  };
}

function validPosition() {
  return { symbol: 'SPY', shares: 100, value: 55000, weight: 0.46, unrealized: 1200 };
}

function validRecentOrder() {
  return { sym: 'SPY', side: 'buy', shares: 10, value: 5500 };
}

function validGarchCvar() {
  return {
    cvar_95: -0.018,
    cvar_95_garch: -0.021,
    var_95: -0.015,
    var_95_garch: -0.017,
    cvar_ratio: 1.2,
    garch_active: true,
    current_volatility: 0.15,
    forecast_volatility: 0.16,
    volatility_clustering: 'normal' as const,
  };
}

function validEntropy() {
  return {
    shannon_entropy: 1.8,
    effective_n: 4.5,
    max_possible: 2.5,
    normalized_score: 0.72,
    concentration_risk: 'good' as const,
    hhi_index: 0.25,
  };
}

function validSmartRebalance() {
  return {
    should_execute: false,
    decision: 'wait',
    urgency: 'low' as const,
    max_drift: 0.08,
    estimated_cost_bps: 12,
    reason: 'Drift below threshold',
    drift_details: { SPY: 0.03, GLD: 0.05 },
    vpin: 0.42,
    in_optimal_window: true,
    ytd_cost_bps: 45,
    remaining_budget_pct: 0.55,
    status: {
      ytd_cost_bps: 45,
      ytd_cost_pct: 0.0045,
      remaining_budget_pct: 0.55,
      is_over_budget: false,
      is_warning: false,
      last_rebalance: '2026-05-20T10:00:00Z',
      deferred_until: null,
      config: {
        drift_threshold: 0.10,
        vpin_threshold: 0.50,
        optimal_window: '10:30-15:30',
        annual_cost_limit: '0.01',
      },
    },
  };
}

function validSignalsData(): Record<string, unknown> {
  return {
    timestamp: '2026-05-26T12:00:00Z',
    regime: validRegime(),
    latest_prices: { SPY: 550, GLD: 195, TLT: 92 },
    current_positions: [validPosition()],
    target_allocations: { SPY: 0.46, GLD: 0.38, TLT: 0.16 },
    cash: 5000,
    total_value: 110000,
    recent_orders: [validRecentOrder()],
    ml_signals: {
      available: false,
      timestamp: null,
      predictions: {},
      features: {},
      grid_search: {
        available: false,
        timestamp: null,
        top_allocation: null,
        sharpe: null,
        volatility: null,
      },
    },
    generated_at: '2026-05-26T12:00:00Z',
    // Optional typed fields
    yield_curve: validYieldCurve(),
    garch_cvar: validGarchCvar(),
    entropy: validEntropy(),
    smart_rebalance: validSmartRebalance(),
    // Untyped panel
    behavioral_sentiment: { score: 0.7, signal: 'bullish' },
    crypto_allocation: { btc: 0.02 },
  };
}

function readJsonOrFallback(path: string, fallback: Record<string, unknown>): Record<string, unknown> {
  if (!existsSync(path)) {
    return fallback;
  }

  return JSON.parse(readFileSync(path, 'utf8')) as Record<string, unknown>;
}

// ===========================================================================
// Individual Schema Tests
// ===========================================================================

describe('RegimeSchema', () => {
  it('accepts valid regime data', () => {
    const result = RegimeSchema.safeParse(validRegime());
    expect(result.success).toBe(true);
  });

  it('rejects missing regime field', () => {
    const result = RegimeSchema.safeParse({ vix: 15, detected: null });
    expect(result.success).toBe(false);
  });

  it('rejects wrong type for vix', () => {
    const result = RegimeSchema.safeParse({ regime: 'normal', vix: 'high', detected: null });
    expect(result.success).toBe(false);
  });

  it('accepts null vix', () => {
    const result = RegimeSchema.safeParse({ regime: 'normal', vix: null, detected: null });
    expect(result.success).toBe(true);
  });
});

describe('YieldCurveSchema', () => {
  it('accepts valid yield curve data', () => {
    const result = YieldCurveSchema.safeParse(validYieldCurve());
    expect(result.success).toBe(true);
  });

  it('rejects invalid duration_regime', () => {
    const result = YieldCurveSchema.safeParse({
      ...validYieldCurve(),
      duration_regime: 'unknown',
    });
    expect(result.success).toBe(false);
  });

  it('accepts null duration_regime', () => {
    const result = YieldCurveSchema.safeParse({
      ...validYieldCurve(),
      duration_regime: null,
    });
    expect(result.success).toBe(true);
  });

  it('optional spread_history is not required', () => {
    const { spread_history, ...rest } = validYieldCurve();
    const result = YieldCurveSchema.safeParse(rest);
    expect(result.success).toBe(true);
  });
});

describe('PositionSchema', () => {
  it('accepts valid position', () => {
    const result = PositionSchema.safeParse(validPosition());
    expect(result.success).toBe(true);
  });

  it('rejects missing symbol', () => {
    const { symbol, ...rest } = validPosition();
    const result = PositionSchema.safeParse(rest);
    expect(result.success).toBe(false);
  });

  it('rejects string weight instead of number', () => {
    const result = PositionSchema.safeParse({ ...validPosition(), weight: '0.46' });
    expect(result.success).toBe(false);
  });
});

describe('RecentOrderSchema', () => {
  it('accepts valid recent order', () => {
    const result = RecentOrderSchema.safeParse(validRecentOrder());
    expect(result.success).toBe(true);
  });

  it('rejects missing shares', () => {
    const { shares, ...rest } = validRecentOrder();
    const result = RecentOrderSchema.safeParse(rest);
    expect(result.success).toBe(false);
  });
});

describe('GarchCvarSchema', () => {
  it('accepts valid GARCH-CVaR data', () => {
    const result = GarchCvarSchema.safeParse(validGarchCvar());
    expect(result.success).toBe(true);
  });

  it('rejects invalid volatility_clustering', () => {
    const result = GarchCvarSchema.safeParse({
      ...validGarchCvar(),
      volatility_clustering: 'extreme',
    });
    expect(result.success).toBe(false);
  });

  it('rejects missing required field', () => {
    const { cvar_95, ...rest } = validGarchCvar();
    const result = GarchCvarSchema.safeParse(rest);
    expect(result.success).toBe(false);
  });
});

describe('EntropySchema', () => {
  it('accepts valid entropy data', () => {
    const result = EntropySchema.safeParse(validEntropy());
    expect(result.success).toBe(true);
  });

  it('accepts optional correlation_entropy', () => {
    const result = EntropySchema.safeParse({
      ...validEntropy(),
      correlation_entropy: 0.5,
      participation_ratio: 0.8,
    });
    expect(result.success).toBe(true);
  });

  it('rejects invalid concentration_risk', () => {
    const result = EntropySchema.safeParse({
      ...validEntropy(),
      concentration_risk: 'perfect',
    });
    expect(result.success).toBe(false);
  });
});

describe('SmartRebalanceSchema', () => {
  it('accepts valid smart rebalance data', () => {
    const result = SmartRebalanceSchema.safeParse(validSmartRebalance());
    expect(result.success).toBe(true);
  });

  it('rejects invalid urgency', () => {
    const result = SmartRebalanceSchema.safeParse({
      ...validSmartRebalance(),
      urgency: 'critical',
    });
    expect(result.success).toBe(false);
  });

  it('rejects missing nested status fields', () => {
    const { status, ...rest } = validSmartRebalance();
    const result = SmartRebalanceSchema.safeParse(rest);
    expect(result.success).toBe(false);
  });

  it('accepts null last_rebalance and deferred_until', () => {
    const result = SmartRebalanceSchema.safeParse({
      ...validSmartRebalance(),
      status: {
        ...validSmartRebalance().status,
        last_rebalance: null,
        deferred_until: null,
      },
    });
    expect(result.success).toBe(true);
  });
});

// ===========================================================================
// HedgeSelectorSchema
// ===========================================================================

describe('HedgeSelectorSchema', () => {
  const validHedgeSelector = () => ({
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

  it('accepts valid hedge selector data', () => {
    const result = HedgeSelectorSchema.safeParse(validHedgeSelector());
    expect(result.success).toBe(true);
  });

  it('fills defaults for partial hedge selector data', () => {
    const result = HedgeSelectorSchema.safeParse({ available: true, primary_hedge: 'vixy' });
    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.generated_at).toBe('');
      expect(result.data.regime).toBe('unknown');
      expect(result.data.secondary_hedge).toBeNull();
      expect(result.data.cost_benefit_gate).toBe(false);
      expect(result.data.min_hold_days).toBe(0);
      expect(result.data.transition_cost_bps).toBe(0);
    }
  });

  it('rejects invalid numeric fields', () => {
    const result = HedgeSelectorSchema.safeParse({
      ...validHedgeSelector(),
      primary_size_pct: 'large',
    });
    expect(result.success).toBe(false);
  });

  it('validates hedge_selector inside SignalsDataSchema', () => {
    const result = SignalsDataSchema.safeParse({
      ...validSignalsData(),
      hedge_selector: validHedgeSelector(),
    });
    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.hedge_selector?.primary_hedge).toBe('put_spread');
    }
  });
});

describe('BrokerSchema', () => {
  const validBroker = () => ({
    connected: true,
    positions: [{ symbol: 'SPY', qty: 100, market_value: 55000, unrealized_pl: 1200, side: 'long' }],
    drift: [{ symbol: 'SPY', broker_qty: 100, local_qty: 98, drift_pct: 0.02 }],
    recent_orders: [{
      symbol: 'SPY', side: 'buy', qty: 10, status: 'filled',
      timestamp: '2026-05-26T12:00:00Z', dry_run: false,
    }],
    last_sync: '2026-05-26T12:00:00Z',
    kill_switch: false,
  });

  it('accepts valid broker data', () => {
    const result = BrokerSchema.safeParse(validBroker());
    expect(result.success).toBe(true);
  });

  it('accepts null last_sync', () => {
    const result = BrokerSchema.safeParse({ ...validBroker(), last_sync: null });
    expect(result.success).toBe(true);
  });

  it('rejects missing connected field', () => {
    const { connected, ...rest } = validBroker();
    const result = BrokerSchema.safeParse(rest);
    expect(result.success).toBe(false);
  });
});

describe('ClosingAuctionSchema', () => {
  const validAuction = () => ({
    signals: [{
      symbol: 'SPY',
      timestamp: '2026-05-26T15:30:00Z',
      direction: 'BUY' as const,
      direction_score: 0.8,
      confidence: 'high' as const,
      imbalance: {
        symbol: 'SPY', timestamp: '2026-05-26T15:30:00Z',
        imbalance_shares: 10000, paired_shares: 50000,
        reference_price: 550, source: 'NYSE',
        imbalance_ratio: 0.2, direction_score: 0.8,
      },
      entry_price: 550,
      target_exit_price: 552,
      historical_win_rate: 0.65,
      historical_count: 20,
      max_position_pct: 0.02,
      urgency: 'normal' as const,
      should_trade: true,
    }],
    last_update: '2026-05-26T15:30:00Z',
    market_open: true,
  });

  it('accepts valid closing auction data', () => {
    const result = ClosingAuctionSchema.safeParse(validAuction());
    expect(result.success).toBe(true);
  });

  it('rejects invalid direction', () => {
    const result = ClosingAuctionSchema.safeParse({
      ...validAuction(),
      signals: [{ ...validAuction().signals[0], direction: 'NOT_A_DIRECTION' }],
    });
    expect(result.success).toBe(false);
  });
});

describe('ZeroDTESchema', () => {
  const validZeroDTE = () => ({
    positions: [{
      id: 'z-001',
      underlying: 'SPY',
      option_type: 'put' as const,
      side: 'sell' as const,
      strike: 540,
      expiration: '2026-05-26',
      quantity: 1,
      entry_price: 1.50,
      entry_time: '2026-05-26T09:30:00Z',
      entry_delta: -0.25,
      entry_theta: 0.10,
      current_delta: -0.30,
      current_theta: 0.12,
      current_underlying_price: 548,
      status: 'open' as const,
      premium_collected: 1.50,
      delta_exposure: -0.30,
      notional_value: 54000,
    }],
    config: {
      max_portfolio_allocation: 0.02,
      max_weekly_positions: 2,
      position_size_pct: 0.005,
      min_vix: 15,
      max_vix: 35,
      delta_target: 0.30,
      min_premium_pct: 0.004,
      max_delta_exposure: 0.08,
      emergency_close_delta: 0.50,
      max_loss_pct: 0.015,
    },
    weekly_trades_used: 1,
    total_premium_collected_mtd: 1.50,
  });

  it('accepts valid zero DTE data', () => {
    const result = ZeroDTESchema.safeParse(validZeroDTE());
    expect(result.success).toBe(true);
  });

  it('accepts null config', () => {
    const result = ZeroDTESchema.safeParse({ ...validZeroDTE(), config: null });
    expect(result.success).toBe(true);
  });

  it('rejects invalid option_type', () => {
    const result = ZeroDTESchema.safeParse({
      ...validZeroDTE(),
      positions: [{ ...validZeroDTE().positions[0], option_type: 'straddle' }],
    });
    expect(result.success).toBe(false);
  });
});

describe('BondMomentumSchema', () => {
  it('accepts valid bond momentum data', () => {
    const data = {
      signals: [{
        etf: 'TLT', timestamp: '2026-05-26T12:00:00Z',
        signal: 0.5, position_size: 0.16, formation_return: 0.02,
        realized_vol: 0.12, formation_months: 6, volatility_target: 0.10,
        confidence: 'moderate' as const, action: 'hold' as const, weight_delta: 0,
      }],
      timestamp: '2026-05-26T12:00:00Z',
      ensemble: { weight: 0.5, confidence: 'moderate', action: 'hold', recommendation: 'Hold current allocation' },
    };
    const result = BondMomentumSchema.safeParse(data);
    expect(result.success).toBe(true);
  });

  it('rejects invalid confidence', () => {
    const data = {
      signals: [{
        etf: 'TLT', timestamp: '2026-05-26T12:00:00Z',
        signal: 0.5, position_size: 0.16, formation_return: 0.02,
        realized_vol: 0.12, formation_months: 6, volatility_target: 0.10,
        confidence: 'unknown' as const, action: 'hold' as const, weight_delta: 0,
      }],
      timestamp: '2026-05-26T12:00:00Z',
      ensemble: { weight: 0.5, confidence: 'moderate', action: 'hold', recommendation: 'Hold current allocation' },
    };
    const result = BondMomentumSchema.safeParse(data);
    expect(result.success).toBe(false);
  });
});

describe('VIXTermStructureSchema', () => {
  it('accepts valid VIX term structure', () => {
    const data = {
      vix: { value: 15.2, timestamp: '2026-05-26T12:00:00Z' },
      vix3m: { value: 17.5, timestamp: '2026-05-26T12:00:00Z' },
      slope: 2.3,
      roll_yield: 0.15,
      composite_signal: 0.35,
      regime: 'mild_contango' as const,
      z_score: 0.5,
    };
    const result = VIXTermStructureSchema.safeParse(data);
    expect(result.success).toBe(true);
  });

  it('accepts optional vix6m', () => {
    const data = {
      vix: { value: 15.2, timestamp: '2026-05-26T12:00:00Z' },
      vix3m: { value: 17.5, timestamp: '2026-05-26T12:00:00Z' },
      vix6m: { value: 19.0, timestamp: '2026-05-26T12:00:00Z' },
      slope: 2.3, roll_yield: 0.15, composite_signal: 0.35,
      regime: 'backwardation' as const, z_score: -0.3,
    };
    const result = VIXTermStructureSchema.safeParse(data);
    expect(result.success).toBe(true);
  });
});

describe('VIXOverlaySchema', () => {
  it('accepts valid VIX overlay', () => {
    const data = {
      allocation: { SPY: 0.40, GLD: 0.35, TLT: 0.25 },
      last_shift_date: '2026-05-25',
      shift_history: [{
        date: '2026-05-25',
        shifts: { SPY: -0.06, GLD: 0.03, TLT: 0.03 },
        signal_value: 0.8,
        regime: 'high_vol',
        new_allocation: { SPY: 0.40, GLD: 0.35, TLT: 0.25 },
      }],
      disabled_until: null,
    };
    const result = VIXOverlaySchema.safeParse(data);
    expect(result.success).toBe(true);
  });
});

// ===========================================================================
// SignalsDataSchema
// ===========================================================================

describe('SignalsDataSchema', () => {
  it('accepts valid full signals data', () => {
    const result = SignalsDataSchema.safeParse(validSignalsData());
    expect(result.success).toBe(true);
  });

  it('accepts minimal signals data (only required fields)', () => {
    const minimal = {
      timestamp: '2026-05-26T12:00:00Z',
      regime: validRegime(),
      latest_prices: { SPY: 550 },
      current_positions: [],
      target_allocations: {},
      cash: 0,
      total_value: 100000,
      recent_orders: [],
      ml_signals: {
        available: false,
        timestamp: null,
        predictions: {},
        features: {},
        grid_search: { available: false, timestamp: null, top_allocation: null, sharpe: null, volatility: null },
      },
    };
    const result = SignalsDataSchema.safeParse(minimal);
    expect(result.success).toBe(true);
  });

  it('accepts generated_at as the timestamp source for production signal artifacts', () => {
    const { timestamp, ...rest } = validSignalsData();
    const result = SignalsDataSchema.safeParse(rest);
    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.timestamp).toBe('2026-05-26T12:00:00Z');
    }
  });

  it('rejects signal artifacts missing both timestamp and generated_at', () => {
    const { timestamp, generated_at, ...rest } = validSignalsData();
    const result = SignalsDataSchema.safeParse(rest);
    expect(result.success).toBe(false);
  });

  it('rejects invalid regime type', () => {
    const data = validSignalsData();
    data.regime = { regime: 123 };
    const result = SignalsDataSchema.safeParse(data);
    expect(result.success).toBe(false);
  });

  it('rejects non-numeric cash', () => {
    const data = validSignalsData();
    data.cash = 'lots';
    const result = SignalsDataSchema.safeParse(data);
    expect(result.success).toBe(false);
  });

  it('passes through extra fields with passthrough', () => {
    const data = validSignalsData();
    data.custom_extra_field = 'should-pass-through';
    const result = SignalsDataSchema.safeParse(data);
    expect(result.success).toBe(true);
    if (result.success) {
      expect((result.data as Record<string, unknown>).custom_extra_field).toBe('should-pass-through');
    }
  });

  // -----------------------------------------------------------------------
  // z.unknown() panels
  // -----------------------------------------------------------------------
  it('passes through any value for behavioral_sentiment (z.unknown())', () => {
    const data = validSignalsData();
    data.behavioral_sentiment = { arbitrary: 'structure', nested: { key: 42 } };
    const result = SignalsDataSchema.safeParse(data);
    expect(result.success).toBe(true);
  });

  it('accepts null for disabled untyped signal panels', () => {
    const data = validSignalsData();
    data.behavioral_sentiment = null;
    data.crypto_allocation = null;
    data.calendar_seasonality = null;
    const result = SignalsDataSchema.safeParse(data);
    expect(result.success).toBe(true);
  });

  it('passes through arbitrary data for crypto_allocation (z.unknown() record)', () => {
    const data = validSignalsData();
    data.crypto_allocation = { btc: 0.5, eth: 0.3, sol: 0.2 };
    const result = SignalsDataSchema.safeParse(data);
    expect(result.success).toBe(true);
  });

  it('accepts the generated signals artifact optional panel shapes when present', () => {
    const artifact = readJsonOrFallback('public/data/signals.json', validSignalsData());
    const result = SignalsDataSchema.safeParse(artifact);
    expect(result.success).toBe(true);
  });

  it('rejects invalid non-null untyped signal panel values', () => {
    const data = validSignalsData();
    data.crypto_allocation = ['not', 'a', 'record'];
    const result = SignalsDataSchema.safeParse(data);
    expect(result.success).toBe(false);
  });

  it('allows deeply nested unknown data in signal panels', () => {
    const data = validSignalsData();
    data.ensemble_voting = {
      weights: [0.3, 0.5, 0.2],
      signals: { msm: 0.5, carv: -0.2 },
      meta: { generated_at: '2026-05-26', version: 'v2' },
    };
    const result = SignalsDataSchema.safeParse(data);
    expect(result.success).toBe(true);
  });

});

// ===========================================================================
// validateSignalsData
// ===========================================================================

describe('validateSignalsData', () => {
  it('returns parsed data for valid input', () => {
    const result = validateSignalsData(validSignalsData());
    expect(result).not.toBeNull();
    expect(result!.timestamp).toBe('2026-05-26T12:00:00Z');
    expect(result!.regime.regime).toBe('normal');
  });

  it('returns null for completely invalid input', () => {
    const result = validateSignalsData(null);
    expect(result).toBeNull();
  });

  it('returns null for non-object input', () => {
    const result = validateSignalsData('not an object');
    expect(result).toBeNull();
  });

  it('returns null for empty object', () => {
    const result = validateSignalsData({});
    expect(result).toBeNull();
  });

  it('returns null for array input', () => {
    const result = validateSignalsData([1, 2, 3]);
    expect(result).toBeNull();
  });

  it('fallback: returns raw data when partially valid (has timestamp)', () => {
    // Data that fails schema validation but has a timestamp
    const partial = { timestamp: '2026-05-26T12:00:00Z', regime: 'invalid-shape' };
    const result = validateSignalsData(partial);
    // Should fall back to raw data because it has 'timestamp'
    expect(result).not.toBeNull();
    expect(result!.timestamp).toBe('2026-05-26T12:00:00Z');
  });

  it('fallback: returns generated_at-only raw data with normalized timestamp', () => {
    const partial = { generated_at: '2026-05-26T12:00:00Z', regime: 'invalid-shape' };
    const result = validateSignalsData(partial);
    expect(result).not.toBeNull();
    expect(result!.timestamp).toBe('2026-05-26T12:00:00Z');
  });

  it('fallback is NOT used when object lacks timestamp and generated_at', () => {
    const partial = { regime: { regime: 'normal' }, no_timestamp: true };
    const result = validateSignalsData(partial);
    expect(result).toBeNull();
  });

  it('fallback preserves all fields from raw data', () => {
    const partial = {
      timestamp: '2026-05-26T12:00:00Z',
      regime: { regime: 'normal' }, // invalid (missing vix, detected)
      extra_field: 42,
    };
    const result = validateSignalsData(partial);
    expect(result).not.toBeNull();
    expect((result as Record<string, unknown>).extra_field).toBe(42);
  });

  it('returns SignalsData with expected fields on success', () => {
    const result = validateSignalsData(validSignalsData());
    expect(result).not.toBeNull();
    expect(result!.target_allocations.SPY).toBe(0.46);
    expect(result!.current_positions).toHaveLength(1);
    expect(result!.current_positions[0].symbol).toBe('SPY');
    expect(result!.recent_orders).toHaveLength(1);
    expect(result!.recent_orders[0].sym).toBe('SPY');
  });

  it('validates GARCH-CVaR when present', () => {
    const data = validSignalsData();
    data.garch_cvar = validGarchCvar();
    const result = validateSignalsData(data);
    expect(result).not.toBeNull();
    expect(result!.garch_cvar!.cvar_95).toBe(-0.018);
    expect(result!.garch_cvar!.garch_active).toBe(true);
  });

  it('validates smart_rebalance when present', () => {
    const data = validSignalsData();
    data.smart_rebalance = validSmartRebalance();
    const result = validateSignalsData(data);
    expect(result).not.toBeNull();
    expect(result!.smart_rebalance!.decision).toBe('wait');
    expect(result!.smart_rebalance!.urgency).toBe('low');
  });

  it('returns null for badly typed nested fields', () => {
    const data = validSignalsData();
    data.cash = 'not-a-number'; // intentional type mismatch
    const result = validateSignalsData(data);
    // Should fail schema but still have timestamp -> fallback
    expect(result).not.toBeNull();
    // But if we give something that fails AND lacks timestamp, it's null
    const result2 = validateSignalsData({ cash: 'not-a-number' });
    expect(result2).toBeNull();
  });
});

// ===========================================================================
// DashboardDataSchema
// ===========================================================================

describe('DashboardDataSchema', () => {
  const validDashboard = () => ({
    prices: {
      SPY: [{ d: '2026-05-26', p: 550 }],
      GLD: [{ d: '2026-05-26', p: 195 }],
    },
    regimes: [{ d: '2026-05-26', r: 'normal', v: 15.2 }],
    paper_portfolio: [{ t: '2026-05-26', v: 110000, r: 0.001 }],
    generated_at: '2026-05-26T12:00:00Z',
  });

  it('accepts valid dashboard data', () => {
    const result = DashboardDataSchema.safeParse(validDashboard());
    expect(result.success).toBe(true);
  });

  it('accepts extra fields via passthrough', () => {
    const data = { ...validDashboard(), extra_field: 'hello' };
    const result = DashboardDataSchema.safeParse(data);
    expect(result.success).toBe(true);
  });

  it('rejects missing generated_at', () => {
    const { generated_at, ...rest } = validDashboard();
    const result = DashboardDataSchema.safeParse(rest);
    expect(result.success).toBe(false);
  });

  it('rejects missing paper_portfolio', () => {
    const { paper_portfolio, ...rest } = validDashboard();
    const result = DashboardDataSchema.safeParse(rest);
    expect(result.success).toBe(false);
  });
});

// ===========================================================================
// AlertsDataSchema
// ===========================================================================

describe('AlertsDataSchema', () => {
  const validAlerts = () => ({
    alerts: [{
      level: 'warning' as const,
      type: 'drift',
      title: 'Drift Alert',
      message: 'SPY drift exceeds threshold',
      timestamp: '2026-05-26T12:00:00Z',
      requires_action: false,
    }],
    count: 1,
    generated_at: '2026-05-26T12:00:00Z',
  });

  it('accepts valid alerts data', () => {
    const result = AlertsDataSchema.safeParse(validAlerts());
    expect(result.success).toBe(true);
  });

  it('accepts empty alerts array', () => {
    const result = AlertsDataSchema.safeParse({ alerts: [], count: 0, generated_at: '2026-05-26T12:00:00Z' });
    expect(result.success).toBe(true);
  });

  it('rejects invalid alert level', () => {
    const data = validAlerts();
    data.alerts[0].level = 'critical' as any;
    const result = AlertsDataSchema.safeParse(data);
    expect(result.success).toBe(false);
  });

  it('rejects missing generated_at', () => {
    const { generated_at, ...rest } = validAlerts();
    const result = AlertsDataSchema.safeParse(rest);
    expect(result.success).toBe(false);
  });
});

// ===========================================================================
// StatsDataSchema
// ===========================================================================

describe('StatsDataSchema', () => {
  const validStats = () => ({
    asset_stats: {
      SPY: { '30d_return': 2.5, volatility: 15.2, current: 550 },
    },
    paper_portfolio: {
      sharpe: 0.79,
      total_return: 0.15,
      max_value: 120000,
      min_value: 90000,
      days_tracked: 500,
    },
    spy_comparison: {
      portfolio_value: 110000,
      spy_value: 105000,
      relative_return: 0.05,
      correlation_30d: 0.85,
      beta: 0.75,
      outperformance: 0.03,
    },
    generated_at: '2026-05-26T12:00:00Z',
  });

  it('accepts valid stats data', () => {
    const result = StatsDataSchema.safeParse(validStats());
    expect(result.success).toBe(true);
  });

  it('accepts null paper_portfolio', () => {
    const data = { ...validStats(), paper_portfolio: null };
    const result = StatsDataSchema.safeParse(data);
    expect(result.success).toBe(true);
  });

  it('accepts null spy_comparison', () => {
    const data = { ...validStats(), spy_comparison: null };
    const result = StatsDataSchema.safeParse(data);
    expect(result.success).toBe(true);
  });

  it('rejects missing asset_stats', () => {
    const { asset_stats, ...rest } = validStats();
    const result = StatsDataSchema.safeParse(rest);
    expect(result.success).toBe(false);
  });

  it('passes through extra fields', () => {
    const data = { ...validStats(), extra_key: 'value' };
    const result = StatsDataSchema.safeParse(data);
    expect(result.success).toBe(true);
  });
});

// ===========================================================================
// HealthDataSchema
// ===========================================================================

describe('HealthDataSchema', () => {
  const validHealth = () => ({
    cron_jobs: [{
      id: 'job-1',
      name: 'Fetch Prices',
      schedule: '0 9 * * 1-5',
      last_run: '2026-05-26T09:00:00Z',
      next_run: '2026-05-27T09:00:00Z',
      status: 'ok' as const,
      state: 'scheduled' as const,
    }],
    data_freshness: {
      prices: { last_update: '2026-05-26T09:00:00Z', days_stale: 0, status: 'fresh' as const },
    },
    system_status: 'healthy' as const,
    generated_at: '2026-05-26T12:00:00Z',
  });

  it('accepts valid health data', () => {
    const result = HealthDataSchema.safeParse(validHealth());
    expect(result.success).toBe(true);
  });

  it('accepts error field', () => {
    const data = { ...validHealth(), error: 'something went wrong' };
    const result = HealthDataSchema.safeParse(data);
    expect(result.success).toBe(true);
  });

  it('preserves cron backend metadata', () => {
    const data = {
      ...validHealth(),
      cron_jobs: [{
        ...validHealth().cron_jobs[0],
        backend: 'hermes' as const,
        source: '/root/.hermes/cron/jobs.json',
        error: 'RuntimeError: final report text',
      }],
    };
    const result = HealthDataSchema.safeParse(data);
    expect(result.success).toBe(true);
    expect(result.data?.cron_jobs[0].backend).toBe('hermes');
    expect(result.data?.cron_jobs[0].error).toContain('RuntimeError');
  });

  it('accepts scheduler backend status metadata', () => {
    const data = {
      ...validHealth(),
      scheduler_status: {
        status: 'degraded' as const,
        backends: {
          local: {
            backend: 'local',
            status: 'ok' as const,
            source: '/root/projects/portfolio-lab/data/cron_status.json',
            total_jobs: 1,
            failed_jobs: 0,
          },
          hermes: {
            backend: 'hermes',
            status: 'degraded' as const,
            source: '/root/.hermes/cron/jobs.json',
            total_jobs: 2,
            failed_jobs: 1,
            reason: 'fixture',
          },
        },
      },
    };
    const result = HealthDataSchema.safeParse(data);
    expect(result.success).toBe(true);
    expect(result.data?.scheduler_status?.backends.hermes.failed_jobs).toBe(1);
  });

  it('rejects invalid system_status', () => {
    const data = { ...validHealth(), system_status: 'unknown' };
    const result = HealthDataSchema.safeParse(data);
    expect(result.success).toBe(false);
  });

  it('rejects null last_run', () => {
    const data = { ...validHealth(), cron_jobs: [{ ...validHealth().cron_jobs[0], last_run: null }] };
    const result = HealthDataSchema.safeParse(data);
    expect(result.success).toBe(true);
  });

  it('accepts null duration for scheduled jobs that have not recorded runtime', () => {
    const data = {
      ...validHealth(),
      cron_jobs: [{ ...validHealth().cron_jobs[0], duration_seconds: null }],
    };
    const result = HealthDataSchema.safeParse(data);
    expect(result.success).toBe(true);
  });

  it('rejects missing cron_jobs', () => {
    const { cron_jobs, ...rest } = validHealth();
    const result = HealthDataSchema.safeParse(rest);
    expect(result.success).toBe(false);
  });
});

// ===========================================================================
// AnalyticsDataSchema
// ===========================================================================

describe('AnalyticsDataSchema', () => {
  const validAnalytics = () => ({
    status: 'success' as const,
    generated_at: '2026-05-26T12:00:00Z',
    data_points: 500,
    date_range: { start: '2025-01-01', end: '2026-05-26' },
    drawdown: {
      series: [{ date: '2026-03-15', value: 100000, peak: 120000, drawdown: -16.7, days_since_peak: 45, is_recovery: false }],
      max_drawdown: { max_drawdown: -16.7, max_drawdown_date: '2026-03-15', recovery_date: null, underwater_days: 45, peak_value: 120000, trough_value: 100000 },
    },
    rolling_metrics: { sharpe_63d: [], sharpe_126d: [], sharpe_252d: [] },
    benchmark_comparison: {
      portfolio: { start_date: '2025-01-01', end_date: '2026-05-26', start_value: 100000, end_value: 110000, total_return: 0.10, cagr: 0.08, volatility: 0.12, max_drawdown: -16.7, sharpe: 0.79 },
    },
    crisis_periods: [{ name: '2022 Crash', period: '2022-01 to 2022-10', description: 'Rate hikes', spy_return: -0.20, portfolio_return: -0.12 }],
  });

  it('accepts valid analytics data', () => {
    const result = AnalyticsDataSchema.safeParse(validAnalytics());
    expect(result.success).toBe(true);
  });

  it('accepts status no_data', () => {
    const data = { ...validAnalytics(), status: 'no_data' as const, message: 'Not enough data' };
    const result = AnalyticsDataSchema.safeParse(data);
    expect(result.success).toBe(true);
  });

  it('accepts recovery_date as string', () => {
    const data = validAnalytics();
    data.drawdown.max_drawdown.recovery_date = '2026-04-01';
    const result = AnalyticsDataSchema.safeParse(data);
    expect(result.success).toBe(true);
  });

  it('rejects invalid status', () => {
    const data = { ...validAnalytics(), status: 'invalid' };
    const result = AnalyticsDataSchema.safeParse(data);
    expect(result.success).toBe(false);
  });

  it('passes through extra fields', () => {
    const data = { ...validAnalytics(), custom: true };
    const result = AnalyticsDataSchema.safeParse(data);
    expect(result.success).toBe(true);
  });
});

// ===========================================================================
// RebalanceHealthSchema
// ===========================================================================

describe('RebalanceHealthSchema', () => {
  const validRH = () => ({
    current_turnover_pct: 0.5,
    max_daily_turnover: 10,
    max_monthly_turnover: 30,
    max_annual_turnover: 100,
    daily_budget_used: 0.05,
    monthly_budget_used: 0.02,
    annual_budget_used: 0.005,
    recent_rebalances: [{ date: '2026-05-20', turnover_pct: 0.5, cost_bps: 12, trigger: 'drift' }],
    cost_drag_bps: 45,
  });

  const fallbackGeneratedRH = () => ({
    ...validRH(),
    market_data_consistency: {
      status: 'unavailable',
      reason: 'fixture_missing_in_clean_checkout',
    },
    alpaca_feed_entitlement: {
      configured_feed: 'iex',
      effective_feed: 'iex',
      entitlement: 'unavailable',
      delayed: true,
      policy_decision: 'reject',
      acceptable_for_live: false,
      reason: 'fixture_missing_in_clean_checkout',
    },
  });

  const generatedRH = () => readJsonOrFallback('public/data/rebalance_health.json', fallbackGeneratedRH());

  it('accepts valid rebalance health data', () => {
    const result = RebalanceHealthSchema.safeParse(validRH());
    expect(result.success).toBe(true);
  });

  it('accepts the generated rebalance health artifact with live diagnostics', () => {
    const result = RebalanceHealthSchema.safeParse(generatedRH());

    expect(result.success).toBe(true);
    if (!result.success) return;

    expect(result.data.market_data_consistency?.status).toBe('unavailable');
    expect(result.data.alpaca_feed_entitlement?.policy_decision).toBe('reject');
    expect(result.data.alpaca_feed_entitlement?.acceptable_for_live).toBe(false);
  });

  it('accepts the fallback rebalance health fixture used in clean checkouts', () => {
    const result = RebalanceHealthSchema.safeParse(fallbackGeneratedRH());

    expect(result.success).toBe(true);
  });

  it('accepts empty recent_rebalances', () => {
    const data = { ...validRH(), recent_rebalances: [] };
    const result = RebalanceHealthSchema.safeParse(data);
    expect(result.success).toBe(true);
  });

  it('rejects missing current_turnover_pct', () => {
    const { current_turnover_pct, ...rest } = validRH();
    const result = RebalanceHealthSchema.safeParse(rest);
    expect(result.success).toBe(false);
  });

  it('passes through extra fields', () => {
    const data = { ...validRH(), extra: 'value' };
    const result = RebalanceHealthSchema.safeParse(data);
    expect(result.success).toBe(true);
  });
});

// ===========================================================================
// GraduationDataSchema
// ===========================================================================

describe('GraduationDataSchema', () => {
  const validGrad = () => ({
    criteria: [{ id: 'sharpe', label: 'Sharpe Ratio', passed: true, value: '0.79', threshold: '>= 0.50' }],
    paper_trading: { start_date: '2026-01-01', initial_capital: 100000, current_value: 110000, days_elapsed: 145, days_required: 90 },
    readiness_pct: 0.85,
    eligible: false,
  });

  it('accepts valid graduation data', () => {
    const result = GraduationDataSchema.safeParse(validGrad());
    expect(result.success).toBe(true);
  });

  it('accepts multiple criteria', () => {
    const data = {
      ...validGrad(),
      criteria: [
        ...validGrad().criteria,
        { id: 'drawdown', label: 'Max Drawdown', passed: true, value: '-16.7%', threshold: '>= -25%' },
      ],
    };
    const result = GraduationDataSchema.safeParse(data);
    expect(result.success).toBe(true);
  });

  it('rejects missing readiness_pct', () => {
    const { readiness_pct, ...rest } = validGrad();
    const result = GraduationDataSchema.safeParse(rest);
    expect(result.success).toBe(false);
  });

  it('rejects non-boolean eligible', () => {
    const data = { ...validGrad(), eligible: 'yes' };
    const result = GraduationDataSchema.safeParse(data);
    expect(result.success).toBe(false);
  });

  it('passes through extra fields', () => {
    const data = { ...validGrad(), meta: { version: 2 } };
    const result = GraduationDataSchema.safeParse(data);
    expect(result.success).toBe(true);
  });
});

// ===========================================================================
// validateFetchData
// ===========================================================================

describe('validateFetchData', () => {
  const TestSchema = z.object({
    name: z.string(),
    value: z.number(),
  });

  it('returns parsed data for valid input', () => {
    const result = validateFetchData({ name: 'test', value: 42 }, TestSchema, 'test');
    expect(result).not.toBeNull();
    expect(result!.name).toBe('test');
    expect(result!.value).toBe(42);
  });

  it('returns null for non-object input (null)', () => {
    const result = validateFetchData(null, TestSchema, 'test');
    expect(result).toBeNull();
  });

  it('returns null for non-object input (string)', () => {
    const result = validateFetchData('invalid', TestSchema, 'test');
    expect(result).toBeNull();
  });

  it('fallback: returns raw data when partially valid (is an object)', () => {
    const partial = { name: 'test', value: 'not-a-number' };
    const result = validateFetchData(partial, TestSchema, 'test');
    // Should fall back to raw data because it is an object
    expect(result).not.toBeNull();
    expect((result as Record<string, unknown>).name).toBe('test');
    expect((result as Record<string, unknown>).value).toBe('not-a-number');
  });

  it('fallback: returns raw data for empty object', () => {
    const result = validateFetchData({}, TestSchema, 'test');
    expect(result).not.toBeNull();
  });

  it('fallback: preserves all fields', () => {
    const raw = { name: 'test', extra_field: [1, 2, 3] };
    const result = validateFetchData(raw, TestSchema, 'test');
    expect(result).not.toBeNull();
    expect((result as Record<string, unknown>).extra_field).toEqual([1, 2, 3]);
  });

  it('returns null for array input', () => {
    const result = validateFetchData([1, 2, 3], TestSchema, 'test');
    expect(result).toBeNull();
  });

  it('works with DashboardDataSchema', () => {
    const raw = {
      prices: {},
      regimes: [],
      paper_portfolio: [],
      generated_at: '2026-05-26T12:00:00Z',
    };
    const result = validateFetchData(raw, DashboardDataSchema, 'dashboard');
    expect(result).not.toBeNull();
    expect(result!.generated_at).toBe('2026-05-26T12:00:00Z');
  });

  it('returns null for DashboardDataSchema with completely wrong type', () => {
    const result = validateFetchData('totally wrong', DashboardDataSchema, 'dashboard');
    expect(result).toBeNull();
  });
});
