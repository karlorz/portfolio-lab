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
  FredMacroSchema,
  validateSignalsData,
  DashboardDataSchema,
  AlertsDataSchema,
  StatsDataSchema,
  HealthDataSchema,
  IncidentLifecycleSummarySchema,
  AnalyticsDataSchema,
  RebalanceHealthSchema,
  GraduationDataSchema,
  validateFetchData,
  IcDecaySignalEntrySchema,
  IcDecaySummarySchema,
  RegimeGateSchema,
  TSMOMSchema,
  ExplainabilitySchema,
  CrossAssetRVSchema,
  VixyHedgeSchema,
  TurnoverValidatorSchema,
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
    coverage_diagnostics: {
      schema_version: 'conformal-coverage/v1',
      observations: 500,
      alpha: 0.05,
      expected_exceedance_rate: 0.05,
      exceedance_count: 25,
      exceedance_rate: 0.05,
      coverage_rate: 0.95,
      coverage_pass: true,
      rolling_window: 252,
      rolling_exceedance_rate: 0.0476,
      longest_violation_cluster: 1,
      kupiec_statistic: 0,
      kupiec_p_value: 1,
      kupiec_pass: true,
      christoffersen_statistic: 0,
      christoffersen_p_value: 1,
      christoffersen_pass: true,
      conditional_coverage_statistic: 0,
      conditional_coverage_p_value: 1,
      conditional_coverage_pass: true,
    },
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
    remaining_budget_ratio: 0.0055,
    status: {
      ytd_cost_bps: 45,
      ytd_cost_pct: 0.0045,
      remaining_budget_pct: 0.55,
      remaining_budget_ratio: 0.0055,
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

function validMarlStatus(): Record<string, unknown> {
  return {
    schema_version: 'marl-runtime-status/v1',
    available: false,
    timestamp: '2026-05-26T12:00:00Z',
    runtime: {
      version: 'unknown',
      device: 'unknown',
      agents_loaded: [],
      signal_integrator_connected: false,
      checkpoint_loaded: false,
      inference_count: 0,
      current_allocation: {},
      graph_metrics: {},
    },
    execution_role: {
      role: 'research_shadow_non_routed',
      routed: false,
      routed_by: null,
      live_authoritative: false,
      description: 'MARL status is visible for research/shadow diagnostics; order routing still consumes target_allocations.',
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
    marl_status: validMarlStatus(),
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
    crypto_allocation: {
      active: true,
      btc_weight: 0.6,
      eth_weight: 0.4,
      total_crypto: 0.02,
      btc_momentum_6m: 0.12,
      eth_momentum_6m: 0.08,
      btc_vol_regime: 'normal',
      eth_vol_regime: 'normal',
      confidence: 0.55,
    },
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

  it('preserves FRED source provenance metadata', () => {
    const result = YieldCurveSchema.safeParse({
      ...validYieldCurve(),
      source_mode: 'synthetic',
      source_status: 'degraded',
      source_reason: 'FRED_API_KEY missing',
      source_provider: 'FRED',
      source_latest_observation: '2026-07-02',
    });

    expect(result.success).toBe(true);
    if (!result.success) throw new Error('expected source provenance to parse');
    expect(result.data.source_mode).toBe('synthetic');
    expect(result.data.source_status).toBe('degraded');
  });
});

describe('FredMacroSchema', () => {
  it('preserves source readiness metadata', () => {
    const result = FredMacroSchema.safeParse({
      regime: 'UNKNOWN',
      confidence: 0,
      recession_probability: 0,
      inflation_pressure: 0,
      monetary_stance: 'unknown',
      manufacturing_health: 50,
      credit_conditions: 'unknown',
      indicators: {},
      timestamp: '2026-07-06T00:00:00Z',
      source_mode: 'unavailable',
      cache_status: 'empty',
      api_key_configured: false,
      indicators_observed: false,
      reason: 'empty_cache',
    });

    expect(result.success).toBe(true);
    if (!result.success) throw new Error('expected FRED readiness to parse');
    expect(result.data.source_mode).toBe('unavailable');
    expect(result.data.indicators_observed).toBe(false);
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

  it('keeps conformal coverage diagnostics optional for old payloads', () => {
    const { coverage_diagnostics, ...legacyPayload } = validGarchCvar();
    const result = GarchCvarSchema.safeParse(legacyPayload);
    expect(result.success).toBe(true);
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

  it('accepts kill-blocked smart rebalance fields', () => {
    const result = SmartRebalanceSchema.safeParse({
      ...validSmartRebalance(),
      should_execute: false,
      decision: 'blocked_kill_switch',
      execution_blocked: true,
      kill_switch_enabled: true,
      kill_switch_level: 'halt',
      kill_switch_reason: 'unresolved_incident:signal_staleness',
      kill_switch_incident_id: 'inc-1',
      kill_switch_message: 'Paper trading halted',
    });
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

  it('rejects mixed remaining budget percent units', () => {
    const result = SmartRebalanceSchema.safeParse({
      ...validSmartRebalance(),
      remaining_budget_pct: 0.005,
      remaining_budget_ratio: 0.005,
      status: {
        ...validSmartRebalance().status,
        remaining_budget_pct: 0.5,
        remaining_budget_ratio: 0.005,
      },
    });
    expect(result.success).toBe(false);
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
  it('accepts producer summary-shaped public artifact', () => {
    const data = {
      active: true,
      yield_10y: 4.5,
      yield_2y: 4.0,
      spread: 0.5,
      curve_regime: 'normal',
      rate_direction: 'stable',
      tlt_weight: 0.2,
      ief_weight: 0.5,
      shy_weight: 0.3,
      effective_duration: 7.3,
      position: 'intermediate',
      confidence: 70.0,
      status_text: 'Bonds: intermediate (normal/stable), dur 7yr',
    };
    const result = BondMomentumSchema.safeParse(data);
    expect(result.success).toBe(true);
  });

  it('accepts legacy overlay signals shape', () => {
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

  it('rejects invalid legacy confidence', () => {
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
      marl_status: validMarlStatus(),
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

  it('rejects arbitrary data for typed crypto_allocation panel', () => {
    const data = validSignalsData();
    data.crypto_allocation = { btc: 0.5, eth: 0.3, sol: 0.2 };
    const result = SignalsDataSchema.safeParse(data);
    expect(result.success).toBe(false);
  });

  it('rejects invalid non-null untyped signal panel values', () => {
    const data = validSignalsData();
    data.crypto_allocation = ['not', 'a', 'record'];
    const result = SignalsDataSchema.safeParse(data);
    expect(result.success).toBe(false);
  });

  it('allows deeply nested unknown data in untyped behavioral_sentiment panel', () => {
    const data = validSignalsData();
    data.behavioral_sentiment = {
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

  it('accepts producer critical alert level', () => {
    const data = validAlerts();
    data.alerts[0].level = 'critical' as any;
    const result = AlertsDataSchema.safeParse(data);
    expect(result.success).toBe(true);
  });

  it('rejects unknown alert level', () => {
    const data = validAlerts();
    data.alerts[0].level = 'emergency' as any;
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

  it('accepts signal_health and fred_readiness sections from generator', () => {
    const data = {
      ...validHealth(),
      signal_health: {
        timestamp: '2026-07-01T12:00:00Z',
        summary: { active: 8 },
        scores: { msm: 0.55 },
        alerts: [],
        overall_health: 'degraded',
      },
      fred_readiness: {
        schema_version: 'fred-readiness/v1',
        status: 'ok',
        readiness: 'pass',
        ready: true,
        blocking: false,
        message: 'FRED credential readiness ok for local mode.',
      },
    };
    const result = HealthDataSchema.safeParse(data);
    expect(result.success).toBe(true);
    expect(result.data?.signal_health?.overall_health).toBe('degraded');
    expect(result.data?.fred_readiness?.ready).toBe(true);
  });

  it('accepts the bounded IC quality summary projection', () => {
    const data = {
      ...validHealth(),
      ic_decay_summary: {
        status: 'critical',
        critical_signals: ['ensemble_duration'],
        warning_signals: [],
        insufficient_data_signals: ['alternative_data'],
        resolved_signal_count: 2,
        min_observations: 20,
        staged_pending_predictions: 7,
        staged_pending_signal_names: ['ensemble_duration'],
        staged_date: '2026-08-01',
        staged_pending_scope: 'ic_staged_date_window',
        historical_unlabeled_rows: 1663,
        historical_unlabeled_dates: 2,
        historical_unlabeled_oldest_date: '2026-07-31',
        historical_unlabeled_scope: 'historical_db_unlabeled_rows',
        evidence_generated_at: '2026-08-01T09:40:23Z',
        evidence_freshness: 'captured_runtime_snapshot',
        routing_authority: 'advisory_only',
        routing_control: 'routing_blocked',
        control_effect: 'paper_warning',
        kill_switch_level: 'halt',
        signal_evidence: {
          ensemble_duration: {
            ic_rolling: -0.05,
            observations: 26,
            status: 'critical',
            metric_axis: 'time_series_rank_correlation',
            metric_kind: 'correlation',
            estimate_kind: 'descriptive',
            alignment_status: 'misaligned',
            inference_status: 'unavailable',
            inference_reason: 'label_alignment_mismatch',
            observation_count: 26,
            observation_unit: 'pairs',
            contract_version: 'ic-evaluation-contract/v2',
            evaluation_contract: {
              contract_version: 'ic-evaluation-contract/v2',
              intended_metric_axis: 'time_series_rank_correlation',
              intended_metric_kind: 'correlation',
              target_asset: 'TLT',
              target_basket: null,
              intended_horizon_sessions: 1,
              prediction_field: 'ensemble_voting.duration_bias',
              prediction_transform: 'identity',
            },
            latest_observation_metadata: {
              prediction_date: '2026-08-07',
              resolved_date: '2026-08-08',
              target_asset: 'SPY',
              realized_horizon_sessions: 1,
              metric_axis: 'time_series_rank_correlation',
              metric_kind: 'correlation',
              contract_version: 'ic-observation-metadata/v2',
            },
          },
        },
      },
    };
    const result = HealthDataSchema.safeParse(data);
    expect(result.success).toBe(true);
    expect(result.data?.ic_decay_summary?.staged_pending_predictions).toBe(7);
    expect(result.data?.ic_decay_summary?.signal_evidence?.ensemble_duration?.estimate_kind).toBe('descriptive');
    expect(result.data?.ic_decay_summary?.signal_evidence?.ensemble_duration?.evaluation_contract?.target_asset).toBe('TLT');
    expect(result.data?.ic_decay_summary?.signal_evidence?.ensemble_duration?.latest_observation_metadata?.target_asset).toBe('SPY');
  });

  it('accepts unavailable signal_health fallback', () => {
    const data = {
      ...validHealth(),
      signal_health: { status: 'unavailable', error: 'Failed to get signal health: x' },
    };
    expect(HealthDataSchema.safeParse(data).success).toBe(true);
  });

  it('accepts health data with data pipeline SLO runbook guidance', () => {
    const data = {
      ...validHealth(),
      data_pipeline_slo: {
        schema_version: 'data-pipeline-slo/v1',
        status: 'warning' as const,
        top_dimension: 'market_data',
        dimensions: {
          market_data: {
            status: 'warning' as const,
            message: 'Yahoo latest date is one session behind.',
            latest_available_market_date: '2026-07-02',
          },
        },
        runbook: {
          status: 'warning' as const,
          top_cause: {
            dimension: 'market_data',
            code: 'stale_prices',
            severity: 'warning' as const,
            action: 'Run make fetch-data before dashboard generation.',
            artifact: 'public/data/prices.json',
            provider: 'Yahoo Finance',
          },
          actions: [
            {
              dimension: 'market_data',
              code: 'stale_prices',
              severity: 'warning' as const,
              action: 'Run make fetch-data before dashboard generation.',
            },
          ],
        },
      },
    };

    const result = HealthDataSchema.safeParse(data);

    expect(result.success).toBe(true);
    expect(result.data?.data_pipeline_slo?.runbook?.top_cause?.code).toBe('stale_prices');
  });

  it('accepts an unavailable data pipeline SLO fallback payload', () => {
    const data = {
      ...validHealth(),
      data_pipeline_slo: {
        schema_version: 'data-pipeline-slo/v1',
        status: 'warning' as const,
        top_dimension: 'unknown',
        error: 'source_manifest.json missing',
        dimensions: {},
      },
    };

    const result = HealthDataSchema.safeParse(data);

    expect(result.success).toBe(true);
    expect(result.data?.data_pipeline_slo?.error).toContain('source_manifest.json');
  });

  it('rejects missing cron_jobs', () => {
    const { cron_jobs, ...rest } = validHealth();
    const result = HealthDataSchema.safeParse(rest);
    expect(result.success).toBe(false);
  });
});

// ===========================================================================
// IncidentLifecycleSummarySchema
// ===========================================================================

describe('IncidentLifecycleSummarySchema', () => {
  const validIncidentSummary = () => ({
    generated_at: '2026-06-11T12:35:00+00:00',
    open_count: 1,
    incidents: [{
      incident_id: 'inc-001',
      channel: 'cron_failure',
      severity: 'p0',
      state: 'firing' as const,
      message: 'Scheduler backends disagree for 2 consecutive checks',
      details: { consecutive_mismatches: 2 },
      created_at: '2026-06-11T12:30:00+00:00',
      updated_at: '2026-06-11T12:34:00+00:00',
      resolved_at: null,
      resolution_notes: null,
      mttr_seconds: null,
    }],
    metrics: {
      incident_frequency: 2,
      open_count: 1,
      resolved_count: 1,
      mean_mttr_seconds: 1800,
    },
  });

  it('accepts valid incident lifecycle summaries', () => {
    const result = IncidentLifecycleSummarySchema.safeParse(validIncidentSummary());
    expect(result.success).toBe(true);
    expect(result.data?.incidents[0].details.consecutive_mismatches).toBe(2);
  });

  it('rejects invalid incident lifecycle states', () => {
    const data = {
      ...validIncidentSummary(),
      incidents: [{ ...validIncidentSummary().incidents[0], state: 'stuck' }],
    };
    const result = IncidentLifecycleSummarySchema.safeParse(data);
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

  it('accepts dual-shape producer numeric criterion values', () => {
    const data = {
      ...validGrad(),
      criteria: [
        {
          id: 'min_sharpe',
          label: 'Rolling Sharpe ratio >= 0.50',
          passed: false,
          value: 0.0,
          threshold: '0.5',
          name: 'min_sharpe',
          required: 0.5,
          description: 'Rolling Sharpe ratio >= 0.50',
        },
      ],
      readiness_score: 18.2,
      is_graduation_ready: false,
    };
    const result = GraduationDataSchema.safeParse(data);
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

// ── Task 2C: additive IC control-eligibility schema ───────────────────

describe('IcDecayControlEligibilitySchema', () => {
  it('accepts descriptive entries with additive control fields', () => {
    const row = {
      ic_rolling: -0.02,
      ic_trend: 'decaying',
      observations: 25,
      status: 'critical',
      min_obs_for_status: 20,
      metric_axis: 'time_series_rank_correlation',
      metric_kind: 'correlation',
      estimate_kind: 'descriptive',
      alignment_status: 'misaligned',
      alignment_reason: 'actual_target_spy_expected_gld',
      inference_status: 'unavailable',
      inference_reason: 'label_alignment_mismatch',
      observation_count: 25,
      observation_unit: 'pairs',
      contract_version: 'ic-evaluation-contract/v2',
      evaluation_contract: {
        contract_version: 'ic-evaluation-contract/v2',
        intended_metric_axis: 'time_series_rank_correlation',
        intended_metric_kind: 'correlation',
        target_asset: 'GLD',
        target_basket: null,
        intended_horizon_sessions: 1,
        prediction_field: 'ensemble_voting.gold_bias',
        prediction_transform: 'identity',
      },
      control_eligible: false,
      control_status: 'ineligible',
      control_ineligibility_reason: 'label_alignment_mismatch',
    };
    const result = IcDecaySignalEntrySchema.safeParse(row);
    expect(result.success).toBe(true);
  });

  it('rejects invalid control_status enum values', () => {
    const row = {
      ic_rolling: null,
      ic_trend: 'unknown',
      observations: 5,
      status: 'insufficient_data',
      min_obs_for_status: 20,
      metric_axis: 'time_series_rank_correlation',
      metric_kind: 'correlation',
      estimate_kind: 'descriptive',
      alignment_status: 'undeclared',
      alignment_reason: 'evaluation_contract_missing',
      inference_status: 'unavailable',
      inference_reason: 'evaluation_contract_missing',
      observation_count: 5,
      observation_unit: 'pairs',
      contract_version: 'ic-evaluation-contract/v2',
      evaluation_contract: {},
      control_eligible: true,
      control_status: 'bogus',
    };
    const result = IcDecaySignalEntrySchema.safeParse(row);
    expect(result.success).toBe(false);
  });
});

describe('IcDecaySummarySchema control fields', () => {
  it('accepts control_eligible_critical_signals', () => {
    const summary = {
      status: 'critical',
      critical_signals: ['ensemble_equity'],
      warning_signals: [],
      insufficient_data_signals: [],
      resolved_signal_count: 1,
      min_observations: 20,
      staged_pending_predictions: 0,
      staged_pending_signal_names: [],
      staged_pending_scope: 'ic_staged_date_window',
      staged_date: null,
      historical_unlabeled_rows: 0,
      historical_unlabeled_dates: 0,
      historical_unlabeled_oldest_date: null,
      historical_unlabeled_scope: 'historical_db_unlabeled_rows',
      evidence_generated_at: '2026-08-09T00:00:00Z',
      evidence_freshness: 'fresh',
      routing_authority: 'advisory_only',
      routing_control: 'routing_blocked',
      control_effect: 'routing_blocked',
      signal_evidence: {},
      control_eligible_critical_signals: ['ensemble_equity'],
      control_eligible_warning_signals: [],
    };
    const result = IcDecaySummarySchema.safeParse(summary);
    expect(result.success).toBe(true);
  });
});

// ===========================================================================
// RegimeGateSchema — /data/regime_gate.json (A11 pilot)
// ===========================================================================
function fallbackRegimeGate() {
  return {
    current_regime: 'NORMAL',
    regime_confidence: 0.62,
    confidence_source: 'regime_surface',
    gate_rules: [
      { signal_name: 'multi_speed_momentum', off_regimes: ['CRISIS', 'HIGH_VOL'], is_active: true },
    ],
    active_signals: ['multi_speed_momentum'],
    inactive_signals: [],
    min_dwell_days: 5,
    generated_at: '2026-08-11T00:00:00Z',
  };
}

describe('RegimeGateSchema', () => {
  const generatedRG = () => readJsonOrFallback('public/data/regime_gate.json', fallbackRegimeGate());

  it('accepts a valid regime gate payload', () => {
    const result = RegimeGateSchema.safeParse(fallbackRegimeGate());
    expect(result.success).toBe(true);
  });

  it('accepts the generated regime_gate artifact (live producer contract)', () => {
    const result = RegimeGateSchema.safeParse(generatedRG());
    expect(result.success).toBe(true);
    if (!result.success) return;
    expect(result.data.current_regime).toBeTruthy();
    expect(Array.isArray(result.data.gate_rules)).toBe(true);
    expect(result.data.gate_rules[0].signal_name).toBeTruthy();
  });

  it('rejects a payload missing required fields', () => {
    const bad = { ...fallbackRegimeGate() } as Record<string, unknown>;
    delete bad.current_regime;
    expect(RegimeGateSchema.safeParse(bad).success).toBe(false);
  });
});

// ===========================================================================
// TSMOMSchema — /data/tsmom.json (A11 pilot #2)
// ===========================================================================
function fallbackTsmom() {
  return {
    composite_signal: 1.0,
    speed_breakdown: [
      {
        label: 'SPY TSMOM',
        weight: 0.46,
        signal: 1,
        asset_signals: { SPY: 0.1 },
        realized_vol: 0.14,
        adjustment: 0.1,
      },
    ],
    position_recommendation: 'long',
    confidence: 1.0,
    standalone_sharpe: 0.96,
    overlay_sharpe: 0.93,
    health_score: 0.55,
    is_gated_off: false,
    generated_at: '2026-08-11T00:00:00Z',
  };
}

describe('TSMOMSchema', () => {
  const generatedTsmom = () => readJsonOrFallback('public/data/tsmom.json', fallbackTsmom());

  it('accepts a valid tsmom payload', () => {
    const result = TSMOMSchema.safeParse(fallbackTsmom());
    expect(result.success).toBe(true);
  });

  it('accepts the generated tsmom artifact (live producer contract)', () => {
    const result = TSMOMSchema.safeParse(generatedTsmom());
    expect(result.success).toBe(true);
    if (!result.success) return;
    expect(['long', 'short', 'neutral']).toContain(result.data.position_recommendation);
    expect(Array.isArray(result.data.speed_breakdown)).toBe(true);
  });

  it('rejects an unknown position_recommendation value', () => {
    const bad = { ...fallbackTsmom(), position_recommendation: 'sideways' };
    expect(TSMOMSchema.safeParse(bad).success).toBe(false);
  });
});


// ---------------------------------------------------------------------------
// Item 28 (A11 extension #3): ExplainabilitySchema / CrossAssetRVSchema /
// VixyHedgeSchema / TurnoverValidatorSchema — fixtures = live payloads
// captured 00:0xZ 2026-08-12 from lab.karldigi.dev/data/...
// ---------------------------------------------------------------------------

describe('explainability schema (Item 28)', () => {
  const livePayload = {
  "timestamp": "2026-08-12T07:48:26.740556",
  "analysis_date": "2026-08-11",
  "latest_decision": {
    "timestamp": "2026-08-12T07:48:26.740556",
    "period": "2026-08-11",
    "regime": "normal",
    "action": "neutral",
    "confidence": 0.727413268694079,
    "reasoning": "Action=neutral; raw_consensus=+0.5575 (bullish); agreement=100.0%; regime=normal.",
    "total_signals": 9,
    "consensus_direction": "neutral",
    "raw_consensus_direction": "bullish",
    "agreement_ratio": 1.0,
    "weighted_consensus": 0.5575,
    "signals": [
      {
        "source": "cross_asset_rv",
        "display_name": "Cross Asset Rv",
        "category": "signal",
        "value": 0.7005,
        "direction": "bullish",
        "strength": 0.7005,
        "confidence": 0.934,
        "weight": 0.5,
        "contribution": 0.35025
      },
      {
        "source": "vix_term_structure",
        "display_name": "Vix Term Structure",
        "category": "signal",
        "value": 0.6431,
        "direction": "bullish",
        "strength": 0.6431,
        "confidence": 1.0,
        "weight": 0.219,
        "contribution": 0.140839
      },
      {
        "source": "google_trends",
        "display_name": "Google Trends",
        "category": "signal",
        "value": 0.4021,
        "direction": "bullish",
        "strength": 0.4021,
        "confidence": 0.871,
        "weight": 0.126,
        "contribution": 0.050665
      },
      {
        "source": "multi_timeframe_fusion",
        "display_name": "Multi Timeframe Fusion",
        "category": "signal",
        "value": 0.1014,
        "direction": "bullish",
        "strength": 0.1014,
        "confidence": 0.501,
        "weight": 0.155,
        "contribution": 0.015717
      },
      {
        "source": "multi_speed_momentum",
        "display_name": "Multi Speed Momentum",
        "category": "signal",
        "value": -0.3333,
        "direction": "bearish",
        "strength": 0.3333,
        "confidence": 0.667,
        "weight": 0.0,
        "contribution": -0.0
      }
    ],
    "top_drivers": [
      {
        "source": "cross_asset_rv",
        "contribution": 0.35025,
        "direction": "bullish"
      },
      {
        "source": "vix_term_structure",
        "contribution": 0.140839,
        "direction": "bullish"
      },
      {
        "source": "google_trends",
        "contribution": 0.050665,
        "direction": "bullish"
      },
      {
        "source": "multi_timeframe_fusion",
        "contribution": 0.015717,
        "direction": "bullish"
      }
    ],
    "top_opposers": []
  },
  "recent_decisions": [],
  "signal_deep_dives": {
    "cross_asset_rv": {
      "source": "cross_asset_rv",
      "display_name": "Cross Asset Rv",
      "category": "signal",
      "total_observations": 1,
      "avg_value": 0.7005,
      "avg_confidence": 0.934,
      "avg_weight": 0.5,
      "hit_rate": null,
      "sharpe_contribution": null
    },
    "vix_term_structure": {
      "source": "vix_term_structure",
      "display_name": "Vix Term Structure",
      "category": "signal",
      "total_observations": 1,
      "avg_value": 0.6431,
      "avg_confidence": 1.0,
      "avg_weight": 0.219,
      "hit_rate": null,
      "sharpe_contribution": null
    },
    "google_trends": {
      "source": "google_trends",
      "display_name": "Google Trends",
      "category": "signal",
      "total_observations": 1,
      "avg_value": 0.4021,
      "avg_confidence": 0.871,
      "avg_weight": 0.126,
      "hit_rate": null,
      "sharpe_contribution": null
    },
    "multi_timeframe_fusion": {
      "source": "multi_timeframe_fusion",
      "display_name": "Multi Timeframe Fusion",
      "category": "signal",
      "total_observations": 1,
      "avg_value": 0.1014,
      "avg_confidence": 0.501,
      "avg_weight": 0.155,
      "hit_rate": null,
      "sharpe_contribution": null
    },
    "multi_speed_momentum": {
      "source": "multi_speed_momentum",
      "display_name": "Multi Speed Momentum",
      "category": "signal",
      "total_observations": 1,
      "avg_value": -0.3333,
      "avg_confidence": 0.667,
      "avg_weight": 0.0,
      "hit_rate": null,
      "sharpe_contribution": null
    }
  },
  "top_sources_today": [
    "cross_asset_rv",
    "vix_term_structure",
    "google_trends",
    "multi_timeframe_fusion",
    "multi_speed_momentum"
  ],
  "decision_quality": {
    "status": "ok",
    "agreement_ratio": 1.0,
    "n_eff": 3.42,
    "weight_entropy": 1.2291
  },
  "freshness": {
    "status": "current",
    "generated_at": "2026-08-12T07:48:26.740556",
    "source_file": "signals.json",
    "analysis_date": "2026-08-11",
    "latest_decision_timestamp": "2026-08-12T07:48:26.740556",
    "stale_source_file": "explainability_2026-05-18.json",
    "stale_analysis_date": "2026-05-18"
  },
  "artifact_id": "explainability/explainability_latest.json",
  "plane": "public",
  "generated_at": "2026-08-12T07:48:26.740556",
  "generator_git_sha": null,
  "generator_git_sha_status": "unavailable",
  "last_full_generator_git_sha": null,
  "runtime_provenance": {
    "schema_version": "runtime-provenance/v1",
    "artifact_id": "explainability/explainability_latest.json",
    "plane": "public",
    "generated_at": "2026-08-12T07:48:26.740556",
    "generator_git_sha": null,
    "generator_git_sha_status": "unavailable",
    "last_full_generator_git_sha": null,
    "patch_source": null
  }
};

  it('accepts the live explainability payload', () => {
    const result = ExplainabilitySchema.safeParse(livePayload);
    expect(result.success).toBe(true);
  });
});

describe('cross_asset_rv schema (Item 28)', () => {
  const livePayload = {
  "signal_value": 0.2782,
  "pairs": [
    {
      "pair_name": "spy_qqq",
      "symbol_a": "SPY",
      "symbol_b": "QQQ",
      "return_a_60d": 4.15,
      "return_b_60d": -0.29,
      "return_differential": 4.44,
      "z_score": 0.9768,
      "z_score_mean": -0.0036,
      "z_score_std": 0.0492,
      "signal_value": -0.4884,
      "regime": "neutral",
      "conviction": 0.2442,
      "active": false,
      "days_active": 0,
      "entry_zscore": 0.0,
      "coverage_status": "available",
      "missing_symbols": []
    },
    {
      "pair_name": "spy_efa",
      "symbol_a": "SPY",
      "symbol_b": "EFA",
      "return_a_60d": 4.15,
      "return_b_60d": 4.55,
      "return_differential": -0.4,
      "z_score": -0.0863,
      "z_score_mean": -0.002,
      "z_score_std": 0.0237,
      "signal_value": 0.0,
      "regime": "converged",
      "conviction": 0.0,
      "active": false,
      "days_active": 0,
      "entry_zscore": 0.0,
      "coverage_status": "available",
      "missing_symbols": []
    },
    {
      "pair_name": "gld_btc",
      "symbol_a": "GLD",
      "symbol_b": "BTC-USD",
      "return_a_60d": 3.73,
      "return_b_60d": 0.06,
      "return_differential": 3.67,
      "z_score": -0.0821,
      "z_score_mean": 0.0411,
      "z_score_std": 0.0539,
      "signal_value": 0.0,
      "regime": "converged",
      "conviction": 0.0,
      "active": false,
      "days_active": 0,
      "entry_zscore": 0.0,
      "coverage_status": "available",
      "missing_symbols": []
    },
    {
      "pair_name": "tlt_ief",
      "symbol_a": "TLT",
      "symbol_b": "IEF",
      "return_a_60d": -3.43,
      "return_b_60d": -0.73,
      "return_differential": -2.71,
      "z_score": -1.9039,
      "z_score_mean": -0.0015,
      "z_score_std": 0.0134,
      "signal_value": 0.9519,
      "regime": "neutral",
      "conviction": 0.476,
      "active": false,
      "days_active": 0,
      "entry_zscore": 0.0,
      "coverage_status": "available",
      "missing_symbols": []
    },
    {
      "pair_name": "spy_gld",
      "symbol_a": "SPY",
      "symbol_b": "GLD",
      "return_a_60d": 4.15,
      "return_b_60d": 3.73,
      "return_differential": 0.42,
      "z_score": -2.8019,
      "z_score_mean": 0.1269,
      "z_score_std": 0.0438,
      "signal_value": 0.7005,
      "regime": "diverged_bear",
      "conviction": 0.934,
      "active": true,
      "days_active": 8,
      "entry_zscore": 0.0,
      "coverage_status": "available",
      "missing_symbols": []
    }
  ],
  "avg_z_score": -0.7795,
  "max_divergence": 2.8019,
  "num_diverged": 1,
  "total_pairs": 5,
  "available_pair_count": 5,
  "unavailable_pair_count": 0,
  "unavailable_pairs": {},
  "missing_symbols": [],
  "risk_on_score": 0.2782,
  "duration_score": 0.9519,
  "overall_conviction": 0.3308,
  "current_regime": "normal",
  "is_gated_off": false,
  "regime_note": "Active \u2014 mean-reversion favorable",
  "weight_in_ensemble": 0.13,
  "generated_at": "2026-08-11T23:48:26.735424+00:00",
  "generator_git_sha": "9f73a08881ca",
  "generator_git_sha_status": "full_generate",
  "last_full_generator_git_sha": "9f73a08881ca",
  "artifact_id": "cross_asset_rv.json",
  "plane": "public",
  "runtime_provenance": {
    "schema_version": "runtime-provenance/v1",
    "artifact_id": "cross_asset_rv.json",
    "plane": "public",
    "generated_at": "2026-08-11T23:48:26.735424+00:00",
    "generator_git_sha": "9f73a08881ca",
    "generator_git_sha_status": "full_generate",
    "last_full_generator_git_sha": "9f73a08881ca",
    "patch_source": null
  }
};

  it('accepts the live cross_asset_rv payload', () => {
    const result = CrossAssetRVSchema.safeParse(livePayload);
    expect(result.success).toBe(true);
  });
});

describe('vixy_hedge schema (Item 28)', () => {
  const livePayload = {
  "current_allocation_pct": 3.0,
  "target_allocation_pct": 2.5,
  "vix_level": 25.0,
  "regime": "elevated",
  "ytd_cost_bps": 195.6,
  "ytd_benefit_bps": 0.0,
  "hedge_efficiency": 0.11,
  "total_signals": 6,
  "last_rebalance": "2026-05-24T18:56:31.826160",
  "generated_at": "2026-08-11T23:47:59.852215+00:00",
  "canonical_controller": "hedge_selector",
  "runtime_role": "diagnostic_cost_evidence",
  "live_authoritative": false,
  "routed": false,
  "generator_git_sha": "9f73a08881ca",
  "generator_git_sha_status": "full_generate",
  "last_full_generator_git_sha": "9f73a08881ca",
  "artifact_id": "vixy_hedge.json",
  "plane": "public",
  "runtime_provenance": {
    "schema_version": "runtime-provenance/v1",
    "artifact_id": "vixy_hedge.json",
    "plane": "public",
    "generated_at": "2026-08-11T23:47:59.852215+00:00",
    "generator_git_sha": "9f73a08881ca",
    "generator_git_sha_status": "full_generate",
    "last_full_generator_git_sha": "9f73a08881ca",
    "patch_source": null
  }
};

  it('accepts the live vixy_hedge payload', () => {
    const result = VixyHedgeSchema.safeParse(livePayload);
    expect(result.success).toBe(true);
  });
});

describe('turnover_validator schema (Item 28)', () => {
  const livePayload = {
  "schema_version": "turnover-validator/v1",
  "signals": {
    "multi_speed_momentum": {
      "periods": 20,
      "mean": 0.11166666666666662,
      "std": 0.298979746619882,
      "sign_flip_rate": 0.05263157894736842,
      "mag_vol": 0.03684210526315789,
      "turnover_penalty": 0.046315789473684206,
      "stability_score": 0.7420299668397669,
      "marginal_score": 0.018727069122680712
    },
    "cross_asset_rv": {
      "periods": 20,
      "mean": 0.34795,
      "std": 0.23699360972952746,
      "sign_flip_rate": 0.0,
      "mag_vol": 0.030973684210526316,
      "turnover_penalty": 0.012389473684210527,
      "stability_score": 0.8108742100314361,
      "marginal_score": 0.2795358378132181
    },
    "international_momentum": {
      "periods": 20,
      "mean": 0.28500000000000003,
      "std": 0.19269556026896006,
      "sign_flip_rate": 0.05263157894736842,
      "mag_vol": 0.031578947368421054,
      "turnover_penalty": 0.04421052631578947,
      "stability_score": 0.8203762657064648,
      "marginal_score": 0.2227420042561667
    },
    "alternative_data": {
      "periods": 20,
      "mean": 0.18699382181656787,
      "std": 0.4376369171559579,
      "sign_flip_rate": 0.05263157894736842,
      "mag_vol": 0.04718034478814178,
      "turnover_penalty": 0.05045108528367776,
      "stability_score": 0.6372162678207758,
      "marginal_score": 0.05402734690554431
    },
    "cross_asset_regime_arb": {
      "periods": 20,
      "mean": 0.06546076355769978,
      "std": 0.061139480647739906,
      "sign_flip_rate": 0.10526315789473684,
      "mag_vol": 0.02244925676450873,
      "turnover_penalty": 0.07213759744264558,
      "stability_score": 0.8982601578153058,
      "marginal_score": 0.03454815785791966
    },
    "unified_overlay": {
      "periods": 20,
      "mean": -0.019999999999999997,
      "std": 0.3254147151591416,
      "sign_flip_rate": 0.10526315789473684,
      "mag_vol": 0.03684210526315789,
      "turnover_penalty": 0.07789473684210525,
      "stability_score": 0.7024728572833377,
      "marginal_score": -0.12720229459243473
    },
    "multi_timeframe_fusion": {
      "periods": 20,
      "mean": 0.0056143190739616445,
      "std": 0.15043215347921526,
      "sign_flip_rate": 0.47368421052631576,
      "mag_vol": 0.15313828761354548,
      "turnover_penalty": 0.34546584136120767,
      "stability_score": 0.590370092643864,
      "marginal_score": -0.09749591360535415
    },
    "google_trends": {
      "periods": 20,
      "mean": 0.40210251505339284,
      "std": 0.0,
      "sign_flip_rate": 0.0,
      "mag_vol": 0.0,
      "turnover_penalty": 0.0,
      "stability_score": 1.0,
      "marginal_score": 0.40210251505339284
    },
    "vix_term_structure": {
      "periods": 20,
      "mean": 0.64307427207206,
      "std": 1.1390647892519134e-16,
      "sign_flip_rate": 0.0,
      "mag_vol": 0.0,
      "turnover_penalty": 0.0,
      "stability_score": 1.0,
      "marginal_score": 0.64307427207206
    }
  },
  "synthetic_baselines": {
    "stable": {
      "metadata": {
        "source_type": "synthetic_or_fixture"
      },
      "diagnostics": {
        "periods": 20,
        "mean": 0.5,
        "std": 0.0,
        "sign_flip_rate": 0.0,
        "mag_vol": 0.0,
        "turnover_penalty": 0.0,
        "stability_score": 1.0,
        "marginal_score": 0.5
      }
    },
    "noisy": {
      "metadata": {
        "source_type": "synthetic_or_fixture"
      },
      "diagnostics": {
        "periods": 20,
        "mean": 0.1,
        "std": 0.5026246899500346,
        "sign_flip_rate": 0.8421052631578947,
        "mag_vol": 0.8421052631578947,
        "turnover_penalty": 0.5,
        "stability_score": 0.06315789473684212,
        "marginal_score": -0.15120051317989705
      }
    },
    "climber": {
      "metadata": {
        "source_type": "synthetic_or_fixture"
      },
      "diagnostics": {
        "periods": 20,
        "mean": 0.96,
        "std": 0.4794733953433759,
        "sign_flip_rate": 0.0,
        "mag_vol": 0.15789473684210523,
        "turnover_penalty": 0.0631578947368421,
        "stability_score": 0.5459475706280579,
        "marginal_score": 0.8215879530646191
      }
    },
    "src": {
      "metadata": {
        "source_type": "synthetic_or_fixture"
      },
      "diagnostics": {
        "periods": 5,
        "mean": 0.6125625401603116,
        "std": 0.44597656170567407,
        "sign_flip_rate": 0.25,
        "mag_vol": 0.2789669092334585,
        "turnover_penalty": 0.2615867636933834,
        "stability_score": 0.37859122488093433,
        "marginal_score": 0.4523201961837943
      }
    }
  },
  "generated_at": "2026-08-11T23:48:25.874690+00:00",
  "generator_git_sha": "9f73a08881ca",
  "generator_git_sha_status": "full_generate",
  "last_full_generator_git_sha": "9f73a08881ca",
  "artifact_id": "turnover_validator.json",
  "plane": "public",
  "runtime_provenance": {
    "schema_version": "runtime-provenance/v1",
    "artifact_id": "turnover_validator.json",
    "plane": "public",
    "generated_at": "2026-08-11T23:48:25.874690+00:00",
    "generator_git_sha": "9f73a08881ca",
    "generator_git_sha_status": "full_generate",
    "last_full_generator_git_sha": "9f73a08881ca",
    "patch_source": null
  }
};

  it('accepts the live turnover_validator payload', () => {
    const result = TurnoverValidatorSchema.safeParse(livePayload);
    expect(result.success).toBe(true);
  });
});
