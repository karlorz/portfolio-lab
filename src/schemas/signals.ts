import { z } from 'zod';
import type { SignalsData } from '../types/live';

// ---------------------------------------------------------------------------
// Regime
// ---------------------------------------------------------------------------
export const RegimeSchema = z.object({
  regime: z.string(),
  vix: z.nullable(z.number()),
  detected: z.nullable(z.string()),
});

// ---------------------------------------------------------------------------
// Yield Curve
// ---------------------------------------------------------------------------
export const YieldCurveSchema = z.object({
  spread2s10s: z.nullable(z.number()),
  dgs2: z.nullable(z.number()),
  dgs10: z.nullable(z.number()),
  duration_regime: z.nullable(
    z.enum(['steep', 'normal', 'flat', 'inverted'])
  ),
  spread_history: z.optional(z.array(z.number())),
});

// ---------------------------------------------------------------------------
// Duration Allocation
// ---------------------------------------------------------------------------
export const DurationAllocationSchema = z.object({
  tlt: z.number(),
  ief: z.number(),
  shy: z.number(),
  bil: z.number(),
});

// ---------------------------------------------------------------------------
// Positions & Orders
// ---------------------------------------------------------------------------
export const PositionSchema = z.object({
  symbol: z.string(),
  shares: z.number(),
  value: z.number(),
  weight: z.number(),
  unrealized: z.number(),
});

export const RecentOrderSchema = z.object({
  sym: z.string(),
  side: z.string(),
  shares: z.number(),
  value: z.number(),
});

// ---------------------------------------------------------------------------
// GARCH-CVaR
// ---------------------------------------------------------------------------
export const GarchCvarSchema = z.object({
  cvar_95: z.number(),
  cvar_95_garch: z.number(),
  var_95: z.number(),
  var_95_garch: z.number(),
  cvar_ratio: z.number(),
  garch_active: z.boolean(),
  current_volatility: z.number(),
  forecast_volatility: z.number(),
  volatility_clustering: z.enum(['low', 'normal', 'elevated', 'high']),
});

// ---------------------------------------------------------------------------
// Entropy
// ---------------------------------------------------------------------------
export const EntropySchema = z.object({
  shannon_entropy: z.number(),
  effective_n: z.number(),
  max_possible: z.number(),
  normalized_score: z.number(),
  concentration_risk: z.enum(['critical', 'high', 'medium', 'low', 'good']),
  hhi_index: z.number(),
  correlation_entropy: z.optional(z.number()),
  participation_ratio: z.optional(z.number()),
});

// ---------------------------------------------------------------------------
// Smart Rebalance
// ---------------------------------------------------------------------------
const SmartRebalanceConfigSchema = z.object({
  drift_threshold: z.number(),
  vpin_threshold: z.number(),
  optimal_window: z.string(),
  annual_cost_limit: z.string(),
});

const SmartRebalanceStatusSchema = z.object({
  ytd_cost_bps: z.number(),
  ytd_cost_pct: z.number(),
  remaining_budget_pct: z.number(),
  is_over_budget: z.boolean(),
  is_warning: z.boolean(),
  last_rebalance: z.nullable(z.string()),
  deferred_until: z.nullable(z.string()),
  config: SmartRebalanceConfigSchema,
});

export const SmartRebalanceSchema = z.object({
  should_execute: z.boolean(),
  decision: z.string(),
  urgency: z.enum(['low', 'moderate', 'high', 'emergency']),
  max_drift: z.number(),
  estimated_cost_bps: z.number(),
  reason: z.string(),
  drift_details: z.record(z.number()),
  vpin: z.number(),
  in_optimal_window: z.boolean(),
  ytd_cost_bps: z.number(),
  remaining_budget_pct: z.number(),
  status: SmartRebalanceStatusSchema,
});

// ---------------------------------------------------------------------------
// Broker
// ---------------------------------------------------------------------------
const BrokerPositionSchema = z.object({
  symbol: z.string(),
  qty: z.number(),
  market_value: z.number(),
  unrealized_pl: z.number(),
  side: z.string(),
});

const BrokerDriftSchema = z.object({
  symbol: z.string(),
  broker_qty: z.number(),
  local_qty: z.number(),
  drift_pct: z.number(),
});

const BrokerOrderSchema = z.object({
  symbol: z.string(),
  side: z.string(),
  qty: z.number(),
  status: z.string(),
  order_id: z.optional(z.string()),
  timestamp: z.string(),
  dry_run: z.boolean(),
  attempts: z.optional(z.number()),
});

export const BrokerSchema = z.object({
  connected: z.boolean(),
  positions: z.array(BrokerPositionSchema),
  drift: z.array(BrokerDriftSchema),
  recent_orders: z.array(BrokerOrderSchema),
  last_sync: z.nullable(z.string()),
  kill_switch: z.boolean(),
});

// ---------------------------------------------------------------------------
// Closing Auction
// ---------------------------------------------------------------------------
const MOCImbalanceSchema = z.object({
  symbol: z.string(),
  timestamp: z.string(),
  imbalance_shares: z.number(),
  paired_shares: z.number(),
  reference_price: z.number(),
  source: z.string(),
  imbalance_ratio: z.number(),
  direction_score: z.number(),
});

const ClosingAuctionSignalSchema = z.object({
  symbol: z.string(),
  timestamp: z.string(),
  direction: z.enum([
    'STRONG_BUY', 'BUY', 'WEAK_BUY', 'NEUTRAL', 'WEAK_SELL', 'SELL', 'STRONG_SELL',
  ]),
  direction_score: z.number(),
  confidence: z.enum(['high', 'medium', 'low', 'insufficient_data']),
  imbalance: MOCImbalanceSchema,
  entry_price: z.number(),
  target_exit_price: z.number(),
  stop_loss_price: z.optional(z.number()),
  historical_win_rate: z.nullable(z.number()),
  historical_count: z.number(),
  max_position_pct: z.number(),
  urgency: z.enum(['immediate', 'high', 'normal']),
  should_trade: z.boolean(),
});

export const ClosingAuctionSchema = z.object({
  signals: z.array(ClosingAuctionSignalSchema),
  last_update: z.nullable(z.string()),
  market_open: z.boolean(),
});

// ---------------------------------------------------------------------------
// Zero DTE
// ---------------------------------------------------------------------------
const ZeroDTEConfigSchema = z.object({
  max_portfolio_allocation: z.number(),
  max_weekly_positions: z.number(),
  position_size_pct: z.number(),
  min_vix: z.number(),
  max_vix: z.number(),
  delta_target: z.number(),
  min_premium_pct: z.number(),
  max_delta_exposure: z.number(),
  emergency_close_delta: z.number(),
  max_loss_pct: z.number(),
});

const ZeroDTEPositionSchema = z.object({
  id: z.string(),
  underlying: z.string(),
  option_type: z.enum(['call', 'put']),
  side: z.enum(['buy', 'sell']),
  strike: z.number(),
  expiration: z.string(),
  quantity: z.number(),
  entry_price: z.number(),
  entry_time: z.string(),
  entry_delta: z.number(),
  entry_theta: z.number(),
  current_delta: z.number(),
  current_theta: z.number(),
  current_underlying_price: z.number(),
  status: z.enum([
    'pending', 'open', 'closed', 'stopped',
    'expired_itm', 'expired_otm', 'rolled',
  ]),
  unrealized_pnl: z.optional(z.number()),
  realized_pnl: z.optional(z.number()),
  premium_collected: z.number(),
  delta_exposure: z.number(),
  notional_value: z.number(),
  close_reason: z.optional(
    z.enum([
      'expiration', 'profit_take', 'stop_loss', 'delta_stop',
      'time_exit', 'manual', 'roll', 'emergency',
    ])
  ),
});

export const ZeroDTESchema = z.object({
  positions: z.array(ZeroDTEPositionSchema),
  config: z.nullable(ZeroDTEConfigSchema),
  weekly_trades_used: z.number(),
  total_premium_collected_mtd: z.number(),
});

// ---------------------------------------------------------------------------
// Bond Momentum
// ---------------------------------------------------------------------------
const BondMomentumSignalSchema = z.object({
  etf: z.string(),
  timestamp: z.string(),
  signal: z.number(),
  position_size: z.number(),
  formation_return: z.number(),
  realized_vol: z.number(),
  formation_months: z.number(),
  volatility_target: z.number(),
  confidence: z.enum(['strong', 'moderate', 'weak']),
  action: z.enum(['increase', 'hold', 'reduce', 'avoid']),
  weight_delta: z.number(),
});

const BondMomentumEnsembleSchema = z.object({
  weight: z.number(),
  confidence: z.string(),
  action: z.string(),
  recommendation: z.string(),
});

export const BondMomentumSchema = z.object({
  signals: z.array(BondMomentumSignalSchema),
  timestamp: z.string(),
  ensemble: BondMomentumEnsembleSchema,
});

// ---------------------------------------------------------------------------
// VIX Term Structure
// ---------------------------------------------------------------------------
const VIXTermStructureLevelSchema = z.object({
  value: z.number(),
  timestamp: z.string(),
});

export const VIXTermStructureSchema = z.object({
  vix: VIXTermStructureLevelSchema,
  vix3m: VIXTermStructureLevelSchema,
  vix6m: z.optional(VIXTermStructureLevelSchema),
  slope: z.number(),
  roll_yield: z.number(),
  composite_signal: z.number(),
  regime: z.enum([
    'extreme_contango', 'steep_contango', 'mild_contango',
    'flat', 'backwardation', 'extreme_backwardation',
  ]),
  z_score: z.number(),
  percentile_1y: z.optional(z.number()),
});

// ---------------------------------------------------------------------------
// VIX Overlay
// ---------------------------------------------------------------------------
const VIXOverlayShiftSchema = z.object({
  date: z.string(),
  shifts: z.record(z.number()),
  signal_value: z.number(),
  regime: z.string(),
  new_allocation: z.record(z.number()),
});

export const VIXOverlaySchema = z.object({
  allocation: z.record(z.number()),
  last_shift_date: z.string(),
  shift_history: z.array(VIXOverlayShiftSchema),
  disabled_until: z.nullable(z.string()),
});

// ---------------------------------------------------------------------------
// ML Signals
// ---------------------------------------------------------------------------
const MLPredictionSchema = z.object({
  predicted_regime: z.string(),
  confidence: z.number(),
  probabilities: z.record(z.number()),
  heuristic: z.boolean(),
});

const MLFeatureSchema = z.object({
  vix_level: z.nullable(z.number()),
  trend_direction: z.number(),
  price_vs_sma20: z.number(),
  return_5d: z.number(),
  spy_correlation: z.number(),
});

const MLGridSearchSchema = z.object({
  available: z.boolean(),
  timestamp: z.nullable(z.string()),
  top_allocation: z.nullable(z.record(z.number())),
  sharpe: z.nullable(z.number()),
  volatility: z.nullable(z.number()),
});

const MLSignalsSchema = z.object({
  available: z.boolean(),
  timestamp: z.nullable(z.string()),
  predictions: z.record(MLPredictionSchema),
  features: z.record(MLFeatureSchema),
  grid_search: MLGridSearchSchema,
});

// ---------------------------------------------------------------------------
// SignalsData — main schema
// ---------------------------------------------------------------------------
export const SignalsDataSchema = z.object({
  // Required fields
  timestamp: z.string(),
  regime: RegimeSchema,
  latest_prices: z.record(z.number()),
  current_positions: z.array(PositionSchema),
  target_allocations: z.record(z.number()),
  cash: z.number(),
  total_value: z.number(),
  recent_orders: z.array(RecentOrderSchema),
  ml_signals: MLSignalsSchema,
  // generated_at is a runtime field not in the TS interface — keep as passthrough
  generated_at: z.optional(z.string()),

  // Optional typed fields
  yield_curve: z.optional(YieldCurveSchema),
  duration_allocation: z.optional(DurationAllocationSchema),
  smart_rebalance: z.optional(SmartRebalanceSchema),
  broker: z.optional(BrokerSchema),
  closing_auction: z.optional(ClosingAuctionSchema),
  zero_dte: z.optional(ZeroDTESchema),
  garch_cvar: z.optional(GarchCvarSchema),
  entropy: z.optional(EntropySchema),
  bond_momentum: z.optional(BondMomentumSchema),
  vix_term_structure: z.optional(VIXTermStructureSchema),
  vix_overlay: z.optional(VIXOverlaySchema),

  // Untyped signal panels — use z.unknown() so any shape passes
  behavioral_sentiment: z.optional(z.record(z.unknown())),
  crypto_allocation: z.optional(z.record(z.unknown())),
  calendar_seasonality: z.optional(z.record(z.unknown())),
  ensemble_voting: z.optional(z.record(z.unknown())),
  alternative_data: z.optional(z.record(z.unknown())),
  factor_rotation: z.optional(z.record(z.unknown())),
  stacking_ensemble: z.optional(z.record(z.unknown())),
  convexity_harvest: z.optional(z.record(z.unknown())),
  llm_sentiment: z.optional(z.record(z.unknown())),
  sector_rotation: z.optional(z.record(z.unknown())),
  factor_rotation_dashboard: z.optional(z.record(z.unknown())),
  collar: z.optional(z.record(z.unknown())),
  kurtosis_regime: z.optional(z.record(z.unknown())),
  volatility_parity: z.optional(z.record(z.unknown())),
  rebalance_health: z.optional(z.record(z.unknown())),
  risk_decomposition: z.optional(z.record(z.unknown())),
  spc_flags: z.optional(z.record(z.unknown())),
  staleness: z.optional(z.record(z.unknown())),
}).passthrough();

// ---------------------------------------------------------------------------
// Validation function with graceful degradation
// ---------------------------------------------------------------------------
export function validateSignalsData(raw: unknown): SignalsData | null {
  const result = SignalsDataSchema.safeParse(raw);
  if (result.success) {
    return result.data as SignalsData;
  }
  // Log validation errors in dev mode
  if (import.meta.env.DEV) {
    console.warn('Signal validation failed:', result.error.issues);
  }
  // Fallback: try to use raw data as-is if it looks like an object
  if (typeof raw === 'object' && raw !== null && 'timestamp' in raw) {
    return raw as SignalsData;
  }
  return null;
}
