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
  // Conformal CVaR cross-check (distribution-free)
  conformal_cvar_95: z.optional(z.nullable(z.number())),
  conformal_var_95: z.optional(z.nullable(z.number())),
  conformal_cvar_ratio: z.optional(z.nullable(z.number())),
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
  drift_details: z.record(z.string(), z.number()),
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
  shifts: z.record(z.string(), z.number()),
  signal_value: z.number(),
  regime: z.string(),
  new_allocation: z.record(z.string(), z.number()),
});

export const VIXOverlaySchema = z.object({
  allocation: z.record(z.string(), z.number()),
  last_shift_date: z.string(),
  shift_history: z.array(VIXOverlayShiftSchema),
  disabled_until: z.nullable(z.string()),
});

// ---------------------------------------------------------------------------
// Hedge Selector
// ---------------------------------------------------------------------------
export const HedgeSelectorSchema = z.object({
  available: z.boolean().default(false),
  generated_at: z.string().default(''),
  regime: z.string().default('unknown'),
  regime_confidence: z.number().default(0),
  primary_hedge: z.string().default('none'),
  primary_size_pct: z.number().default(0),
  secondary_hedge: z.nullable(z.string()).default(null),
  secondary_size_pct: z.number().default(0),
  expected_benefit_bps: z.number().default(0),
  expected_cost_bps: z.number().default(0),
  net_benefit_bps: z.number().default(0),
  cost_benefit_gate: z.boolean().default(false),
  kelly_fraction: z.number().default(0),
  confidence_scaled_size: z.number().default(0),
  min_hold_days: z.number().default(0),
  transition_cost_bps: z.number().default(0),
}).passthrough();

// ---------------------------------------------------------------------------
// ML Signals
// ---------------------------------------------------------------------------
const MLPredictionSchema = z.object({
  predicted_regime: z.string(),
  confidence: z.number(),
  probabilities: z.record(z.string(), z.number()),
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
  top_allocation: z.nullable(z.record(z.string(), z.number())),
  sharpe: z.nullable(z.number()),
  volatility: z.nullable(z.number()),
});

const MLSignalsSchema = z.object({
  available: z.boolean(),
  timestamp: z.nullable(z.string()),
  predictions: z.record(z.string(), MLPredictionSchema),
  features: z.record(z.string(), MLFeatureSchema),
  grid_search: MLGridSearchSchema,
});

// ---------------------------------------------------------------------------
// IcDecaySchema — IC decay monitoring for signal quality tracking
// ---------------------------------------------------------------------------
const IcDecaySignalEntrySchema = z.object({
  ic_rolling: z.nullable(z.number()),
  ic_trend: z.enum(['stable', 'decaying', 'improving', 'unknown']),
  observations: z.number(),
  status: z.enum(['healthy', 'warning', 'critical', 'insufficient_data']),
});

export const IcDecaySchema = z.object({
  signals: z.record(z.string(), IcDecaySignalEntrySchema).optional(),
  error: z.optional(z.string()),
}).passthrough();

// ---------------------------------------------------------------------------
// SignalWFESchema — Per-signal walk-forward validation
// ---------------------------------------------------------------------------
const SignalWFEEntrySchema = z.object({
  signal_name: z.string(),
  wfe: z.number(),
  mean_is_ic: z.number(),
  mean_oos_ic: z.number(),
  std_oos_ic: z.number(),
  n_windows: z.number(),
  positive_oos_ratio: z.number(),
  status: z.enum(['validated', 'weak', 'unvalidated', 'insufficient_data']),
});

export const SignalWFESchema = z.object({
  signals: z.record(z.string(), SignalWFEEntrySchema).optional(),
  error: z.optional(z.string()),
}).passthrough();

// ---------------------------------------------------------------------------
// FredMacroSchema — FRED-MD macro regime signal (before SignalsDataObjectSchema)
// ---------------------------------------------------------------------------
export const FredMacroSchema = z.object({
  regime: z.string(),
  confidence: z.number(),
  recession_probability: z.number(),
  inflation_pressure: z.number(),
  monetary_stance: z.string(),
  manufacturing_health: z.number(),
  credit_conditions: z.string(),
  indicators: z.record(z.string(), z.number()),
  timestamp: z.string(),
}).passthrough();

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

const OptionalPanelObjectSchema = z.union([
  z.record(z.string(), z.unknown()),
  z.null(),
]);

export const StackingEnsembleSchema = z.object({
  active: z.boolean(),
  stacking_available: z.boolean(),
  prediction_direction: z.string(),
  confidence: z.number(),
  probability_bullish: z.number(),
  probability_bearish: z.number(),
  probability_neutral: z.number(),
  fallback_used: z.boolean(),
  model_version: z.string(),
  voting_accuracy: z.number(),
  stacking_accuracy: z.number(),
  feature_count: z.nullable(z.number()),
  feature_count_metadata_available: z.boolean(),
  feature_count_source: z.enum([
    'model_metadata',
    'unavailable_no_model',
    'unavailable_missing_metadata',
  ]),
  runtime_mode: z.enum([
    'model_backed',
    'fallback_no_model',
    'fallback_weighted_voting',
  ]),
  model_backed: z.boolean(),
  operator_disclosure: z.string().min(1),
  latency_ms: z.number(),
  top_features: z.optional(z.array(z.object({
    name: z.string(),
    importance: z.number(),
  }).passthrough())),
  backtest_finding: z.optional(z.string()),
}).passthrough();

// Generated optional panels may be disabled as null, emitted as {}, or emitted
// in a legacy summary shape while their panel code remains defensive.
const optionalPanel = <T extends z.ZodTypeAny>(schema: T) => (
  z.optional(z.union([schema, OptionalPanelObjectSchema]))
);

function normalizeSignalsTimestamp(raw: unknown): unknown {
  if (!isPlainRecord(raw)) return raw;
  if (typeof raw.timestamp === 'string') return raw;
  if (typeof raw.generated_at !== 'string') return raw;
  return { ...raw, timestamp: raw.generated_at };
}

// ---------------------------------------------------------------------------
// SignalsData — main schema
// ---------------------------------------------------------------------------
const SignalsDataObjectSchema = z.object({
  // Required fields
  timestamp: z.string(),
  regime: RegimeSchema,
  latest_prices: z.record(z.string(), z.number()),
  current_positions: z.array(PositionSchema),
  target_allocations: z.record(z.string(), z.number()),
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
  closing_auction: optionalPanel(ClosingAuctionSchema),
  zero_dte: optionalPanel(ZeroDTESchema),
  garch_cvar: z.optional(GarchCvarSchema),
  entropy: z.optional(EntropySchema),
  bond_momentum: optionalPanel(BondMomentumSchema),
  vix_term_structure: optionalPanel(VIXTermStructureSchema),
  vix_overlay: z.optional(VIXOverlaySchema),
  hedge_selector: z.optional(z.nullable(HedgeSelectorSchema)),

  // Untyped signal panels can be null when the signal is disabled/unavailable.
  behavioral_sentiment: z.optional(OptionalPanelObjectSchema),
  crypto_allocation: z.optional(OptionalPanelObjectSchema),
  calendar_seasonality: z.optional(OptionalPanelObjectSchema),
  ensemble_voting: z.optional(OptionalPanelObjectSchema),
  alternative_data: z.optional(OptionalPanelObjectSchema),
  factor_rotation: z.optional(OptionalPanelObjectSchema),
  stacking_ensemble: optionalPanel(StackingEnsembleSchema),
  convexity_harvest: z.optional(OptionalPanelObjectSchema),
  llm_sentiment: z.optional(OptionalPanelObjectSchema),
  sector_rotation: z.optional(OptionalPanelObjectSchema),
  factor_rotation_dashboard: z.optional(OptionalPanelObjectSchema),
  collar: z.optional(OptionalPanelObjectSchema),
  kurtosis_regime: z.optional(OptionalPanelObjectSchema),
  bocd_regime: z.optional(z.object({
    regime: z.number(),
    regime_change_prob: z.number(),
    changepoint_count: z.number(),
    current_run_length: z.number(),
    hazard_rate: z.number(),
    threshold: z.number(),
    n_observations: z.number(),
    description: z.string(),
    timestamp: z.string(),
  })),
  volatility_parity: z.optional(z.record(z.string(), z.unknown())),
  rebalance_health: z.optional(z.record(z.string(), z.unknown())),
  broker_circuit_breaker: z.optional(z.object({
    state: z.enum(['closed', 'open', 'half-open']),
    fail_count: z.number(),
    reset_timeout: z.number(),
  })),
  risk_decomposition: z.optional(z.record(z.string(), z.unknown())),
  spc_flags: z.optional(z.record(z.string(), z.unknown())),
  staleness: z.optional(z.record(z.string(), z.unknown())),
  ic_decay: z.optional(IcDecaySchema),
  signal_wfe: z.optional(SignalWFESchema),
  fred_macro: z.optional(FredMacroSchema),
  ramp: z.optional(z.object({
    phase: z.string(),
    allocation_pct: z.number(),
    days_at_phase: z.number(),
    min_days_required: z.number(),
    max_drawdown_pct: z.number(),
    max_drawdown_allowed: z.number(),
    can_advance: z.boolean(),
    alpaca_status: z.record(z.string(), z.unknown()),
  })),
  gold_tlt_correlation: z.optional(z.object({
    current_correlation: z.number(),
    current_regime: z.string(),
    correlation_trend: z.string(),
    mean_correlation: z.number(),
    min_correlation: z.number(),
    max_correlation: z.number(),
    structural_breaks_count: z.number(),
    regimes_count: z.number(),
    implications: z.string(),
  })),
}).passthrough();

export const SignalsDataSchema = z.preprocess(normalizeSignalsTimestamp, SignalsDataObjectSchema);

// ---------------------------------------------------------------------------
// DashboardDataSchema — /data/dashboard.json
// ---------------------------------------------------------------------------
const RegimeEntrySchema = z.object({
  d: z.string(),
  r: z.string(),
  v: z.nullable(z.number()),
});

const PerformanceEntrySchema = z.object({
  t: z.string(),
  v: z.number(),
  r: z.number(),
});

export const DashboardDataSchema = z.object({
  prices: z.record(z.string(), z.array(z.object({
    d: z.string(),
    p: z.number(),
  }))),
  regimes: z.array(RegimeEntrySchema),
  paper_portfolio: z.array(PerformanceEntrySchema),
  generated_at: z.string(),
}).passthrough();

// ---------------------------------------------------------------------------
// AlertsDataSchema — /data/alerts.json
// ---------------------------------------------------------------------------
export const AlertsDataSchema = z.object({
  alerts: z.array(z.object({
    level: z.enum(['success', 'warning', 'error', 'info']),
    type: z.string(),
    title: z.string(),
    message: z.string(),
    timestamp: z.optional(z.string()),
    requires_action: z.boolean(),
  })),
  count: z.number(),
  generated_at: z.string(),
}).passthrough();

// ---------------------------------------------------------------------------
// StatsDataSchema — /data/stats.json
// ---------------------------------------------------------------------------
const PaperPortfolioStatsSchema = z.object({
  sharpe: z.number(),
  total_return: z.number(),
  max_value: z.number(),
  min_value: z.number(),
  days_tracked: z.number(),
});

const SPYComparisonSchema = z.object({
  portfolio_value: z.number(),
  spy_value: z.number(),
  relative_return: z.number(),
  correlation_30d: z.number(),
  beta: z.number(),
  outperformance: z.number(),
});

export const StatsDataSchema = z.object({
  asset_stats: z.record(z.string(), z.object({
    '30d_return': z.number(),
    volatility: z.number(),
    current: z.number(),
  })),
  paper_portfolio: z.nullable(PaperPortfolioStatsSchema),
  spy_comparison: z.nullable(SPYComparisonSchema),
  generated_at: z.string(),
}).passthrough();

// ---------------------------------------------------------------------------
// HealthDataSchema — /data/health.json
// ---------------------------------------------------------------------------
const CronJobStatusSchema = z.object({
  id: z.string(),
  name: z.string(),
  schedule: z.string(),
  last_run: z.nullable(z.string()),
  next_run: z.nullable(z.string()),
  status: z.enum(['ok', 'error', 'unknown']),
  state: z.enum(['scheduled', 'paused', 'running']),
  backend: z.optional(z.string()),
  source: z.optional(z.string()),
  error: z.optional(z.string()),
  duration_seconds: z.optional(z.nullable(z.number())),
}).passthrough();

const DataFreshnessSchema = z.object({
  last_update: z.string(),
  days_stale: z.number(),
  market_lag_days: z.optional(z.number()),
  latest_available_market_date: z.optional(z.nullable(z.string())),
  status: z.enum(['fresh', 'stale', 'critical']),
});

const SchedulerBackendSchema = z.object({
  backend: z.string(),
  status: z.enum(['ok', 'degraded', 'warning', 'unavailable', 'error', 'unknown']),
  source: z.string(),
  total_jobs: z.number(),
  failed_jobs: z.number(),
  reason: z.optional(z.string()),
}).passthrough();

const SchedulerStatusSchema = z.object({
  status: z.enum(['ok', 'degraded', 'warning', 'unavailable', 'unknown']),
  backends: z.record(z.string(), SchedulerBackendSchema),
}).passthrough();

const DataPipelineSloDimensionSchema = z.object({
  status: z.enum(['ok', 'warning', 'critical', 'unknown']),
  message: z.optional(z.string()),
}).passthrough();

const DataPipelineRunbookActionSchema = z.object({
  dimension: z.string(),
  code: z.string(),
  severity: z.enum(['ok', 'warning', 'critical', 'unknown']),
  action: z.string(),
  artifact: z.optional(z.string()),
  provider: z.optional(z.string()),
  reason: z.optional(z.string()),
}).passthrough();

const DataPipelineRunbookSchema = z.object({
  status: z.enum(['ok', 'warning', 'critical', 'unknown']),
  top_cause: z.nullable(DataPipelineRunbookActionSchema),
  actions: z.array(DataPipelineRunbookActionSchema),
}).passthrough();

const DataPipelineSloSchema = z.object({
  schema_version: z.string(),
  status: z.enum(['ok', 'warning', 'critical', 'unknown']),
  top_dimension: z.nullable(z.string()),
  dimensions: z.record(z.string(), DataPipelineSloDimensionSchema),
  runbook: z.optional(DataPipelineRunbookSchema),
  error: z.optional(z.string()),
}).passthrough();

const SignalHealthSectionSchema = z.object({
  timestamp: z.optional(z.string()),
  summary: z.optional(z.record(z.string(), z.unknown())),
  scores: z.optional(z.record(z.string(), z.number())),
  alerts: z.optional(z.array(z.unknown())),
  overall_health: z.optional(z.string()),
  error: z.optional(z.string()),
  status: z.optional(z.string()),
}).passthrough();

const FredReadinessSectionSchema = z.object({
  schema_version: z.optional(z.string()),
  status: z.enum(['ok', 'warning', 'critical', 'unknown']).optional(),
  readiness: z.optional(z.string()),
  ready: z.optional(z.boolean()),
  blocking: z.optional(z.boolean()),
  reason: z.optional(z.string().nullable()),
  message: z.optional(z.string()),
  remediation: z.optional(z.string().nullable()),
}).passthrough();

export const HealthDataSchema = z.object({
  cron_jobs: z.array(CronJobStatusSchema),
  data_freshness: z.record(z.string(), DataFreshnessSchema),
  system_status: z.enum(['healthy', 'warning', 'critical', 'degraded']),
  generated_at: z.string(),
  scheduler_status: z.optional(SchedulerStatusSchema),
  data_pipeline_slo: z.optional(DataPipelineSloSchema),
  signal_health: z.optional(SignalHealthSectionSchema),
  fred_readiness: z.optional(FredReadinessSectionSchema),
  error: z.optional(z.string()),
}).passthrough();

// ---------------------------------------------------------------------------
// IncidentLifecycleSummarySchema — /data/incidents.json
// ---------------------------------------------------------------------------
const IncidentLifecycleIncidentSchema = z.object({
  incident_id: z.string(),
  channel: z.string(),
  severity: z.string(),
  state: z.enum(['firing', 'acknowledged', 'resolving', 'resolved']),
  message: z.string(),
  details: z.record(z.string(), z.unknown()),
  created_at: z.string(),
  updated_at: z.string(),
  resolved_at: z.nullable(z.string()),
  resolution_notes: z.nullable(z.string()),
  mttr_seconds: z.nullable(z.number()),
}).passthrough();

const IncidentLifecycleMetricsSchema = z.object({
  incident_frequency: z.number(),
  open_count: z.number(),
  resolved_count: z.number(),
  mean_mttr_seconds: z.nullable(z.number()),
}).passthrough();

export const IncidentLifecycleSummarySchema = z.object({
  generated_at: z.string(),
  open_count: z.number(),
  incidents: z.array(IncidentLifecycleIncidentSchema),
  metrics: IncidentLifecycleMetricsSchema,
}).passthrough();

// ---------------------------------------------------------------------------
// AnalyticsDataSchema — /data/analytics.json
// ---------------------------------------------------------------------------
const DrawdownPointSchema = z.object({
  date: z.string(),
  value: z.number(),
  peak: z.number(),
  drawdown: z.number(),
  days_since_peak: z.number(),
  is_recovery: z.boolean(),
});

const MaxDrawdownDataSchema = z.object({
  max_drawdown: z.number(),
  max_drawdown_date: z.string(),
  recovery_date: z.nullable(z.string()),
  underwater_days: z.number(),
  peak_value: z.number(),
  trough_value: z.number(),
});

const RollingMetricPointSchema = z.object({
  date: z.string(),
  sharpe: z.number(),
  volatility: z.number(),
  mean_return: z.number(),
  window_days: z.number(),
});

const PortfolioBenchmarkDataSchema = z.object({
  start_date: z.string(),
  end_date: z.string(),
  start_value: z.number(),
  end_value: z.number(),
  total_return: z.number(),
  cagr: z.nullable(z.number()),
  volatility: z.number(),
  max_drawdown: z.number(),
  sharpe: z.nullable(z.number()),
});

const CrisisPeriodDataSchema = z.object({
  name: z.string(),
  period: z.string(),
  description: z.string(),
  spy_return: z.number(),
  portfolio_return: z.nullable(z.number()),
});

export const AnalyticsDataSchema = z.object({
  status: z.enum(['success', 'no_data', 'error']),
  message: z.optional(z.string()),
  generated_at: z.string(),
  data_points: z.number(),
  date_range: z.object({
    start: z.nullable(z.string()),
    end: z.nullable(z.string()),
  }),
  drawdown: z.object({
    series: z.array(DrawdownPointSchema),
    max_drawdown: MaxDrawdownDataSchema,
  }),
  rolling_metrics: z.object({
    sharpe_63d: z.array(RollingMetricPointSchema),
    sharpe_126d: z.array(RollingMetricPointSchema),
    sharpe_252d: z.array(RollingMetricPointSchema),
  }),
  benchmark_comparison: z.object({
    portfolio: PortfolioBenchmarkDataSchema,
  }),
  crisis_periods: z.array(CrisisPeriodDataSchema),
}).passthrough();

// ---------------------------------------------------------------------------
// RebalanceHealthSchema — /data/rebalance_health.json
// ---------------------------------------------------------------------------
const RebalanceExecutionSchema = z.object({
  date: z.string(),
  time: z.string(),
  orders: z.number(),
  total_value: z.number(),
  symbols: z.array(z.string()),
}).passthrough();

const MarketDataConsistencySchema = z.object({
  status: z.string(),
  reason: z.optional(z.string()),
  checked_at: z.optional(z.string()),
  rows: z.optional(z.array(z.record(z.string(), z.unknown()))),
  warnings: z.optional(z.array(z.string())),
}).passthrough();

const AlpacaFeedEntitlementSchema = z.object({
  configured_feed: z.string(),
  effective_feed: z.string(),
  entitlement: z.string(),
  delayed: z.boolean(),
  acceptable_for_live: z.boolean(),
  policy_decision: z.string(),
  reason: z.optional(z.string()),
}).passthrough();

export const RebalanceHealthSchema = z.object({
  generated: z.optional(z.string()),
  next_rebalance: z.optional(z.object({
    date: z.string(),
    days_until: z.number(),
    frequency: z.string(),
  }).passthrough()),
  schedule_compliance: z.optional(z.object({
    on_time: z.number(),
    delayed: z.number(),
    total: z.number(),
    compliance_pct: z.number(),
  }).passthrough()),
  execution_history: z.optional(z.array(RebalanceExecutionSchema)),
  total_executions: z.optional(z.number()),
  market_data_consistency: z.optional(MarketDataConsistencySchema),
  alpaca_feed_entitlement: z.optional(AlpacaFeedEntitlementSchema),
  current_turnover_pct: z.optional(z.number()),
  max_daily_turnover: z.optional(z.number()),
  max_monthly_turnover: z.optional(z.number()),
  max_annual_turnover: z.optional(z.number()),
  daily_budget_used: z.optional(z.number()),
  monthly_budget_used: z.optional(z.number()),
  annual_budget_used: z.optional(z.number()),
  recent_rebalances: z.optional(z.array(z.object({
    date: z.string(),
    turnover_pct: z.number(),
    cost_bps: z.number(),
    trigger: z.string(),
  }).passthrough())),
  cost_drag_bps: z.optional(z.number()),
}).passthrough().superRefine((data, ctx) => {
  const hasGeneratedContract = Boolean(
    data.generated
    && data.next_rebalance
    && data.schedule_compliance
    && data.execution_history
    && typeof data.total_executions === 'number'
  );
  const hasLegacyTurnoverContract = Boolean(
    typeof data.current_turnover_pct === 'number'
    && typeof data.max_daily_turnover === 'number'
    && typeof data.max_monthly_turnover === 'number'
    && typeof data.max_annual_turnover === 'number'
    && typeof data.daily_budget_used === 'number'
    && typeof data.monthly_budget_used === 'number'
    && typeof data.annual_budget_used === 'number'
    && data.recent_rebalances
    && typeof data.cost_drag_bps === 'number'
  );

  if (!hasGeneratedContract && !hasLegacyTurnoverContract) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: 'Expected generated schedule contract or legacy turnover contract',
    });
  }
});

// ---------------------------------------------------------------------------
// GraduationDataSchema — /data/graduation.json
// ---------------------------------------------------------------------------
export const GraduationDataSchema = z.object({
  criteria: z.array(z.object({
    id: z.string(),
    label: z.string(),
    passed: z.boolean(),
    value: z.string(),
    threshold: z.string(),
  })),
  paper_trading: z.object({
    start_date: z.string(),
    initial_capital: z.number(),
    current_value: z.number(),
    days_elapsed: z.number(),
    days_required: z.number(),
  }),
  readiness_pct: z.number(),
  eligible: z.boolean(),
}).passthrough();

// ---------------------------------------------------------------------------
// Generic validation helper with graceful degradation
// ---------------------------------------------------------------------------
export function validateFetchData<T>(
  raw: unknown,
  schema: z.ZodType<T>,
  _endpoint: string,
): T | null {
  const result = schema.safeParse(raw);
  if (result.success) {
    return result.data as T;
  }
  // Log validation errors in dev mode
  if (import.meta.env.DEV) {
    console.warn(`[${_endpoint}] Validation failed:`, result.error.issues);
  }
  // Fallback: try to use raw data as-is if it looks like an object (but not an array)
  if (typeof raw === 'object' && raw !== null && !Array.isArray(raw)) {
    return raw as T;
  }
  return null;
}

// ---------------------------------------------------------------------------
// validateSignalsData — specific validator for signals.json (retained)
// ---------------------------------------------------------------------------
export function validateSignalsData(raw: unknown): SignalsData | null {
  const normalized = normalizeSignalsTimestamp(raw);
  const result = SignalsDataSchema.safeParse(normalized);
  if (result.success) {
    return result.data as SignalsData;
  }
  // Log validation errors in dev mode
  if (import.meta.env.DEV) {
    console.warn('Signal validation failed:', result.error.issues);
  }
  // Fallback: try to use raw data as-is if it looks like a timestamped object
  if (isPlainRecord(normalized) && typeof normalized.timestamp === 'string') {
    return normalized as unknown as SignalsData;
  }
  return null;
}
