import { z } from 'zod';
import type { Alert, AlertsData, SignalsData } from '../types/live';

// ---------------------------------------------------------------------------
// Runtime provenance
// ---------------------------------------------------------------------------
export const RuntimeProvenanceSchema = z.object({
  schema_version: z.optional(z.string()),
  artifact_id: z.optional(z.nullable(z.string())),
  plane: z.optional(z.nullable(z.string())),
  generated_at: z.optional(z.nullable(z.string())),
  generator_git_sha: z.optional(z.nullable(z.string())),
  generator_git_sha_status: z.optional(z.nullable(z.string())),
  last_full_generator_git_sha: z.optional(z.nullable(z.string())),
  patch_source: z.optional(z.nullable(z.string())),
}).passthrough();

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
  source_mode: z.optional(z.string()),
  source_status: z.optional(z.string()),
  source_reason: z.optional(z.nullable(z.string())),
  source_provider: z.optional(z.nullable(z.string())),
  source_generated_at: z.optional(z.nullable(z.string())),
  source_latest_observation: z.optional(z.nullable(z.string())),
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
const ConformalCoverageDiagnosticsSchema = z.object({
  schema_version: z.literal('conformal-coverage/v1'),
  observations: z.number(),
  alpha: z.number(),
  expected_exceedance_rate: z.number(),
  exceedance_count: z.number(),
  exceedance_rate: z.number(),
  coverage_rate: z.number(),
  coverage_pass: z.boolean(),
  rolling_window: z.number(),
  rolling_exceedance_rate: z.number(),
  longest_violation_cluster: z.number(),
  kupiec_statistic: z.number(),
  kupiec_p_value: z.number(),
  kupiec_pass: z.boolean(),
  christoffersen_statistic: z.number(),
  christoffersen_p_value: z.number(),
  christoffersen_pass: z.boolean(),
  conditional_coverage_statistic: z.number(),
  conditional_coverage_p_value: z.number(),
  conditional_coverage_pass: z.boolean(),
  by_regime: z.optional(z.record(z.string(), z.record(z.string(), z.unknown()))),
});

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
  coverage_diagnostics: z.optional(z.nullable(ConformalCoverageDiagnosticsSchema)),
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
  correlation_entropy: z.optional(z.nullable(z.number())),
  participation_ratio: z.optional(z.nullable(z.number())),
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
  remaining_budget_ratio: z.optional(z.number().min(0).max(1)),
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
  remaining_budget_ratio: z.optional(z.number().min(0).max(1)),
  /** True when kill_switch.json blocks execution (order_router SSOT). */
  execution_blocked: z.optional(z.boolean()),
  kill_switch_enabled: z.optional(z.boolean()),
  kill_switch_level: z.optional(z.string().nullable()),
  kill_switch_reason: z.optional(z.string().nullable()),
  kill_switch_incident_id: z.optional(z.string().nullable()),
  kill_switch_message: z.optional(z.string().nullable()),
  status: SmartRebalanceStatusSchema,
}).superRefine((data, ctx) => {
  const hasRatioContract =
    data.remaining_budget_ratio !== undefined || data.status.remaining_budget_ratio !== undefined;
  if (
    hasRatioContract
    && Math.abs(data.remaining_budget_pct - data.status.remaining_budget_pct) > 0.000001
  ) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: ['remaining_budget_pct'],
      message: 'remaining_budget_pct must match status.remaining_budget_pct display units',
    });
  }
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
  kill_switch_level: z.optional(z.nullable(z.string())),
  kill_switch_source: z.optional(z.nullable(z.string())),
  kill_switch_reason: z.optional(z.nullable(z.string())),
  kill_switch_incident_id: z.optional(z.nullable(z.string())),
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
// Bond Momentum (producer summary + legacy overlay rows)
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

/** Legacy TSMOM per-ETF overlay contract. */
const BondMomentumLegacySchema = z.object({
  signals: z.array(BondMomentumSignalSchema),
  timestamp: z.string().optional(),
  ensemble: BondMomentumEnsembleSchema.optional(),
});

/**
 * Canonical public producer shape from DashboardGenerator bond_duration /
 * overlay bond_momentum summary (flat recommendation, no signals[]).
 */
const BondMomentumSummarySchema = z.object({
  active: z.boolean().optional().default(true),
  yield_10y: z.number(),
  yield_2y: z.number(),
  spread: z.number(),
  curve_regime: z.string(),
  rate_direction: z.string(),
  tlt_weight: z.number(),
  ief_weight: z.number(),
  shy_weight: z.number(),
  effective_duration: z.number(),
  position: z.string(),
  confidence: z.number(),
  status_text: z.string().optional().default(''),
  generated_at: z.string().optional(),
  timestamp: z.string().optional(),
}).passthrough();

export const BondMomentumSchema = z.union([
  BondMomentumSummarySchema,
  BondMomentumLegacySchema,
]);

// ---------------------------------------------------------------------------
// VIX Term Structure
// ---------------------------------------------------------------------------
const VIXTermStructureLevelSchema = z.object({
  value: z.number(),
  timestamp: z.string(),
});

const VIX_TERM_STRUCTURE_REGIMES = [
  'extreme_contango',
  'steep_contango',
  'mild_contango',
  'flat',
  'backwardation',
  'extreme_backwardation',
] as const;

const VIXTermStructureViewSchema = z.object({
  vix: VIXTermStructureLevelSchema,
  vix3m: VIXTermStructureLevelSchema,
  vix6m: z.optional(VIXTermStructureLevelSchema),
  slope: z.number(),
  roll_yield: z.number(),
  composite_signal: z.number(),
  regime: z.enum(VIX_TERM_STRUCTURE_REGIMES),
  z_score: z.number(),
  percentile_1y: z.optional(z.number()),
  timestamp: z.optional(z.string()),
});

function isVixTermStructureRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function finiteNumberOrUndefined(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    const n = Number(value);
    if (Number.isFinite(n)) return n;
  }
  return undefined;
}

function vixLevelFrom(
  nested: unknown,
  flatValue: unknown,
  timestamp: string,
): { value: number; timestamp: string } | undefined {
  if (isVixTermStructureRecord(nested)) {
    const value = finiteNumberOrUndefined(nested.value);
    if (value !== undefined) {
      return {
        value,
        timestamp:
          typeof nested.timestamp === 'string' && nested.timestamp
            ? nested.timestamp
            : timestamp,
      };
    }
  }
  const value = finiteNumberOrUndefined(flatValue);
  if (value === undefined) return undefined;
  return { value, timestamp };
}

/**
 * Accept producer-shaped public artifact keys (vix_spot, slope_vix3m_vix, …)
 * and nested legacy view-model; emit the panel/view schema.
 */
function normalizeVixTermStructureInput(raw: unknown): unknown {
  if (!isVixTermStructureRecord(raw)) return raw;

  const timestamp =
    typeof raw.timestamp === 'string'
      ? raw.timestamp
      : typeof raw.generated_at === 'string'
        ? raw.generated_at
        : '';

  const vix = vixLevelFrom(raw.vix, raw.vix_spot, timestamp);
  const vix3m = vixLevelFrom(raw.vix3m, raw.vix3m, timestamp);
  const vix6m = vixLevelFrom(raw.vix6m, raw.vix6m_spot ?? raw.vix6m_value, timestamp);

  const slope = finiteNumberOrUndefined(
    raw.slope ?? raw.slope_vix3m_vix ?? raw.slope_signal,
  );
  const roll_yield = finiteNumberOrUndefined(raw.roll_yield ?? raw.roll_yield_signal);
  const composite_signal = finiteNumberOrUndefined(
    raw.composite_signal ?? raw.signal_value,
  );
  const z_score = finiteNumberOrUndefined(raw.z_score ?? raw.vix_zscore_signal);
  const percentile_1y = finiteNumberOrUndefined(raw.percentile_1y);
  const regime = typeof raw.regime === 'string' ? raw.regime : undefined;

  // If already nested-complete, keep extras via reconstruction of required fields.
  if (
    vix === undefined
    || vix3m === undefined
    || slope === undefined
    || roll_yield === undefined
    || composite_signal === undefined
    || z_score === undefined
    || !regime
  ) {
    // Fall through to raw so Zod reports structured issues when incomplete.
    // Still prefer mapped fields when partially available for better errors.
  }

  if (
    vix !== undefined
    && vix3m !== undefined
    && slope !== undefined
    && roll_yield !== undefined
    && composite_signal !== undefined
    && z_score !== undefined
    && regime !== undefined
  ) {
    return {
      vix,
      vix3m,
      ...(vix6m ? { vix6m } : {}),
      slope,
      roll_yield,
      composite_signal,
      regime,
      z_score,
      ...(percentile_1y !== undefined ? { percentile_1y } : {}),
      ...(timestamp ? { timestamp } : {}),
    };
  }

  return raw;
}

export const VIXTermStructureSchema = z.preprocess(
  normalizeVixTermStructureInput,
  VIXTermStructureViewSchema,
);

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
  canonical_controller: z.string().default('hedge_selector'),
  vixy_role: z.string().default('diagnostic_sizing_helper'),
  term_structure_role: z.string().default('gate_discount_multiplier'),
  term_structure_gate: z.boolean().default(false),
  term_structure_multiplier: z.number().default(0),
  term_structure_signal: z.nullable(z.number()).default(null),
  gate_reason: z.string().default('unknown'),
}).passthrough();

// ---------------------------------------------------------------------------
// ML Signals
// ---------------------------------------------------------------------------
const MLPredictionSchema = z.object({
  predicted_regime: z.string(),
  confidence: z.number(),
  probabilities: z.record(z.string(), z.number()),
  heuristic: z.boolean(),
  feature_timestamp: z.optional(z.nullable(z.string())),
  feature_freshness_status: z.optional(z.nullable(z.string())),
  source_artifact: z.optional(z.nullable(z.string())),
});

const MLFeatureSchema = z.object({
  vix_level: z.nullable(z.number()),
  trend_direction: z.number(),
  price_vs_sma20: z.number(),
  return_5d: z.number(),
  spy_correlation: z.number(),
  feature_timestamp: z.optional(z.nullable(z.string())),
});

const MLGridSearchSchema = z.object({
  available: z.boolean(),
  timestamp: z.nullable(z.string()),
  top_allocation: z.nullable(z.record(z.string(), z.number())),
  sharpe: z.nullable(z.number()),
  volatility: z.nullable(z.number()),
  source_artifact: z.optional(z.nullable(z.string())),
  benchmark_timestamp: z.optional(z.nullable(z.string())),
  observation_semantics: z.optional(z.nullable(z.string())),
  freshness_status: z.optional(z.nullable(z.string())),
  staleness_days: z.optional(z.nullable(z.number())),
  live_authoritative: z.optional(z.boolean()),
});

const MLExecutionRoleSchema = z.object({
  role: z.string(),
  routed: z.boolean(),
  routed_by: z.optional(z.nullable(z.string())),
  live_authoritative: z.boolean(),
});

const MLSignalsSchema = z.object({
  available: z.boolean(),
  timestamp: z.nullable(z.string()),
  generated_at: z.optional(z.nullable(z.string())),
  feature_source_artifact: z.optional(z.nullable(z.string())),
  feature_as_of: z.optional(z.nullable(z.string())),
  feature_freshness_status: z.optional(z.nullable(z.string())),
  feature_staleness_days: z.optional(z.nullable(z.number())),
  prediction_source_mode: z.optional(z.nullable(z.string())),
  execution_role: z.optional(MLExecutionRoleSchema),
  predictions: z.record(z.string(), MLPredictionSchema),
  features: z.record(z.string(), MLFeatureSchema),
  grid_search: MLGridSearchSchema,
});

export const MarlStatusSchema = z.object({
  schema_version: z.literal('marl-runtime-status/v1'),
  available: z.boolean(),
  timestamp: z.nullable(z.string()),
  runtime: z.object({
    version: z.string(),
    device: z.string(),
    agents_loaded: z.array(z.string()),
    signal_integrator_connected: z.boolean(),
    checkpoint_loaded: z.boolean(),
    inference_count: z.number(),
    current_allocation: z.record(z.string(), z.number()),
    graph_metrics: z.record(z.string(), z.unknown()),
  }),
  execution_role: z.object({
    role: z.literal('research_shadow_non_routed'),
    routed: z.literal(false),
    routed_by: z.null(),
    live_authoritative: z.literal(false),
    description: z.string(),
  }),
  error: z.optional(z.nullable(z.string())),
});

// ---------------------------------------------------------------------------
// IcDecaySchema — IC decay monitoring for signal quality tracking
// ---------------------------------------------------------------------------
const IcMetricAxisSchema = z.enum([
  'time_series_rank_correlation',
  'cross_sectional_ic',
  'calibration_proper_score',
]);

const IcMetricKindSchema = z.enum(['correlation', 'calibration_proper_score']);
const IcAlignmentStatusSchema = z.enum([
  'aligned',
  'provisional',
  'misaligned',
  'ambiguous',
  'metric_mismatch',
  'undeclared',
]);
const IcInferenceReasonSchema = z.enum([
  'legacy_rows_missing_alignment_metadata',
  'observation_metadata_incomplete',
  'label_alignment_mismatch',
  'metric_contract_mismatch',
  'evaluation_contract_missing',
  'dependence_not_characterized',
]);

const IcEvaluationContractSchema = z.object({
  contract_version: z.literal('ic-evaluation-contract/v2'),
  intended_metric_axis: IcMetricAxisSchema,
  intended_metric_kind: IcMetricKindSchema,
  target_asset: z.nullable(z.string()),
  target_basket: z.nullable(z.string()),
  intended_horizon_sessions: z.nullable(z.number().int().nonnegative()),
  prediction_field: z.nullable(z.string()),
  prediction_transform: z.nullable(z.string()),
});

const IcObservationMetadataSchema = z.object({
  prediction_date: z.optional(z.string()),
  realized_start_date: z.optional(z.string()),
  resolved_date: z.optional(z.string()),
  target_asset: z.optional(z.string()),
  intended_horizon_sessions: z.optional(z.number().int().nonnegative()),
  realized_horizon_sessions: z.optional(z.number().int().nonnegative()),
  prediction_field: z.optional(z.string()),
  prediction_transform: z.optional(z.string()),
  metric_axis: z.optional(IcMetricAxisSchema),
  metric_kind: z.optional(IcMetricKindSchema),
  contract_version: z.optional(z.literal('ic-observation-metadata/v2')),
});

const IcDecayEvidenceFieldsSchema = z.object({
  metric_axis: z.optional(IcMetricAxisSchema),
  metric_kind: z.optional(IcMetricKindSchema),
  estimate_kind: z.optional(z.literal('descriptive')),
  alignment_status: z.optional(IcAlignmentStatusSchema),
  alignment_reason: z.optional(z.string()),
  inference_status: z.optional(z.literal('unavailable')),
  inference_reason: z.optional(IcInferenceReasonSchema),
  observation_count: z.optional(z.number().int().nonnegative()),
  observation_unit: z.optional(z.enum(['pairs', 'dates', 'cross_sectional_periods'])),
  contract_version: z.optional(z.literal('ic-evaluation-contract/v2')),
  evaluation_contract: z.optional(IcEvaluationContractSchema),
  latest_observation_metadata: z.optional(IcObservationMetadataSchema),
});

export const IcDecaySignalEntrySchema = IcDecayEvidenceFieldsSchema.extend({
  ic_rolling: z.nullable(z.number()),
  ic_trend: z.enum(['stable', 'decaying', 'improving', 'unknown']),
  observations: z.number(),
  status: z.enum(['healthy', 'warning', 'critical', 'insufficient_data']),
  min_obs_for_status: z.optional(z.number().int().nonnegative()),
  // Task 2A: control eligibility derived from complete contract alignment —
  // never from coefficient magnitude. Additive; descriptive fields preserved.
  control_eligible: z.optional(z.boolean()),
  control_status: z.optional(z.enum(['eligible', 'ineligible'])),
  control_ineligibility_reason: z.optional(z.nullable(z.string())),
});

const IcDecayEvidenceEntrySchema = IcDecayEvidenceFieldsSchema.extend({
  ic_rolling: z.optional(z.nullable(z.number())),
  observations: z.optional(z.number()),
  status: z.optional(z.enum(['healthy', 'warning', 'critical', 'insufficient_data'])),
});

export const IcDecaySchema = z.object({
  status: z.optional(z.enum([
    'healthy',
    'warning',
    'critical',
    'insufficient_resolved_history',
    'waiting_for_forward_returns',
    'no_data',
  ])),
  signals: z.record(z.string(), IcDecaySignalEntrySchema).optional(),
  resolved_signal_count: z.optional(z.number()),
  pending_predictions: z.optional(z.number()),
  staged_date: z.optional(z.nullable(z.string())),
  label_horizon: z.optional(z.string()),
  error: z.optional(z.string()),
}).passthrough();

// Bounded IC quality projection carried by private/public health surfaces.
// This is intentionally separate from the raw signals.ic_decay report.
export const IcDecaySummarySchema = z.object({
  status: z.enum([
    'healthy',
    'warning',
    'critical',
    'insufficient_resolved_history',
    'waiting_for_forward_returns',
    'no_data',
    'unknown',
  ]),
  critical_signals: z.array(z.string()),
  warning_signals: z.array(z.string()),
  insufficient_data_signals: z.optional(z.array(z.string())),
  resolved_signal_count: z.number(),
  min_observations: z.number(),
  staged_pending_predictions: z.number(),
  staged_pending_signal_names: z.optional(z.array(z.string())),
  staged_date: z.nullable(z.string()),
  staged_pending_scope: z.string(),
  historical_unlabeled_rows: z.number(),
  historical_unlabeled_dates: z.number(),
  historical_unlabeled_oldest_date: z.optional(z.nullable(z.string())),
  historical_unlabeled_scope: z.string(),
  evidence_generated_at: z.optional(z.nullable(z.string())),
  evidence_freshness: z.string(),
  routing_authority: z.string(),
  routing_control: z.string(),
  control_effect: z.string(),
  // Task 2A: control-eligible subsets of the descriptive signal lists.
  control_eligible_critical_signals: z.optional(z.array(z.string())),
  control_eligible_warning_signals: z.optional(z.array(z.string())),
  kill_switch_level: z.optional(z.nullable(z.string())),
  signal_evidence: z.optional(z.record(z.string(), IcDecayEvidenceEntrySchema)),
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
  status: z.optional(z.enum([
    'validated',
    'weak',
    'unvalidated',
    'insufficient_data',
    'insufficient_resolved_history',
    'waiting_for_forward_returns',
    'no_data',
  ])),
  signals: z.record(z.string(), SignalWFEEntrySchema).optional(),
  resolved_signal_count: z.optional(z.number()),
  pending_predictions: z.optional(z.number()),
  staged_date: z.optional(z.nullable(z.string())),
  label_horizon: z.optional(z.string()),
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
  status: z.optional(z.string()),
  source_mode: z.optional(z.string()),
  cache_status: z.optional(z.string()),
  api_key_configured: z.optional(z.boolean()),
  reason: z.optional(z.nullable(z.string())),
  latest_fetched_at: z.optional(z.nullable(z.string())),
  row_count: z.optional(z.nullable(z.number())),
  age_hours: z.optional(z.nullable(z.number())),
  ttl_hours: z.optional(z.nullable(z.number())),
  indicators_observed: z.optional(z.boolean()),
}).passthrough();

// ---------------------------------------------------------------------------
// Active signals.json panel schemas
// ---------------------------------------------------------------------------
const NullableNumberSchema = z.nullable(z.number());

export const CryptoAllocationSchema = z.object({
  active: z.boolean(),
  btc_weight: z.number(),
  eth_weight: z.number(),
  total_crypto: z.number(),
  btc_momentum_6m: z.number(),
  eth_momentum_6m: z.number(),
  btc_vol_regime: z.string(),
  eth_vol_regime: z.string(),
  confidence: z.number(),
}).passthrough();

export const CalendarSeasonalitySchema = z.object({
  active: z.boolean(),
  modifier: z.number(),
  active_windows: z.array(z.string()),
  next_window: z.string(),
  days_to_next: z.number(),
  recommendation: z.string(),
  effect: z.string(),
}).passthrough();

const EnsembleSourceVoteSchema = z.object({
  source: z.string(),
  direction: z.union([z.string(), z.number()]),
  strength: z.number(),
  confidence: z.number(),
  weight: z.number(),
  weight_original: z.optional(z.number()),
  staleness_decay: z.optional(z.number()),
}).passthrough();

const ConfiguredSourceStatusSchema = z.object({
  source: z.string(),
  label: z.optional(z.string()),
  configured: z.boolean(),
  configured_weight: z.optional(z.number()),
  collected: z.boolean(),
  active: z.boolean(),
  contributing: z.boolean(),
  status: z.string(),
  reason: z.optional(z.string()),
}).passthrough();

const AdaptiveLearningStatusSchema = z.enum([
  'active',
  'disabled',
  'unavailable',
  'non_effective',
]);

const AdaptiveLearningBranchSchema = z.object({
  status: AdaptiveLearningStatusSchema,
  enabled: z.boolean(),
  reason: z.string(),
  observations: z.optional(z.number()),
  warmup_days: z.optional(z.number()),
  max_blend: z.optional(z.number()),
  current_blend: z.optional(z.number()),
  state_available: z.optional(z.boolean()),
  blend_alpha: z.optional(z.number()),
}).passthrough();

const AdaptiveLearningDisclosureSchema = z.object({
  bandit: AdaptiveLearningBranchSchema,
  online_ic: AdaptiveLearningBranchSchema,
}).passthrough();

export const EnsembleVotingSchema = z.object({
  regime: z.string(),
  regime_confidence: z.number(),
  weighted_consensus: z.number(),
  agreement_ratio: z.number(),
  action: z.string(),
  confidence: z.number(),
  equity_bias: z.number(),
  duration_bias: z.number(),
  gold_bias: z.number(),
  num_sources: z.number(),
  configured_source_count: z.optional(z.number()),
  collected_source_count: z.optional(z.number()),
  contributing_source_count: z.optional(z.number()),
  inactive_source_count: z.optional(z.number()),
  inactive_sources: z.optional(z.array(z.string())),
  configured_source_status: z.optional(z.array(ConfiguredSourceStatusSchema)),
  source_breakdown: z.array(EnsembleSourceVoteSchema),
  n_eff: z.optional(z.number()),
  weight_entropy: z.optional(z.number()),
  adaptive_learning: z.optional(AdaptiveLearningDisclosureSchema),
  total_weight_after_decay: z.optional(z.number()),
}).passthrough();

const AlternativeDataComponentSchema = z.object({
  score: NullableNumberSchema,
  confidence: NullableNumberSchema,
  weight: NullableNumberSchema,
}).passthrough();

export const AlternativeDataSchema = z.object({
  regime: z.optional(z.string()),
  probability: z.optional(z.number()),
  confidence: z.optional(z.number()),
  timestamp: z.string(),
  components: z.optional(z.record(z.string(), AlternativeDataComponentSchema)),
  composite_score: z.optional(NullableNumberSchema),
  z_score: z.optional(NullableNumberSchema),
  sources_count: z.optional(NullableNumberSchema),
  data_freshness_hours: z.optional(NullableNumberSchema),
}).passthrough();

export const FactorRotationSignalSchema = z.object({
  selected_factors: z.optional(z.array(z.string())),
  allocation: z.optional(z.record(z.string(), z.unknown())),
  signal_strength: z.optional(z.unknown()),
  recommendation: z.optional(z.string()),
}).passthrough().superRefine((data, ctx) => {
  const hasProductionSignal =
    data.selected_factors !== undefined
    || data.allocation !== undefined
    || data.signal_strength !== undefined
    || data.recommendation !== undefined;
  if (!hasProductionSignal) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: 'factor_rotation must include at least one production signal field',
    });
  }
});

export const StackingEnsembleSchema = z.object({
  active: z.boolean(),
  stacking_available: z.boolean(),
  runtime_role: z.enum(['research_dormant', 'model_backed_advisory']),
  runtime_status: z.enum(['unavailable_no_model', 'model_loaded']),
  live_authoritative: z.boolean(),
  routed: z.boolean(),
  routed_by: z.nullable(z.string()),
  prediction_available: z.boolean(),
  prediction_direction: z.string(),
  confidence: z.number(),
  probability_bullish: z.number(),
  probability_bearish: z.number(),
  probability_neutral: z.number(),
  fallback_used: z.boolean(),
  model_version: z.string(),
  voting_accuracy: NullableNumberSchema,
  stacking_accuracy: NullableNumberSchema,
  accuracy_metrics_available: z.boolean(),
  feature_count: NullableNumberSchema,
  feature_count_metadata_available: z.boolean(),
  feature_count_source: z.enum([
    'model_metadata',
    'unavailable_no_model',
    'unavailable_missing_metadata',
  ]),
  source_roster: z.array(z.string()),
  source_roster_version: z.string(),
  fallback_semantics: z.string(),
  latency_ms: z.number(),
  status_reason: z.string(),
  operator_message: z.string(),
  top_features: z.optional(z.array(z.object({
    name: z.string(),
    importance: z.number(),
  }).passthrough())),
  backtest_finding: z.optional(z.string()),
}).passthrough();

export const ConvexityHarvestSchema = z.object({
  date: z.string(),
  allocation_pct: z.number(),
  position_type: z.string(),
  vix_level: z.number(),
  contango_pct: z.number(),
  expected_roll_yield: z.number(),
  risk_score: z.number(),
  exit_triggered: z.boolean(),
  exit_reason: z.nullable(z.string()),
}).passthrough();

export const LLMSentimentSchema = z.object({
  timestamp: z.string(),
  technical_regime: z.string(),
  technical_confidence: z.number(),
  sentiment_regime: z.string(),
  sentiment_confidence: z.number(),
  combined_score: z.number(),
  combined_regime: z.string(),
  technical_weight: z.number(),
  sentiment_weight: z.number(),
  circuit_breaker_level: z.string(),
  position_scaling_factor: z.number(),
  equity_tilt: z.number(),
  bond_duration_tilt: z.number(),
  gold_tilt: z.number(),
}).passthrough();

const SectorEntrySchema = z.object({
  symbol: z.string(),
  name: z.string(),
  momentumScore: z.number(),
  allocation: z.number(),
  rank: z.number(),
  longMomentum: z.optional(z.number()),
  shortMomentum: z.optional(z.number()),
  volatility: z.optional(z.number()),
}).passthrough();

const SectorAllocationEntrySchema = z.object({
  symbol: z.string(),
  weight: z.number(),
  momentum: z.number(),
  rank: z.number(),
}).passthrough();

const SectorAllocationSchema = z.object({
  spy_core: z.number(),
  spy_total: z.number(),
  sector_overlay: z.number(),
  sectors: z.array(SectorAllocationEntrySchema),
}).passthrough();

export const SectorRotationSchema = z.object({
  timestamp: z.string(),
  status: z.string(),
  vix: z.optional(z.number()),
  regime: z.optional(z.nullable(z.string())),
  methodology: z.optional(z.string()),
  overlay_pct: z.optional(z.number()),
  top_sectors: z.optional(z.array(SectorEntrySchema)),
  allocation: z.optional(SectorAllocationSchema),
  rebalanceRecommended: z.optional(z.boolean()),
  rebalanceReason: z.optional(z.nullable(z.string())),
  spAllocation: z.optional(z.number()),
  sectorAllocations: z.optional(z.array(z.unknown())),
  totalEquityWeight: z.optional(z.number()),
  regimeAdjusted: z.optional(z.boolean()),
}).passthrough();

export const FactorRotationDashboardSchema = z.object({
  active: z.boolean(),
  selected_factors: z.array(z.string()),
  signal_strength: z.number(),
  factor_allocations: z.record(z.string(), z.number()),
  backtest_finding: z.optional(z.string()),
}).passthrough();

export const CollarSchema = z.object({
  active: z.boolean(),
  regime: z.string(),
  call_strike: z.number(),
  put_strike: z.number(),
  net_premium: z.number(),
  is_cashless: z.boolean(),
  max_upside_pct: z.number(),
  max_downside_pct: z.number(),
  vix_level: z.number(),
  confidence: z.number(),
}).passthrough();

export const KurtosisRegimeSchema = z.object({
  active: z.boolean(),
  kurtosis_20d: z.number(),
  kurtosis_60d: z.number(),
  ker_ratio: z.number(),
  regime: z.string(),
  transitioning: z.boolean(),
  strategy_preference: z.string(),
  tsom_weight: z.number(),
  mr_weight: z.number(),
  fat_tail_risk: z.number(),
}).passthrough();

// Public producer stores allocation/risk as percentage points (spy_pct: 40 = 40%).
export const VolatilityParitySchema = z.object({
  date: z.string(),
  /** Percentage points (10 = 10% target vol). */
  target_volatility: z.number(),
  /** Percentage points (40 = 40% weight). */
  spy_pct: z.number(),
  gld_pct: z.number(),
  tlt_pct: z.number(),
  core_vol_contribution: z.number(),
  vix_short_pct: z.number(),
  vix_tail_pct: z.number(),
  vix_vol_contribution: z.number(),
  cash_pct: z.number(),
  expected_portfolio_vol: z.number(),
  expected_max_dd: z.number(),
  rebalance_triggered: z.boolean(),
  rebalance_reason: z.nullable(z.string()),
}).passthrough();

const AllocationSurfaceRoleSchema = z.object({
  label: z.string(),
  // execution_blocked: kill-switch still routes via target_allocations but blocks execution
  role: z.enum(['execution_routed', 'execution_blocked', 'advisory_non_routed']),
  routed: z.boolean(),
  routed_by: z.nullable(z.string()),
  live_authoritative: z.optional(z.boolean()),
  canonical_controller: z.optional(z.string()),
  description: z.string(),
}).passthrough();

const AdvisoryAllocationArtifactRoleSchema = z.object({
  schema_version: z.literal('allocation-artifact-role/v1'),
  surface: z.string(),
  allocation_field: z.string(),
  runtime_role: z.literal('advisory_non_routed'),
  live_authoritative: z.literal(false),
  routed: z.literal(false),
  routed_by: z.null(),
  canonical_controller: z.literal('signals.json.target_allocations'),
  routed_surface: z.literal('target_allocations'),
  routed_surface_path: z.literal('public/data/signals.json#target_allocations'),
  description: z.string(),
}).passthrough();

const uppercaseSymbolWeightsSchema = () => (
  z.record(z.string(), z.number()).refine(
    (weights) => Object.keys(weights).every((symbol) => symbol === symbol.toUpperCase()),
    { message: 'allocation symbols must be uppercase canonical identifiers' },
  )
);

export const AllocationSurfaceRolesSchema = z.object({
  schema_version: z.literal('allocation-surface-roles/v1'),
  routed_surface: z.string(),
  routed_by: z.string(),
  surfaces: z.object({
    target_allocations: AllocationSurfaceRoleSchema.extend({
      // Routed surface stays order_router; role flips to execution_blocked under kill switch.
      role: z.enum(['execution_routed', 'execution_blocked']),
      routed: z.literal(true),
      routed_by: z.string(),
    }),
    ensemble_voting: AllocationSurfaceRoleSchema.extend({
      role: z.literal('advisory_non_routed'),
      routed: z.literal(false),
      routed_by: z.null(),
    }),
    adaptive_sizing: AllocationSurfaceRoleSchema.extend({
      role: z.literal('advisory_non_routed'),
      routed: z.literal(false),
      routed_by: z.null(),
      live_authoritative: z.literal(false),
    }),
    black_litterman: AllocationSurfaceRoleSchema.extend({
      role: z.literal('advisory_non_routed'),
      routed: z.literal(false),
      routed_by: z.null(),
      live_authoritative: z.literal(false),
    }),
  }).passthrough(),
}).passthrough();

// Producer emits per-asset adjustment maps (SPY/GLD/TLT). Older fixtures may
// use scalar totals; accept both so runtime validation matches public JSON.
const adaptiveAdjustmentFieldSchema = z.union([
  z.number(),
  uppercaseSymbolWeightsSchema(),
]);

export const AdaptiveSizingSchema = z.object({
  base_allocation: uppercaseSymbolWeightsSchema().optional(),
  adjusted_allocation: uppercaseSymbolWeightsSchema(),
  adjustments: uppercaseSymbolWeightsSchema().optional(),
  regime_adjustment: adaptiveAdjustmentFieldSchema.optional(),
  volatility_adjustment: adaptiveAdjustmentFieldSchema.optional(),
  signal_adjustment: adaptiveAdjustmentFieldSchema.optional(),
  drawdown_adjustment: adaptiveAdjustmentFieldSchema.optional(),
  factors: z.optional(z.record(z.string(), z.unknown())),
  authority: AdvisoryAllocationArtifactRoleSchema.extend({
    surface: z.literal('adaptive_sizing'),
    allocation_field: z.literal('adjusted_allocation'),
  }),
  generated_at: z.string(),
}).passthrough();

const BlackLittermanViewSchema = z.object({
  signal_name: z.string(),
  asset: z.string().refine((asset) => asset === asset.toUpperCase(), {
    message: 'view asset must be uppercase canonical identifier',
  }),
  direction: z.enum(['bullish', 'bearish', 'neutral']),
  confidence: z.number(),
  expected_return_delta: z.number(),
});

export const BlackLittermanSchema = z.object({
  prior_weights: uppercaseSymbolWeightsSchema(),
  posterior_weights: uppercaseSymbolWeightsSchema(),
  posterior_returns: uppercaseSymbolWeightsSchema().optional(),
  views: z.array(BlackLittermanViewSchema),
  tau: z.number(),
  view_confidence_method: z.string(),
  optimization_available: z.optional(z.boolean()),
  excluded_assets: z.array(z.string()).optional(),
  zero_weight_assets: z.array(z.string()).optional(),
  authority: AdvisoryAllocationArtifactRoleSchema.extend({
    surface: z.literal('black_litterman'),
    allocation_field: z.literal('posterior_weights'),
  }),
  generated_at: z.optional(z.string()),
}).passthrough();

const AdvancedRegimeSignalAuthoritySchema = z.object({
  role: z.literal('advisory_shadow'),
  routed: z.literal(false),
  availability: z.optional(z.enum(['present', 'unavailable', 'stale', 'error', 'unknown'])),
  published: z.optional(z.boolean()),
  description: z.optional(z.string()),
}).passthrough();

export const RegimeAuthoritySchema = z.object({
  schema_version: z.literal('regime-authority/v1'),
  live_controller: z.literal('signals.json.target_allocations'),
  live_controller_module: z.literal('src.broker.order_router'),
  live_regime: z.string(),
  allocation_regime: z.string(),
  routed_surface: z.literal('target_allocations'),
  target_allocations: z.record(z.string(), z.number()),
  regime_controller: z.optional(z.literal('classify_vix_regime')),
  regime_controller_module: z.optional(z.literal('src.utils.classify_vix_regime')),
  regime_routed: z.optional(z.literal(false)),
  advanced_regime_signals: z.object({
    two_stage_regime: AdvancedRegimeSignalAuthoritySchema,
    bocd_regime: AdvancedRegimeSignalAuthoritySchema,
    regime_transition: AdvancedRegimeSignalAuthoritySchema,
  }).passthrough(),
}).passthrough();

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

const OptionalPanelObjectSchema = z.union([
  z.record(z.string(), z.unknown()),
  z.null(),
]);

// Generated optional panels may be disabled as null, emitted as {}, or emitted
// in a legacy summary shape while their panel code remains defensive.
const optionalPanel = <T extends z.ZodTypeAny>(schema: T) => (
  z.optional(z.union([schema, OptionalPanelObjectSchema]))
);

const activePanel = <T extends z.ZodTypeAny>(schema: T) => (
  z.optional(z.nullable(schema))
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
  allocation_surface_roles: z.optional(AllocationSurfaceRolesSchema),
  regime_authority: z.optional(RegimeAuthoritySchema),
  artifact_id: z.optional(z.nullable(z.string())),
  plane: z.optional(z.nullable(z.string())),
  generator_git_sha: z.optional(z.nullable(z.string())),
  generator_git_sha_status: z.optional(z.nullable(z.string())),
  last_full_generator_git_sha: z.optional(z.nullable(z.string())),
  patch_source: z.optional(z.nullable(z.string())),
  runtime_provenance: z.optional(z.nullable(RuntimeProvenanceSchema)),
  cash: z.number(),
  total_value: z.number(),
  recent_orders: z.array(RecentOrderSchema),
  ml_signals: MLSignalsSchema,
  marl_status: MarlStatusSchema,
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

  // Active signal panels can be null when the signal is disabled/unavailable.
  behavioral_sentiment: z.optional(OptionalPanelObjectSchema),
  crypto_allocation: activePanel(CryptoAllocationSchema),
  calendar_seasonality: activePanel(CalendarSeasonalitySchema),
  ensemble_voting: activePanel(EnsembleVotingSchema),
  alternative_data: activePanel(AlternativeDataSchema),
  factor_rotation: activePanel(FactorRotationSignalSchema),
  stacking_ensemble: activePanel(StackingEnsembleSchema),
  convexity_harvest: activePanel(ConvexityHarvestSchema),
  llm_sentiment: activePanel(LLMSentimentSchema),
  sector_rotation: activePanel(SectorRotationSchema),
  factor_rotation_dashboard: activePanel(FactorRotationDashboardSchema),
  collar: activePanel(CollarSchema),
  kurtosis_regime: activePanel(KurtosisRegimeSchema),
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
  volatility_parity: activePanel(VolatilityParitySchema),
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
const AlertLevelSchema = z.enum(['success', 'warning', 'error', 'critical', 'info']);

/**
 * Tolerant producer-facing alert contract. Health-only jobs historically omit
 * presentation fields, so those fields are normalized at the fetch boundary.
 */
export const AlertWireSchema = z.object({
  level: AlertLevelSchema,
  type: z.string(),
  title: z.optional(z.string()),
  message: z.string(),
  timestamp: z.optional(z.string()),
  requires_action: z.optional(z.boolean()),
  reason: z.optional(z.string()),
  incident_id: z.optional(z.string()),
  enabled: z.optional(z.boolean()),
  channel: z.optional(z.string()),
  kill_switch_level: z.optional(z.string().nullable()),
}).passthrough();

export const AlertsWireDataSchema = z.object({
  alerts: z.array(z.unknown()),
  count: z.optional(z.number()),
  generated_at: z.optional(z.string()),
}).passthrough();

const NormalizedAlertSchema = AlertWireSchema.extend({
  type: z.string().trim().min(1),
  title: z.string().trim().min(1),
  message: z.string().trim().min(1),
  timestamp: z.optional(z.string().trim().min(1)),
  requires_action: z.boolean(),
  // Legacy in-memory fixtures predate stable_id; validateAlertsData always
  // supplies it for fetched data.
  stable_id: z.optional(z.string().trim().min(1)),
});

export const AlertsDataSchema = z.object({
  alerts: z.array(NormalizedAlertSchema),
  count: z.number(),
  generated_at: z.string(),
}).passthrough();

function alertTypeLabel(type: string): string {
  return type
    .split(/[_-]+/)
    .filter(Boolean)
    .map((part) => `${part.charAt(0).toUpperCase()}${part.slice(1)}`)
    .join(' ') || 'Dashboard Alert';
}

function alertIdentityPart(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '') || 'alert';
}

/**
 * Validate and normalize alerts without using validateFetchData's legacy raw
 * object fallback. Invalid rows are dropped together and reported once.
 */
export function validateAlertsData(raw: unknown): AlertsData | null {
  const envelope = AlertsWireDataSchema.safeParse(raw);
  if (!envelope.success) {
    if (import.meta.env.DEV) {
      console.warn('[alerts] Unusable alerts envelope:', envelope.error.issues);
    }
    return null;
  }

  const generatedAt = envelope.data.generated_at?.trim() || undefined;
  const alerts: Alert[] = [];
  let malformedRows = 0;

  for (const candidate of envelope.data.alerts) {
    const parsed = AlertWireSchema.safeParse(candidate);
    if (!parsed.success) {
      malformedRows += 1;
      continue;
    }

    const type = parsed.data.type.trim();
    const message = parsed.data.message.trim();
    if (!type || !message) {
      malformedRows += 1;
      continue;
    }

    const title = parsed.data.title?.trim() || alertTypeLabel(type);
    const timestamp = parsed.data.timestamp?.trim() || generatedAt;
    const incidentId = parsed.data.incident_id?.trim() || undefined;
    const stableId = incidentId || `${type}:${alertIdentityPart(message)}`;
    const requiresAction = parsed.data.requires_action
      ?? (type === 'kill_switch' || parsed.data.level === 'error' || parsed.data.level === 'warning');

    alerts.push({
      ...parsed.data,
      type,
      title,
      message,
      timestamp,
      requires_action: requiresAction,
      stable_id: stableId,
      incident_id: incidentId,
      reason: parsed.data.reason?.trim() || undefined,
      channel: parsed.data.channel?.trim() || undefined,
      kill_switch_level: parsed.data.kill_switch_level?.trim() || undefined,
    });
  }

  if (malformedRows > 0) {
    console.warn(`[alerts] Omitted ${malformedRows} malformed alert rows`);
  }

  return {
    alerts,
    count: alerts.length,
    generated_at: generatedAt ?? '',
  };
}

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
  status: z.enum(['ok', 'error', 'unknown', 'disabled', 'pending']),
  state: z.enum(['scheduled', 'paused', 'running', 'manual_only']),
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
  // Current health producers publish diagnostic score objects (value,
  // status, reason, and provenance), while older snapshots used scalars.
  scores: z.optional(z.record(z.string(), z.unknown())),
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
  artifact_id: z.optional(z.nullable(z.string())),
  plane: z.optional(z.nullable(z.string())),
  generator_git_sha: z.optional(z.nullable(z.string())),
  generator_git_sha_status: z.optional(z.nullable(z.string())),
  last_full_generator_git_sha: z.optional(z.nullable(z.string())),
  patch_source: z.optional(z.nullable(z.string())),
  runtime_provenance: z.optional(z.nullable(RuntimeProvenanceSchema)),
  scheduler_status: z.optional(SchedulerStatusSchema),
  data_pipeline_slo: z.optional(DataPipelineSloSchema),
  signal_health: z.optional(SignalHealthSectionSchema),
  ic_decay_summary: z.optional(IcDecaySummarySchema),
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
  portfolio_return_available: z.optional(z.boolean()),
  availability: z.optional(z.enum(['available', 'unavailable', 'partial'])),
}).passthrough();

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
  // Section-level availability: global status=success does not imply complete crisis comparison
  crisis_periods_status: z.optional(z.enum(['success', 'partial', 'unavailable'])),
  crisis_periods_reason: z.optional(z.nullable(z.string())),
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
    // Producer emits numeric value/required; panel String()-coerces for display.
    // Accept both so dual-shape graduation.json validates without fallback.
    value: z.union([z.string(), z.number()]),
    threshold: z.union([z.string(), z.number()]),
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
// RegimeGateSchema — /data/regime_gate.json (A11 pilot; live producer
// generator.py generate_regime_gate_json, extras passthrough-covered:
// confidence_source, generator_git_sha*, artifact_id, plane, runtime_provenance)
// ---------------------------------------------------------------------------
export const RegimeGateSchema = z.object({
  current_regime: z.string(),
  regime_confidence: z.number(),
  gate_rules: z.array(z.object({
    signal_name: z.string(),
    off_regimes: z.array(z.string()),
    is_active: z.boolean(),
  })),
  active_signals: z.array(z.string()),
  inactive_signals: z.array(z.string()),
  min_dwell_days: z.number(),
  generated_at: z.string(),
}).passthrough();

// ---------------------------------------------------------------------------
// TSMOMSchema — /data/tsmom.json (A11 pilot #2; speed items carry extras
// realized_vol/adjustment not in the panel interface — passthrough required)
// ---------------------------------------------------------------------------
export const TSMOMSpeedItemSchema = z.object({
  label: z.string(),
  weight: z.number(),
  signal: z.number(),
  asset_signals: z.record(z.string(), z.number()),
}).passthrough();

export const TSMOMSchema = z.object({
  composite_signal: z.number(),
  speed_breakdown: z.array(TSMOMSpeedItemSchema),
  position_recommendation: z.enum(['long', 'short', 'neutral']),
  confidence: z.number(),
  standalone_sharpe: z.number(),
  overlay_sharpe: z.number(),
  health_score: z.number(),
  is_gated_off: z.boolean(),
  generated_at: z.string(),
}).passthrough();

// ---------------------------------------------------------------------------
// ExplainabilitySchema — /data/explainability/explainability_latest.json
// (A11 extension #3; freshness carries stale_* keys only when stale —
// .partial() required; latest_decision null when no current decision)
// ---------------------------------------------------------------------------
export const ExplainabilitySchema = z.object({
  timestamp: z.string(),
  analysis_date: z.string(),
  latest_decision: z.unknown().nullable(),
  recent_decisions: z.array(z.unknown()),
  signal_deep_dives: z.record(z.string(), z.unknown()),
  top_sources_today: z.array(z.string()),
  decision_quality: z.record(z.string(), z.unknown()),
  freshness: z
    .object({
      status: z.string(),
      generated_at: z.string(),
      source_file: z.string(),
      analysis_date: z.string(),
      latest_decision_timestamp: z.unknown(),
      stale_source_file: z.unknown(),
      stale_analysis_date: z.unknown(),
    })
    .partial()
    .optional(),
}).passthrough();

// ---------------------------------------------------------------------------
// CrossAssetRVSchema — /data/cross_asset_rv.json (A11 extension #3; pair
// entries carry optional computed fields — 9 required + .passthrough())
// ---------------------------------------------------------------------------
export const CrossAssetRVPairSchema = z.object({
  pair_name: z.string(),
  symbol_a: z.string(),
  symbol_b: z.string(),
  return_a_60d: z.number(),
  return_b_60d: z.number(),
  return_differential: z.number(),
  z_score: z.number(),
  z_score_mean: z.number(),
  z_score_std: z.number(),
}).passthrough();

export const CrossAssetRVSchema = z.object({
  signal_value: z.number(),
  pairs: z.array(CrossAssetRVPairSchema),
  current_regime: z.string(),
  is_gated_off: z.boolean(),
}).passthrough();

// ---------------------------------------------------------------------------
// VixyHedgeSchema — /data/vixy_hedge.json (A11 extension #3; schema is on the
// LIVE status shape, NOT the panel interface shape — the panel normalizes
// client-side; canonical_controller/runtime_role are the live controller
// payload keys, not the panel's controller_role/runtimeStatus aliases)
// ---------------------------------------------------------------------------
export const VixyHedgeSchema = z.object({
  current_allocation_pct: z.number(),
  target_allocation_pct: z.number(),
  vix_level: z.number(),
  regime: z.string(),
  ytd_cost_bps: z.number(),
  ytd_benefit_bps: z.number(),
  hedge_efficiency: z.number(),
  total_signals: z.number(),
  last_rebalance: z.string(),
  generated_at: z.string(),
  canonical_controller: z.string(),
  runtime_role: z.string(),
  live_authoritative: z.boolean(),
  routed: z.boolean(),
}).passthrough();

// ---------------------------------------------------------------------------
// TurnoverValidatorSchema — /data/turnover_validator.json (A11 follow-up
// post-Item 29; nested turnover-validator/v1 shape: signals → per-source
// diagnostics, synthetic_baselines disclosed separately)
// ---------------------------------------------------------------------------
export const TurnoverSignalDiagnosticsSchema = z.object({
  periods: z.number(),
  mean: z.number(),
  std: z.number(),
  sign_flip_rate: z.number(),
  mag_vol: z.number(),
  turnover_penalty: z.number(),
  stability_score: z.number(),
  marginal_score: z.number(),
}).passthrough();

export const TurnoverValidatorSchema = z.object({
  schema_version: z.string(),
  signals: z.record(z.string(), TurnoverSignalDiagnosticsSchema),
  synthetic_baselines: z.record(z.string(), z.unknown()).optional(),
  generated_at: z.string(),
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
