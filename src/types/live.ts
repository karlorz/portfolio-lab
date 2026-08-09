// Types for live trading dashboard data

import type { AlternativeData } from '../components/AlternativeDataPanel';
import type { CalendarData } from '../components/CalendarSeasonalityPanel';
import type { CollarData } from '../components/CollarPanel';
import type { ConvexityHarvestData } from '../components/ConvexityHarvestPanel';
import type { CryptoData } from '../components/CryptoAllocationPanel';
import type { AllocationSurfaceRole, EnsembleVotingData } from '../components/EnsembleVotingPanel';
import type { FactorRotationDashboardData } from '../components/FactorRotationDashboardPanel';
import type { KurtosisData } from '../components/KurtosisRegimePanel';
import type { LLMSentimentData } from '../components/LLMSentimentPanel';
import type { MarlRuntimeStatusData } from '../components/MarlRuntimeStatusPanel';
import type { SectorRotationData } from '../components/SectorRotationPanel';
import type { StackingEnsembleData } from '../components/StackingEnsemblePanel';

export interface HedgeSelectorData {
  available: boolean;
  generated_at: string;
  regime: string;
  regime_confidence: number;
  primary_hedge: string;
  primary_size_pct: number;
  secondary_hedge: string | null;
  secondary_size_pct: number;
  expected_benefit_bps: number;
  expected_cost_bps: number;
  net_benefit_bps: number;
  cost_benefit_gate: boolean;
  kelly_fraction: number;
  confidence_scaled_size: number;
  min_hold_days: number;
  transition_cost_bps: number;
}

export interface FactorRotationSignalData {
  selected_factors?: string[];
  allocation?: Record<string, unknown>;
  signal_strength?: unknown;
  recommendation?: string;
  [key: string]: unknown;
}

export interface AdvancedRegimeSignalAuthority {
  role: 'advisory_shadow';
  routed: false;
  availability?: 'present' | 'unavailable' | 'stale' | 'error' | 'unknown';
  published?: boolean;
  description?: string;
  [key: string]: unknown;
}

export type IcMetricAxis =
  | 'time_series_rank_correlation'
  | 'cross_sectional_ic'
  | 'calibration_proper_score';

export type IcMetricKind = 'correlation' | 'calibration_proper_score';

export interface IcEvaluationContract {
  contract_version: 'ic-evaluation-contract/v2';
  intended_metric_axis: IcMetricAxis;
  intended_metric_kind: IcMetricKind;
  target_asset: string | null;
  target_basket: string | null;
  intended_horizon_sessions: number | null;
  prediction_field: string | null;
  prediction_transform: string | null;
}

export interface IcObservationMetadata {
  prediction_date?: string;
  realized_start_date?: string;
  resolved_date?: string;
  target_asset?: string;
  intended_horizon_sessions?: number;
  realized_horizon_sessions?: number;
  prediction_field?: string;
  prediction_transform?: string;
  metric_axis?: IcMetricAxis;
  metric_kind?: IcMetricKind;
  contract_version?: 'ic-observation-metadata/v2';
}

export interface IcDecayEvidence {
  ic_rolling?: number | null;
  observations?: number;
  status?: 'healthy' | 'warning' | 'critical' | 'insufficient_data';
  metric_axis?: IcMetricAxis;
  metric_kind?: IcMetricKind;
  estimate_kind?: 'descriptive';
  alignment_status?: 'aligned' | 'provisional' | 'misaligned' | 'ambiguous' | 'metric_mismatch' | 'undeclared';
  alignment_reason?: string;
  inference_status?: 'unavailable';
  inference_reason?:
    | 'legacy_rows_missing_alignment_metadata'
    | 'observation_metadata_incomplete'
    | 'label_alignment_mismatch'
    | 'metric_contract_mismatch'
    | 'evaluation_contract_missing'
    | 'dependence_not_characterized';
  observation_count?: number;
  observation_unit?: 'pairs' | 'dates' | 'cross_sectional_periods';
  contract_version?: 'ic-evaluation-contract/v2';
  evaluation_contract?: IcEvaluationContract;
  latest_observation_metadata?: IcObservationMetadata;
}

export interface RegimeAuthority {
  schema_version: 'regime-authority/v1';
  live_controller: 'signals.json.target_allocations';
  live_controller_module: 'src.broker.order_router';
  live_regime: string;
  allocation_regime: string;
  routed_surface: 'target_allocations';
  target_allocations: Record<string, number>;
  regime_controller?: 'classify_vix_regime';
  regime_controller_module?: 'src.utils.classify_vix_regime';
  regime_routed?: false;
  advanced_regime_signals: {
    two_stage_regime: AdvancedRegimeSignalAuthority;
    bocd_regime: AdvancedRegimeSignalAuthority;
    regime_transition: AdvancedRegimeSignalAuthority;
    [key: string]: AdvancedRegimeSignalAuthority;
  };
}

/**
 * Public `signals.json.volatility_parity` allocation shape.
 * Allocation and risk fields are **percentage points** (spy_pct: 40 = 40%,
 * target_volatility: 10 = 10% vol) — not decimal fractions.
 */
export interface VolatilityParitySignalData {
  date: string;
  /** Percentage points (10 = 10% target vol). */
  target_volatility: number;
  /** Percentage points (40 = 40% weight). */
  spy_pct: number;
  gld_pct: number;
  tlt_pct: number;
  core_vol_contribution: number;
  vix_short_pct: number;
  vix_tail_pct: number;
  vix_vol_contribution: number;
  cash_pct: number;
  expected_portfolio_vol: number;
  expected_max_dd: number;
  rebalance_triggered: boolean;
  rebalance_reason: string | null;
  [key: string]: unknown;
}

export interface SignalsData {
  timestamp: string;
  generated_at?: string;
  artifact_id?: string | null;
  plane?: string | null;
  generator_git_sha?: string | null;
  generator_git_sha_status?: string | null;
  last_full_generator_git_sha?: string | null;
  patch_source?: string | null;
  runtime_provenance?: RuntimeProvenance | null;
  regime: {
    regime: string;
    vix: number | null;
    detected: string | null;
  };
  yield_curve?: {
    spread2s10s: number | null;
    dgs2: number | null;
    dgs10: number | null;
    duration_regime: 'steep' | 'normal' | 'flat' | 'inverted' | null;
    spread_history?: number[];
    source_mode?: string;
    source_status?: string;
    source_reason?: string | null;
    source_provider?: string | null;
    source_generated_at?: string | null;
    source_latest_observation?: string | null;
  };
  duration_allocation?: {
    tlt: number;
    ief: number;
    shy: number;
    bil: number;
  };
  latest_prices: Record<string, number>;
  current_positions: Array<{
    symbol: string;
    shares: number;
    value: number;
    weight: number;
    unrealized: number;
  }>;
  target_allocations: Record<string, number>;
  regime_authority?: RegimeAuthority;
  allocation_surface_roles?: {
    schema_version: 'allocation-surface-roles/v1';
    routed_surface: string;
    routed_by: string;
    surfaces: {
      target_allocations: AllocationSurfaceRole;
      ensemble_voting: AllocationSurfaceRole;
      [key: string]: AllocationSurfaceRole;
    };
  };
  cash: number;
  total_value: number;
  recent_orders: Array<{
    sym: string;
    side: string;
    shares: number;
    value: number;
  }>;
  ml_signals: {
    available: boolean;
    timestamp: string | null;
    generated_at?: string | null;
    feature_source_artifact?: string | null;
    feature_as_of?: string | null;
    feature_freshness_status?: string | null;
    feature_staleness_days?: number | null;
    prediction_source_mode?: string | null;
    execution_role?: {
      role: string;
      routed: boolean;
      routed_by?: string | null;
      live_authoritative: boolean;
    };
    predictions: Record<string, {
      predicted_regime: string;
      confidence: number;
      probabilities: Record<string, number>;
      heuristic: boolean;
      feature_timestamp?: string | null;
      feature_freshness_status?: string | null;
      source_artifact?: string | null;
    }>;
    features: Record<string, {
      vix_level: number | null;
      trend_direction: number;
      price_vs_sma20: number;
      return_5d: number;
      spy_correlation: number;
      feature_timestamp?: string | null;
    }>;
    grid_search: {
      available: boolean;
      timestamp: string | null;
      top_allocation: Record<string, number> | null;
      sharpe: number | null;
      volatility: number | null;
      source_artifact?: string | null;
      benchmark_timestamp?: string | null;
      observation_semantics?: string | null;
      freshness_status?: string | null;
      staleness_days?: number | null;
      live_authoritative?: boolean;
    };
  };
  marl_status: MarlRuntimeStatusData;
  smart_rebalance?: SmartRebalanceData;
  broker?: BrokerData;
  closing_auction?: {
    signals: ClosingAuctionSignal[];
    last_update: string | null;
    market_open: boolean;
  };
  zero_dte?: {
    positions: ZeroDTEPosition[];
    config: ZeroDTEConfig | null;
    weekly_trades_used: number;
    total_premium_collected_mtd: number;
  };
  garch_cvar?: GarchCvarData;
  entropy?: EntropyData;
  /** Producer summary shape (canonical) or legacy overlay rows. */
  bond_momentum?: BondMomentumSummaryData | BondMomentumLegacyData | null;
  vix_term_structure?: VIXTermStructureData;
  vix_overlay?: VIXOverlayState;
  hedge_selector?: HedgeSelectorData | null;
  // Signal panel data
  behavioral_sentiment?: Record<string, unknown> | null;
  crypto_allocation?: CryptoData | null;
  calendar_seasonality?: CalendarData | null;
  ensemble_voting?: EnsembleVotingData | null;
  alternative_data?: AlternativeData | null;
  factor_rotation?: FactorRotationSignalData | null;
  stacking_ensemble?: StackingEnsembleData | null;
  convexity_harvest?: ConvexityHarvestData | null;
  llm_sentiment?: LLMSentimentData | null;
  sector_rotation?: SectorRotationData | null;
  factor_rotation_dashboard?: FactorRotationDashboardData | null;
  collar?: CollarData | null;
  kurtosis_regime?: KurtosisData | null;
  volatility_parity?: VolatilityParitySignalData | null;
  // Rebalance health
  rebalance_health?: Record<string, unknown>;
  // Circuit breaker state
  broker_circuit_breaker?: {
    state: 'closed' | 'open' | 'half-open';
    fail_count: number;
    reset_timeout: number;
  };
  // Risk decomposition
  risk_decomposition?: Record<string, unknown>;
  // SPC monitoring
  spc_flags?: Record<string, unknown>;
  // Staleness info
  staleness?: Record<string, unknown>;
  // FRED-MD macro regime signal
  fred_macro?: {
    regime: string;
    confidence: number;
    recession_probability: number;
    inflation_pressure: number;
    monetary_stance: string;
    manufacturing_health: number;
    credit_conditions: string;
    indicators: Record<string, number>;
    timestamp: string;
    status?: string;
    source_mode?: string;
    cache_status?: string;
    api_key_configured?: boolean;
    reason?: string | null;
    latest_fetched_at?: string | null;
    row_count?: number | null;
    age_hours?: number | null;
    ttl_hours?: number | null;
    indicators_observed?: boolean;
  };
  // Two-stage k-means macro regime classifier (Oliveira et al. 2025)
  two_stage_regime?: {
    regime: string;
    confidence: number;
    crisis_probability: number;
    probabilities: Record<string, number>;
    n_pca_components: number;
    variance_retained: number;
    n_observations: number;
    n_series: number;
    method: string;
    timestamp: string;
  };
  // Bayesian Online Changepoint Detection (Adams & MacKay 2007)
  bocd_regime?: {
    regime: number;
    regime_change_prob: number;
    changepoint_count: number;
    current_run_length: number;
    hazard_rate: number;
    threshold: number;
    n_observations: number;
    description: string;
    timestamp: string;
  };
  // IC decay monitoring for signal quality tracking
  ic_decay?: {
    status?: 'healthy' | 'warning' | 'critical' | 'insufficient_resolved_history' | 'waiting_for_forward_returns' | 'no_data';
    signals?: Record<string, {
      ic_rolling: number | null;
      ic_trend: 'stable' | 'decaying' | 'improving' | 'unknown';
      observations: number;
      status: 'healthy' | 'warning' | 'critical' | 'insufficient_data';
      min_obs_for_status?: number;
    } & IcDecayEvidence>;
    resolved_signal_count?: number;
    pending_predictions?: number;
    staged_prediction_names?: string[];
    staged_date?: string | null;
    label_horizon?: string;
    error?: string;
  };
  // Per-signal walk-forward validation
  signal_wfe?: {
    status?: 'validated' | 'weak' | 'unvalidated' | 'insufficient_data' | 'insufficient_resolved_history' | 'waiting_for_forward_returns' | 'no_data';
    signals?: Record<string, {
      signal_name: string;
      wfe: number;
      mean_is_ic: number;
      mean_oos_ic: number;
      std_oos_ic: number;
      n_windows: number;
      positive_oos_ratio: number;
      status: 'validated' | 'weak' | 'unvalidated' | 'insufficient_data';
    }>;
    resolved_signal_count?: number;
    pending_predictions?: number;
    staged_date?: string | null;
    label_horizon?: string;
    error?: string;
  };
}

export interface Position {
  symbol: string;
  shares: number;
  value: number;
  weight: number;
  unrealized: number;
}

export interface RecentOrder {
  sym: string;
  side: string;
  shares: number;
  value: number;
}

export interface PerformanceEntry {
  t: string;  // date
  v: number;  // total value
  r: number;  // daily return
}

export interface Alert {
  level: 'success' | 'warning' | 'error' | 'critical' | 'info';
  type: string;
  title: string;
  message: string;
  timestamp?: string;
  requires_action: boolean;
  /** Stable presentation identity, normalized from incident_id or type/message. */
  stable_id?: string;
  /** Producer lifecycle identity when one is available. */
  incident_id?: string;
  reason?: string;
  enabled?: boolean;
  channel?: string;
  kill_switch_level?: string | null;
}

export type IncidentLifecycleState = 'firing' | 'acknowledged' | 'resolving' | 'resolved';

export interface IncidentLifecycleIncident {
  incident_id: string;
  channel: string;
  severity: string;
  state: IncidentLifecycleState;
  message: string;
  details: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  resolved_at: string | null;
  resolution_notes: string | null;
  mttr_seconds: number | null;
}

export interface IncidentLifecycleMetrics {
  incident_frequency: number;
  open_count: number;
  resolved_count: number;
  mean_mttr_seconds: number | null;
}

export interface IncidentLifecycleSummary {
  generated_at: string;
  open_count: number;
  incidents: IncidentLifecycleIncident[];
  metrics: IncidentLifecycleMetrics;
}

export interface AssetStat {
  '30d_return': number;
  volatility: number;
  current: number;
}

export interface KillSwitchHealth {
  status?: string;
  enabled?: boolean;
  level?: string | null;
  reason?: string | null;
  source?: string | null;
  message?: string | null;
  timestamp?: string | null;
  incident_id?: string | null;
  mode?: string | null;
  channel?: string | null;
}

export interface OpenIncidentsHealth {
  status?: string;
  open_count?: number;
  incidents?: Array<Record<string, unknown>>;
}

/** Dual-write provenance_completeness block (Batch AS+; advisory split-brain forensics). */
export interface ProvenanceCompleteness {
  generator_git_sha_present?: boolean;
  dual_write_attempted?: boolean;
  dual_write_ok?: boolean | null;
  dual_write_lag_seconds?: number | null;
  dual_write_lag_stale?: boolean;
  dual_write_lag_threshold_seconds?: number;
  dual_write_lag_unit?: string;
  paths_identical?: boolean | null;
  private_path?: string | null;
  public_path?: string | null;
  note?: string;
  disclosure?: string;
  [key: string]: unknown;
}

/** Additive runtime identity shared by public operator artifacts. */
export interface RuntimeProvenance {
  schema_version?: string;
  artifact_id?: string | null;
  plane?: 'public' | 'private' | 'unknown' | string | null;
  generated_at?: string | null;
  generator_git_sha?: string | null;
  generator_git_sha_status?: string | null;
  last_full_generator_git_sha?: string | null;
  patch_source?: string | null;
  [key: string]: unknown;
}

export interface HealthData {
  cron_jobs: CronJobStatus[];
  data_freshness: Record<string, DataFreshness>;
  system_status: 'healthy' | 'warning' | 'critical' | 'degraded';
  generated_at: string;
  scheduler_status?: SchedulerStatus;
  data_pipeline_slo?: DataPipelineSlo;
  ic_decay_summary?: IcDecaySummary;
  kill_switch?: KillSwitchHealth;
  open_incidents?: OpenIncidentsHealth;
  /** Code tip that produced this artifact (when stamped). */
  generator_git_sha?: string | null;
  generator_git_sha_status?: string | null;
  artifact_id?: string | null;
  plane?: string | null;
  runtime_provenance?: RuntimeProvenance | null;
  /** Dual-write lag / split-brain forensics (health_ops or dual-write producers). */
  provenance_completeness?: ProvenanceCompleteness | null;
  error?: string;
}

export interface IcDecaySummary {
  status:
    | 'healthy'
    | 'warning'
    | 'critical'
    | 'insufficient_resolved_history'
    | 'waiting_for_forward_returns'
    | 'no_data'
    | 'unknown';
  critical_signals: string[];
  warning_signals: string[];
  insufficient_data_signals?: string[];
  resolved_signal_count: number;
  min_observations: number;
  staged_pending_predictions: number;
  staged_pending_signal_names?: string[];
  staged_date: string | null;
  staged_pending_scope: string;
  historical_unlabeled_rows: number;
  historical_unlabeled_dates: number;
  historical_unlabeled_oldest_date?: string | null;
  historical_unlabeled_scope: string;
  evidence_generated_at?: string | null;
  evidence_freshness: string;
  routing_authority: string;
  routing_control: string;
  control_effect: string;
  kill_switch_level?: string | null;
  signal_evidence?: Record<string, IcDecayEvidence>;
}

export interface DataPipelineSlo {
  schema_version: string;
  status: 'ok' | 'warning' | 'critical' | 'unknown';
  top_dimension: string | null;
  dimensions: Record<string, DataPipelineSloDimension>;
  runbook?: DataPipelineRunbook;
  error?: string;
}

export interface DataPipelineRunbook {
  status: 'ok' | 'warning' | 'critical' | 'unknown';
  top_cause: DataPipelineRunbookAction | null;
  actions: DataPipelineRunbookAction[];
}

export interface DataPipelineRunbookAction {
  dimension: string;
  code: string;
  severity: 'ok' | 'warning' | 'critical' | 'unknown';
  action: string;
  artifact?: string;
  provider?: string;
  reason?: string;
  [key: string]: unknown;
}

export interface DataPipelineSloDimension {
  status: 'ok' | 'warning' | 'critical' | 'unknown';
  message?: string;
  [key: string]: unknown;
}

export interface CronJobStatus {
  id: string;
  name: string;
  schedule: string;
  last_run: string | null;
  next_run: string | null;
  status: 'ok' | 'error' | 'unknown' | 'disabled';
  state: 'scheduled' | 'paused' | 'running' | 'manual_only';
  backend?: string;
  source?: string;
  error?: string;
  duration_seconds?: number | null;
}

export interface SchedulerStatus {
  status: 'ok' | 'degraded' | 'warning' | 'unavailable' | 'unknown';
  backends: Record<string, SchedulerBackendStatus>;
}

export interface SchedulerBackendStatus {
  backend: string;
  status: 'ok' | 'degraded' | 'warning' | 'unavailable' | 'error' | 'unknown';
  source: string;
  total_jobs: number;
  failed_jobs: number;
  reason?: string;
}

export interface DataFreshness {
  last_update: string;
  days_stale: number;
  market_lag_days?: number;
  latest_available_market_date?: string | null;
  status: 'fresh' | 'stale' | 'critical';
}

export interface RegimeEntry {
  d: string;
  r: string;
  v: number | null;
}

export interface DashboardData {
  prices: Record<string, Array<{ d: string; p: number }>>;
  regimes: RegimeEntry[];
  paper_portfolio: PerformanceEntry[];
  generated_at: string;
}

export interface StatsData {
  asset_stats: Record<string, AssetStat>;
  paper_portfolio: {
    sharpe: number;
    total_return: number;
    max_value: number;
    min_value: number;
    days_tracked: number;
  } | null;
  spy_comparison?: {
    portfolio_value: number;
    spy_value: number;
    relative_return: number;
    correlation_30d: number;
    beta: number;
    outperformance: number;
  } | null;
  generated_at: string;
}

// Analytics Types (v2.5)
export interface AnalyticsData {
  status: 'success' | 'no_data' | 'error';
  message?: string;
  generated_at: string;
  data_points: number;
  date_range: {
    start: string | null;
    end: string | null;
  };
  drawdown: {
    series: DrawdownPoint[];
    max_drawdown: MaxDrawdownData;
  };
  rolling_metrics: {
    sharpe_63d: RollingMetricPoint[];
    sharpe_126d: RollingMetricPoint[];
    sharpe_252d: RollingMetricPoint[];
  };
  benchmark_comparison: {
    portfolio: PortfolioBenchmarkData;
  };
  crisis_periods: CrisisPeriodData[];
  /** Section-level: global status=success does not imply complete crisis comparison. */
  crisis_periods_status?: 'success' | 'partial' | 'unavailable';
  crisis_periods_reason?: string | null;
}

export interface DrawdownPoint {
  date: string;
  value: number;
  peak: number;
  drawdown: number;
  days_since_peak: number;
  is_recovery: boolean;
}

export interface MaxDrawdownData {
  max_drawdown: number;
  max_drawdown_date: string;
  recovery_date: string | null;
  underwater_days: number;
  peak_value: number;
  trough_value: number;
}

export interface RollingMetricPoint {
  date: string;
  sharpe: number;
  volatility: number;
  mean_return: number;
  window_days: number;
}

export interface PortfolioBenchmarkData {
  start_date: string;
  end_date: string;
  start_value: number;
  end_value: number;
  total_return: number;
  cagr: number | null;
  volatility: number;
  max_drawdown: number;
  sharpe: number | null;
}

export interface CrisisPeriodData {
  name: string;
  period: string;
  description: string;
  spy_return: number;
  portfolio_return: number | null;
  portfolio_return_available?: boolean;
  availability?: 'available' | 'unavailable' | 'partial';
}

export interface AlertsData {
  alerts: Alert[];
  count: number;
  generated_at: string;
}

// Broker Integration Types (v2.3 Phase 4)
export interface BrokerData {
  connected: boolean;
  positions: Array<{
    symbol: string;
    qty: number;
    market_value: number;
    unrealized_pl: number;
    side: string;
  }>;
  drift: Array<{
    symbol: string;
    broker_qty: number;
    local_qty: number;
    drift_pct: number;
  }>;
  recent_orders: Array<{
    symbol: string;
    side: string;
    qty: number;
    status: string;
    order_id?: string;
    timestamp: string;
    dry_run: boolean;
    attempts?: number;
  }>;
  last_sync: string | null;
  kill_switch: boolean;
  kill_switch_level?: string | null;
  kill_switch_source?: string | null;
  kill_switch_reason?: string | null;
  kill_switch_incident_id?: string | null;
}

// Smart Rebalance Types (v2.90)
export interface SmartRebalanceData {
  should_execute: boolean;
  decision: string;
  urgency: 'low' | 'moderate' | 'high' | 'emergency';
  max_drift: number;
  estimated_cost_bps: number;
  reason: string;
  drift_details: Record<string, number>;
  vpin: number;
  in_optimal_window: boolean;
  ytd_cost_bps: number;
  remaining_budget_pct: number;
  remaining_budget_ratio?: number;
  /** Kill authority blocks actionable rebalance (mirrors order_router). */
  execution_blocked?: boolean;
  kill_switch_enabled?: boolean;
  kill_switch_level?: string | null;
  kill_switch_reason?: string | null;
  kill_switch_incident_id?: string | null;
  kill_switch_message?: string | null;
  status: {
    ytd_cost_bps: number;
    ytd_cost_pct: number;
    remaining_budget_pct: number;
    remaining_budget_ratio?: number;
    is_over_budget: boolean;
    is_warning: boolean;
    last_rebalance: string | null;
    deferred_until: string | null;
    config: {
      drift_threshold: number;
      vpin_threshold: number;
      optimal_window: string;
      annual_cost_limit: string;
    };
  };
}

// 0DTE Options Types (v3.12)
export interface ZeroDTEConfig {
  max_portfolio_allocation: number;  // e.g., 0.02 = 2%
  max_weekly_positions: number;      // e.g., 2
  position_size_pct: number;          // e.g., 0.005 = 0.5%
  min_vix: number;                    // e.g., 15
  max_vix: number;                    // e.g., 35
  delta_target: number;               // e.g., 0.30
  min_premium_pct: number;            // e.g., 0.004 = 0.4%
  max_delta_exposure: number;         // e.g., 0.08 = 8%
  emergency_close_delta: number;      // e.g., 0.50
  max_loss_pct: number;               // e.g., 0.015 = 1.5%
}

export interface ZeroDTETrade {
  id: string;
  underlying: string;
  option_type: 'call' | 'put';
  side: 'buy' | 'sell';
  quantity: number;
  strike: number;
  expiration: string;
  entry_price: number;
  entry_time: string;
  exit_price?: number;
  exit_time?: string;
  premium_collected: number;
  realized_pnl?: number;
}

export interface ZeroDTEPosition {
  id: string;
  underlying: string;
  option_type: 'call' | 'put';
  side: 'buy' | 'sell';
  strike: number;
  expiration: string;
  quantity: number;
  entry_price: number;
  entry_time: string;
  entry_delta: number;
  entry_theta: number;
  current_delta: number;
  current_theta: number;
  current_underlying_price: number;
  status: 'pending' | 'open' | 'closed' | 'stopped' | 'expired_itm' | 'expired_otm' | 'rolled';
  unrealized_pnl?: number;
  realized_pnl?: number;
  premium_collected: number;
  delta_exposure: number;
  notional_value: number;
  close_reason?: 'expiration' | 'profit_take' | 'stop_loss' | 'delta_stop' | 'time_exit' | 'manual' | 'roll' | 'emergency';
}

// GARCH-CVaR Types (v3.21)
export interface GarchCvarData {
  cvar_95: number;
  cvar_95_garch: number;
  var_95: number;
  var_95_garch: number;
  cvar_ratio: number;
  garch_active: boolean;
  current_volatility: number;
  forecast_volatility: number;
  volatility_clustering: 'low' | 'normal' | 'elevated' | 'high';
  // Conformal CVaR cross-check (distribution-free)
  conformal_cvar_95?: number | null;
  conformal_var_95?: number | null;
  conformal_cvar_ratio?: number | null;
  coverage_diagnostics?: ConformalCoverageDiagnostics | null;
}

export interface ConformalCoverageDiagnostics {
  schema_version: 'conformal-coverage/v1';
  observations: number;
  alpha: number;
  expected_exceedance_rate: number;
  exceedance_count: number;
  exceedance_rate: number;
  coverage_rate: number;
  coverage_pass: boolean;
  rolling_window: number;
  rolling_exceedance_rate: number;
  longest_violation_cluster: number;
  kupiec_statistic: number;
  kupiec_p_value: number;
  kupiec_pass: boolean;
  christoffersen_statistic: number;
  christoffersen_p_value: number;
  christoffersen_pass: boolean;
  conditional_coverage_statistic: number;
  conditional_coverage_p_value: number;
  conditional_coverage_pass: boolean;
  by_regime?: Record<string, Record<string, unknown>>;
}

// Entropy Monitor Types (v3.22)
export interface EntropyData {
  shannon_entropy: number;
  effective_n: number;
  max_possible: number;
  normalized_score: number;
  concentration_risk: 'critical' | 'high' | 'medium' | 'low' | 'good';
  hhi_index: number;
  correlation_entropy?: number;
  participation_ratio?: number;
}

// Closing Auction Types (v3.17)
export interface MOCImbalance {
  symbol: string;
  timestamp: string;
  imbalance_shares: number;
  paired_shares: number;
  reference_price: number;
  source: string;
  imbalance_ratio: number;
  direction_score: number;
}

export interface ClosingAuctionSignal {
  symbol: string;
  timestamp: string;
  direction: 'STRONG_BUY' | 'BUY' | 'WEAK_BUY' | 'NEUTRAL' | 'WEAK_SELL' | 'SELL' | 'STRONG_SELL';
  direction_score: number;
  confidence: 'high' | 'medium' | 'low' | 'insufficient_data';
  imbalance: MOCImbalance;
  entry_price: number;
  target_exit_price: number;
  stop_loss_price?: number;
  historical_win_rate: number | null;
  historical_count: number;
  max_position_pct: number;
  urgency: 'immediate' | 'high' | 'normal';
  should_trade: boolean;
}

// Bond Momentum Types (v3.30 legacy overlay + current duration summary)
export interface BondMomentumSignal {
  etf: string;
  timestamp: string;
  signal: number;
  position_size: number;
  formation_return: number;
  realized_vol: number;
  formation_months: number;
  volatility_target: number;
  confidence: 'strong' | 'moderate' | 'weak';
  action: 'increase' | 'hold' | 'reduce' | 'avoid';
  weight_delta: number;
}

export interface BondMomentumEnsemble {
  weight: number;
  confidence: string;
  action: string;
  recommendation: string;
}

/** Legacy TSMOM per-ETF overlay rows (optional consumer shape). */
export interface BondMomentumLegacyData {
  signals: BondMomentumSignal[];
  timestamp?: string;
  ensemble?: BondMomentumEnsemble;
}

/**
 * Canonical public producer shape from DashboardGenerator bond_duration /
 * overlay bond_momentum summary (no signals[] rows).
 */
export interface BondMomentumSummaryData {
  active: boolean;
  yield_10y: number;
  yield_2y: number;
  spread: number;
  curve_regime: string;
  rate_direction: string;
  tlt_weight: number;
  ief_weight: number;
  shy_weight: number;
  effective_duration: number;
  position: string;
  confidence: number;
  status_text: string;
  generated_at?: string;
  timestamp?: string;
}

// VIX Term Structure Types (v4.50)
export interface VIXTermStructureLevel {
  value: number;
  timestamp: string;
}

export interface VIXTermStructureData {
  vix: VIXTermStructureLevel;
  vix3m: VIXTermStructureLevel;
  vix6m?: VIXTermStructureLevel;
  slope: number;
  roll_yield: number;
  composite_signal: number;
  regime: 'extreme_contango' | 'steep_contango' | 'mild_contango' | 'flat' | 'backwardation' | 'extreme_backwardation';
  z_score: number;
  percentile_1y?: number;
}

export interface VIXOverlayShift {
  date: string;
  shifts: Record<string, number>;
  signal_value: number;
  regime: string;
  new_allocation: Record<string, number>;
}

export interface VIXOverlayState {
  allocation: Record<string, number>;
  last_shift_date: string;
  shift_history: VIXOverlayShift[];
  disabled_until: string | null;
}

// Chat Types (v7.10)
export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

export interface ChatQuery {
  question: string;
  answer: string;
  timestamp: string;
  fallback: boolean;
}

export interface ChatSuggestion {
  label: string;
  query: string;
  category: 'portfolio' | 'risk' | 'signals' | 'overlays' | 'costs';
}
