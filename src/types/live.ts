// Types for live trading dashboard data

export interface SignalsData {
  timestamp: string;
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
    predictions: Record<string, {
      predicted_regime: string;
      confidence: number;
      probabilities: Record<string, number>;
      heuristic: boolean;
    }>;
    features: Record<string, {
      vix_level: number | null;
      trend_direction: number;
      price_vs_sma20: number;
      return_5d: number;
      spy_correlation: number;
    }>;
    grid_search: {
      available: boolean;
      timestamp: string | null;
      top_allocation: Record<string, number> | null;
      sharpe: number | null;
      volatility: number | null;
    };
  };
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
  level: 'success' | 'warning' | 'error' | 'info';
  type: string;
  title: string;
  message: string;
  timestamp?: string;
  requires_action: boolean;
}

export interface AssetStat {
  '30d_return': number;
  volatility: number;
  current: number;
}

export interface HealthData {
  cron_jobs: CronJobStatus[];
  data_freshness: Record<string, DataFreshness>;
  system_status: 'healthy' | 'warning' | 'critical' | 'degraded';
  generated_at: string;
  error?: string;
}

export interface CronJobStatus {
  id: string;
  name: string;
  schedule: string;
  last_run: string | null;
  next_run: string | null;
  status: 'ok' | 'error' | 'unknown';
  state: 'scheduled' | 'paused' | 'running';
}

export interface DataFreshness {
  last_update: string;
  days_stale: number;
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
  status: {
    ytd_cost_bps: number;
    ytd_cost_pct: number;
    remaining_budget_pct: number;
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
