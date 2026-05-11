// Types for live trading dashboard data

export interface SignalsData {
  timestamp: string;
  regime: {
    regime: string;
    vix: number | null;
    detected: string | null;
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
