// Types for live trading dashboard data

export interface SignalsData {
  regime: {
    regime: string;
    vix: number | null;
    detected: string | null;
  };
  latest_prices: Record<string, number>;
  portfolio: {
    total_value: number;
    cash: number;
    positions: Position[];
  };
  target_allocation: Record<string, number>;
  recent_orders: RecentOrder[];
  generated_at: string;
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
    outperformance: number;  // cumulative
  } | null;
  generated_at: string;
}

export interface AlertsData {
  alerts: Alert[];
  count: number;
  generated_at: string;
}
